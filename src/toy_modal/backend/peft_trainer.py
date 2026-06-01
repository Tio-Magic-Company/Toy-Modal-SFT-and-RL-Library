"""PEFT-backed trainer engine behind the TrainerEngine worker interface."""

from __future__ import annotations

import base64
from contextlib import nullcontext
import json
import os
from pathlib import Path
from typing import Any

from toy_modal import types
from toy_modal.backend.lora_mapping import build_peft_lora_config, lora_mapping_manifest
from toy_modal.backend.loss_inputs import (
    DPOBatchItem,
    ImportanceSamplingBatchItem,
    SupervisedBatchItem,
    prepare_dpo_batch_items,
    prepare_importance_sampling_batch_items,
    prepare_supervised_batch_items,
    validate_training_batch,
)
from toy_modal.backend.losses import (
    cispo_loss,
    dpo_loss,
    importance_sampling_loss,
    ppo_loss,
    token_logprobs_from_logits,
    weighted_cross_entropy_loss,
)
from toy_modal.backend.storage import ArtifactStore
from toy_modal.backend.trainer_worker import _expires_at_from_ttl, _now
from toy_modal.errors import (
    BackendUnavailableError,
    BadRequestError,
    CheckpointNotFoundError,
    StaleModelSequenceError,
)
from toy_modal.paths import build_toy_path, parse_toy_path


class PeftTrainerEngine:
    """Real single-process LoRA SFT/RL trainer using Transformers and PEFT.

    This class intentionally mirrors ``TrainerEngine``'s public worker methods
    so Modal routing can swap between the deterministic tiny backend and this
    PEFT backend without changing SDK clients.
    """

    backend_name = "peft"

    @classmethod
    def load_or_initialize(
        cls,
        run_id: str,
        registry,
        model_root: str,
        run_root: str,
    ) -> "PeftTrainerEngine":
        return cls(run_id=run_id, registry=registry, model_root=model_root, run_root=run_root)

    def __init__(
        self,
        *,
        run_id: str,
        registry=None,
        model_root: str = "/models",
        run_root: str = "/runs",
    ) -> None:
        self.run_id = run_id
        self.registry = registry if registry is not None else {}
        self.model_root = Path(model_root)
        self.run_root = Path(run_root)
        self.store = ArtifactStore.from_runs_root(self.run_root)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.optimizer_config: dict[str, Any] | None = None
        self.device = None
        self._loaded_adapter_dir: Path | None = None

    def forward(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._loss_response(payload, with_gradient=False)

    def forward_backward(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._loss_response(payload, with_gradient=True)

    def optim_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = self._record()
        self._ensure_expected_sequence(record, payload.get("expected_model_seq_id"))
        if not record.get("latest_gradient_id"):
            raise BadRequestError("optim_step requires a prior forward_backward gradient")

        adam = types.AdamParams.model_validate(payload["adam_params"])
        self._ensure_model_loaded()
        torch, _, _, _ = _backend_deps()
        trainable_params = self._trainable_params()
        if not trainable_params:
            raise BadRequestError("PEFT model has no trainable parameters")
        self._ensure_optimizer(adam)

        grad_norm = 0.0
        if adam.grad_clip_norm and adam.grad_clip_norm > 0:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(trainable_params, adam.grad_clip_norm)
            grad_norm = float(grad_norm_tensor.detach().cpu())

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        record["model_seq_id"] += 1
        record["optimizer_step"] += 1
        record["latest_gradient_id"] = None
        record["updated_at"] = _now()
        record["optimizer_state"] = {
            "step": record["optimizer_step"],
            **self.optimizer_config,
        }
        self._save_record(record)
        return types.OptimStepResponse(
            model_seq_id=record["model_seq_id"],
            optimizer_step=record["optimizer_step"],
            metrics={"grad_norm": grad_norm, "trainable_params": float(len(trainable_params))},
        ).model_dump(mode="json")

    def save_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._save_checkpoint(payload, checkpoint_type="training")

    def load_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_toy_path(
            payload["path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        manifest_path = self.store.layout.artifact_manifest_path(
            parsed.project_id,
            parsed.run_id,
            parsed.artifact_type,
            parsed.name,
        )
        if not manifest_path.exists():
            raise CheckpointNotFoundError(payload["path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.store.raise_if_manifest_expired(manifest)
        self.store.validate_manifest_files(manifest)

        adapter_dir = manifest_path.parent / manifest.get("adapter_path", "adapter")
        if not adapter_dir.exists():
            raise CheckpointNotFoundError(f"adapter directory missing for {payload['path']}")

        record = self._record()
        record["base_model"] = manifest["base_model"]
        record["lora_config"] = manifest["lora_config"]
        record["lora_rank"] = manifest["lora_config"].get("rank")
        record["model_seq_id"] = manifest["model_seq_id"]
        record["optimizer_step"] = manifest["optimizer_step"] if payload.get("optimizer") else 0
        record["latest_gradient_id"] = None
        record["updated_at"] = _now()
        self._save_record(record)

        self._reset_loaded_state()
        self._ensure_model_loaded(adapter_dir=adapter_dir)

        if payload.get("optimizer"):
            self._load_optimizer_state(manifest_path.parent, manifest)
        else:
            self.optimizer = None
            self.optimizer_config = None

        self._save_record(record)
        return types.LoadWeightsResponse(
            path=payload["path"],
            training_run_id=self.run_id,
            model_seq_id=record["model_seq_id"],
            optimizer_step=record["optimizer_step"],
            lora_config=types.LoraConfig.model_validate(record["lora_config"]),
        ).model_dump(mode="json")

    def save_weights_for_sampler(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._save_checkpoint(payload, checkpoint_type="sampler")

    def validate_old_logprobs_sequence(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data", [])
        loss_fn = payload.get("loss_fn", "importance_sampling")
        validate_training_batch(data, loss_fn)
        record = self._record()
        try:
            self._ensure_old_logprobs_sequence(record, data)
        except StaleModelSequenceError as exc:
            return {
                "accepted": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "model_seq_id": record["model_seq_id"],
            }
        return {
            "accepted": True,
            "error_type": None,
            "error": None,
            "model_seq_id": record["model_seq_id"],
        }

    def trainable_parameter_count(self) -> tuple[int, int]:
        self._ensure_model_loaded()
        trainable = 0
        total = 0
        for parameter in self.model.parameters():
            count = parameter.numel()
            total += count
            if parameter.requires_grad:
                trainable += count
        return trainable, total

    def completion_logprobs(self, data: list[Any]) -> list[list[float]]:
        items = prepare_supervised_batch_items(data)
        self._ensure_model_loaded()
        tensors = self._tensorize_supervised(items)
        self.model.eval()
        torch, _, _, _ = _backend_deps()
        with torch.no_grad():
            logits = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits
        logprobs, mask = token_logprobs_from_logits(logits, tensors["labels"])
        return _unpadded_rows(logprobs, mask)

    def _loss_response(self, payload: dict[str, Any], *, with_gradient: bool) -> dict[str, Any]:
        record = self._record()
        self._ensure_expected_sequence(record, payload.get("expected_model_seq_id"))
        loss_fn = payload.get("loss_fn")
        data = payload.get("data", [])
        self._ensure_old_logprobs_sequence(record, data)
        self._ensure_model_loaded()

        if loss_fn == "cross_entropy":
            loss, metrics, outputs, num_tokens = self._cross_entropy(data, with_gradient=with_gradient)
            output_type = "CrossEntropyLossReturn"
        elif loss_fn == "dpo":
            loss, metrics, outputs, num_tokens = self._dpo_loss(
                data,
                loss_fn_config=payload.get("loss_fn_config") or {},
                with_gradient=with_gradient,
            )
            output_type = "DPOLossReturn"
        elif loss_fn in {"importance_sampling", "ppo", "cispo"}:
            loss, metrics, outputs, num_tokens = self._rl_loss(
                data,
                loss_fn=loss_fn,
                loss_fn_config=payload.get("loss_fn_config") or {},
                with_gradient=with_gradient,
            )
            output_type = _rl_output_type(loss_fn)
        else:
            raise BadRequestError(f"unsupported loss_fn: {loss_fn!r}")

        gradient_id = None
        if with_gradient:
            gradient_id = f"grad_{record['training_run_id']}_{record['model_seq_id']}"
            record["latest_gradient_id"] = gradient_id
        record["last_request_time"] = _now()
        self._save_record(record)
        return types.ForwardBackwardOutput(
            loss=float(loss.detach().cpu()),
            loss_fn_output_type=output_type,
            loss_fn_outputs=outputs,
            metrics=metrics,
            num_tokens=num_tokens,
            gradient_id=gradient_id,
            model_seq_id=record["model_seq_id"],
        ).model_dump(mode="json")

    def _cross_entropy(
        self,
        data: list[Any],
        *,
        with_gradient: bool,
    ) -> tuple[Any, dict[str, float], dict[str, Any], int]:
        items = prepare_supervised_batch_items(data)
        tensors = self._tensorize_supervised(items)
        context = nullcontext() if with_gradient else self._torch_no_grad()
        self.model.train(mode=with_gradient)
        if with_gradient:
            self.model.zero_grad(set_to_none=True)
        with context:
            logits = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits
            loss, metrics = weighted_cross_entropy_loss(logits, tensors["labels"], tensors["weights"])
        if with_gradient:
            loss.backward()
        outputs = {
            "loss": types.TensorData(data=[float(loss.detach().cpu())], dtype="float32"),
        }
        return loss, metrics, outputs, sum(len(item.tokens) for item in items)

    def _rl_loss(
        self,
        data: list[Any],
        *,
        loss_fn: str,
        loss_fn_config: dict[str, float],
        with_gradient: bool,
    ) -> tuple[Any, dict[str, float], dict[str, Any], int]:
        torch, _, _, _ = _backend_deps()
        items = prepare_importance_sampling_batch_items(data)
        tensors = self._tensorize_importance_sampling(items)
        context = nullcontext() if with_gradient else self._torch_no_grad()
        self.model.train(mode=with_gradient)
        if with_gradient:
            self.model.zero_grad(set_to_none=True)
        with context:
            logits = self.model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
            ).logits
            selected_logprobs, label_mask = token_logprobs_from_logits(logits, tensors["labels"])
            new_logprobs = _pad_selected_rows(
                selected_logprobs,
                label_mask,
                width=tensors["old_logprobs"].shape[1],
                device=self.device,
            )
            common_kwargs = {
                "new_logprobs": new_logprobs,
                "old_logprobs": tensors["old_logprobs"],
                "advantages": tensors["advantages"],
                "weights": tensors["weights"],
                "masks": tensors["masks"],
            }
            if loss_fn == "importance_sampling":
                loss, metrics = importance_sampling_loss(
                    **common_kwargs,
                    ref_logprobs=tensors.get("ref_logprobs"),
                    kl_coef=float(loss_fn_config.get("kl_coef", 0.0)),
                )
            elif loss_fn == "ppo":
                loss, metrics = ppo_loss(
                    **common_kwargs,
                    clip_low_threshold=float(loss_fn_config.get("clip_low_threshold", 0.8)),
                    clip_high_threshold=float(loss_fn_config.get("clip_high_threshold", 1.2)),
                )
            elif loss_fn == "cispo":
                loss, metrics = cispo_loss(
                    **common_kwargs,
                    clip_low_threshold=float(loss_fn_config.get("clip_low_threshold", 0.8)),
                    clip_high_threshold=float(loss_fn_config.get("clip_high_threshold", 1.2)),
                )
            else:
                raise BadRequestError(f"unsupported loss_fn: {loss_fn!r}")
        if with_gradient:
            loss.backward()
        outputs = {
            "loss": types.TensorData(data=[float(loss.detach().cpu())], dtype="float32"),
            "new_logprobs": types.TensorData(
                data=new_logprobs.detach().cpu().reshape(-1).tolist(),
                dtype="float32",
                shape=tuple(int(dim) for dim in new_logprobs.shape),
            ),
        }
        metrics["completion_tokens"] = float(sum(len(item.completion_tokens) for item in items))
        return loss, metrics, outputs, sum(len(item.tokens) for item in items)

    def _dpo_loss(
        self,
        data: list[Any],
        *,
        loss_fn_config: dict[str, float],
        with_gradient: bool,
    ) -> tuple[Any, dict[str, float], dict[str, Any], int]:
        torch, _, _, _ = _backend_deps()
        items = prepare_dpo_batch_items(data)
        chosen = self._tensorize_dpo_side(items, side="chosen")
        rejected = self._tensorize_dpo_side(items, side="rejected")
        context = nullcontext() if with_gradient else self._torch_no_grad()
        self.model.train(mode=with_gradient)
        if with_gradient:
            self.model.zero_grad(set_to_none=True)
        with context:
            chosen_logits = self.model(
                input_ids=chosen["input_ids"],
                attention_mask=chosen["attention_mask"],
            ).logits
            rejected_logits = self.model(
                input_ids=rejected["input_ids"],
                attention_mask=rejected["attention_mask"],
            ).logits
            chosen_logprobs, chosen_mask = token_logprobs_from_logits(chosen_logits, chosen["labels"])
            rejected_logprobs, rejected_mask = token_logprobs_from_logits(rejected_logits, rejected["labels"])
            chosen_sequence_logprobs = (chosen_logprobs * chosen_mask.to(chosen_logprobs.dtype)).sum(dim=1)
            rejected_sequence_logprobs = (
                rejected_logprobs * rejected_mask.to(rejected_logprobs.dtype)
            ).sum(dim=1)
            beta = torch.tensor(
                [float(loss_fn_config.get("beta", item.beta)) for item in items],
                dtype=torch.float32,
                device=self.device,
            )
            reference_chosen = torch.tensor(
                [item.reference_chosen_logprob for item in items],
                dtype=torch.float32,
                device=self.device,
            )
            reference_rejected = torch.tensor(
                [item.reference_rejected_logprob for item in items],
                dtype=torch.float32,
                device=self.device,
            )
            loss, metrics = dpo_loss(
                chosen_logprobs=chosen_sequence_logprobs,
                rejected_logprobs=rejected_sequence_logprobs,
                reference_chosen_logprobs=reference_chosen,
                reference_rejected_logprobs=reference_rejected,
                beta=beta,
            )
        if with_gradient:
            loss.backward()
        outputs = {
            "loss": types.TensorData(data=[float(loss.detach().cpu())], dtype="float32"),
            "chosen_logprobs": types.TensorData(
                data=chosen_sequence_logprobs.detach().cpu().tolist(),
                dtype="float32",
                shape=(len(items),),
            ),
            "rejected_logprobs": types.TensorData(
                data=rejected_sequence_logprobs.detach().cpu().tolist(),
                dtype="float32",
                shape=(len(items),),
            ),
        }
        return loss, metrics, outputs, sum(
            len(item.chosen_tokens) + len(item.rejected_tokens) for item in items
        )

    def _tensorize_supervised(self, items: list[SupervisedBatchItem]) -> dict[str, Any]:
        torch, _, _, _ = _backend_deps()
        max_len = max(len(item.tokens) for item in items)
        pad_id = self._pad_token_id()
        input_ids = []
        attention_mask = []
        labels = []
        weights = []
        for item in items:
            pad = max_len - len(item.tokens)
            input_ids.append(item.tokens + ([pad_id] * pad))
            attention_mask.append(([1] * len(item.tokens)) + ([0] * pad))
            labels.append(item.labels + ([-100] * pad))
            weights.append(item.weights + ([0.0] * pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=self.device),
        }

    def _tensorize_importance_sampling(
        self,
        items: list[ImportanceSamplingBatchItem],
    ) -> dict[str, Any]:
        tensors = self._tensorize_supervised(
            [
                SupervisedBatchItem(
                    tokens=item.tokens,
                    labels=item.labels,
                    weights=[1.0 if label != -100 else 0.0 for label in item.labels],
                    prompt_length=len(item.tokens) - len(item.completion_tokens),
                    target_length=len(item.completion_tokens),
                )
                for item in items
            ]
        )
        torch, _, _, _ = _backend_deps()
        max_completion = max(len(item.completion_tokens) for item in items)
        tensors["old_logprobs"] = _pad_float_rows(
            [item.old_logprobs for item in items],
            width=max_completion,
            device=self.device,
            torch=torch,
        )
        tensors["advantages"] = _pad_float_rows(
            [item.advantages for item in items],
            width=max_completion,
            device=self.device,
            torch=torch,
        )
        tensors["weights"] = _pad_float_rows(
            [item.weights for item in items],
            width=max_completion,
            device=self.device,
            torch=torch,
        )
        tensors["masks"] = _pad_float_rows(
            [item.masks for item in items],
            width=max_completion,
            device=self.device,
            torch=torch,
        )
        if any(item.ref_logprobs is not None for item in items):
            tensors["ref_logprobs"] = _pad_float_rows(
                [
                    item.ref_logprobs
                    if item.ref_logprobs is not None
                    else [0.0] * len(item.completion_tokens)
                    for item in items
                ],
                width=max_completion,
                device=self.device,
                torch=torch,
            )
        return tensors

    def _tensorize_dpo_side(self, items: list[DPOBatchItem], *, side: str) -> dict[str, Any]:
        torch, _, _, _ = _backend_deps()
        tokens_attr = f"{side}_tokens"
        labels_attr = f"{side}_labels"
        weights_attr = f"{side}_weights"
        max_len = max(len(getattr(item, tokens_attr)) for item in items)
        pad_id = self._pad_token_id()
        input_ids = []
        attention_mask = []
        labels = []
        weights = []
        for item in items:
            item_tokens = list(getattr(item, tokens_attr))
            item_labels = list(getattr(item, labels_attr))
            item_weights = list(getattr(item, weights_attr))
            pad = max_len - len(item_tokens)
            input_ids.append(item_tokens + ([pad_id] * pad))
            attention_mask.append(([1] * len(item_tokens)) + ([0] * pad))
            labels.append(item_labels + ([-100] * pad))
            weights.append(item_weights + ([0.0] * pad))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self.device),
            "weights": torch.tensor(weights, dtype=torch.float32, device=self.device),
        }

    def _save_checkpoint(self, payload: dict[str, Any], *, checkpoint_type: str) -> dict[str, Any]:
        self._ensure_model_loaded()
        torch, _, _, _ = _backend_deps()
        record = self._record()
        name = payload.get("name") or f"seq-{record['model_seq_id']}"
        project_id = record.get("project_id") or "default"
        artifact_type = "checkpoints" if checkpoint_type == "training" else "sampler_weights"
        artifact_dir = self.store.temp_artifact_dir(project_id, self.run_id, artifact_type, name)
        adapter_dir = artifact_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self._save_adapter(adapter_dir)
        adapter_files = _adapter_manifest_files(adapter_dir)

        optimizer_path = None
        if checkpoint_type == "training":
            optimizer_path = artifact_dir / "optimizer.pt"
            torch.save(
                {
                    "optimizer_step": record.get("optimizer_step", 0),
                    "state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
                    "config": self.optimizer_config,
                },
                optimizer_path,
            )

        toy_path = build_toy_path(project_id, self.run_id, artifact_type, name)
        size_bytes = _dir_size(artifact_dir)
        checkpoint = types.Checkpoint(
            checkpoint_id=name,
            checkpoint_type=checkpoint_type,
            toy_path=toy_path,
            size_bytes=size_bytes,
            expires_at=_expires_at_from_ttl(payload.get("ttl_seconds")),
        )
        checkpoint_payload = checkpoint.model_dump(mode="json")
        manifest = {
            "schema_version": 1,
            "version": 1,
            "backend": self.backend_name,
            "artifact_type": artifact_type,
            "path": toy_path,
            "checkpoint": checkpoint_payload,
            "training_run_id": self.run_id,
            "project_id": project_id,
            "base_model": record["base_model"],
            "lora_config": record["lora_config"],
            "lora_mapping": lora_mapping_manifest(
                record["base_model"],
                types.LoraConfig.model_validate(record["lora_config"]),
            ),
            "model_seq_id": record["model_seq_id"],
            "optimizer_step": record["optimizer_step"],
            "optimizer_state": record.get("optimizer_state", {}),
            "adapter_path": "adapter",
            "adapter_files": adapter_files,
            "optimizer_path": optimizer_path.name if optimizer_path is not None else None,
            "created_at": _now(),
        }
        backend_manifest = self._backend_manifest(record)
        if backend_manifest:
            manifest.update(backend_manifest)
        if checkpoint_type == "sampler":
            sampler_id = f"{self.run_id}:sample:{len(record.get('samplers', []))}"
            sampler_payload = {
                "sampler_id": sampler_id,
                "base_model": record["base_model"],
                "model_path": toy_path,
                "sampling_session_id": record.get("session_id") or f"session_{self.run_id}",
                "created_at": _now(),
            }
            manifest["sampler"] = sampler_payload
            manifest["sampler_id"] = sampler_id
            record["session_id"] = sampler_payload["sampling_session_id"]
            record["samplers"] = [
                item
                for item in record.get("samplers", [])
                if item.get("model_path") != toy_path
            ]
            record["samplers"].append(sampler_payload)
        manifest = self.store.enrich_manifest(manifest, artifact_dir)
        self.store.write_json(artifact_dir / "manifest.json", manifest)
        self.store.validate_manifest_files(manifest, artifact_dir)
        self.store.promote_artifact_dir(artifact_dir, project_id, self.run_id, artifact_type, name)

        checkpoints = [
            item
            for item in record.get("checkpoints", [])
            if item["toy_path"] != toy_path
        ]
        checkpoint_payload = manifest["checkpoint"]
        checkpoints.append(checkpoint_payload)
        record["checkpoints"] = checkpoints
        if checkpoint_type == "training":
            record["last_checkpoint"] = checkpoint_payload
        else:
            record["last_sampler_checkpoint"] = checkpoint_payload
        record["updated_at"] = _now()
        self._save_record(record)
        response_cls = (
            types.SaveWeightsResponse
            if checkpoint_type == "training"
            else types.SaveWeightsForSamplerResponse
        )
        return response_cls(
            path=toy_path,
            checkpoint_id=name,
            model_seq_id=record["model_seq_id"],
        ).model_dump(mode="json")

    def _save_adapter(self, adapter_dir: Path) -> None:
        try:
            self.model.save_pretrained(adapter_dir, safe_serialization=True)
        except TypeError:
            self.model.save_pretrained(adapter_dir)

    def _backend_manifest(self, record: dict[str, Any]) -> dict[str, Any]:
        return {}

    def _load_optimizer_state(self, artifact_dir: Path, manifest: dict[str, Any]) -> None:
        torch, _, _, _ = _backend_deps()
        optimizer_name = manifest.get("optimizer_path")
        if not optimizer_name:
            self.optimizer = None
            self.optimizer_config = None
            return
        try:
            optimizer_payload = torch.load(
                artifact_dir / optimizer_name,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:
            optimizer_payload = torch.load(
                artifact_dir / optimizer_name,
                map_location=self.device,
            )
        config = optimizer_payload.get("config")
        state_dict = optimizer_payload.get("state_dict")
        if not config or state_dict is None:
            self.optimizer = None
            self.optimizer_config = None
            return
        adam = types.AdamParams(
            learning_rate=float(config["learning_rate"]),
            beta1=float(config.get("beta1", 0.9)),
            beta2=float(config.get("beta2", 0.999)),
            eps=float(config.get("eps", 1e-8)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            grad_clip_norm=float(config.get("grad_clip_norm", 0.0)),
        )
        self._ensure_optimizer(adam)
        self.optimizer.load_state_dict(state_dict)

    def _ensure_model_loaded(self, adapter_dir: Path | None = None) -> None:
        torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = _backend_deps()
        from peft import get_peft_model

        record = self._record()
        pending_manifest = None
        if adapter_dir is None and record.get("pending_load_state_path"):
            parsed = parse_toy_path(record["pending_load_state_path"])
            pending_manifest_path = self.store.layout.artifact_manifest_path(
                parsed.project_id,
                parsed.run_id,
                parsed.artifact_type,
                parsed.name,
            )
            if not pending_manifest_path.exists():
                raise CheckpointNotFoundError(record["pending_load_state_path"])
            pending_manifest = json.loads(pending_manifest_path.read_text(encoding="utf-8"))
            self.store.raise_if_manifest_expired(pending_manifest)
            self.store.validate_manifest_files(pending_manifest, pending_manifest_path.parent)
            adapter_dir = pending_manifest_path.parent / pending_manifest.get("adapter_path", "adapter")
            if not adapter_dir.exists():
                raise CheckpointNotFoundError(
                    f"adapter directory missing for {record['pending_load_state_path']}"
                )
        if self.model is not None and adapter_dir == self._loaded_adapter_dir:
            return

        lora_config = types.LoraConfig.model_validate(record["lora_config"])
        if lora_config.seed is not None:
            torch.manual_seed(lora_config.seed)

        self.model_root.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        pretrained_kwargs: dict[str, Any] = {"cache_dir": str(self.model_root)}
        if token:
            pretrained_kwargs["token"] = token

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(record["base_model"], **pretrained_kwargs)
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base_model = AutoModelForCausalLM.from_pretrained(record["base_model"], **pretrained_kwargs)
        except Exception as exc:
            raise BackendUnavailableError(
                f"failed to load base model/tokenizer {record['base_model']!r}"
            ) from exc

        if self.tokenizer is not None and getattr(base_model.config, "pad_token_id", None) is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
        else:
            self.model = get_peft_model(
                base_model,
                build_peft_lora_config(record["base_model"], lora_config),
            )
        self.model.to(self.device)
        self.model.train()
        self._loaded_adapter_dir = adapter_dir
        self.optimizer = None
        self.optimizer_config = None
        if pending_manifest is not None:
            if record.get("pending_load_optimizer"):
                self._load_optimizer_state(adapter_dir.parent, pending_manifest)
            record.pop("pending_load_state_path", None)
            record.pop("pending_load_optimizer", None)
            self._save_record(record)

    def _ensure_optimizer(self, adam: types.AdamParams) -> None:
        torch, _, _, _ = _backend_deps()
        config = {
            "learning_rate": adam.learning_rate,
            "beta1": adam.beta1,
            "beta2": adam.beta2,
            "eps": adam.eps,
            "weight_decay": adam.weight_decay,
            "grad_clip_norm": adam.grad_clip_norm,
        }
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self._trainable_params(),
                lr=adam.learning_rate,
                betas=(adam.beta1, adam.beta2),
                eps=adam.eps,
                weight_decay=adam.weight_decay,
            )
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = adam.learning_rate
                group["betas"] = (adam.beta1, adam.beta2)
                group["eps"] = adam.eps
                group["weight_decay"] = adam.weight_decay
        self.optimizer_config = config

    def _trainable_params(self) -> list[Any]:
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def _pad_token_id(self) -> int:
        if self.tokenizer is not None and self.tokenizer.pad_token_id is not None:
            return int(self.tokenizer.pad_token_id)
        config = getattr(self.model, "config", None)
        if config is not None and getattr(config, "pad_token_id", None) is not None:
            return int(config.pad_token_id)
        if config is not None and getattr(config, "eos_token_id", None) is not None:
            return int(config.eos_token_id)
        return 0

    def _torch_no_grad(self):
        torch, _, _, _ = _backend_deps()
        return torch.no_grad()

    def _record(self) -> dict[str, Any]:
        key = f"run:{self.run_id}"
        try:
            return dict(self.registry[key])
        except Exception:
            pass
        stored = self.store.find_run_metadata(self.run_id)
        if stored is not None:
            return stored
        raise BadRequestError(f"training run not found: {self.run_id}")

    def _save_record(self, record: dict[str, Any]) -> None:
        key = f"run:{self.run_id}"
        try:
            self.registry[key] = record
        except Exception:
            pass
        project_id = record.get("project_id") or "default"
        self.store.write_run_metadata(project_id, self.run_id, record)

    def _manifest_path(self, run_id: str, artifact_type: str, name: str) -> Path:
        record = self._record()
        project_id = record.get("project_id") or "default"
        return self.store.layout.artifact_manifest_path(project_id, run_id, artifact_type, name)

    def _reset_loaded_state(self) -> None:
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.optimizer_config = None
        self._loaded_adapter_dir = None

    @staticmethod
    def _ensure_expected_sequence(record: dict[str, Any], expected: int | None) -> None:
        if expected is not None and expected != record["model_seq_id"]:
            raise StaleModelSequenceError(
                f"expected model_seq_id {expected}, current is {record['model_seq_id']}"
            )

    @staticmethod
    def _ensure_old_logprobs_sequence(record: dict[str, Any], data: list[Any]) -> None:
        for raw_datum in data:
            datum = types.Datum.model_validate(raw_datum)
            expected = datum.loss_fn_inputs.get("old_logprobs_model_seq_id")
            if expected is not None and int(expected) != record["model_seq_id"]:
                raise StaleModelSequenceError(
                    "old_logprobs_model_seq_id "
                    f"{expected} does not match current model_seq_id {record['model_seq_id']}"
                )


def _backend_deps():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise BackendUnavailableError(
            "PeftTrainerEngine requires optional backend dependencies: "
            "torch, transformers, and peft"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, PeftModel


def _adapter_manifest_files(adapter_dir: Path) -> list[dict[str, str]]:
    config_path = adapter_dir / "adapter_config.json"
    weight_paths = [
        adapter_dir / "adapter_model.safetensors",
        adapter_dir / "adapter_model.bin",
    ]
    if not config_path.exists() or not any(path.exists() for path in weight_paths):
        entries = sorted(str(path.relative_to(adapter_dir)) for path in adapter_dir.rglob("*"))
        raise CheckpointNotFoundError(
            f"PEFT adapter save did not produce required files in {adapter_dir}; "
            f"visible entries: {entries}"
        )

    max_bytes = _embedded_adapter_max_bytes()
    files = [path for path in sorted(adapter_dir.rglob("*")) if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    if total_bytes > max_bytes:
        return []
    return [
        {
            "path": str(path.relative_to(adapter_dir)),
            "encoding": "base64",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        for path in files
    ]


def _embedded_adapter_max_bytes() -> int:
    configured = os.getenv("TOY_MODAL_EMBED_ADAPTER_MAX_BYTES")
    if configured is not None:
        try:
            return max(0, int(configured))
        except ValueError:
            return 0
    return 8 * 1024 * 1024


def _rl_output_type(loss_fn: str) -> str:
    if loss_fn == "dpo":
        return "DPOLossReturn"
    if loss_fn == "ppo":
        return "PPOLossReturn"
    if loss_fn == "cispo":
        return "CISPOLossReturn"
    return "ImportanceSamplingLossReturn"


def _pad_float_rows(rows: list[list[float]], *, width: int, device: Any, torch: Any):
    return torch.tensor(
        [row + ([0.0] * (width - len(row))) for row in rows],
        dtype=torch.float32,
        device=device,
    )


def _pad_selected_rows(selected_logprobs: Any, mask: Any, *, width: int, device: Any) -> Any:
    torch, _, _, _ = _backend_deps()
    rows = []
    for row, row_mask in zip(selected_logprobs, mask):
        values = row[row_mask]
        if values.numel() > width:
            raise BadRequestError(
                f"selected completion logprobs length {values.numel()} exceeds expected width {width}"
            )
        if values.numel() < width:
            values = torch.cat([values, torch.zeros(width - values.numel(), device=device)])
        rows.append(values)
    return torch.stack(rows)


def _unpadded_rows(selected_logprobs: Any, mask: Any) -> list[list[float]]:
    rows: list[list[float]] = []
    for row, row_mask in zip(selected_logprobs, mask):
        rows.append(row[row_mask].detach().cpu().tolist())
    return rows


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
