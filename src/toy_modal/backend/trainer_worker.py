"""Trainer engine used by Modal workers and local tiny-backend tests.

Heavy ML dependencies stay optional. The default implementation is a deterministic
tiny engine that exercises the same state, checkpoint, and sequencing contracts
without GPU cost. A future PEFT-backed implementation can replace the loss/model
math behind this interface without changing SDK routes.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from toy_modal import types
from toy_modal.backend.loss_inputs import validate_training_batch
from toy_modal.backend.storage import ArtifactStore
from toy_modal.errors import BadRequestError, CheckpointNotFoundError, StaleModelSequenceError
from toy_modal.paths import build_toy_path, parse_toy_path


class TrainerEngine:
    @classmethod
    def load_or_initialize(cls, run_id: str, registry, model_root: str, run_root: str) -> "TrainerEngine":
        return cls(run_id=run_id, registry=registry, model_root=model_root, run_root=run_root)

    def __init__(self, *, run_id: str, registry=None, model_root: str = "/models", run_root: str = "/runs") -> None:
        self.run_id = run_id
        self.registry = registry if registry is not None else {}
        self.model_root = Path(model_root)
        self.run_root = Path(run_root)
        self.store = ArtifactStore.from_runs_root(self.run_root)

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
        record["model_seq_id"] += 1
        record["optimizer_step"] += 1
        record["latest_gradient_id"] = None
        record["updated_at"] = _now()
        record["optimizer_state"] = {
            "step": record["optimizer_step"],
            "learning_rate": adam.learning_rate,
            "beta1": adam.beta1,
            "beta2": adam.beta2,
            "eps": adam.eps,
            "weight_decay": adam.weight_decay,
            "grad_clip_norm": adam.grad_clip_norm,
        }
        self._save_record(record)
        return types.OptimStepResponse(
            model_seq_id=record["model_seq_id"],
            optimizer_step=record["optimizer_step"],
            metrics={"tiny_optimizer": 1.0},
        ).model_dump(mode="json")

    def save_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._save_checkpoint(payload, checkpoint_type="training")

    def load_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_toy_path(
            payload["path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        manifest = self._manifest_path(parsed.run_id, parsed.artifact_type, parsed.name)
        if not manifest.exists():
            raise CheckpointNotFoundError(payload["path"])
        artifact = json.loads(manifest.read_text())
        self.store.raise_if_manifest_expired(artifact)
        self.store.validate_manifest_files(artifact)

        record = self._record()
        record["base_model"] = artifact["base_model"]
        record["lora_config"] = artifact["lora_config"]
        record["lora_rank"] = artifact["lora_config"].get("rank")
        record["model_seq_id"] = artifact["model_seq_id"]
        record["optimizer_step"] = artifact["optimizer_step"] if payload.get("optimizer") else 0
        record["updated_at"] = _now()
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

    def _loss_response(self, payload: dict[str, Any], *, with_gradient: bool) -> dict[str, Any]:
        record = self._record()
        self._ensure_expected_sequence(record, payload.get("expected_model_seq_id"))
        data = payload.get("data", [])
        validate_training_batch(data, payload.get("loss_fn", "cross_entropy"))
        self._ensure_old_logprobs_sequence(record, data)
        token_count = 0
        weighted_targets = 0.0
        for datum in data:
            item = types.Datum.model_validate(datum)
            token_count += item.model_input.length()
            inputs = item.loss_fn_inputs
            weights = _tensor_or_list(inputs.get("weights", [1.0]))
            targets = _tensor_or_list(inputs.get("target_tokens", []))
            weighted_targets += sum(float(value) for value in weights) + len(targets)

        base_loss = max(0.01, 2.0 - (0.05 * record.get("optimizer_step", 0)))
        loss = round(base_loss + ((token_count + weighted_targets) % 17) / 100, 4)
        gradient_id = None
        if with_gradient:
            gradient_id = f"grad_{record['training_run_id']}_{record['model_seq_id']}"
            record["latest_gradient_id"] = gradient_id
        record["last_request_time"] = _now()
        self._save_record(record)
        return types.ForwardBackwardOutput(
            loss=loss,
            loss_fn_output_type="TinyLossReturn",
            loss_fn_outputs={"loss": types.TensorData(data=[loss], dtype="float32")},
            metrics={"tiny_loss": loss},
            num_tokens=token_count,
            gradient_id=gradient_id,
            model_seq_id=record["model_seq_id"],
        ).model_dump(mode="json")

    def _save_checkpoint(self, payload: dict[str, Any], *, checkpoint_type: str) -> dict[str, Any]:
        record = self._record()
        name = payload.get("name") or f"seq-{record['model_seq_id']}"
        project_id = record.get("project_id") or "default"
        artifact_type = "checkpoints" if checkpoint_type == "training" else "sampler_weights"
        artifact_dir = self.store.temp_artifact_dir(project_id, self.run_id, artifact_type, name)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "state.json").write_text(
            json.dumps(
                {
                    "training_run_id": self.run_id,
                    "checkpoint_type": checkpoint_type,
                    "model_seq_id": record["model_seq_id"],
                    "optimizer_step": record["optimizer_step"],
                    "optimizer_state": record.get("optimizer_state", {}),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        toy_path = build_toy_path(project_id, self.run_id, artifact_type, name)
        checkpoint = types.Checkpoint(
            checkpoint_id=name,
            checkpoint_type=checkpoint_type,
            toy_path=toy_path,
            size_bytes=0,
            expires_at=_expires_at_from_ttl(payload.get("ttl_seconds")),
        )
        checkpoint_payload = checkpoint.model_dump(mode="json")
        manifest = {
            "schema_version": 1,
            "version": 1,
            "backend": "tiny",
            "artifact_type": artifact_type,
            "path": toy_path,
            "checkpoint": checkpoint_payload,
            "training_run_id": self.run_id,
            "project_id": project_id,
            "base_model": record["base_model"],
            "lora_config": record["lora_config"],
            "model_seq_id": record["model_seq_id"],
            "optimizer_step": record["optimizer_step"],
            "optimizer_state": record.get("optimizer_state", {}),
            "optimizer_path": "state.json" if checkpoint_type == "training" else None,
            "created_at": _now(),
        }
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
        response_cls = types.SaveWeightsResponse if checkpoint_type == "training" else types.SaveWeightsForSamplerResponse
        return response_cls(
            path=toy_path,
            checkpoint_id=name,
            model_seq_id=record["model_seq_id"],
        ).model_dump(mode="json")

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


def load_trainer_engine(
    engine_name: str,
    *,
    run_id: str,
    registry,
    model_root: str,
    run_root: str,
) -> TrainerEngine:
    normalized = engine_name.lower().replace("_", "-")
    if normalized in {"tiny", "deterministic"}:
        return TrainerEngine.load_or_initialize(
            run_id=run_id,
            registry=registry,
            model_root=model_root,
            run_root=run_root,
        )
    if normalized == "peft":
        from toy_modal.backend.peft_trainer import PeftTrainerEngine

        return PeftTrainerEngine.load_or_initialize(
            run_id=run_id,
            registry=registry,
            model_root=model_root,
            run_root=run_root,
        )
    if normalized in {"unsloth", "unsloth-peft", "unsloth-lora"}:
        from toy_modal.backend.unsloth_engines import UnslothTrainerEngine

        return UnslothTrainerEngine.load_or_initialize(
            run_id=run_id,
            registry=registry,
            model_root=model_root,
            run_root=run_root,
        )
    raise BadRequestError(f"unsupported trainer engine: {engine_name!r}")


def _tensor_or_list(value: Any) -> list[Any]:
    if isinstance(value, types.TensorData):
        return value.data
    if isinstance(value, dict) and "data" in value:
        return value["data"]
    if isinstance(value, list):
        return value
    return [value]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_at_from_ttl(ttl_seconds: int | None) -> datetime | None:
    if ttl_seconds is None:
        return None
    if ttl_seconds <= 0:
        raise BadRequestError("ttl_seconds must be positive or None")
    return datetime.now(timezone.utc) + timedelta(seconds=int(ttl_seconds))
