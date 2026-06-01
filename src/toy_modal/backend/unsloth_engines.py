"""Unsloth-backed trainer and sampler engines.

These engines keep the public Tinker-style `toy_modal` worker contract intact
while delegating model loading, LoRA wrapping, quantized modes, and inference
patching to Unsloth Core.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from toy_modal import types
from toy_modal.backend.lora_mapping import build_peft_lora_kwargs, lora_mapping_manifest
from toy_modal.backend.peft_trainer import PeftTrainerEngine
from toy_modal.backend.sampler_worker import TransformersSamplerEngine, _resolve_model_reference
from toy_modal.backend.trainer_worker import _now
from toy_modal.backend.unsloth_config import (
    UnslothEngineConfig,
    load_unsloth_engine_config,
    unsloth_runtime_versions,
)
from toy_modal.errors import BackendUnavailableError, CheckpointNotFoundError
from toy_modal.paths import parse_toy_path


class UnslothTrainerEngine(PeftTrainerEngine):
    """LoRA trainer that uses Unsloth Core behind the existing worker API."""

    backend_name = "unsloth-peft"

    def _ensure_model_loaded(self, adapter_dir: Path | None = None) -> None:
        torch, PeftModel, FastLanguageModel = _unsloth_deps()
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
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        config = load_unsloth_engine_config()
        model_kwargs = _resolve_model_kwargs(config.model_kwargs(token=token), torch)
        model_kwargs["cache_dir"] = str(self.model_root)
        try:
            base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=record["base_model"],
                **model_kwargs,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            raise BackendUnavailableError(
                f"failed to load base model/tokenizer {record['base_model']!r} with Unsloth"
            ) from exc

        if self.tokenizer is not None and getattr(base_model.config, "pad_token_id", None) is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

        if adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=True)
            self.model = _for_training(FastLanguageModel, self.model, config)
        else:
            self.model = FastLanguageModel.get_peft_model(
                base_model,
                **_unsloth_lora_kwargs(record["base_model"], lora_config, config),
            )
        self.model.train()
        self.device = _model_device(torch, self.model)
        self._loaded_adapter_dir = adapter_dir
        self.optimizer = None
        self.optimizer_config = None
        if pending_manifest is not None:
            if record.get("pending_load_optimizer"):
                self._load_optimizer_state(adapter_dir.parent, pending_manifest)
            record.pop("pending_load_state_path", None)
            record.pop("pending_load_optimizer", None)
            self._save_record(record)

    def _backend_manifest(self, record: dict[str, Any]) -> dict[str, Any]:
        config = load_unsloth_engine_config()
        return {
            "unsloth": {
                "engine": self.backend_name,
                "config": config.manifest(),
                "package_versions": unsloth_runtime_versions(),
                "loaded_at": _now(),
            },
            "lora_mapping": lora_mapping_manifest(
                record["base_model"],
                types.LoraConfig.model_validate(record["lora_config"]),
            ),
        }


class UnslothSamplerEngine(TransformersSamplerEngine):
    """Sampler that uses Unsloth Core while preserving SamplingClient outputs."""

    backend_name = "unsloth"

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        torch, PeftModel, FastLanguageModel = _unsloth_deps()
        self.model_root.mkdir(parents=True, exist_ok=True)
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        config = load_unsloth_engine_config()
        model_kwargs = _resolve_model_kwargs(config.model_kwargs(token=token), torch)
        model_kwargs["cache_dir"] = str(self.model_root)

        try:
            base_model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model,
                **model_kwargs,
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as exc:
            raise BackendUnavailableError(
                f"failed to load base model/tokenizer {self.base_model!r} with Unsloth"
            ) from exc

        if self.tokenizer is not None and getattr(base_model.config, "pad_token_id", None) is None:
            base_model.config.pad_token_id = self.tokenizer.pad_token_id

        if self.adapter_dir is not None:
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir), is_trainable=False)
        else:
            self.model = base_model
        self.model = _for_inference(FastLanguageModel, self.model)
        self.model.eval()
        self.device = _model_device(torch, self.model)


def load_unsloth_sampler_engine(
    *,
    base_model: str | None,
    model_path: str | None,
    model_root: str,
    run_root: str,
) -> UnslothSamplerEngine:
    resolved_base_model, adapter_dir, manifest = _resolve_model_reference(
        base_model=base_model,
        model_path=model_path,
        run_root=Path(run_root),
    )
    if resolved_base_model is None:
        raise CheckpointNotFoundError(model_path or "<missing model reference>")
    engine = UnslothSamplerEngine(
        base_model=resolved_base_model,
        model_path=model_path,
        model_root=model_root,
        run_root=run_root,
    )
    engine.adapter_dir = adapter_dir
    engine.manifest = manifest
    return engine


def _unsloth_lora_kwargs(
    base_model: str,
    config: types.LoraConfig,
    unsloth_config: UnslothEngineConfig,
) -> dict[str, Any]:
    kwargs = build_peft_lora_kwargs(base_model, config)
    return {
        "r": kwargs["r"],
        "target_modules": kwargs["target_modules"],
        "lora_alpha": kwargs["lora_alpha"],
        "lora_dropout": kwargs["lora_dropout"],
        "bias": kwargs["bias"],
        "use_gradient_checkpointing": unsloth_config.use_gradient_checkpointing,
        "random_state": config.seed if config.seed is not None else 3407,
        "max_seq_length": unsloth_config.max_seq_length,
    }


def _unsloth_deps():
    try:
        from unsloth import FastLanguageModel
        import torch
        from peft import PeftModel
    except ImportError as exc:
        raise BackendUnavailableError(
            "Unsloth engines require optional backend dependencies: "
            "unsloth, unsloth_zoo, torch, peft, transformers, and trl"
        ) from exc
    return torch, PeftModel, FastLanguageModel


def _resolve_model_kwargs(kwargs: dict[str, Any], torch: Any) -> dict[str, Any]:
    dtype = kwargs.get("dtype")
    if isinstance(dtype, str):
        try:
            kwargs["dtype"] = getattr(torch, dtype)
        except AttributeError as exc:
            raise BackendUnavailableError(f"unsupported Unsloth dtype: {dtype!r}") from exc
    return kwargs


def _model_device(torch: Any, model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _for_training(FastLanguageModel: Any, model: Any, config: UnslothEngineConfig) -> Any:
    if not hasattr(FastLanguageModel, "for_training") and hasattr(FastLanguageModel, "patch_peft_model"):
        patched = FastLanguageModel.patch_peft_model(
            model,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        return patched if patched is not None else model
    try:
        patched = FastLanguageModel.for_training(
            model,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
    except TypeError:
        patched = FastLanguageModel.for_training(model)
    return patched if patched is not None else model


def _for_inference(FastLanguageModel: Any, model: Any) -> Any:
    patched = FastLanguageModel.for_inference(model)
    return patched if patched is not None else model
