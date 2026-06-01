"""Backend configuration read from environment variables."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass

from toy_modal.backend.unsloth_config import UnslothEngineConfig
from toy_modal.defaults import (
    DEFAULT_TRANSFORMERS_CAPABILITY_MODELS,
    DEFAULT_UNSLOTH_CAPABILITY_MODELS,
)


@dataclass(frozen=True)
class BackendConfig:
    app_name: str = "toy-modal-backend"
    train_gpu: str = "A100"
    sample_gpu: str = "L40S"
    prefetch_gpu: str | None = None
    trainer_engine: str = "unsloth-peft"
    sampler_engine: str = "unsloth"
    model_volume: str = "toy-modal-model-cache"
    run_volume: str = "toy-modal-runs"
    registry_dict: str = "toy-modal-registry"
    hf_secret_name: str | None = None
    allow_unsafe_custom_loss: bool = False
    sample_max_containers: int = 2
    http_api_key: str | None = None
    http_large_result_inline_bytes: int = 512_000
    unsloth_load_in_4bit: bool = True
    unsloth_load_in_8bit: bool = False
    unsloth_load_in_16bit: bool = False
    unsloth_load_in_fp8: str | None = None
    unsloth_max_seq_length: int = 2048
    unsloth_dtype: str | None = None
    unsloth_use_gradient_checkpointing: str = "unsloth"
    unsloth_trust_remote_code: bool = False
    unsloth_fast_inference: bool = False
    unsloth_gpu_memory_utilization: float = 0.5
    unsloth_use_exact_model_name: bool = False
    unsloth_package: str = "unsloth[base]"
    unsloth_bitsandbytes_package: str = "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0"
    unsloth_extra_pip_packages: tuple[str, ...] = ()
    supported_models: tuple[str, ...] = ()

    @property
    def uses_unsloth(self) -> bool:
        return _is_unsloth_engine(self.trainer_engine) or _is_unsloth_engine(self.sampler_engine)

    @property
    def resolved_prefetch_gpu(self) -> str | None:
        if self.prefetch_gpu:
            return self.prefetch_gpu
        if self.uses_unsloth:
            return self.sample_gpu
        return None

    def unsloth_engine_config(self) -> UnslothEngineConfig:
        return UnslothEngineConfig(
            load_in_4bit=self.unsloth_load_in_4bit,
            load_in_8bit=self.unsloth_load_in_8bit,
            load_in_16bit=self.unsloth_load_in_16bit,
            load_in_fp8=_fp8_config_value(self.unsloth_load_in_fp8),
            max_seq_length=self.unsloth_max_seq_length,
            dtype=self.unsloth_dtype,
            use_gradient_checkpointing=_gradient_checkpointing_config_value(
                self.unsloth_use_gradient_checkpointing
            ),
            trust_remote_code=self.unsloth_trust_remote_code,
            fast_inference=self.unsloth_fast_inference,
            gpu_memory_utilization=self.unsloth_gpu_memory_utilization,
            use_exact_model_name=self.unsloth_use_exact_model_name,
        )

    def validate(self) -> None:
        if self.uses_unsloth:
            self.unsloth_engine_config().validate()

    @property
    def unsloth_pip_packages(self) -> tuple[str, ...]:
        packages = [
            self.unsloth_package,
            self.unsloth_bitsandbytes_package,
            *self.unsloth_extra_pip_packages,
        ]
        return tuple(package for package in packages if package)

    @property
    def resolved_supported_models(self) -> tuple[str, ...]:
        if self.supported_models:
            return self.supported_models
        if self.uses_unsloth:
            return DEFAULT_UNSLOTH_CAPABILITY_MODELS
        return DEFAULT_TRANSFORMERS_CAPABILITY_MODELS


def load_config() -> BackendConfig:
    config = BackendConfig(
        app_name=os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"),
        train_gpu=os.getenv("TOY_MODAL_TRAIN_GPU", "A100"),
        sample_gpu=os.getenv("TOY_MODAL_SAMPLE_GPU", "L40S"),
        prefetch_gpu=os.getenv("TOY_MODAL_PREFETCH_GPU") or None,
        trainer_engine=os.getenv("TOY_MODAL_TRAINER_ENGINE", "unsloth-peft"),
        sampler_engine=os.getenv("TOY_MODAL_SAMPLER_ENGINE", "unsloth"),
        model_volume=os.getenv("TOY_MODAL_MODEL_VOLUME", "toy-modal-model-cache"),
        run_volume=os.getenv("TOY_MODAL_RUN_VOLUME", "toy-modal-runs"),
        registry_dict=os.getenv("TOY_MODAL_REGISTRY_DICT", "toy-modal-registry"),
        hf_secret_name=os.getenv("TOY_MODAL_HF_SECRET_NAME") or None,
        allow_unsafe_custom_loss=os.getenv("TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS", "0") == "1",
        sample_max_containers=int(os.getenv("TOY_MODAL_SAMPLE_MAX_CONTAINERS", "2")),
        http_api_key=os.getenv("TOY_MODAL_HTTP_API_KEY") or None,
        http_large_result_inline_bytes=int(os.getenv("TOY_MODAL_HTTP_LARGE_RESULT_INLINE_BYTES", "512000")),
        unsloth_load_in_4bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", True),
        unsloth_load_in_8bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_8BIT", False),
        unsloth_load_in_16bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_16BIT", False),
        unsloth_load_in_fp8=os.getenv("TOY_MODAL_UNSLOTH_LOAD_IN_FP8") or None,
        unsloth_max_seq_length=int(os.getenv("TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH", "2048")),
        unsloth_dtype=os.getenv("TOY_MODAL_UNSLOTH_DTYPE") or None,
        unsloth_use_gradient_checkpointing=os.getenv(
            "TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING",
            "unsloth",
        ),
        unsloth_trust_remote_code=_bool_env("TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE", False),
        unsloth_fast_inference=_bool_env("TOY_MODAL_UNSLOTH_FAST_INFERENCE", False),
        unsloth_gpu_memory_utilization=float(
            os.getenv("TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION", "0.5")
        ),
        unsloth_use_exact_model_name=_bool_env("TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME", False),
        unsloth_package=os.getenv("TOY_MODAL_UNSLOTH_PACKAGE", "unsloth[base]"),
        unsloth_bitsandbytes_package=os.getenv(
            "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE",
            "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
        ),
        unsloth_extra_pip_packages=_package_tuple_env("TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES"),
        supported_models=_model_tuple_env("TOY_MODAL_SUPPORTED_MODELS"),
    )
    config.validate()
    return config


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_unsloth_engine(engine_name: str) -> bool:
    return engine_name.lower().replace("_", "-").startswith("unsloth")


def _fp8_config_value(value: str | None) -> bool | str:
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return normalized


def _gradient_checkpointing_config_value(value: str) -> bool | str:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value


def _package_tuple_env(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return ()
    return tuple(part for part in shlex.split(value) if part)


def _model_tuple_env(name: str) -> tuple[str, ...]:
    value = os.getenv(name)
    if not value:
        return ()
    if "," in value:
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(part for part in shlex.split(value) if part)
