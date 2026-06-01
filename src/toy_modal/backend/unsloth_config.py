"""Runtime configuration helpers for Unsloth-backed engines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from toy_modal.errors import BadRequestError

UNSLOTH_RUNTIME_PACKAGES = (
    "unsloth",
    "unsloth_zoo",
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
)


@dataclass(frozen=True)
class UnslothEngineConfig:
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    load_in_16bit: bool = False
    load_in_fp8: bool | str = False
    max_seq_length: int = 2048
    dtype: str | None = None
    use_gradient_checkpointing: bool | str = "unsloth"
    trust_remote_code: bool = False
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.5
    use_exact_model_name: bool = False

    def validate(self) -> None:
        quantized_modes = [
            self.load_in_4bit,
            self.load_in_8bit,
            self.load_in_16bit,
            self.load_in_fp8 is not False,
        ]
        if sum(bool(value) for value in quantized_modes) > 1:
            raise BadRequestError(
                "Unsloth backend can load in only one of 4-bit, 8-bit, 16-bit, or FP8 mode"
            )
        if self.max_seq_length <= 0:
            raise BadRequestError("TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH must be positive")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise BadRequestError(
                "TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION must be in the interval (0, 1]"
            )

    def model_kwargs(self, *, token: str | None = None) -> dict[str, Any]:
        self.validate()
        kwargs: dict[str, Any] = {
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "load_in_16bit": self.load_in_16bit,
            "load_in_fp8": self.load_in_fp8,
            "trust_remote_code": self.trust_remote_code,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "fast_inference": self.fast_inference,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_exact_model_name": self.use_exact_model_name,
        }
        if self.dtype:
            kwargs["dtype"] = self.dtype
        if token:
            kwargs["token"] = token
        return kwargs

    def manifest(self) -> dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "load_in_16bit": self.load_in_16bit,
            "load_in_fp8": self.load_in_fp8,
            "max_seq_length": self.max_seq_length,
            "dtype": self.dtype,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "trust_remote_code": self.trust_remote_code,
            "fast_inference": self.fast_inference,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_exact_model_name": self.use_exact_model_name,
        }


def load_unsloth_engine_config() -> UnslothEngineConfig:
    return UnslothEngineConfig(
        load_in_4bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", True),
        load_in_8bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_8BIT", False),
        load_in_16bit=_bool_env("TOY_MODAL_UNSLOTH_LOAD_IN_16BIT", False),
        load_in_fp8=_fp8_env("TOY_MODAL_UNSLOTH_LOAD_IN_FP8"),
        max_seq_length=int(os.getenv("TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH", "2048")),
        dtype=os.getenv("TOY_MODAL_UNSLOTH_DTYPE") or None,
        use_gradient_checkpointing=_gradient_checkpointing_env(
            "TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING",
            "unsloth",
        ),
        trust_remote_code=_bool_env("TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE", False),
        fast_inference=_bool_env("TOY_MODAL_UNSLOTH_FAST_INFERENCE", False),
        gpu_memory_utilization=float(
            os.getenv("TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION", "0.5")
        ),
        use_exact_model_name=_bool_env("TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME", False),
    )


def unsloth_runtime_versions() -> dict[str, str | None]:
    """Return installed backend package versions without importing patching libraries."""

    result: dict[str, str | None] = {}
    for package in UNSLOTH_RUNTIME_PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _fp8_env(name: str) -> bool | str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return False
    normalized = value.strip().lower()
    if normalized in {"0", "false", "no", "off"}:
        return False
    if normalized in {"1", "true", "yes", "on"}:
        return True
    return normalized


def _gradient_checkpointing_env(name: str, default: str) -> bool | str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return value
