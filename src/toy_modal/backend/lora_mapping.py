"""LoRA configuration mapping for the PEFT trainer backend.

The public SDK keeps Tinker-compatible fields on ``types.LoraConfig``. This
module is the backend-only translation layer that maps those fields onto PEFT's
``LoraConfig`` names and model-family target module conventions.
"""

from __future__ import annotations

from typing import Any

from toy_modal import types
from toy_modal.errors import BadRequestError


_DEFAULT_TARGETS = {
    "attn": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "mlp": ("gate_proj", "up_proj", "down_proj"),
    "unembed": ("lm_head",),
}

_FAMILY_TARGETS: tuple[tuple[tuple[str, ...], dict[str, tuple[str, ...]]], ...] = (
    (
        ("qwen", "llama", "mistral", "mixtral", "gemma", "yi", "deepseek", "phi"),
        _DEFAULT_TARGETS,
    ),
    (
        ("gpt2", "gpt-2", "distilgpt2"),
        {
            "attn": ("c_attn", "c_proj"),
            "mlp": ("c_fc", "c_proj"),
            "unembed": ("lm_head",),
        },
    ),
    (
        ("gpt_neox", "gpt-neox", "pythia", "dolly"),
        {
            "attn": ("query_key_value", "dense"),
            "mlp": ("dense_h_to_4h", "dense_4h_to_h"),
            "unembed": ("embed_out",),
        },
    ),
    (
        ("falcon",),
        {
            "attn": ("query_key_value", "dense"),
            "mlp": ("dense_h_to_4h", "dense_4h_to_h"),
            "unembed": ("lm_head",),
        },
    ),
    (
        ("opt-", "facebook/opt", "/opt"),
        {
            "attn": ("q_proj", "k_proj", "v_proj", "out_proj"),
            "mlp": ("fc1", "fc2"),
            "unembed": ("lm_head",),
        },
    ),
)


def resolve_lora_target_modules(base_model: str, config: types.LoraConfig) -> list[str]:
    """Resolve public train_* switches into PEFT target module suffixes."""

    if config.target_modules:
        return _dedupe(config.target_modules)

    family_targets = _targets_for_model(base_model)
    modules: list[str] = []
    if config.train_attn:
        modules.extend(family_targets["attn"])
    if config.train_mlp:
        modules.extend(family_targets["mlp"])
    if config.train_unembed:
        modules.extend(family_targets["unembed"])

    modules = _dedupe(modules)
    if not modules:
        raise BadRequestError(
            "LoRA config must enable at least one of train_attn, train_mlp, "
            "train_unembed, or provide target_modules"
        )
    return modules


def build_peft_lora_kwargs(base_model: str, config: types.LoraConfig) -> dict[str, Any]:
    """Return PEFT ``LoraConfig`` keyword arguments without importing PEFT."""

    return {
        "r": config.rank,
        "lora_alpha": config.alpha if config.alpha is not None else config.rank,
        "lora_dropout": config.dropout,
        "target_modules": resolve_lora_target_modules(base_model, config),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def build_peft_lora_config(base_model: str, config: types.LoraConfig):
    """Create a PEFT ``LoraConfig`` from the public ``types.LoraConfig``."""

    from peft import LoraConfig

    return LoraConfig(**build_peft_lora_kwargs(base_model, config))


def lora_mapping_manifest(base_model: str, config: types.LoraConfig) -> dict[str, Any]:
    """Serializable mapping details for checkpoint manifests and docs."""

    kwargs = build_peft_lora_kwargs(base_model, config)
    return {
        "public_config": config.model_dump(mode="json"),
        "peft_config": kwargs,
        "model_family": _family_name(base_model),
    }


def _targets_for_model(base_model: str) -> dict[str, tuple[str, ...]]:
    model_key = base_model.lower()
    for markers, targets in _FAMILY_TARGETS:
        if any(marker in model_key for marker in markers):
            return targets
    return _DEFAULT_TARGETS


def _family_name(base_model: str) -> str:
    model_key = base_model.lower()
    for markers, _ in _FAMILY_TARGETS:
        for marker in markers:
            if marker in model_key:
                return marker
    return "default-causal-lm"


def _dedupe(values: list[str] | tuple[str, ...]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
