"""Model-cache preflight and prefetch helpers for Modal Volumes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex
from typing import Any


GATED_MODEL_PREFIXES = (
    "meta-llama/",
    "google/",
    "mistralai/",
)


@dataclass(frozen=True)
class ModelPrefetchPlan:
    model_id: str
    cache_dir: str
    backend: str
    include_model: bool
    include_tokenizer: bool
    dry_run: bool
    appears_gated: bool
    has_hf_token: bool
    required_packages: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "cache_dir": self.cache_dir,
            "backend": self.backend,
            "include_model": self.include_model,
            "include_tokenizer": self.include_tokenizer,
            "dry_run": self.dry_run,
            "appears_gated": self.appears_gated,
            "has_hf_token": self.has_hf_token,
            "required_packages": self.required_packages,
            "notes": self.notes,
        }


def preflight_model_prefetch(
    model_id: str,
    *,
    model_root: str | Path,
    include_model: bool = True,
    include_tokenizer: bool = True,
    dry_run: bool = True,
    backend: str = "auto",
) -> ModelPrefetchPlan:
    """Return a no-network plan describing a model prefetch request."""

    cache_dir = Path(model_root).expanduser().resolve()
    normalized_backend = _normalize_backend(backend)
    appears_gated = model_id.startswith(GATED_MODEL_PREFIXES)
    has_token = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    packages = ["huggingface_hub", "transformers"]
    if include_model:
        packages.extend(["torch", "accelerate"])
    if normalized_backend == "unsloth" and include_model:
        packages.extend(_unsloth_pip_packages_from_env())
    notes = [
        "Dry run only; no model files are downloaded." if dry_run else "Real prefetch requested.",
        "Model files will be cached in the configured Modal model Volume.",
    ]
    if normalized_backend == "unsloth":
        notes.append("Unsloth backend prefetch uses FastLanguageModel for model loads.")
        if include_tokenizer and not include_model:
            notes.append("Tokenizer-only prefetch still uses Transformers AutoTokenizer.")
    if appears_gated and not has_token:
        notes.append("This model appears gated; configure HF_TOKEN through a Modal Secret before real prefetch.")
    return ModelPrefetchPlan(
        model_id=model_id,
        cache_dir=str(cache_dir),
        backend=normalized_backend,
        include_model=include_model,
        include_tokenizer=include_tokenizer,
        dry_run=dry_run,
        appears_gated=appears_gated,
        has_hf_token=has_token,
        required_packages=packages,
        notes=notes,
    )


def prefetch_model(
    model_id: str,
    *,
    model_root: str | Path,
    include_model: bool = True,
    include_tokenizer: bool = True,
    dry_run: bool = False,
    local_files_only: bool = False,
    backend: str = "auto",
) -> dict[str, Any]:
    """Download tokenizer/model files into a cache directory.

    The function is intentionally import-light until a real prefetch is
    requested, so CLI dry runs and unit tests do not need heavy ML packages.
    """

    plan = preflight_model_prefetch(
        model_id,
        model_root=model_root,
        include_model=include_model,
        include_tokenizer=include_tokenizer,
        dry_run=dry_run,
        backend=backend,
    )
    if dry_run:
        return plan.to_dict()

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if plan.appears_gated and not token:
        raise RuntimeError(
            f"{model_id!r} appears gated; set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before prefetching"
        )

    cache_dir = Path(model_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"cache_dir": str(cache_dir), "local_files_only": local_files_only}
    if token:
        kwargs["token"] = token

    loaded: list[str] = []
    if plan.backend == "unsloth" and include_model:
        try:
            from unsloth import FastLanguageModel
            import torch

            from toy_modal.backend.unsloth_config import load_unsloth_engine_config
            from toy_modal.backend.unsloth_engines import _resolve_model_kwargs
        except ImportError as exc:
            raise RuntimeError(
                "Unsloth model prefetch requires the unsloth backend extras"
            ) from exc
        model_kwargs = _resolve_model_kwargs(
            load_unsloth_engine_config().model_kwargs(token=token),
            torch,
        )
        model_kwargs.update({"cache_dir": str(cache_dir), "local_files_only": local_files_only})
        FastLanguageModel.from_pretrained(model_name=model_id, **model_kwargs)
        loaded.extend(["tokenizer", "model"])
    else:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "model prefetch requires the backend extras: transformers and torch"
            ) from exc
        if include_tokenizer:
            AutoTokenizer.from_pretrained(model_id, **kwargs)
            loaded.append("tokenizer")
        if include_model:
            AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            loaded.append("model")

    result = plan.to_dict()
    result.update({"dry_run": False, "loaded": loaded, "cache_exists": cache_dir.exists()})
    return result


def _normalize_backend(backend: str) -> str:
    normalized = (backend or "auto").lower().replace("_", "-")
    if normalized == "auto":
        configured = os.getenv("TOY_MODAL_PREFETCH_BACKEND")
        if configured:
            return _normalize_backend(configured)
        engine_names = [
            os.getenv("TOY_MODAL_TRAINER_ENGINE"),
            os.getenv("TOY_MODAL_SAMPLER_ENGINE"),
        ]
        if any((name or "").lower().replace("_", "-").startswith("unsloth") for name in engine_names):
            return "unsloth"
        if any((name or "").lower().replace("_", "-") in {"transformers", "hf", "huggingface", "peft"} for name in engine_names):
            return "transformers"
        return "unsloth"
    if normalized.startswith("unsloth"):
        return "unsloth"
    if normalized in {"transformers", "hf", "huggingface", "peft"}:
        return "transformers"
    raise ValueError(f"unsupported model prefetch backend: {backend!r}")


def _unsloth_pip_packages_from_env() -> list[str]:
    packages = [
        os.getenv("TOY_MODAL_UNSLOTH_PACKAGE", "unsloth[base]"),
        os.getenv(
            "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE",
            "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
        ),
        *shlex.split(os.getenv("TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES", "")),
    ]
    return [package for package in packages if package]
