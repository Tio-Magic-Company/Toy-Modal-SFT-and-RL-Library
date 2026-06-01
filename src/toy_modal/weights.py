"""Weight export, adapter, and Hub publishing helpers.

The helpers are local/offline by default. Functions that need optional
Hugging Face, Unsloth, Transformers, or PEFT dependencies import them lazily
and fail with an actionable error when the dependency is missing.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@dataclass(frozen=True)
class ModelCardConfig:
    model_name: str
    base_model: str
    description: str = "A toy_modal-trained adapter or model artifact."
    license: str = "apache-2.0"
    tags: list[str] = field(default_factory=lambda: ["toy_modal", "lora"])
    metrics: dict[str, float] = field(default_factory=dict)


def download(rest_client, toy_path: str, destination: str | Path) -> Path:
    """Download or materialize checkpoint archive metadata.

    Local/mock archive URLs are written as JSON metadata. HTTP(S) URLs are
    downloaded directly. Modal Volume URLs are recorded as metadata because
    direct Volume download remains environment-specific.
    """

    target = Path(destination)
    archive = rest_client.get_checkpoint_archive_url_from_toy_path(toy_path).result()
    target.parent.mkdir(parents=True, exist_ok=True)
    if archive.url.startswith(("http://", "https://")):
        with urllib.request.urlopen(archive.url) as response:
            target.write_bytes(response.read())
    else:
        target.write_text(json.dumps(archive.model_dump(mode="json"), indent=2, sort_keys=True))
    return target


def build_lora_adapter(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    adapter_config: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> Path:
    """Build a PEFT-style adapter directory from local checkpoint artifacts."""

    source = Path(checkpoint_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 4,
        "lora_alpha": 8,
        "target_modules": ["q_proj", "v_proj"],
        **(adapter_config or {}),
    }
    (target / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    copied = False
    if source.exists() and source.is_dir():
        for candidate in source.glob("*.safetensors"):
            shutil.copy2(candidate, target / "adapter_model.safetensors")
            copied = True
            break
        if not copied:
            for candidate in source.glob("*.bin"):
                shutil.copy2(candidate, target / "adapter_model.bin")
                copied = True
                break
    if not copied and not dry_run:
        raise FileNotFoundError(
            f"no adapter weight file found in {source}; pass dry_run=True to create a placeholder"
        )
    if not copied:
        (target / "adapter_model.json").write_text(
            json.dumps({"source": str(source), "placeholder": True}, indent=2),
            encoding="utf-8",
        )
    return target


def build_hf_model(
    *,
    base_model: str,
    adapter_dir: str | Path,
    output_dir: str | Path,
    merge: bool = False,
    local_files_only: bool = False,
    dry_run: bool = False,
    backend: str = "auto",
    save_method: str | None = None,
) -> Path:
    """Build a Hugging Face-compatible model directory.

    ``backend="auto"`` follows the framework default and uses Unsloth's loader
    and merged-save helpers. Pass ``backend="transformers"`` to use the
    historical Transformers/PEFT path. Without real dependencies, ``dry_run``
    writes an explicit manifest so tutorials still run in offline smoke mode.
    """

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    resolved_backend = _resolve_export_backend(backend)
    if dry_run:
        (target / "toy_modal_hf_manifest.json").write_text(
            json.dumps(
                {
                    "base_model": base_model,
                    "adapter_dir": str(adapter_dir),
                    "backend": resolved_backend,
                    "merge": merge,
                    "save_method": _resolve_unsloth_save_method(merge, save_method)
                    if resolved_backend == "unsloth"
                    else save_method,
                    "local_files_only": local_files_only,
                    "dry_run": True,
                    "requires": _export_requirements(resolved_backend),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return target

    if resolved_backend == "unsloth":
        return _build_hf_model_unsloth(
            base_model=base_model,
            adapter_dir=adapter_dir,
            output_dir=target,
            merge=merge,
            local_files_only=local_files_only,
            save_method=save_method,
        )
    return _build_hf_model_transformers(
        base_model=base_model,
        adapter_dir=adapter_dir,
        output_dir=target,
        merge=merge,
        local_files_only=local_files_only,
    )


def _build_hf_model_transformers(
    *,
    base_model: str,
    adapter_dir: str | Path,
    output_dir: Path,
    merge: bool,
    local_files_only: bool,
) -> Path:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "build_hf_model with backend='transformers' requires transformers and peft; "
            "pass dry_run=True for a manifest-only plan"
        )

    model = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=local_files_only)
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    model_to_save = peft_model.merge_and_unload() if merge else peft_model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _build_hf_model_unsloth(
    *,
    base_model: str,
    adapter_dir: str | Path,
    output_dir: Path,
    merge: bool,
    local_files_only: bool,
    save_method: str | None,
) -> Path:
    try:
        from unsloth import FastLanguageModel
        import torch
        from peft import PeftModel

        from toy_modal.backend.unsloth_config import load_unsloth_engine_config
        from toy_modal.backend.unsloth_engines import _for_inference, _resolve_model_kwargs
    except ImportError as exc:
        raise ImportError(
            "build_hf_model with backend='unsloth' requires the unsloth export dependencies; "
            "install with `python -m pip install -e '.[export,unsloth]'` or pass "
            "backend='transformers'"
        ) from exc

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    config = load_unsloth_engine_config()
    model_kwargs = _resolve_model_kwargs(config.model_kwargs(token=token), torch)
    model_kwargs["local_files_only"] = local_files_only

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base_model, **model_kwargs)
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    peft_model = _for_inference(FastLanguageModel, peft_model)
    resolved_save_method = _resolve_unsloth_save_method(merge, save_method)
    if hasattr(peft_model, "save_pretrained_merged"):
        peft_model.save_pretrained_merged(
            str(output_dir),
            tokenizer,
            save_method=resolved_save_method,
        )
        return output_dir

    model_to_save = peft_model.merge_and_unload() if merge else peft_model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _resolve_export_backend(backend: str) -> str:
    normalized = (backend or "auto").lower().replace("_", "-")
    if normalized == "auto":
        return "unsloth"
    if normalized in {"unsloth", "unsloth-peft"}:
        return "unsloth"
    if normalized in {"transformers", "peft", "hf", "huggingface"}:
        return "transformers"
    raise ValueError(f"unsupported HF export backend: {backend!r}")


def _resolve_unsloth_save_method(merge: bool, save_method: str | None) -> str:
    if save_method:
        return save_method
    return "merged_16bit" if merge else "lora"


def _export_requirements(backend: str) -> list[str]:
    if backend == "unsloth":
        return ["unsloth", "unsloth_zoo", "torch", "peft", "transformers", "safetensors"]
    return ["transformers", "peft", "torch", "safetensors"]


def generate_model_card(config: ModelCardConfig) -> str:
    metrics = "\n".join(f"- {key}: {value}" for key, value in sorted(config.metrics.items()))
    tags = "\n".join(f"- {tag}" for tag in config.tags)
    return (
        f"---\nlicense: {config.license}\ntags:\n{tags}\n---\n\n"
        f"# {config.model_name}\n\n"
        f"{config.description}\n\n"
        f"Base model: `{config.base_model}`\n\n"
        "## Metrics\n\n"
        f"{metrics or '- No metrics recorded.'}\n"
    )


def publish_to_hf_hub(
    folder_path: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    dry_run: bool = True,
) -> dict[str, object]:
    """Publish a local model folder to the Hugging Face Hub.

    ``dry_run=True`` is the default and never contacts the network.
    """

    folder = Path(folder_path)
    if dry_run:
        return {"repo_id": repo_id, "folder_path": str(folder), "dry_run": True}
    resolved_token = token or os.getenv("HF_TOKEN")
    if not resolved_token:
        raise ValueError("HF_TOKEN or token=... is required when dry_run=False")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("publish_to_hf_hub requires huggingface_hub") from exc
    api = HfApi(token=resolved_token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit_info = api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
    )
    return {"repo_id": repo_id, "commit": str(commit_info), "dry_run": False}


def package_and_publish(
    *,
    folder_path: str | Path,
    card: ModelCardConfig,
    repo_id: str,
    dry_run: bool = True,
) -> dict[str, object]:
    with TemporaryDirectory() as tmpdir:
        target = Path(tmpdir)
        if Path(folder_path).exists():
            shutil.copytree(folder_path, target / "model", dirs_exist_ok=True)
        (target / "README.md").write_text(generate_model_card(card), encoding="utf-8")
        (target / "toy_modal_publish.json").write_text(
            json.dumps(asdict(card), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return publish_to_hf_hub(target, repo_id=repo_id, dry_run=dry_run)
