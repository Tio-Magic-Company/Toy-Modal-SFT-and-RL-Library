import builtins
import sys
from types import SimpleNamespace

from toy_modal.backend.model_cache import prefetch_model, preflight_model_prefetch


def test_model_prefetch_dry_run_reports_gated_models(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    result = prefetch_model(
        "meta-llama/Llama-3.1-8B",
        model_root=tmp_path / "models",
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["backend"] == "unsloth"
    assert result["appears_gated"] is True
    assert result["has_hf_token"] is False
    assert "transformers" in result["required_packages"]
    assert "unsloth[base]" in result["required_packages"]
    assert "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0" in result["required_packages"]


def test_model_prefetch_preflight_for_public_qwen_model(tmp_path) -> None:
    plan = preflight_model_prefetch(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        include_model=False,
        dry_run=True,
    )

    assert plan.model_id == "Qwen/Qwen3.5-4B"
    assert plan.backend == "unsloth"
    assert plan.include_model is False
    assert plan.appears_gated is False


def test_model_prefetch_preflight_can_use_transformers_backend(tmp_path) -> None:
    plan = preflight_model_prefetch(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        backend="transformers",
    )

    assert plan.backend == "transformers"
    assert "transformers" in plan.required_packages
    assert "unsloth" not in plan.required_packages


def test_model_prefetch_auto_backend_prefers_unsloth_if_either_engine_uses_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOY_MODAL_TRAINER_ENGINE", "unsloth-peft")
    monkeypatch.setenv("TOY_MODAL_SAMPLER_ENGINE", "transformers")

    plan = preflight_model_prefetch(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        backend="auto",
    )

    assert plan.backend == "unsloth"


def test_model_prefetch_auto_backend_can_be_overridden(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOY_MODAL_TRAINER_ENGINE", "unsloth-peft")
    monkeypatch.setenv("TOY_MODAL_PREFETCH_BACKEND", "transformers")

    plan = preflight_model_prefetch(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        backend="auto",
    )

    assert plan.backend == "transformers"


def test_model_prefetch_dry_run_reports_configured_unsloth_packages(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_PACKAGE", "unsloth[huggingface]")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE", "")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES", "xformers triton==3.1.0")

    plan = preflight_model_prefetch(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        backend="unsloth",
    )

    assert plan.required_packages[-3:] == ["unsloth[huggingface]", "xformers", "triton==3.1.0"]


def test_model_prefetch_unsloth_backend_uses_fast_language_model(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    calls = []

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(*, model_name: str, **kwargs):
            calls.append({"model_name": model_name, **kwargs})
            return object(), object()

    monkeypatch.setitem(
        sys.modules,
        "unsloth",
        SimpleNamespace(FastLanguageModel=FakeFastLanguageModel),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())

    result = prefetch_model(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        dry_run=False,
        backend="unsloth",
        local_files_only=True,
    )

    assert result["backend"] == "unsloth"
    assert result["loaded"] == ["tokenizer", "model"]
    assert result["cache_exists"] is True
    assert calls == [
        {
            "model_name": "Qwen/Qwen3.5-4B",
            "cache_dir": str(tmp_path / "models"),
            "local_files_only": True,
            "max_seq_length": 2048,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "load_in_16bit": False,
            "load_in_fp8": False,
            "trust_remote_code": False,
            "use_gradient_checkpointing": "unsloth",
            "fast_inference": False,
            "gpu_memory_utilization": 0.5,
            "use_exact_model_name": False,
        }
    ]


def test_model_prefetch_imports_unsloth_before_torch(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    calls: list[str] = []
    original_import = builtins.__import__

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(*, model_name: str, **kwargs):
            return object(), object()

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root == "unsloth":
            calls.append("unsloth")
            return SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)
        if root == "torch":
            calls.append("torch")
            return SimpleNamespace()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    prefetch_model(
        "Qwen/Qwen3.5-4B",
        model_root=tmp_path / "models",
        dry_run=False,
        backend="unsloth",
        local_files_only=True,
    )

    assert calls[0] == "unsloth"
    assert calls.index("unsloth") < calls.index("torch")
