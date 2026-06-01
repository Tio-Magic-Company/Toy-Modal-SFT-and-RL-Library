import builtins
from types import SimpleNamespace

from toy_modal import types
from toy_modal.backend.registry import create_run
from toy_modal.backend.sampler_worker import load_sampler_engine
from toy_modal.backend.trainer_worker import load_trainer_engine
from toy_modal.backend.unsloth_config import UnslothEngineConfig
from toy_modal.backend.unsloth_engines import (
    UnslothSamplerEngine,
    UnslothTrainerEngine,
    _resolve_model_kwargs,
    _unsloth_deps,
    _unsloth_lora_kwargs,
)


def test_trainer_loader_selects_unsloth_engine(tmp_path) -> None:
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "unsloth",
                "base_model": "Qwen/Qwen3-4B",
                "lora_config": types.LoraConfig(rank=4, seed=123),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )

    engine = load_trainer_engine(
        "unsloth-peft",
        run_id=response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )

    assert isinstance(engine, UnslothTrainerEngine)


def test_sampler_loader_selects_unsloth_engine(tmp_path) -> None:
    engine = load_sampler_engine(
        "unsloth",
        base_model="Qwen/Qwen3-4B",
        model_path=None,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )

    assert isinstance(engine, UnslothSamplerEngine)
    assert engine.base_model == "Qwen/Qwen3-4B"
    assert engine.model is None


def test_unsloth_lora_kwargs_preserve_public_lora_config() -> None:
    lora_config = types.LoraConfig(
        rank=8,
        alpha=16,
        dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        seed=99,
    )
    unsloth_config = UnslothEngineConfig(use_gradient_checkpointing="unsloth")

    kwargs = _unsloth_lora_kwargs("meta-llama/Llama-3.1-8B", lora_config, unsloth_config)

    assert kwargs == {
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 99,
        "max_seq_length": 2048,
    }


def test_unsloth_dtype_string_resolves_to_torch_dtype() -> None:
    class FakeTorch:
        bfloat16 = object()

    kwargs = _resolve_model_kwargs({"dtype": "bfloat16"}, FakeTorch)

    assert kwargs["dtype"] is FakeTorch.bfloat16


def test_unsloth_deps_imports_unsloth_before_peft(monkeypatch) -> None:
    calls: list[str] = []
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root == "unsloth":
            calls.append("unsloth")
            return SimpleNamespace(FastLanguageModel=object())
        if root == "peft":
            calls.append("peft")
            return SimpleNamespace(PeftModel=object())
        if root == "torch":
            calls.append("torch")
            return SimpleNamespace()
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    _unsloth_deps()

    assert calls[0] == "unsloth"
    assert calls.index("unsloth") < calls.index("peft")
