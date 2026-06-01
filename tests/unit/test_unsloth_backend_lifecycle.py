from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from toy_modal import types
from toy_modal.backend import peft_trainer, unsloth_engines
from toy_modal.backend.peft_trainer import PeftTrainerEngine
from toy_modal.backend.registry import create_run
from toy_modal.backend.storage import ArtifactStore
from toy_modal.backend.unsloth_engines import UnslothSamplerEngine, UnslothTrainerEngine
from toy_modal.paths import build_toy_path


def test_unsloth_trainer_checkpoint_manifest_and_sampler_adapter_load(tmp_path, monkeypatch) -> None:
    fake = _install_fake_unsloth_deps(monkeypatch)
    monkeypatch.setattr(
        unsloth_engines,
        "unsloth_runtime_versions",
        lambda: {"unsloth": "2026.5.2", "unsloth_zoo": "2026.5.2"},
    )
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "unsloth-life",
                "base_model": "Qwen/Qwen3-4B",
                "lora_config": types.LoraConfig(rank=8, alpha=16, seed=123),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )
    engine = UnslothTrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )

    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": response.training_run_id, "name": "step-1"})
    )
    checkpoint_manifest, checkpoint_dir = _manifest_for_path(tmp_path, checkpoint.path)
    sampler_weights = types.SaveWeightsForSamplerResponse.model_validate(
        engine.save_weights_for_sampler(
            {"training_run_id": response.training_run_id, "name": "sampler-1"}
        )
    )

    assert checkpoint_manifest["backend"] == "unsloth-peft"
    assert checkpoint_manifest["unsloth"]["engine"] == "unsloth-peft"
    assert checkpoint_manifest["unsloth"]["config"]["load_in_4bit"] is True
    assert checkpoint_manifest["unsloth"]["package_versions"]["unsloth"] == "2026.5.2"
    assert checkpoint_manifest["adapter_path"] == "adapter"
    assert (checkpoint_dir / "adapter" / "adapter_config.json").exists()
    assert (checkpoint_dir / "optimizer.pt").exists()
    assert fake.FastLanguageModel.from_pretrained_calls[0]["model_name"] == "Qwen/Qwen3-4B"
    assert fake.FastLanguageModel.from_pretrained_calls[0]["cache_dir"] == str(tmp_path / "models")
    assert fake.FastLanguageModel.get_peft_model_calls[0]["r"] == 8
    assert fake.Torch.manual_seed_calls == [123]

    sampler = UnslothSamplerEngine.load(
        base_model=None,
        model_path=sampler_weights.path,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    sampler._ensure_model_loaded()

    assert fake.PeftModel.from_pretrained_calls[-1]["adapter_dir"].endswith("/adapter")
    assert fake.PeftModel.from_pretrained_calls[-1]["is_trainable"] is False
    assert fake.FastLanguageModel.for_inference_calls[-1] is sampler.model
    assert sampler.model.eval_called is True


def test_unsloth_import_path_runs_before_plain_backend_deps_for_fresh_state_calls(
    tmp_path,
    monkeypatch,
) -> None:
    _install_fake_unsloth_deps(monkeypatch)
    call_order: list[str] = []

    def fake_unsloth_deps():
        call_order.append("unsloth")
        return _FakeTorch, _FakePeftModel, _FakeFastLanguageModel

    def fake_backend_deps():
        call_order.append("backend")
        return _FakeTorch, None, None, None

    monkeypatch.setattr(unsloth_engines, "_unsloth_deps", fake_unsloth_deps)
    monkeypatch.setattr(peft_trainer, "_backend_deps", fake_backend_deps)
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "fresh-calls",
                "base_model": "unsloth/tinyllama-bnb-4bit",
                "lora_config": types.LoraConfig(rank=4),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )

    save_engine = UnslothTrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    save_engine.save_state({"training_run_id": response.training_run_id, "name": "fresh-save"})

    assert call_order[0] == "unsloth"

    call_order.clear()
    record = dict(registry[f"run:{response.training_run_id}"])
    record["latest_gradient_id"] = "grad_fresh"
    registry[f"run:{response.training_run_id}"] = record
    optim_engine = UnslothTrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )

    optim_engine.optim_step(
        {
            "training_run_id": response.training_run_id,
            "adam_params": types.AdamParams(learning_rate=1e-4).model_dump(mode="json"),
        }
    )

    assert call_order[0] == "unsloth"


def test_load_state_updates_record_before_adapter_load(tmp_path) -> None:
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "restore-target",
                "base_model": "old-base",
                "lora_config": types.LoraConfig(rank=2),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )
    engine = _RecordingLoadStateTrainer(
        run_id=response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    checkpoint_path = _write_minimal_checkpoint_manifest(
        tmp_path,
        project_id="source-project",
        run_id="source-run",
        base_model="new-base",
        lora_config=types.LoraConfig(rank=12),
    )

    result = types.LoadWeightsResponse.model_validate(
        engine.load_state({"path": checkpoint_path, "optimizer": False})
    )

    assert result.model_seq_id == 7
    assert engine.loaded_base_model == "new-base"
    assert engine.loaded_lora_rank == 12
    assert engine._record()["base_model"] == "new-base"
    assert engine._record()["lora_config"]["rank"] == 12


class _RecordingLoadStateTrainer(PeftTrainerEngine):
    def _ensure_model_loaded(self, adapter_dir: Path | None = None) -> None:
        record = self._record()
        self.loaded_base_model = record["base_model"]
        self.loaded_lora_rank = record["lora_config"]["rank"]
        self._loaded_adapter_dir = adapter_dir


def _write_minimal_checkpoint_manifest(
    tmp_path: Path,
    *,
    project_id: str,
    run_id: str,
    base_model: str,
    lora_config: types.LoraConfig,
) -> str:
    run_root = tmp_path / "runs"
    store = ArtifactStore.from_runs_root(run_root)
    name = "restore"
    artifact_dir = run_root / project_id / run_id / "checkpoints" / name
    adapter_dir = artifact_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")
    toy_path = build_toy_path(project_id, run_id, "checkpoints", name)
    store.write_artifact_manifest(
        project_id,
        run_id,
        "checkpoints",
        name,
        {
            "path": toy_path,
            "backend": "unsloth-peft",
            "training_run_id": run_id,
            "project_id": project_id,
            "base_model": base_model,
            "lora_config": lora_config.model_dump(mode="json"),
            "model_seq_id": 7,
            "optimizer_step": 0,
            "adapter_path": "adapter",
            "optimizer_path": None,
            "checkpoint": {
                "checkpoint_id": name,
                "checkpoint_type": "training",
                "toy_path": toy_path,
                "size_bytes": 0,
            },
        },
    )
    return toy_path


def _manifest_for_path(tmp_path: Path, toy_path: str) -> tuple[dict[str, Any], Path]:
    _, _, rest = toy_path.partition("://")
    project_id, run_id, artifact_type, name = rest.split("/", 3)
    path = tmp_path / "runs" / project_id / run_id / artifact_type / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8")), path.parent


def _install_fake_unsloth_deps(monkeypatch):
    fake = SimpleNamespace(
        Torch=_FakeTorch,
        PeftModel=_FakePeftModel,
        FastLanguageModel=_FakeFastLanguageModel,
    )
    _FakeTorch.manual_seed_calls = []
    _FakePeftModel.from_pretrained_calls = []
    _FakeFastLanguageModel.from_pretrained_calls = []
    _FakeFastLanguageModel.get_peft_model_calls = []
    _FakeFastLanguageModel.for_training_calls = []
    _FakeFastLanguageModel.for_inference_calls = []
    monkeypatch.setattr(unsloth_engines, "_unsloth_deps", lambda: (_FakeTorch, _FakePeftModel, _FakeFastLanguageModel))
    monkeypatch.setattr(peft_trainer, "_backend_deps", lambda: (_FakeTorch, None, None, None))
    return fake


class _FakeTorch:
    manual_seed_calls: list[int] = []

    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    @classmethod
    def manual_seed(cls, seed: int) -> None:
        cls.manual_seed_calls.append(seed)

    @staticmethod
    def device(name: str) -> str:
        return name

    @staticmethod
    def save(payload: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    class optim:
        class AdamW:
            def __init__(
                self,
                params: list[Any],
                *,
                lr: float,
                betas: tuple[float, float],
                eps: float,
                weight_decay: float,
            ) -> None:
                self.params = list(params)
                self.param_groups = [
                    {
                        "lr": lr,
                        "betas": betas,
                        "eps": eps,
                        "weight_decay": weight_decay,
                    }
                ]

            def step(self) -> None:
                pass

            def zero_grad(self, *, set_to_none: bool = True) -> None:
                pass

            def state_dict(self) -> dict[str, Any]:
                return {"param_count": len(self.params)}

            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                self.loaded_state_dict = state_dict


class _FakeParameter:
    device = "fake-device"
    requires_grad = True


class _FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self) -> None:
        self.pad_token_id = None
        self._pad_token = None

    @property
    def pad_token(self) -> str | None:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, value: str) -> None:
        self._pad_token = value
        if value == self.eos_token:
            self.pad_token_id = self.eos_token_id


class _FakeModel:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=2)
        self.train_called = False
        self.eval_called = False
        self._parameters = [_FakeParameter()]

    def parameters(self):
        return iter(self._parameters)

    def train(self) -> None:
        self.train_called = True

    def eval(self) -> None:
        self.eval_called = True

    def save_pretrained(self, adapter_dir: Path, safe_serialization: bool = True) -> None:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"safe_serialization": safe_serialization}),
            encoding="utf-8",
        )
        (adapter_dir / "adapter_model.safetensors").write_text("weights", encoding="utf-8")


class _FakePeftModel:
    from_pretrained_calls: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, base_model: _FakeModel, adapter_dir: str, *, is_trainable: bool) -> _FakeModel:
        cls.from_pretrained_calls.append(
            {
                "base_model": base_model,
                "adapter_dir": adapter_dir,
                "is_trainable": is_trainable,
            }
        )
        return _FakeModel("adapter")


class _FakeFastLanguageModel:
    from_pretrained_calls: list[dict[str, Any]] = []
    get_peft_model_calls: list[dict[str, Any]] = []
    for_training_calls: list[_FakeModel] = []
    for_inference_calls: list[_FakeModel] = []

    @classmethod
    def from_pretrained(cls, *, model_name: str, **kwargs: Any) -> tuple[_FakeModel, _FakeTokenizer]:
        cls.from_pretrained_calls.append({"model_name": model_name, **kwargs})
        return _FakeModel(model_name), _FakeTokenizer()

    @classmethod
    def get_peft_model(cls, base_model: _FakeModel, **kwargs: Any) -> _FakeModel:
        cls.get_peft_model_calls.append(kwargs)
        return _FakeModel("lora")

    @classmethod
    def for_training(cls, model: _FakeModel, **kwargs: Any) -> None:
        cls.for_training_calls.append(model)

    @classmethod
    def for_inference(cls, model: _FakeModel) -> None:
        cls.for_inference_calls.append(model)
