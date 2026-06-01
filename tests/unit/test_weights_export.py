from __future__ import annotations

import json
import sys
import types as module_types
from pathlib import Path
from typing import Any

import pytest

from toy_modal import weights


def test_build_hf_model_dry_run_defaults_to_unsloth_export_plan(tmp_path) -> None:
    output = weights.build_hf_model(
        base_model="unsloth/tinyllama-bnb-4bit",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "hf",
        merge=True,
        dry_run=True,
    )

    manifest = json.loads((output / "toy_modal_hf_manifest.json").read_text(encoding="utf-8"))

    assert manifest["backend"] == "unsloth"
    assert manifest["save_method"] == "merged_16bit"
    assert "unsloth" in manifest["requires"]
    assert "peft" in manifest["requires"]


def test_build_hf_model_can_plan_transformers_export(tmp_path) -> None:
    output = weights.build_hf_model(
        base_model="hf-internal-testing/tiny-random-gpt2",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "hf",
        backend="transformers",
        dry_run=True,
    )

    manifest = json.loads((output / "toy_modal_hf_manifest.json").read_text(encoding="utf-8"))

    assert manifest["backend"] == "transformers"
    assert "transformers" in manifest["requires"]
    assert "unsloth" not in manifest["requires"]


def test_build_hf_model_unsloth_uses_fast_language_model_before_peft(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[str] = []
    unsloth_module = module_types.ModuleType("unsloth")
    peft_module = module_types.ModuleType("peft")
    unsloth_module.FastLanguageModel = _FakeFastLanguageModel
    peft_module.PeftModel = _FakePeftModel
    _FakeFastLanguageModel.calls = calls
    _FakePeftModel.calls = calls

    monkeypatch.setitem(sys.modules, "unsloth", unsloth_module)
    monkeypatch.setitem(sys.modules, "peft", peft_module)
    monkeypatch.delenv("TOY_MODAL_UNSLOTH_DTYPE", raising=False)

    output = weights.build_hf_model(
        base_model="unsloth/tinyllama-bnb-4bit",
        adapter_dir=tmp_path / "adapter",
        output_dir=tmp_path / "hf",
        backend="unsloth",
        merge=True,
        local_files_only=True,
    )

    assert calls[0] == "fast_language_model.from_pretrained"
    assert calls.index("fast_language_model.from_pretrained") < calls.index("peft.from_pretrained")
    assert (output / "unsloth-save.json").exists()
    saved = json.loads((output / "unsloth-save.json").read_text(encoding="utf-8"))
    assert saved["save_method"] == "merged_16bit"
    assert saved["tokenizer"] == "saved"


def test_build_hf_model_rejects_unknown_backend(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported HF export backend"):
        weights.build_hf_model(
            base_model="model",
            adapter_dir=tmp_path / "adapter",
            output_dir=tmp_path / "hf",
            backend="unknown",
            dry_run=True,
        )


class _FakeFastLanguageModel:
    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, *, model_name: str, **kwargs: Any):
        cls.calls.append("fast_language_model.from_pretrained")
        assert model_name == "unsloth/tinyllama-bnb-4bit"
        assert kwargs["local_files_only"] is True
        return _FakeModel(), _FakeTokenizer()

    @classmethod
    def for_inference(cls, model: Any) -> Any:
        cls.calls.append("fast_language_model.for_inference")
        model.inference = True
        return model


class _FakePeftModel:
    calls: list[str] = []

    @classmethod
    def from_pretrained(cls, model: Any, adapter_dir: str, *, is_trainable: bool):
        cls.calls.append("peft.from_pretrained")
        assert is_trainable is False
        return _FakePeftExportModel()


class _FakeModel:
    pass


class _FakeTokenizer:
    def save_pretrained(self, output_dir: Path) -> None:
        Path(output_dir, "tokenizer.json").write_text("{}", encoding="utf-8")


class _FakePeftExportModel:
    def save_pretrained_merged(
        self,
        output_dir: str,
        tokenizer: _FakeTokenizer,
        *,
        save_method: str,
    ) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(path)
        (path / "unsloth-save.json").write_text(
            json.dumps({"save_method": save_method, "tokenizer": "saved"}),
            encoding="utf-8",
        )
