import importlib.util
import sys
from pathlib import Path

import pytest

from toy_modal import types


def test_artifact_lifecycle_defaults_to_unsloth_profile() -> None:
    module = _load_artifact_lifecycle_validation()
    args = module.build_parser().parse_args([])

    assert args.base_model == "unsloth/tinyllama-bnb-4bit"
    assert args.expected_trainer_engine == "unsloth-peft"
    assert args.expected_sampler_engine == "unsloth"


def test_artifact_lifecycle_accepts_matching_unsloth_capabilities() -> None:
    module = _load_artifact_lifecycle_validation()
    args = module.build_parser().parse_args([])
    capabilities = types.GetServerCapabilitiesResponse(
        supported_models=[args.base_model],
        trainer_engine="unsloth-peft",
        sampler_engine="unsloth",
        backend_profile={
            "uses_unsloth": True,
            "unsloth": {
                "engine_config": {"load_in_4bit": True},
                "pip_packages": ["unsloth[base]"],
                "model_families": ["Qwen"],
            },
        },
    )

    result = module._assert_backend_capabilities(capabilities, args)

    assert result["trainer_engine"] == "unsloth-peft"
    assert result["backend_profile"]["uses_unsloth"] is True


def test_artifact_lifecycle_rejects_mismatched_backend_profile() -> None:
    module = _load_artifact_lifecycle_validation()
    args = module.build_parser().parse_args([])
    capabilities = types.GetServerCapabilitiesResponse(
        supported_models=[args.base_model],
        trainer_engine="peft",
        sampler_engine="transformers",
        backend_profile={"uses_unsloth": False},
    )

    with pytest.raises(AssertionError, match="trainer_engine"):
        module._assert_backend_capabilities(capabilities, args)


def test_artifact_lifecycle_rejects_missing_supported_base_model() -> None:
    module = _load_artifact_lifecycle_validation()
    args = module.build_parser().parse_args(["--base-model", "custom/model"])
    capabilities = types.GetServerCapabilitiesResponse(
        supported_models=["unsloth/tinyllama-bnb-4bit"],
        trainer_engine="unsloth-peft",
        sampler_engine="unsloth",
        backend_profile={
            "uses_unsloth": True,
            "unsloth": {
                "engine_config": {"load_in_4bit": True},
                "pip_packages": ["unsloth[base]"],
                "model_families": ["Qwen"],
            },
        },
    )

    with pytest.raises(AssertionError, match="supported_models"):
        module._assert_backend_capabilities(capabilities, args)


def _load_artifact_lifecycle_validation():
    path = Path(__file__).parents[2] / "dev_notes" / "validation" / "artifact_lifecycle_validation.py"
    spec = importlib.util.spec_from_file_location("artifact_lifecycle_validation_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
