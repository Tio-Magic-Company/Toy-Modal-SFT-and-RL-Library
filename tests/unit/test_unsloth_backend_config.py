import pytest

import toy_modal as tinker
from toy_modal.backend import unsloth_config
from toy_modal.backend.config import BackendConfig, load_config
from toy_modal.backend.unsloth_config import UnslothEngineConfig, load_unsloth_engine_config
from toy_modal.cookbook import DEFAULT_MODAL_BASE_MODEL
from toy_modal.errors import BadRequestError


UNSLOTH_ENV_VARS = [
    "TOY_MODAL_SAMPLE_GPU",
    "TOY_MODAL_PREFETCH_GPU",
    "TOY_MODAL_TRAINER_ENGINE",
    "TOY_MODAL_SAMPLER_ENGINE",
    "TOY_MODAL_UNSLOTH_LOAD_IN_4BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_8BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_16BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_FP8",
    "TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH",
    "TOY_MODAL_UNSLOTH_DTYPE",
    "TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING",
    "TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE",
    "TOY_MODAL_UNSLOTH_FAST_INFERENCE",
    "TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION",
    "TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME",
    "TOY_MODAL_UNSLOTH_PACKAGE",
    "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE",
    "TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES",
    "TOY_MODAL_SUPPORTED_MODELS",
]


def test_backend_defaults_select_unsloth(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = load_config()

    assert config.trainer_engine == "unsloth-peft"
    assert config.sampler_engine == "unsloth"
    assert config.uses_unsloth is True
    assert config.resolved_prefetch_gpu == "L40S"
    assert config.unsloth_pip_packages == (
        "unsloth[base]",
        "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0",
    )
    assert "unsloth/tinyllama-bnb-4bit" in config.resolved_supported_models
    assert "unsloth/Qwen2.5-7B-Instruct-bnb-4bit" in config.resolved_supported_models


def test_public_default_model_matches_unsloth_backend_default() -> None:
    assert tinker.DEFAULT_BASE_MODEL == "unsloth/tinyllama-bnb-4bit"
    assert DEFAULT_MODAL_BASE_MODEL == tinker.DEFAULT_BASE_MODEL


def test_backend_uses_unsloth_when_either_engine_is_unsloth() -> None:
    assert BackendConfig(trainer_engine="peft", sampler_engine="transformers").uses_unsloth is False
    assert BackendConfig(trainer_engine="unsloth-peft", sampler_engine="transformers").uses_unsloth is True
    assert BackendConfig(trainer_engine="peft", sampler_engine="unsloth").uses_unsloth is True


def test_backend_prefetch_gpu_defaults_to_unsloth_sample_gpu() -> None:
    assert BackendConfig(trainer_engine="peft", sampler_engine="transformers").resolved_prefetch_gpu is None
    assert BackendConfig(sampler_engine="unsloth", sample_gpu="A10G").resolved_prefetch_gpu == "A10G"
    assert BackendConfig(prefetch_gpu="H100", sampler_engine="unsloth").resolved_prefetch_gpu == "H100"


def test_backend_supported_models_can_be_overridden(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_SUPPORTED_MODELS", "custom/a, custom/b")

    config = load_config()

    assert config.resolved_supported_models == ("custom/a", "custom/b")


def test_backend_parses_unsloth_package_overrides(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_PACKAGE", "unsloth[huggingface]")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE", "")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES", "xformers triton==3.1.0")

    config = load_config()

    assert config.unsloth_pip_packages == ("unsloth[huggingface]", "xformers", "triton==3.1.0")


def test_backend_validates_unsloth_quantization_at_load_time(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", "1")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_8BIT", "1")

    with pytest.raises(BadRequestError, match="only one"):
        load_config()


def test_backend_skips_unsloth_validation_for_plain_engines(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_TRAINER_ENGINE", "peft")
    monkeypatch.setenv("TOY_MODAL_SAMPLER_ENGINE", "transformers")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", "1")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_8BIT", "1")

    config = load_config()

    assert config.uses_unsloth is False


def test_unsloth_env_parsing_and_model_kwargs(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", "0")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_16BIT", "1")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH", "4096")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_DTYPE", "bfloat16")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING", "0")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE", "1")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_FAST_INFERENCE", "1")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION", "0.75")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME", "1")

    config = load_unsloth_engine_config()
    kwargs = config.model_kwargs(token="token")

    assert kwargs["load_in_4bit"] is False
    assert kwargs["load_in_16bit"] is True
    assert kwargs["max_seq_length"] == 4096
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["use_gradient_checkpointing"] is False
    assert kwargs["trust_remote_code"] is True
    assert kwargs["fast_inference"] is True
    assert kwargs["gpu_memory_utilization"] == 0.75
    assert kwargs["use_exact_model_name"] is True
    assert kwargs["token"] == "token"


def test_unsloth_fp8_env_accepts_named_mode(monkeypatch) -> None:
    for name in UNSLOTH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_4BIT", "0")
    monkeypatch.setenv("TOY_MODAL_UNSLOTH_LOAD_IN_FP8", "block")

    config = load_unsloth_engine_config()

    assert config.model_kwargs()["load_in_fp8"] == "block"


def test_unsloth_rejects_multiple_quant_modes() -> None:
    config = UnslothEngineConfig(load_in_4bit=True, load_in_8bit=True)

    with pytest.raises(BadRequestError, match="only one"):
        config.validate()


def test_unsloth_rejects_invalid_memory_fraction() -> None:
    config = UnslothEngineConfig(gpu_memory_utilization=1.5)

    with pytest.raises(BadRequestError, match="interval"):
        config.validate()


def test_unsloth_runtime_versions_are_metadata_only(monkeypatch) -> None:
    monkeypatch.setattr(unsloth_config, "UNSLOTH_RUNTIME_PACKAGES", ("unsloth", "missing"))

    def fake_version(package: str) -> str:
        if package == "unsloth":
            return "2026.5.2"
        raise unsloth_config.PackageNotFoundError(package)

    monkeypatch.setattr(unsloth_config, "version", fake_version)

    assert unsloth_config.unsloth_runtime_versions() == {
        "unsloth": "2026.5.2",
        "missing": None,
    }
