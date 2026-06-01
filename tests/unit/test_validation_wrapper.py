import importlib.util
import sys
from pathlib import Path


def test_modal_validation_wrapper_defaults_to_unsloth_profile() -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args([])

    assert args.app_name == "toy-modal-unsloth-validation"
    assert args.project_id == "modal-unsloth-validation"
    assert args.trainer_engine == "unsloth-peft"
    assert args.sampler_engine == "unsloth"
    assert args.base_model is None
    assert args.prefetch_gpu == ""
    assert args.run_training_report is False
    assert args.require_unsloth_import is False
    assert args.unsloth_package == "unsloth[base]"
    assert args.unsloth_bitsandbytes_package == "bitsandbytes>=0.45.5,!=0.46.0,!=0.48.0"
    assert args.unsloth_extra_pip_packages == ""
    assert args.supported_models == ""
    assert args.skip_supported_model_check is False
    assert module._default_base_model(args.trainer_engine, args.sampler_engine) == "unsloth/tinyllama-bnb-4bit"
    assert module._install_extras(args) == ".[backend,unsloth,dev]"


def test_modal_validation_wrapper_can_install_plain_peft_profile() -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(
        ["--trainer-engine", "peft", "--sampler-engine", "transformers"]
    )

    assert module._default_base_model(args.trainer_engine, args.sampler_engine) == "hf-internal-testing/tiny-random-gpt2"
    assert module._install_extras(args) == ".[backend,dev]"


def test_modal_validation_wrapper_builds_optional_unsloth_dependency_probe() -> None:
    module = _load_run_modal_validation()
    command = module._unsloth_dependency_probe_command("/python", required=False)

    assert command[:2] == ["/python", "-c"]
    assert "from unsloth import FastLanguageModel" in command[2]
    assert command[2].index("from unsloth import FastLanguageModel") < command[2].index(
        "from peft import PeftModel"
    )
    assert "required = False" in command[2]
    assert "bitsandbytes" in command[2]


def test_modal_validation_wrapper_can_require_unsloth_dependency_probe() -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(["--require-unsloth-import"])
    command = module._unsloth_dependency_probe_command("/python", required=args.require_unsloth_import)

    assert args.require_unsloth_import is True
    assert "required = True" in command[2]


def test_modal_validation_wrapper_preserves_explicit_base_model() -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(
        ["--base-model", "custom/model", "--trainer-engine", "unsloth-peft"]
    )

    assert args.base_model == "custom/model"


def test_modal_validation_wrapper_training_report_uses_generic_entrypoint(tmp_path) -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(
        [
            "--report-root",
            str(tmp_path),
            "--validation-id",
            "unit",
            "--run-training-report",
        ]
    )
    args.base_model = module._default_base_model(args.trainer_engine, args.sampler_engine)
    runner = module.ValidationRun(args)
    command = runner._training_report_command()

    assert args.run_training_report is True
    assert "dev_notes/validation/modal_training_validation.py" in command
    assert "--expected-trainer-engine" in command
    assert "unsloth-peft" in command
    assert "--expected-sampler-engine" in command
    assert "unsloth" in command


def test_modal_validation_wrapper_propagates_supported_model_options(tmp_path) -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(
        [
            "--report-root",
            str(tmp_path),
            "--validation-id",
            "unit",
            "--supported-models",
            "custom/model",
            "--skip-supported-model-check",
        ]
    )
    args.base_model = "custom/model"
    runner = module.ValidationRun(args)

    assert runner.env["TOY_MODAL_SUPPORTED_MODELS"] == "custom/model"
    assert "--skip-supported-model-check" in runner._modal_report_command()
    assert "--skip-supported-model-check" in runner._training_report_command()
    assert "--skip-supported-model-check" in runner._artifact_lifecycle_command()


def test_modal_validation_wrapper_accepts_legacy_training_report_alias() -> None:
    module = _load_run_modal_validation()
    args = module.build_parser().parse_args(["--run-legacy-report"])

    assert args.run_training_report is True


def _load_run_modal_validation():
    path = Path(__file__).parents[2] / "dev_notes" / "validation" / "run_modal_validation.py"
    spec = importlib.util.spec_from_file_location("run_modal_validation_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
