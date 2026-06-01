import importlib.util
import json
import sys
from pathlib import Path

from fake_modal_backend import install_fake_modal


def test_stage1_modal_parity_report_covers_fake_modal_workflow(monkeypatch, tmp_path) -> None:
    install_fake_modal(monkeypatch)
    module = _load_modal_parity_report()
    report_dir = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "modal_parity_report.py",
            "--app-name",
            "toy-modal-test",
            "--report-dir",
            str(report_dir),
            "--timeout",
            "5",
            "--sample-max-tokens",
            "2",
            "--num-samples",
            "1",
            "--quiet",
        ],
    )

    module.main()

    reports = sorted(report_dir.glob("modal_parity_*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["summary"]["fail"] == 0

    steps = {step["name"]: step for step in report["steps"]}
    capabilities = steps["server.capabilities"]
    assert capabilities["status"] == "pass"
    assert capabilities["result"]["trainer_engine"] == "unsloth-peft"
    assert capabilities["result"]["sampler_engine"] == "unsloth"
    assert capabilities["result"]["backend_profile"]["uses_unsloth"] is True
    assert "unsloth" in capabilities["result"]["backend_profile"]
    for name in [
        "training.forward_cross_entropy",
        "training.forward_backward_then_optim_step",
        "training.forward_rl_losses",
        "service.create_training_client_from_state_variants",
        "sampling.service_model_path_and_helper_clients",
        "rest.training_checkpoint_metadata",
    ]:
        assert steps[name]["status"] == "pass"

    stale = steps["training.stale_old_logprobs_guard"]
    assert stale["status"] == "pass"
    assert stale["result"]["accepted"] is False
    assert stale["result"]["error_type"] == "StaleModelSequenceError"


def test_modal_parity_report_default_model_tracks_backend(monkeypatch) -> None:
    module = _load_modal_parity_report()
    monkeypatch.delenv("TOY_MODAL_TRAINER_ENGINE", raising=False)
    monkeypatch.delenv("TOY_MODAL_SAMPLER_ENGINE", raising=False)

    assert module._default_base_model_from_env() == "unsloth/tinyllama-bnb-4bit"

    monkeypatch.setenv("TOY_MODAL_TRAINER_ENGINE", "peft")
    monkeypatch.setenv("TOY_MODAL_SAMPLER_ENGINE", "transformers")

    assert module._default_base_model_from_env() == "hf-internal-testing/tiny-random-gpt2"


def _load_modal_parity_report():
    path = Path(__file__).parents[2] / "dev_notes" / "validation" / "modal_parity_report.py"
    spec = importlib.util.spec_from_file_location("modal_parity_report_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
