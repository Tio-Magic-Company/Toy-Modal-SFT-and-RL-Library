from toy_modal.cli import main
from toy_modal.defaults import DEFAULT_BASE_MODEL
from fake_modal_backend import install_fake_modal


def test_cli_smoke_test_modal_direct_fake(capsys, monkeypatch) -> None:
    backend = install_fake_modal(monkeypatch)
    assert main(["smoke-test"]) == 0
    output = capsys.readouterr().out
    assert "loss=" in output
    assert "step=1" in output
    assert any(call[3].get("base_model") == DEFAULT_BASE_MODEL for call in backend.function_calls)


def test_cli_run_list_modal_direct_fake(capsys, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    assert main(["run", "list"]) == 0
    output = capsys.readouterr().out
    assert '"training_runs"' in output


def test_cli_cookbook_list(capsys) -> None:
    assert main(["cookbook", "list"]) == 0
    output = capsys.readouterr().out
    assert "sl_loop" in output
    assert "true_thinking_score" in output


def test_cli_cookbook_smoke(tmp_path, capsys, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    assert main(["cookbook", "smoke", "sl_loop", "--log-path", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert '"recipe": "sl_loop"' in output
    assert '"optimizer_step": 1' in output
    assert (tmp_path / "metrics.jsonl").exists()


def test_cli_cookbook_smoke_accepts_modal_target_after_subcommand(capsys, monkeypatch) -> None:
    backend = install_fake_modal(monkeypatch)
    assert (
        main(
            [
                "cookbook",
                "smoke",
                "sl_loop",
                "--app-name",
                "prod-toy-modal",
                "--environment-name",
                "production",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert backend.function_calls[0][0] == "prod-toy-modal"
    assert backend.function_calls[0][2] == "production"


def test_cli_backend_check_modal_direct_fake(capsys, monkeypatch) -> None:
    backend = install_fake_modal(monkeypatch)
    assert main(["backend", "check"]) == 0
    output = capsys.readouterr().out
    assert '"transport": "modal-direct"' in output
    assert '"training_run_id": "run_' in output
    assert any(call[3].get("base_model") == DEFAULT_BASE_MODEL for call in backend.function_calls)


def test_cli_prefetch_model_dry_run(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "backend",
                "prefetch-model",
                "Qwen/Qwen3.5-4B",
                "--dry-run",
                "--model-root",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"model_id": "Qwen/Qwen3.5-4B"' in output
    assert '"dry_run": true' in output
    assert '"backend": "unsloth"' in output


def test_cli_prefetch_model_dry_run_transformers_backend(tmp_path, capsys) -> None:
    assert (
        main(
            [
                "backend",
                "prefetch-model",
                "Qwen/Qwen3.5-4B",
                "--dry-run",
                "--backend",
                "transformers",
                "--model-root",
                str(tmp_path / "models"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"backend": "transformers"' in output
