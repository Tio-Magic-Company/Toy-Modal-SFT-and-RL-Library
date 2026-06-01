import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest


TUTORIAL_FILES = [
    "000_setup_tutorial.py",
    "101_hello_toy_modal.py",
    "102_first_sft.py",
    "103_async_patterns.py",
    "104_first_rl.py",
    "201_rendering.py",
    "202_loss_functions.py",
    "203_completers.py",
    "204_weights.py",
    "205_evaluations.py",
    "301_cookbook_abstractions.py",
    "302_custom_environment.py",
    "303_sft_with_config.py",
    "304_rl_with_config.py",
    "401_sl_hyperparams.py",
    "402_rl_hyperparams.py",
    "403_dpo_preferences.py",
    "404_sequence_extension.py",
    "405_multi_agent.py",
    "406_prompt_distillation.py",
    "407_rlhf_pipeline.py",
    "501_export_hf.py",
    "502_lora_adapter.py",
    "503_publish_hub.py",
]

TUTORIAL_MODEL_DEFAULTS = {
    "101_hello_toy_modal.py": "Qwen/Qwen3-4B-Instruct-2507",
    "102_first_sft.py": "Qwen/Qwen3.5-4B",
    "103_async_patterns.py": "Qwen/Qwen3.5-4B",
    "104_first_rl.py": "meta-llama/Llama-3.1-8B",
    "201_rendering.py": "Qwen/Qwen3-30B-A3B",
    "202_loss_functions.py": "Qwen/Qwen3-4B-Instruct-2507",
    "203_completers.py": "Qwen/Qwen3-4B-Instruct-2507",
    "204_weights.py": "Qwen/Qwen3-4B-Instruct-2507",
    "205_evaluations.py": "Qwen/Qwen3-4B-Instruct-2507",
    "301_cookbook_abstractions.py": "Qwen/Qwen3.5-4B",
    "302_custom_environment.py": "Qwen/Qwen3.5-4B",
    "303_sft_with_config.py": "Qwen/Qwen3-4B-Instruct-2507",
    "304_rl_with_config.py": "Qwen/Qwen3-4B-Instruct-2507",
    "401_sl_hyperparams.py": "Qwen/Qwen3-4B-Instruct-2507",
    "402_rl_hyperparams.py": "Qwen/Qwen3-4B-Instruct-2507",
    "403_dpo_preferences.py": "Qwen/Qwen3-4B-Instruct-2507",
    "404_sequence_extension.py": "Qwen/Qwen3-4B-Instruct-2507",
    "405_multi_agent.py": "Qwen/Qwen3-4B-Instruct-2507",
    "406_prompt_distillation.py": "Qwen/Qwen3-4B-Instruct-2507",
    "407_rlhf_pipeline.py": "meta-llama/Llama-3.2-3B",
    "501_export_hf.py": "Qwen/Qwen3.5-4B",
    "502_lora_adapter.py": "Qwen/Qwen3.5-4B",
    "503_publish_hub.py": "Qwen/Qwen3.5-4B",
}


def test_tutorial_inventory_is_marimo_notebooks() -> None:
    repo_root = Path(__file__).parents[2]
    tutorials_dir = repo_root / "docs" / "tutorials" / "notebooks"
    for filename in TUTORIAL_FILES:
        path = tutorials_dir / filename
        assert path.exists(), filename
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert "import marimo" in source
        assert "app = marimo.App()" in source
        assert source.count("@app.cell") >= 8, f"{filename} has too few notebook cells"
        assert source.count("mo.md") >= 5, f"{filename} lacks teaching markdown cells"
        assert "LIMITATIONS" not in source
        assert "Please note:" in source
        assert "<details>" in source
        assert "</details>" in source
        assert "Understanding " in source
        assert "Further reading:" in source
        assert "https://" in source
        assert "import toy_modal as tinker" in source
        assert "from common" not in source
        assert 'transport="modal-direct"' in source or '"modal-direct"' in source
        assert "--i-understand-costs" in source
        assert "toy-modal backend prefetch-model" in source
        assert 'if __name__ == "__main__":' in source
        assert "app.run()" in source


def test_tutorial_model_defaults_match_parity_targets() -> None:
    repo_root = Path(__file__).parents[2]
    for filename, model_id in TUTORIAL_MODEL_DEFAULTS.items():
        source = (repo_root / "docs" / "tutorials" / "notebooks" / filename).read_text(encoding="utf-8")
        assert f'value="{model_id}"' in source, filename
    rendering = (repo_root / "docs" / "tutorials" / "notebooks" / "201_rendering.py").read_text(encoding="utf-8")
    assert "Qwen/Qwen3-VL-235B-A22B-Instruct" in rendering


def test_tutorials_are_not_thin_wrappers() -> None:
    repo_root = Path(__file__).parents[2]
    for filename in TUTORIAL_FILES:
        source = (repo_root / "docs" / "tutorials" / "notebooks" / filename).read_text(encoding="utf-8")
        assert source.count("\n") >= 150, f"{filename} is too thin to teach the workflow"
        assert "Toy Modal" in source or "toy_modal" in source
        assert "Next steps" in source
        assert "Please note:" in source


def test_setup_tutorial_documents_modal_deployment() -> None:
    repo_root = Path(__file__).parents[2]
    source = (repo_root / "docs" / "tutorials" / "notebooks" / "000_setup_tutorial.py").read_text(encoding="utf-8")
    assert "toy-modal backend deploy" in source
    assert "modal secret create huggingface-token" in source
    assert "TOY_MODAL_APP_NAME" in source
    assert "TOY_MODAL_TRAIN_GPU" in source
    assert "ServiceClient" in source
    assert "--deploy --i-understand-costs" in source


@pytest.mark.skipif(
    os.getenv("RUN_MARIMO_TUTORIAL_IMPORTS") != "1",
    reason="Marimo notebook imports are opt-in because marimo is a tutorial extra.",
)
def test_tutorials_import_when_marimo_is_installed() -> None:
    if importlib.util.find_spec("marimo") is None:
        pytest.skip("marimo is not installed")
    repo_root = Path(__file__).parents[2]
    tutorials_dir = repo_root / "docs" / "tutorials" / "notebooks"
    sys.path.insert(0, str(tutorials_dir))
    try:
        for filename in TUTORIAL_FILES:
            spec = importlib.util.spec_from_file_location(filename[:-3], tutorials_dir / filename)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert module.app is not None
    finally:
        sys.path.remove(str(tutorials_dir))


@pytest.mark.skipif(
    os.getenv("RUN_MODAL_TUTORIALS") != "1"
    or os.getenv("TOY_MODAL_I_UNDERSTAND_COSTS") != "1",
    reason="Modal tutorial execution is cost-bearing and opt-in.",
)
def test_modal_tutorial_execution_is_manual_for_marimo() -> None:
    pytest.skip(
        "Marimo notebooks are executed interactively with `marimo edit`. "
        "Use the notebook UI cost guards before running remote cells."
    )
