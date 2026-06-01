import os
import subprocess
import sys
from pathlib import Path


RECIPE_FILES = [
    "sl_loop.py",
    "rl_loop.py",
    "sl_basic.py",
    "rl_basic.py",
    "chat_sft.py",
    "math_rl.py",
    "code_rl.py",
    "preference_dpo.py",
    "prompt_distillation.py",
    "model_distillation.py",
    "tool_use.py",
    "multi_agent.py",
    "rubric_grading.py",
    "verifier_environment.py",
    "vlm_image_classification.py",
    "harbor_rl.py",
    "sdft.py",
    "true_thinking_score.py",
    "eval_scaffold.py",
    "tiny_sft_workflow.py",
    "on_policy_rl_workflow.py",
]


def test_all_cookbook_recipes_parse_help_without_modal_credentials() -> None:
    repo_root = Path(__file__).parents[2]
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repo_root / "src"),
    }
    for recipe in RECIPE_FILES:
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "docs" / "recipes" / recipe),
                "--help",
            ],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "--transport" in result.stdout
        assert "local-mock" not in result.stdout
