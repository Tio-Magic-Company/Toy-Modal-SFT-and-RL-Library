import ast
from pathlib import Path


EXAMPLES = [
    "sft_minimal.py",
    "rl_minimal.py",
    "sampling_batch.py",
    "checkpoint_resume.py",
    "logprob_sampling.py",
    "rest_metadata.py",
    "cookbook_smoke.py",
]


def test_examples_are_modal_first_and_parse() -> None:
    repo_root = Path(__file__).parents[2]
    for example in EXAMPLES:
        path = repo_root / "docs" / "examples" / example
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        assert "local-mock" not in source
        assert "inprocess://local-mock" not in source
        assert "modal-direct" in source or "RecipeConfig" in source or "service_from_args" in source


def test_inprocess_http_example_removed() -> None:
    repo_root = Path(__file__).parents[2]
    assert not (repo_root / "docs" / "examples" / "http_inprocess_smoke.py").exists()
