import asyncio
import os
import subprocess
import sys
from functools import partial
from pathlib import Path

import toy_modal as tinker
from toy_modal import completers, renderers
from toy_modal.cookbook import (
    ProblemEnv,
    ProblemGroupBuilder,
    assemble_training_data,
    compute_advantages,
    do_group_rollout,
)
from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal


def test_renderer_registry_supervised_weights_and_parse(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    tokenizer = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL).get_tokenizer()
    renderer = renderers.get_renderer("qwen3", tokenizer)
    messages = [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hi"},
        {"role": "assistant", "content": "Hello", "train": True},
    ]

    model_input, weights = renderer.build_supervised_example(
        messages,
        train_on_what=renderers.TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    datum = renderers.conversation_to_datum(tokenizer, messages, renderer=renderer)
    parsed, termination = renderer.parse_response(tokenizer.encode("answer\nUser: next"))

    assert model_input.length() == len(weights)
    assert (weights > 0).sum().item() > 0
    assert datum.loss_fn_inputs["target_tokens"]
    assert parsed["content"] == "answer"
    assert termination.is_stop_sequence
    assert "qwen3" in renderers.get_registered_renderer_names()


def test_completers_and_problem_rollout(monkeypatch) -> None:
    install_fake_modal(monkeypatch)

    async def scenario() -> None:
        service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
        sampler = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL)
        tokenizer = sampler.get_tokenizer()
        renderer = renderers.get_renderer("role_colon", tokenizer)
        prompt = renderer.build_generation_prompt([{"role": "user", "content": "2+2?"}])
        token_completer = completers.TinkerTokenCompleter(sampler, max_tokens=2, seed=1)

        token_result = await token_completer(prompt, stop=renderer.get_stop_sequences())
        message_result = await completers.TinkerMessageCompleter(
            sampler,
            renderer,
            max_tokens=2,
            seed=2,
        )([{"role": "user", "content": "Say OK"}])

        class TinyEnv(ProblemEnv):
            def check_answer(self, text: str) -> bool:
                return bool(text)

        group = await do_group_rollout(
            ProblemGroupBuilder(partial(TinyEnv, "question", "", renderer), num_envs=2),
            token_completer,
        )
        advantages = compute_advantages([group])
        datums, metadata = assemble_training_data([group], advantages)

        assert token_result.tokens
        assert message_result["role"] == "assistant"
        assert group.trajectories_G
        assert datums
        assert metadata
        assert advantages

    asyncio.run(scenario())


def test_tutorial_parity_example_is_modal_first() -> None:
    repo_root = Path(__file__).parents[2]
    source = (repo_root / "docs" / "examples" / "tutorial_parity.py").read_text(encoding="utf-8")
    assert 'default="modal-direct"' in source
    assert "tinker.DEFAULT_BASE_MODEL" in source
