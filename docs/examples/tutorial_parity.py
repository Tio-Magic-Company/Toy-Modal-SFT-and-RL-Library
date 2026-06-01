"""Tutorial-parity walkthrough using ``import toy_modal as tinker``.

This file mirrors the workflow categories in the checked-in original tutorials
without depending on the original package name. It uses the deployed
``modal-direct`` path and therefore requires Modal credentials and a deployed
backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from functools import partial
from pathlib import Path

import toy_modal as tinker
from toy_modal import completers, renderers
from toy_modal.cookbook import (
    ChatDatasetBuilderCommonConfig,
    InMemorySupervisedDataset,
    ProblemEnv,
    ProblemGroupBuilder,
    SupervisedTrainConfig,
    assemble_training_data,
    compute_advantages,
    do_group_rollout,
    remove_constant_reward_groups,
    run_supervised_config,
)


MODEL_NAME = tinker.DEFAULT_BASE_MODEL


class TinyChatDatasetBuilder:
    def __init__(self, tokenizer, renderer) -> None:
        self.tokenizer = tokenizer
        self.renderer = renderer
        self.common_config = ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL_NAME,
            renderer_name="role_colon",
            batch_size=2,
        )

    def __call__(self):
        examples = [
            [
                {"role": "user", "content": "What is 2 + 3?"},
                {"role": "assistant", "content": "2 + 3 = 5"},
            ],
            [
                {"role": "user", "content": "Translate hello to French."},
                {"role": "assistant", "content": "Bonjour"},
            ],
        ]
        datums = [
            renderers.conversation_to_datum(self.tokenizer, messages, renderer=self.renderer)
            for messages in examples
        ]
        return InMemorySupervisedDataset(datums, batch_size=2), None


class MathEnv(ProblemEnv):
    def get_question(self) -> str:
        return f"{self.problem}\nWrite the answer plainly."

    def check_format(self, text: str) -> bool:
        return bool(text.strip())


async def run_tutorial_parity(
    *,
    transport: str,
    app_name: str,
    environment_name: str | None,
    base_url: str | None,
    api_key: str | None,
    log_path: Path | None,
) -> dict[str, object]:
    service_kwargs = {
        "project_id": "tutorial-parity",
        "transport": transport,
        "app_name": app_name,
        "environment_name": environment_name,
    }
    if transport == "http":
        service_kwargs["base_url"] = base_url
        service_kwargs["api_key"] = api_key
    service = tinker.ServiceClient(**service_kwargs)
    capabilities = await service.get_server_capabilities_async()
    sampler = await service.create_sampling_client_async(base_model=MODEL_NAME)
    tokenizer = sampler.get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)

    messages = [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "What is the longest-lived rodent species?"},
        {"role": "assistant", "content": "The naked mole rat."},
    ]

    prompt = await asyncio.to_thread(renderer.build_generation_prompt, messages[:-1])
    sampled = await sampler.sample_async(
        prompt,
        2,
        tinker.SamplingParams(max_tokens=4, temperature=0.0, seed=1),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=2,
    )

    datum = await asyncio.to_thread(renderers.conversation_to_datum, tokenizer, messages, renderer=renderer)
    training = await service.create_lora_training_client_async(MODEL_NAME, rank=4)
    loss_future = await training.forward_backward_async([datum], "cross_entropy")
    loss = await loss_future.result_async()
    optim_future = await training.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    optim = await optim_future.result_async()
    trained_sampler = await training.save_weights_and_get_sampling_client_async("tutorial-sft")

    token_completer = completers.TinkerTokenCompleter(
        trained_sampler,
        max_tokens=4,
        temperature=0.0,
        seed=3,
    )
    token_result = await token_completer(prompt, stop=renderer.get_stop_sequences())
    message_completer = completers.TinkerMessageCompleter(
        trained_sampler,
        renderer,
        max_tokens=4,
        temperature=0.0,
        seed=4,
    )
    message_result = await message_completer([{"role": "user", "content": "Say OK."}])

    builder = ProblemGroupBuilder(
        env_thunk=partial(MathEnv, "What is 2 + 2?", "4", renderer),
        num_envs=2,
    )
    rollout_group = await do_group_rollout(builder, token_completer)
    filtered_groups = remove_constant_reward_groups([rollout_group])
    rl_groups = filtered_groups or [rollout_group]
    advantages = compute_advantages(rl_groups)
    rl_datums, metadata = assemble_training_data(rl_groups, advantages, model_seq_id=training.model_seq_id)
    if rl_datums:
        rl_loss_future = await training.forward_backward_async(rl_datums, "importance_sampling")
        rl_loss = await rl_loss_future.result_async()
        rl_optim_future = await training.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
        await rl_optim_future.result_async()
    else:
        rl_loss = None

    config_result = await asyncio.to_thread(
        run_supervised_config,
        SupervisedTrainConfig(
            dataset_builder=TinyChatDatasetBuilder(tokenizer, renderer),
            model_name=MODEL_NAME,
            transport=transport,
            app_name=app_name,
            environment_name=environment_name,
            max_steps=1,
            log_path=log_path,
        )
    )

    report = {
        "supported_models": capabilities.supported_model_names,
        "num_samples": len(sampled.samples),
        "sft_loss": loss.loss,
        "optimizer_step": optim.optimizer_step,
        "token_completion_tokens": token_result.tokens,
        "message_completion": renderers.get_text_content(message_result),
        "rollout_rewards": rollout_group.get_total_rewards(),
        "rl_loss": rl_loss.loss if rl_loss is not None else None,
        "rl_metadata_count": len(metadata),
        "config_optimizer_step": config_result.optimizer_step,
    }
    if log_path is not None:
        log_path.mkdir(parents=True, exist_ok=True)
        (log_path / "tutorial_parity_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="modal-direct")
    parser.add_argument("--app-name", default=os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    parser.add_argument("--environment-name", default=os.getenv("TOY_MODAL_ENVIRONMENT"))
    parser.add_argument("--base-url", default=os.getenv("TOY_MODAL_HTTP_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("TOY_MODAL_HTTP_API_KEY"))
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    report = asyncio.run(
        run_tutorial_parity(
            transport=args.transport,
            app_name=args.app_name,
            environment_name=args.environment_name,
            base_url=args.base_url,
            api_key=args.api_key,
            log_path=args.log_path,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
