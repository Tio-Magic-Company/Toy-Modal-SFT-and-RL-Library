"""Math RL workflow with grouped rollouts and verifiable rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import toy_modal as tinker
from toy_modal import types
from toy_modal.cookbook import (
    RolloutStore,
    collect_grouped_rollouts,
    grpo_datums_from_trajectory_groups,
)
from common import RecipeRunLogger, add_service_args, append_jsonl, service_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser)
    parser.add_argument("--log-path")
    parser.add_argument("--prompt", default="Question: 2+2? Answer with a single number:")
    parser.add_argument("--answer", default="4")
    parser.add_argument("--resume")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--loss-fn", choices=["importance_sampling", "ppo", "cispo"], default="ppo")
    args = parser.parse_args()
    logger = RecipeRunLogger(recipe="math_rl", args=args)
    try:
        record = _run(args, logger)
        logger.success(record)
        print(json.dumps(record, sort_keys=True))
    except Exception as exc:
        logger.failure(exc)
        raise


def _run(args: argparse.Namespace, logger: RecipeRunLogger) -> dict[str, object]:
    service_kwargs = service_kwargs_from_args(args)
    logger.event("create_service", transport=args.transport, project_id=args.project_id)
    service = tinker.ServiceClient(**service_kwargs)
    logger.event("create_training_client", resume=bool(args.resume), base_model=args.base_model)
    training = (
        service.create_training_client_from_state_with_optimizer(args.resume)
        if args.resume
        else service.create_lora_training_client(args.base_model, rank=4, train_unembed=False)
    )
    logger.event("get_tokenizer", training_run_id=training.training_run_id)
    tokenizer = training.get_tokenizer()
    logger.event("save_rollout_sampler")
    rollout_sampler = training.save_weights_and_get_sampling_client("math-rl-rollout")
    logger.event("rollout_sampler_ready", model_path=rollout_sampler.model_path)

    log_path = Path(args.log_path) if args.log_path else None
    store = RolloutStore(log_path / "trajectories.jsonl") if log_path else None
    logger.event("collect_rollouts", group_size=args.group_size, prompt=args.prompt)
    groups = collect_grouped_rollouts(
        sampler=rollout_sampler,
        tokenizer=tokenizer,
        prompts=[args.prompt],
        group_size=args.group_size,
        sampling_params=types.SamplingParams(max_tokens=4, temperature=0.7, top_k=8, seed=23),
        reward_fn=lambda _prompt, completion: _verifiable_math_reward(completion, args.answer),
        store=store,
    )
    logger.event(
        "rollouts_collected",
        num_groups=len(groups),
        num_trajectories=sum(len(group.trajectories) for group in groups),
        rewards=[trajectory.reward for group in groups for trajectory in group.trajectories],
    )
    datums = grpo_datums_from_trajectory_groups(
        groups,
        loss_fn=args.loss_fn,
        model_seq_id=training.model_seq_id,
    )
    if not datums:
        logger.event("fallback_non_degenerate_group")
        datums = grpo_datums_from_trajectory_groups(
            [_non_degenerate_fallback_group(tokenizer, args.prompt, args.answer)],
            loss_fn=args.loss_fn,
            model_seq_id=training.model_seq_id,
        )

    logger.event("forward_backward", loss_fn=args.loss_fn, num_datums=len(datums))
    loss = training.forward_backward(
        datums,
        args.loss_fn,
        loss_fn_config={"clip_low_threshold": 0.8, "clip_high_threshold": 1.2},
    ).result()
    logger.event("optim_step")
    step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    logger.event("save_state")
    checkpoint = training.save_state("math-rl-grpo").result()
    record = {
        "recipe": "math_rl",
        "training_run_id": training.training_run_id,
        "checkpoint": checkpoint.path,
        "rollout_model_path": rollout_sampler.model_path,
        "loss_fn": args.loss_fn,
        "loss": loss.loss,
        "optimizer_step": step.optimizer_step,
        "num_datums": len(datums),
        "rewards": [trajectory.reward for group in groups for trajectory in group.trajectories],
    }
    _write_outputs(log_path, record)
    return record


def _verifiable_math_reward(completion: str, answer: str) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", completion)
    if not numbers:
        return -1.0
    return 1.0 if numbers[-1] == answer else -1.0


def _non_degenerate_fallback_group(tokenizer, prompt: str, answer: str):
    from toy_modal.cookbook import Trajectory, TrajectoryGroup

    prompt_tokens = tokenizer.encode(prompt)
    good = tokenizer.encode(f" {answer}")
    bad = tokenizer.encode(" 0" if answer != "0" else " 1")
    return TrajectoryGroup(
        prompt=prompt,
        trajectories=[
            Trajectory(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=good,
                old_logprobs=[-0.25] * len(good),
                reward=1.0,
                text=answer,
            ),
            Trajectory(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=bad,
                old_logprobs=[-0.5] * len(bad),
                reward=-1.0,
                text="0",
            ),
        ],
    )


def _write_outputs(log_path: Path | None, record: dict[str, object]) -> None:
    if log_path is None:
        return
    log_path.mkdir(parents=True, exist_ok=True)
    append_jsonl(log_path / "metrics.jsonl", record)
    append_jsonl(
        log_path / "checkpoints.jsonl",
        {
            "recipe": record["recipe"],
            "training_run_id": record["training_run_id"],
            "checkpoint": record["checkpoint"],
            "rollout_model_path": record["rollout_model_path"],
        },
    )


if __name__ == "__main__":
    main()
