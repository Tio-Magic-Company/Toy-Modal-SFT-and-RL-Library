"""Basic on-policy RL workflow using sampler rollout logprobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import toy_modal as tinker
from toy_modal import types
from toy_modal.backend.loss_inputs import validate_training_batch
from common import RecipeRunLogger, add_service_args, append_jsonl, service_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser)
    parser.add_argument("--prompt", default="Question: 2+2? Answer:")
    parser.add_argument("--answer", default="4")
    parser.add_argument("--resume")
    parser.add_argument("--eval-output")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    logger = RecipeRunLogger(recipe="on_policy_rl_workflow", args=args)
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
    prompt_tokens = tokenizer.encode(args.prompt)
    logger.event("save_rollout_sampler")
    rollout_sampler = training.save_weights_and_get_sampling_client("on-policy-rollout")
    logger.event("rollout_sampler_ready", model_path=rollout_sampler.model_path)
    logger.event("sample_rollout", prompt_token_count=len(prompt_tokens))
    rollout = rollout_sampler.sample(
        types.ModelInput.from_ints(prompt_tokens),
        1,
        types.SamplingParams(max_tokens=4, temperature=0.7, top_k=8, seed=19),
    ).result()
    sequence = rollout.samples[0]
    completion = sequence.tokens[len(prompt_tokens) :]
    old_logprobs = sequence.logprobs or rollout_sampler.compute_logprobs(
        types.ModelInput.from_ints(sequence.tokens)
    ).result()[len(prompt_tokens) :]
    reward = 1.0 if args.answer in tokenizer.decode(completion) else -0.25
    logger.event(
        "rollout_complete",
        completion_tokens=completion,
        reward=reward,
        old_logprob_count=len(old_logprobs),
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(sequence.tokens),
        loss_fn_inputs={
            "target_tokens": completion,
            "old_logprobs": old_logprobs,
            "advantages": [reward] * len(completion),
            "weights": [1.0] * len(completion),
            "old_logprobs_model_seq_id": training.model_seq_id,
        },
    )
    validate_training_batch([datum], "importance_sampling")
    logger.event("forward_backward", loss_fn="importance_sampling")
    loss = training.forward_backward([datum], "importance_sampling").result()
    logger.event("optim_step")
    step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    logger.event("save_state")
    checkpoint = training.save_state("on-policy-rl-workflow").result()
    record = {
        "recipe": "on_policy_rl_workflow",
        "training_run_id": training.training_run_id,
        "checkpoint": checkpoint.path,
        "rollout_model_path": rollout_sampler.model_path,
        "reward": reward,
        "loss": loss.loss,
        "optimizer_step": step.optimizer_step,
        "sample_tokens": sequence.tokens,
        "old_logprobs": old_logprobs,
    }
    _write_outputs(args.log_path, args.eval_output, record)
    return record


def _write_outputs(log_path: str | None, eval_output: str | None, record: dict) -> None:
    if log_path:
        path = Path(log_path)
        path.mkdir(parents=True, exist_ok=True)
        append_jsonl(path / "metrics.jsonl", record)
        append_jsonl(
            path / "checkpoints.jsonl",
            {
                "recipe": record["recipe"],
                "training_run_id": record["training_run_id"],
                "checkpoint": record["checkpoint"],
                "rollout_model_path": record["rollout_model_path"],
            },
        )
    if eval_output:
        Path(eval_output).write_text(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
