"""Tiny supervised fine-tuning workflow with eval and resume hooks."""

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
    parser.add_argument("--dataset", help="Optional JSONL with prompt/completion fields")
    parser.add_argument("--resume")
    parser.add_argument("--eval-output")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    logger = RecipeRunLogger(recipe="tiny_sft_workflow", args=args)
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
    rows = _load_rows(args.dataset) if args.dataset else [
        {"prompt": "Question: 2+2? Answer:", "completion": " 4"},
        {"prompt": "Question: 3+5? Answer:", "completion": " 8"},
    ]
    logger.event("load_dataset", num_rows=len(rows), dataset=args.dataset)
    data = []
    for row in rows:
        prompt = tokenizer.encode(row["prompt"])
        completion = tokenizer.encode(row["completion"])
        data.append(
            types.Datum(
                model_input=types.ModelInput.from_ints(prompt + completion),
                loss_fn_inputs={
                    "target_tokens": completion,
                    "weights": [0.0] * len(prompt) + [1.0] * len(completion),
                },
            )
        )
    validate_training_batch(data, "cross_entropy")
    logger.event("forward_backward", loss_fn="cross_entropy", num_datums=len(data))
    loss = training.forward_backward(data, "cross_entropy").result()
    logger.event("optim_step")
    step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    logger.event("save_state")
    checkpoint = training.save_state("tiny-sft-workflow").result()
    logger.event("save_sampler")
    sampler = training.save_weights_and_get_sampling_client("tiny-sft-workflow-sampler")
    logger.event("sampler_saved", model_path=sampler.model_path)
    sample = sampler.sample(
        types.ModelInput.from_ints(tokenizer.encode("Question: 4+6? Answer:")),
        1,
        types.SamplingParams(max_tokens=4, seed=11),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=2,
    ).result()
    logger.event("sample_complete", sample_tokens=sample.samples[0].tokens)
    record = {
        "recipe": "tiny_sft_workflow",
        "training_run_id": training.training_run_id,
        "checkpoint": checkpoint.path,
        "model_path": sampler.model_path,
        "loss": loss.loss,
        "optimizer_step": step.optimizer_step,
        "sample": sample.model_dump(mode="json"),
    }
    _write_outputs(args.log_path, args.eval_output, record)
    return record


def _load_rows(path: str) -> list[dict[str, str]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
                "model_path": record["model_path"],
            },
        )
    if eval_output:
        Path(eval_output).write_text(json.dumps(record["sample"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
