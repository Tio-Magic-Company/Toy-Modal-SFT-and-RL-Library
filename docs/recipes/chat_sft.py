"""Conversational SFT recipe with JSONL loading and chat rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import toy_modal as tinker
from toy_modal import types
from toy_modal.cookbook import (
    TrainLoopConfig,
    load_conversation_jsonl,
    render_conversation_datums,
    run_supervised_train_loop,
)
from common import RecipeRunLogger, add_service_args, append_jsonl, service_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser)
    parser.add_argument("--dataset")
    parser.add_argument("--resume")
    parser.add_argument("--log-path")
    args = parser.parse_args()
    logger = RecipeRunLogger(recipe="chat_sft", args=args)
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
    dataset_path = Path(args.dataset) if args.dataset else _default_dataset()
    logger.event("load_dataset", dataset_path=str(dataset_path))
    datums = render_conversation_datums(tokenizer, load_conversation_jsonl(dataset_path))
    logger.event("render_datums", num_datums=len(datums))
    loop = run_supervised_train_loop(
        training,
        datums,
        TrainLoopConfig(
            loss_fn="cross_entropy",
            learning_rate=1e-4,
            checkpoint_prefix="chat-sft",
        ),
    )
    logger.event("train_loop_complete", optimizer_step=loop.optimizer_step, losses=loop.losses)
    sampler = training.save_weights_and_get_sampling_client("chat-sft-sampler")
    logger.event("sampler_saved", model_path=sampler.model_path)
    sample = sampler.sample(
        types.ModelInput.from_ints(tokenizer.encode("User: Say hello\nAssistant:")),
        1,
        types.SamplingParams(max_tokens=3, seed=11),
    ).result()
    logger.event("sample_complete", sample_tokens=sample.samples[0].tokens)
    record = {
        "recipe": "chat_sft",
        "training_run_id": training.training_run_id,
        "checkpoint": loop.checkpoints[-1] if loop.checkpoints else None,
        "model_path": sampler.model_path,
        "loss": loop.losses[-1],
        "optimizer_step": loop.optimizer_step,
        "sample_tokens": sample.samples[0].tokens,
        "num_datums": len(datums),
    }
    _write_outputs(Path(args.log_path) if args.log_path else None, record)
    return record


def _default_dataset() -> Path:
    path = Path(tempfile.gettempdir()) / "toy-modal-chat-sft-default.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Say hello."},
                    {"role": "assistant", "content": "Hello."},
                ]
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


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
            "model_path": record["model_path"],
        },
    )


if __name__ == "__main__":
    main()
