"""Code RL workflow with user-owned sandbox execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import toy_modal as tinker
from toy_modal import types
from toy_modal.cookbook import Trajectory, TrajectoryGroup, grpo_datums_from_trajectory_groups
from common import RecipeRunLogger, add_service_args, append_jsonl, service_kwargs_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_service_args(parser)
    parser.add_argument("--log-path")
    parser.add_argument("--resume")
    parser.add_argument("--loss-fn", choices=["importance_sampling", "ppo", "cispo"], default="ppo")
    parser.add_argument("--timeout-seconds", type=float, default=1.0)
    args = parser.parse_args()
    logger = RecipeRunLogger(recipe="code_rl", args=args)
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
    prompt = "Write Python code that prints 4."
    prompt_tokens = tokenizer.encode(prompt)
    candidates = [
        "print(2 + 2)",
        "print(5)",
    ]
    trajectories = []
    sandbox_records = []
    for candidate in candidates:
        logger.event("sandbox_candidate", candidate=candidate, timeout_seconds=args.timeout_seconds)
        sandbox = _run_python_sandbox(candidate, expected_stdout="4", timeout=args.timeout_seconds)
        logger.event("sandbox_result", candidate=candidate, passed=sandbox["passed"])
        sandbox_records.append(sandbox)
        completion_tokens = tokenizer.encode(candidate)
        trajectories.append(
            Trajectory(
                prompt=prompt,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                old_logprobs=[-0.4] * len(completion_tokens),
                reward=1.0 if sandbox["passed"] else -1.0,
                text=candidate,
                metadata={"sandbox": sandbox},
            )
        )

    logger.event("build_datums", loss_fn=args.loss_fn, num_trajectories=len(trajectories))
    datums = grpo_datums_from_trajectory_groups(
        [TrajectoryGroup(prompt=prompt, trajectories=trajectories)],
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
    checkpoint = training.save_state("code-rl-sandbox").result()
    record = {
        "recipe": "code_rl",
        "training_run_id": training.training_run_id,
        "checkpoint": checkpoint.path,
        "loss_fn": args.loss_fn,
        "loss": loss.loss,
        "optimizer_step": step.optimizer_step,
        "pass_rate": sum(1 for item in sandbox_records if item["passed"]) / len(sandbox_records),
        "sandbox": sandbox_records,
    }
    _write_outputs(Path(args.log_path) if args.log_path else None, record)
    return record


def _run_python_sandbox(code: str, *, expected_stdout: str, timeout: float) -> dict[str, object]:
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "timeout": True, "stdout": "", "stderr": "timeout"}
    stdout = result.stdout.strip()
    return {
        "passed": result.returncode == 0 and stdout == expected_stdout,
        "timeout": False,
        "returncode": result.returncode,
        "stdout": stdout,
        "stderr": result.stderr.strip(),
    }


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
        },
    )


if __name__ == "__main__":
    main()
