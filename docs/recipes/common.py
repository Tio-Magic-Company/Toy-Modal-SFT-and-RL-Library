"""Shared helpers for Modal-backed cookbook smoke recipes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from toy_modal.cookbook import RecipeConfig, run_smoke_recipe
from toy_modal.defaults import DEFAULT_BASE_MODEL

DEFAULT_MODAL_BASE_MODEL = DEFAULT_BASE_MODEL


SAFE_ENV_KEYS = (
    "TOY_MODAL_APP_NAME",
    "TOY_MODAL_ENVIRONMENT",
    "TOY_MODAL_TRAINER_ENGINE",
    "TOY_MODAL_SAMPLER_ENGINE",
    "TOY_MODAL_TRAIN_GPU",
    "TOY_MODAL_SAMPLE_GPU",
    "TOY_MODAL_PREFETCH_GPU",
    "TOY_MODAL_MODEL_VOLUME",
    "TOY_MODAL_RUN_VOLUME",
    "TOY_MODAL_REGISTRY_DICT",
    "TOY_MODAL_SAMPLE_MAX_CONTAINERS",
    "TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS",
    "TOY_MODAL_HF_SECRET_NAME",
    "TOY_MODAL_UNSLOTH_LOAD_IN_4BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_8BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_16BIT",
    "TOY_MODAL_UNSLOTH_LOAD_IN_FP8",
    "TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH",
    "TOY_MODAL_UNSLOTH_DTYPE",
    "TOY_MODAL_UNSLOTH_USE_GRADIENT_CHECKPOINTING",
    "TOY_MODAL_UNSLOTH_TRUST_REMOTE_CODE",
    "TOY_MODAL_UNSLOTH_FAST_INFERENCE",
    "TOY_MODAL_UNSLOTH_GPU_MEMORY_UTILIZATION",
    "TOY_MODAL_UNSLOTH_USE_EXACT_MODEL_NAME",
    "TOY_MODAL_UNSLOTH_PACKAGE",
    "TOY_MODAL_UNSLOTH_BITSANDBYTES_PACKAGE",
    "TOY_MODAL_UNSLOTH_EXTRA_PIP_PACKAGES",
    "MODAL_PROFILE",
    "MODAL_LOGLEVEL",
)

PACKAGE_NAMES = (
    "toy-modal",
    "modal",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "unsloth",
    "unsloth_zoo",
    "trl",
    "bitsandbytes",
)


class RecipeRunLogger:
    def __init__(self, *, recipe: str, args: argparse.Namespace) -> None:
        self.recipe = recipe
        self.log_path = Path(args.log_path) if getattr(args, "log_path", None) else None
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        if self.log_path is None:
            return
        self.log_path.mkdir(parents=True, exist_ok=True)
        config = {
            "recipe": recipe,
            "run_id": self.run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "arguments": _safe_args(args),
            "environment": _safe_env(),
            "packages": _package_versions(),
            "python": sys.version,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
        }
        _write_json(self.log_path / "run_config.json", config)
        _append_jsonl(self.log_path / "run_history.jsonl", config)
        self.event("start")

    def event(self, name: str, **fields: object) -> None:
        if self.log_path is None:
            return
        _append_jsonl(
            self.log_path / "events.jsonl",
            {
                "recipe": self.recipe,
                "run_id": self.run_id,
                "event": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **fields,
            },
        )

    def success(self, record: dict[str, object]) -> None:
        if self.log_path is None:
            return
        payload = {
            "recipe": self.recipe,
            "run_id": self.run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "result": record,
        }
        _write_json(self.log_path / "result.json", payload)
        self.event("success", result_summary=_summary_fields(record))

    def failure(self, exc: BaseException) -> None:
        if self.log_path is None:
            return
        payload = {
            "recipe": self.recipe,
            "run_id": self.run_id,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(self.log_path / "error.json", payload)
        _append_jsonl(self.log_path / "errors.jsonl", payload)
        self.event("failure", error_type=type(exc).__name__, error=str(exc))


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    _append_jsonl(path, record)


def parser(description: str) -> argparse.ArgumentParser:
    arg_parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    arg_parser.add_argument("--transport", default="modal-direct")
    arg_parser.add_argument("--base-model", default=DEFAULT_MODAL_BASE_MODEL)
    arg_parser.add_argument("--project-id", default="recipe")
    arg_parser.add_argument("--app-name", default=os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    arg_parser.add_argument("--environment-name", default=os.getenv("TOY_MODAL_ENVIRONMENT"))
    arg_parser.add_argument("--base-url")
    arg_parser.add_argument("--api-key", default=os.getenv("TOY_MODAL_HTTP_API_KEY"))
    arg_parser.add_argument("--log-path")
    return arg_parser


def add_service_args(arg_parser: argparse.ArgumentParser) -> None:
    arg_parser.add_argument("--transport", default="modal-direct")
    arg_parser.add_argument("--base-model", default=DEFAULT_MODAL_BASE_MODEL)
    arg_parser.add_argument("--project-id", default="recipe")
    arg_parser.add_argument("--app-name", default=os.getenv("TOY_MODAL_APP_NAME", "toy-modal-backend"))
    arg_parser.add_argument("--environment-name", default=os.getenv("TOY_MODAL_ENVIRONMENT"))
    arg_parser.add_argument("--base-url")
    arg_parser.add_argument("--api-key", default=os.getenv("TOY_MODAL_HTTP_API_KEY"))


def service_kwargs_from_args(args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "project_id": args.project_id,
        "transport": args.transport,
        "app_name": args.app_name,
        "environment_name": args.environment_name,
    }
    if args.base_url:
        kwargs["base_url"] = args.base_url
    if args.api_key:
        kwargs["api_key"] = args.api_key
    return kwargs


def run_recipe(name: str, *, description: str) -> dict[str, object]:
    args = parser(description).parse_args()
    logger = RecipeRunLogger(recipe=name, args=args)
    try:
        result = run_smoke_recipe(
            RecipeConfig(
                name=name,
                transport=args.transport,
                base_model=args.base_model,
                project_id=args.project_id,
                app_name=args.app_name,
                environment_name=args.environment_name,
                base_url=args.base_url,
                api_key=args.api_key,
                log_path=Path(args.log_path) if args.log_path else None,
            )
        )
        record = result.to_record()
        logger.success(record)
        print(json.dumps(record, sort_keys=True))
        return record
    except Exception as exc:
        logger.failure(exc)
        raise


def _safe_args(args: argparse.Namespace) -> dict[str, object]:
    result = vars(args).copy()
    for key in list(result):
        if any(secret in key.lower() for secret in ("key", "token", "secret")) and result[key]:
            result[key] = "<set>"
    return result


def _safe_env() -> dict[str, str | None]:
    result = {key: os.getenv(key) for key in SAFE_ENV_KEYS if os.getenv(key) is not None}
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "TOY_MODAL_HTTP_API_KEY"):
        result[key] = "<set>" if os.getenv(key) else None
    return result


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _summary_fields(record: dict[str, object]) -> dict[str, object]:
    keys = (
        "training_run_id",
        "checkpoint",
        "model_path",
        "rollout_model_path",
        "loss_fn",
        "loss",
        "optimizer_step",
        "num_datums",
        "reward",
    )
    return {key: record[key] for key in keys if key in record}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
