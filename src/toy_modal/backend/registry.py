"""Small metadata helpers for Modal Dict plus Volume-backed registry files."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from toy_modal import types


def create_run(payload: dict[str, Any], *, registry, run_root: str) -> dict[str, Any]:
    run_id = f"run_{uuid4().hex[:12]}"
    project_id = payload.get("project_id") or "default"
    lora_config = types.LoraConfig.model_validate(payload["lora_config"])
    run = types.TrainingRun(
        training_run_id=run_id,
        project_id=project_id,
        base_model=payload["base_model"],
        lora_rank=lora_config.rank,
        user_metadata=payload.get("user_metadata") or {},
    )
    record = {
        **run.model_dump(mode="json"),
        "lora_config": lora_config.model_dump(mode="json"),
        "session_id": f"session_{run_id}",
        "samplers": [],
        "latest_adapter_path": f"/runs/{project_id}/{run_id}/adapters/current",
        "latest_optimizer_path": f"/runs/{project_id}/{run_id}/optimizer/current.pt",
        "latest_gradient_id": None,
    }
    _write_registry_backup(run_root=run_root, project_id=project_id, run_id=run_id, record=record)
    registry[f"run:{run_id}"] = record
    return types.CreateTrainingRunResponse(
        training_run_id=run_id,
        project_id=project_id,
        base_model=run.base_model,
        lora_config=lora_config,
        model_seq_id=run.model_seq_id,
        optimizer_step=run.optimizer_step,
        user_metadata=run.user_metadata,
    ).model_dump(mode="json")


def mark_run_status(registry, run_id: str, status: types.RunStatus) -> dict[str, Any]:
    record = dict(registry[f"run:{run_id}"])
    record["status"] = status
    record["updated_at"] = datetime.now(timezone.utc).isoformat()
    registry[f"run:{run_id}"] = record
    return record


def _write_registry_backup(
    *,
    run_root: str,
    project_id: str,
    run_id: str,
    record: dict[str, Any],
) -> None:
    run_dir = Path(run_root) / project_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata.json").write_text(json.dumps(record, indent=2, sort_keys=True))
