"""Volume-backed metadata and artifact route handlers for deployed backends."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from toy_modal import types
from toy_modal.backend.storage import ArtifactStore, file_checksum, manifest_expired
from toy_modal.errors import BadRequestError, CheckpointNotFoundError, NotFoundError, RunNotFoundError
from toy_modal.paths import build_toy_path, parse_toy_path


def handle_metadata_route(
    route: str,
    payload: dict[str, Any],
    *,
    registry,
    run_root: str,
    archive_url_prefix: str,
) -> Any:
    service = MetadataService(registry=registry, run_root=run_root, archive_url_prefix=archive_url_prefix)
    handlers = {
        "rest.get_training_run": service.get_training_run,
        "rest.get_training_run_by_toy_path": service.get_training_run_by_toy_path,
        "rest.get_weights_info_by_toy_path": service.get_weights_info_by_toy_path,
        "rest.list_training_runs": service.list_training_runs,
        "rest.list_checkpoints": service.list_checkpoints,
        "rest.list_user_checkpoints": service.list_user_checkpoints,
        "rest.get_checkpoint_archive_url": service.get_checkpoint_archive_url,
        "rest.get_checkpoint_archive_url_from_toy_path": service.get_checkpoint_archive_url_from_toy_path,
        "rest.inspect_checkpoint_artifact_from_toy_path": service.inspect_checkpoint_artifact_from_toy_path,
        "rest.delete_checkpoint": service.delete_checkpoint,
        "rest.delete_checkpoint_from_toy_path": service.delete_checkpoint_from_toy_path,
        "rest.set_checkpoint_ttl_from_toy_path": service.set_checkpoint_ttl_from_toy_path,
        "rest.set_checkpoint_public": service.set_checkpoint_public,
        "rest.get_session": service.get_session,
        "rest.list_sessions": service.list_sessions,
        "rest.get_sampler": service.get_sampler,
    }
    try:
        return handlers[route](payload)
    except KeyError as exc:
        raise BadRequestError(f"Unsupported metadata route: {route}") from exc


class MetadataService:
    def __init__(self, *, registry, run_root: str, archive_url_prefix: str) -> None:
        self.registry = registry
        self.store = ArtifactStore.from_runs_root(run_root)
        self.run_root = Path(run_root)
        self.archive_url_prefix = archive_url_prefix.rstrip("/")

    def get_training_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._public_run(payload["training_run_id"])

    def get_training_run_by_toy_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        return self._public_run(parsed.run_id)

    def get_weights_info_by_toy_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest = self._manifest_from_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        checkpoint_type = (
            "training" if manifest.get("checkpoint", {}).get("checkpoint_type") == "training" else "sampler"
        )
        return types.WeightsInfoResponse(
            path=payload["toy_path"],
            base_model=manifest["base_model"],
            lora_rank=(manifest.get("lora_config") or {}).get("rank"),
            checkpoint_type=checkpoint_type,
        ).model_dump(mode="json")

    def list_training_runs(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = int(payload.get("limit", 20))
        offset = int(payload.get("offset", 0))
        runs = [
            types.TrainingRun.model_validate(self._public_record(record))
            for record in self._all_run_records()
        ]
        runs.sort(key=lambda run: run.created_at, reverse=True)
        return types.TrainingRunsResponse(
            training_runs=runs[offset : offset + limit],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(runs)),
        ).model_dump(mode="json")

    def list_checkpoints(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = self._record(payload["training_run_id"])
        self._cleanup_expired_artifacts(run)
        manifests = self.store.list_artifact_manifests(
            run.get("project_id") or "default",
            payload["training_run_id"],
        )
        checkpoints = self._checkpoints_from_manifests(manifests)
        return types.CheckpointsListResponse(
            checkpoints=checkpoints,
            cursor=types.Cursor(limit=len(checkpoints), offset=0, total_count=len(checkpoints)),
        ).model_dump(mode="json")

    def list_user_checkpoints(self, payload: dict[str, Any]) -> dict[str, Any]:
        for record in self._all_run_records():
            self._cleanup_expired_artifacts(record)
        limit = int(payload.get("limit", 100))
        offset = int(payload.get("offset", 0))
        manifests: list[dict[str, Any]] = []
        for record in self._all_run_records():
            project_id = record.get("project_id") or "default"
            run_id = record["training_run_id"]
            manifests.extend(self.store.list_artifact_manifests(project_id, run_id))
        checkpoints = self._checkpoints_from_manifests(manifests)
        checkpoints.sort(key=lambda checkpoint: checkpoint.time, reverse=True)
        return types.CheckpointsListResponse(
            checkpoints=checkpoints[offset : offset + limit],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(checkpoints)),
        ).model_dump(mode="json")

    def get_checkpoint_archive_url(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = self._record(payload["training_run_id"])
        project_id = run.get("project_id") or "default"
        manifest = self._manifest_by_checkpoint_id(
            project_id,
            payload["training_run_id"],
            payload["checkpoint_id"],
        )
        archive_name = payload["checkpoint_id"].replace("/", "_")
        archive_path = self.store.layout.archive_path(project_id, payload["training_run_id"], archive_name)
        self._write_checkpoint_archive(manifest, archive_path)
        return types.CheckpointArchiveUrlResponse(
            url=f"{self.archive_url_prefix}/{archive_path.relative_to(self.run_root)}",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        ).model_dump(mode="json")

    def get_checkpoint_archive_url_from_toy_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        parsed = parse_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        return self.get_checkpoint_archive_url(
            {"training_run_id": parsed.run_id, "checkpoint_id": parsed.name}
        )

    def inspect_checkpoint_artifact_from_toy_path(self, payload: dict[str, Any]) -> dict[str, Any]:
        manifest, manifest_path = self._manifest_and_path_from_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        artifact_dir = manifest_path.parent
        parsed = parse_toy_path(manifest["path"])
        archive_name = parsed.name.replace("/", "_")
        archive_path = self.store.layout.archive_path(parsed.project_id, parsed.run_id, archive_name)
        archive_metadata_path = self.store.layout.archive_metadata_path(parsed.project_id, parsed.run_id, archive_name)
        archive_metadata = None
        if archive_metadata_path.exists():
            archive_metadata = self.store.read_json(archive_metadata_path)
        volume_paths = [
            {
                "path": str(path.relative_to(self.run_root)),
                "size_bytes": path.stat().st_size,
                "sha256": file_checksum(path),
            }
            for path in sorted(artifact_dir.rglob("*"))
            if path.is_file() and not path.is_symlink()
        ]
        archive_files = []
        for path in (archive_path, archive_metadata_path):
            if path.exists():
                archive_files.append(
                    {
                        "path": str(path.relative_to(self.run_root)),
                        "size_bytes": path.stat().st_size,
                        "sha256": file_checksum(path),
                    }
                )
        return {
            "toy_path": manifest["path"],
            "artifact_dir": str(artifact_dir.relative_to(self.run_root)),
            "manifest_path": str(manifest_path.relative_to(self.run_root)),
            "manifest": manifest,
            "volume_files": volume_paths,
            "archive_files": archive_files,
            "archive_metadata": archive_metadata,
        }

    def delete_checkpoint(self, payload: dict[str, Any]) -> None:
        run = self._record(payload["training_run_id"])
        project_id = run.get("project_id") or "default"
        manifest = self._manifest_by_checkpoint_id(
            project_id,
            payload["training_run_id"],
            payload["checkpoint_id"],
            validate_files=False,
        )
        artifact_dir = self._artifact_dir_for_manifest(manifest)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        self._remove_archives_for_checkpoint(project_id, payload["training_run_id"], payload["checkpoint_id"])
        self._remove_checkpoint_from_run(run, manifest["path"])
        self._save_record(run)
        return None

    def delete_checkpoint_from_toy_path(self, payload: dict[str, Any]) -> None:
        parsed = parse_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        return self.delete_checkpoint(
            {"training_run_id": parsed.run_id, "checkpoint_id": parsed.name}
        )

    def set_checkpoint_ttl_from_toy_path(self, payload: dict[str, Any]) -> None:
        ttl_seconds = payload.get("ttl_seconds")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise BadRequestError("ttl_seconds must be positive or None")
        manifest, path = self._manifest_and_path_from_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        expires_at = None
        if ttl_seconds is not None:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(ttl_seconds))
        manifest["checkpoint"]["expires_at"] = (
            expires_at.isoformat() if expires_at is not None else None
        )
        self.store.write_json(path, manifest)
        self._replace_checkpoint_in_run(manifest)
        return None

    def set_checkpoint_public(self, payload: dict[str, Any]) -> None:
        manifest, path = self._manifest_and_path_from_toy_path(
            payload["toy_path"],
            accept_tinker_paths=payload.get("accept_tinker_paths", False),
        )
        manifest["checkpoint"]["public"] = bool(payload["public"])
        self.store.write_json(path, manifest)
        self._replace_checkpoint_in_run(manifest)
        return None

    def get_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = payload["session_id"]
        runs = [
            record
            for record in self._all_run_records()
            if record.get("session_id") == session_id
        ]
        if not runs:
            raise NotFoundError(f"session not found: {session_id}")
        sampler_ids = [
            sampler["sampler_id"]
            for record in runs
            for sampler in record.get("samplers", [])
        ]
        return types.GetSessionResponse(
            session_id=session_id,
            training_run_ids=[record["training_run_id"] for record in runs],
            sampler_ids=sampler_ids,
            user_metadata=runs[0].get("user_metadata") or {},
        ).model_dump(mode="json")

    def list_sessions(self, payload: dict[str, Any]) -> dict[str, Any]:
        limit = int(payload.get("limit", 20))
        offset = int(payload.get("offset", 0))
        session_ids = sorted(
            {
                record.get("session_id")
                for record in self._all_run_records()
                if record.get("session_id")
            }
        )
        return types.ListSessionsResponse(
            sessions=session_ids[offset : offset + limit],
            cursor=types.Cursor(limit=limit, offset=offset, total_count=len(session_ids)),
        ).model_dump(mode="json")

    def get_sampler(self, payload: dict[str, Any]) -> dict[str, Any]:
        for record in self._all_run_records():
            for sampler in record.get("samplers", []):
                if sampler.get("sampler_id") == payload["sampler_id"]:
                    return types.GetSamplerResponse.model_validate(sampler).model_dump(mode="json")
        raise NotFoundError(f"sampler not found: {payload['sampler_id']}")

    def _record(self, run_id: str) -> dict[str, Any]:
        key = f"run:{run_id}"
        try:
            return dict(self.registry[key])
        except Exception:
            pass
        record = self.store.find_run_metadata(run_id)
        if record is None:
            raise RunNotFoundError(run_id)
        return record

    def _save_record(self, record: dict[str, Any]) -> None:
        key = f"run:{record['training_run_id']}"
        try:
            self.registry[key] = record
        except Exception:
            pass
        self.store.write_run_metadata(
            record.get("project_id") or "default",
            record["training_run_id"],
            record,
        )

    def _all_run_records(self) -> list[dict[str, Any]]:
        records: dict[str, dict[str, Any]] = {}
        try:
            for key in self.registry:
                if str(key).startswith("run:"):
                    record = dict(self.registry[key])
                    records[record["training_run_id"]] = record
        except Exception:
            pass
        for path in sorted(self.store.layout.runs_dir.glob("*/*/metadata.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records.setdefault(record["training_run_id"], record)
        return list(records.values())

    def _public_run(self, run_id: str) -> dict[str, Any]:
        return self._public_record(self._record(run_id))

    @staticmethod
    def _public_record(record: dict[str, Any]) -> dict[str, Any]:
        public_fields = types.TrainingRun.model_fields
        return {key: value for key, value in record.items() if key in public_fields}

    def _manifest_from_path(self, toy_path: str, *, accept_tinker_paths: bool = False) -> dict[str, Any]:
        manifest, _ = self._manifest_and_path_from_toy_path(
            toy_path,
            accept_tinker_paths=accept_tinker_paths,
        )
        return manifest

    def _manifest_and_path_from_toy_path(
        self,
        toy_path: str,
        *,
        accept_tinker_paths: bool = False,
    ) -> tuple[dict[str, Any], Path]:
        parsed = parse_toy_path(toy_path, accept_tinker_paths=accept_tinker_paths)
        path = self.store.layout.artifact_manifest_path(
            parsed.project_id,
            parsed.run_id,
            parsed.artifact_type,
            parsed.name,
        )
        if not path.exists():
            raise CheckpointNotFoundError(toy_path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self._raise_if_expired(manifest)
        self.store.validate_manifest_files(manifest)
        return manifest, path

    def _manifest_by_checkpoint_id(
        self,
        project_id: str,
        run_id: str,
        checkpoint_id: str,
        *,
        validate_files: bool = True,
    ) -> dict[str, Any]:
        for manifest in self.store.list_artifact_manifests(project_id, run_id):
            self._raise_if_expired(manifest)
            checkpoint = manifest.get("checkpoint") or {}
            if checkpoint.get("checkpoint_id") == checkpoint_id:
                if validate_files:
                    self.store.validate_manifest_files(manifest)
                return manifest
        raise CheckpointNotFoundError(checkpoint_id)

    def _artifact_dir_for_manifest(self, manifest: dict[str, Any]) -> Path:
        parsed = parse_toy_path(manifest["path"])
        return self.store.layout.artifact_dir(
            parsed.project_id,
            parsed.run_id,
            parsed.artifact_type,
            parsed.name,
        )

    @staticmethod
    def _checkpoints_from_manifests(manifests: list[dict[str, Any]]) -> list[types.Checkpoint]:
        checkpoints = []
        for manifest in manifests:
            checkpoint = manifest.get("checkpoint")
            if checkpoint is not None:
                checkpoints.append(types.Checkpoint.model_validate(checkpoint))
        return checkpoints

    def _replace_checkpoint_in_run(self, manifest: dict[str, Any]) -> None:
        parsed = parse_toy_path(manifest["path"])
        run = self._record(parsed.run_id)
        checkpoints = run.get("checkpoints", [])
        run["checkpoints"] = [
            manifest["checkpoint"] if item.get("toy_path") == manifest["path"] else item
            for item in checkpoints
        ]
        if manifest["checkpoint"]["checkpoint_type"] == "training":
            run["last_checkpoint"] = manifest["checkpoint"]
        else:
            run["last_sampler_checkpoint"] = manifest["checkpoint"]
        run["updated_at"] = _now()
        self._save_record(run)

    def _remove_checkpoint_from_run(self, run: dict[str, Any], toy_path: str) -> None:
        run["checkpoints"] = [
            checkpoint
            for checkpoint in run.get("checkpoints", [])
            if checkpoint.get("toy_path") != toy_path
        ]
        run["samplers"] = [
            sampler
            for sampler in run.get("samplers", [])
            if sampler.get("model_path") != toy_path
        ]
        if (run.get("last_checkpoint") or {}).get("toy_path") == toy_path:
            run["last_checkpoint"] = None
        if (run.get("last_sampler_checkpoint") or {}).get("toy_path") == toy_path:
            run["last_sampler_checkpoint"] = None
        run["updated_at"] = _now()

    def _cleanup_expired_artifacts(self, run: dict[str, Any]) -> None:
        project_id = run.get("project_id") or "default"
        changed = False
        for manifest in list(self.store.list_artifact_manifests(project_id, run["training_run_id"])):
            if self._is_expired(manifest):
                artifact_dir = self._artifact_dir_for_manifest(manifest)
                if artifact_dir.exists():
                    shutil.rmtree(artifact_dir)
                checkpoint_id = (manifest.get("checkpoint") or {}).get("checkpoint_id")
                if checkpoint_id:
                    self._remove_archives_for_checkpoint(project_id, run["training_run_id"], checkpoint_id)
                self._remove_checkpoint_from_run(run, manifest["path"])
                changed = True
        if changed:
            self._save_record(run)

    @staticmethod
    def _is_expired(manifest: dict[str, Any]) -> bool:
        return manifest_expired(manifest)

    def _raise_if_expired(self, manifest: dict[str, Any]) -> None:
        if self._is_expired(manifest):
            artifact_dir = self._artifact_dir_for_manifest(manifest)
            if artifact_dir.exists():
                shutil.rmtree(artifact_dir)
            run = self._record(parse_toy_path(manifest["path"]).run_id)
            project_id = run.get("project_id") or "default"
            checkpoint_id = (manifest.get("checkpoint") or {}).get("checkpoint_id")
            if checkpoint_id:
                self._remove_archives_for_checkpoint(project_id, run["training_run_id"], checkpoint_id)
            self._remove_checkpoint_from_run(run, manifest["path"])
            self._save_record(run)
            raise CheckpointNotFoundError(manifest["path"])

    def _write_checkpoint_archive(self, manifest: dict[str, Any], archive_path: Path) -> dict[str, Any]:
        artifact_dir = self._artifact_dir_for_manifest(manifest)
        self.store.validate_manifest_files(manifest, artifact_dir)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
        members = []
        root_name = artifact_dir.name
        with tarfile.open(tmp_path, "w:gz") as archive:
            for item in manifest.get("artifact_files", []):
                relative = Path(item["path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe artifact path in manifest: {item['path']!r}")
                source = artifact_dir / relative
                if source.is_symlink():
                    raise ValueError(f"unsafe symlink in artifact directory: {source}")
                arcname = Path(root_name) / relative
                archive.add(source, arcname=str(arcname), recursive=False)
                members.append(
                    {
                        "path": str(arcname),
                        "size_bytes": int(item["size_bytes"]),
                        "sha256": item["sha256"],
                    }
                )
            manifest_source = artifact_dir / "manifest.json"
            archive.add(manifest_source, arcname=str(Path(root_name) / "manifest.json"), recursive=False)
            members.append(
                {
                    "path": str(Path(root_name) / "manifest.json"),
                    "size_bytes": manifest_source.stat().st_size,
                    "sha256": file_checksum(manifest_source),
                }
            )
        tmp_path.replace(archive_path)
        metadata = {
            "schema_version": 1,
            "archive_path": str(archive_path.relative_to(self.run_root)),
            "source_toy_path": manifest["path"],
            "created_at": _now(),
            "size_bytes": archive_path.stat().st_size,
            "sha256": file_checksum(archive_path),
            "members": members,
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }
        archive_name = archive_path.name.removesuffix(".tar.gz")
        parsed = parse_toy_path(manifest["path"])
        metadata_path = self.store.layout.archive_metadata_path(parsed.project_id, parsed.run_id, archive_name)
        self.store.write_json(metadata_path, metadata)
        return metadata

    def _remove_archives_for_checkpoint(self, project_id: str, run_id: str, checkpoint_id: str) -> None:
        archive_name = checkpoint_id.replace("/", "_")
        archive_path = self.store.layout.archive_path(project_id, run_id, archive_name)
        metadata_path = self.store.layout.archive_metadata_path(project_id, run_id, archive_name)
        for path in (
            archive_path,
            metadata_path,
            archive_path.with_suffix(archive_path.suffix + ".tmp"),
        ):
            if path.exists():
                path.unlink()


def create_training_run_from_state(
    payload: dict[str, Any],
    *,
    registry,
    run_root: str,
) -> dict[str, Any]:
    from toy_modal.backend.registry import create_run

    store = ArtifactStore.from_runs_root(run_root)
    parsed = parse_toy_path(
        payload["path"],
        accept_tinker_paths=payload.get("accept_tinker_paths", False),
    )
    manifest_path = store.layout.artifact_manifest_path(
        parsed.project_id,
        parsed.run_id,
        parsed.artifact_type,
        parsed.name,
    )
    if not manifest_path.exists():
        raise CheckpointNotFoundError(payload["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    store.raise_if_manifest_expired(manifest)
    store.validate_manifest_files(manifest)
    optimizer_path = manifest.get("optimizer_path")
    restore_optimizer = bool(payload.get("optimizer")) and bool(optimizer_path)
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": payload.get("project_id") or manifest.get("project_id") or parsed.project_id,
                "base_model": manifest["base_model"],
                "lora_config": types.LoraConfig.model_validate(manifest["lora_config"]),
                "user_metadata": payload.get("user_metadata") or {},
            },
            registry=registry,
            run_root=run_root,
        )
    )
    key = f"run:{response.training_run_id}"
    record = dict(registry[key])
    record["model_seq_id"] = manifest["model_seq_id"]
    record["optimizer_step"] = manifest["optimizer_step"] if restore_optimizer else 0
    record["optimizer_state"] = manifest.get("optimizer_state", {}) if restore_optimizer else {}
    record["latest_gradient_id"] = None
    record["pending_load_state_path"] = manifest["path"]
    record["pending_load_optimizer"] = restore_optimizer
    record["updated_at"] = _now()
    registry[key] = record
    store.write_run_metadata(record.get("project_id") or "default", response.training_run_id, record)
    return types.LoadWeightsResponse(
        path=payload["path"],
        training_run_id=response.training_run_id,
        model_seq_id=record["model_seq_id"],
        optimizer_step=record["optimizer_step"],
        lora_config=types.LoraConfig.model_validate(record["lora_config"]),
    ).model_dump(mode="json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
