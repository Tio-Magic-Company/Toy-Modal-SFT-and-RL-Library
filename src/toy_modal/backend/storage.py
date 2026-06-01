"""Durable storage helpers for backend run and artifact metadata."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class VolumeLayout:
    """Path builder for the canonical toy_modal backend Volume layout."""

    root: Path
    runs_dir_name: str | None = "runs"

    @classmethod
    def from_root(cls, root: str | Path) -> "VolumeLayout":
        return cls(root=Path(root))

    @classmethod
    def from_runs_root(cls, root: str | Path) -> "VolumeLayout":
        return cls(root=Path(root), runs_dir_name=None)

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def runs_dir(self) -> Path:
        return self.root if self.runs_dir_name is None else self.root / self.runs_dir_name

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def archives_dir(self) -> Path:
        return self.root / "archives"

    def run_dir(self, project_id: str, run_id: str) -> Path:
        return self.runs_dir / project_id / run_id

    def run_metadata_path(self, project_id: str, run_id: str) -> Path:
        return self.run_dir(project_id, run_id) / "metadata.json"

    def artifact_dir(self, project_id: str, run_id: str, artifact_type: str, name: str) -> Path:
        return self.run_dir(project_id, run_id) / artifact_type / name

    def artifact_manifest_path(
        self,
        project_id: str,
        run_id: str,
        artifact_type: str,
        name: str,
    ) -> Path:
        return self.artifact_dir(project_id, run_id, artifact_type, name) / "manifest.json"

    def log_path(self, project_id: str, run_id: str, log_name: str = "events.jsonl") -> Path:
        return self.logs_dir / project_id / run_id / log_name

    def archive_path(self, project_id: str, run_id: str, name: str) -> Path:
        return self.archives_dir / project_id / run_id / f"{name}.tar.gz"

    def archive_metadata_path(self, project_id: str, run_id: str, name: str) -> Path:
        archive_path = self.archive_path(project_id, run_id, name)
        return archive_path.with_suffix(archive_path.suffix + ".metadata.json")


class ArtifactStore:
    """Small JSON store used by local tests and Modal workers.

    Modal Volumes require explicit commits in the worker layer. This class only
    handles deterministic paths and atomic local writes; callers remain
    responsible for committing/reloading the backing Volume when running on
    Modal.
    """

    def __init__(self, root: str | Path) -> None:
        self.layout = VolumeLayout.from_root(root)

    @classmethod
    def from_runs_root(cls, root: str | Path) -> "ArtifactStore":
        store = cls.__new__(cls)
        store.layout = VolumeLayout.from_runs_root(root)
        return store

    def ensure_layout(self) -> None:
        for path in (
            self.layout.models_dir,
            self.layout.runs_dir,
            self.layout.logs_dir,
            self.layout.archives_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def write_run_metadata(self, project_id: str, run_id: str, record: dict[str, Any]) -> Path:
        path = self.layout.run_metadata_path(project_id, run_id)
        self.write_json(path, record)
        return path

    def read_run_metadata(self, project_id: str, run_id: str) -> dict[str, Any]:
        return self.read_json(self.layout.run_metadata_path(project_id, run_id))

    def find_run_metadata(self, run_id: str) -> dict[str, Any] | None:
        matches = sorted(self.layout.runs_dir.glob(f"*/{run_id}/metadata.json"))
        if not matches:
            return None
        return self.read_json(matches[0])

    def write_artifact_manifest(
        self,
        project_id: str,
        run_id: str,
        artifact_type: str,
        name: str,
        manifest: dict[str, Any],
    ) -> Path:
        path = self.layout.artifact_manifest_path(project_id, run_id, artifact_type, name)
        manifest = self.enrich_manifest(manifest, path.parent)
        self.write_json(path, manifest)
        return path

    def temp_artifact_dir(self, project_id: str, run_id: str, artifact_type: str, name: str) -> Path:
        final_dir = self.layout.artifact_dir(project_id, run_id, artifact_type, name)
        return final_dir.parent / f".{name}.{uuid4().hex}.tmp"

    def promote_artifact_dir(
        self,
        temp_dir: str | Path,
        project_id: str,
        run_id: str,
        artifact_type: str,
        name: str,
    ) -> Path:
        source = Path(temp_dir)
        target = self.layout.artifact_dir(project_id, run_id, artifact_type, name)
        if not source.exists():
            raise FileNotFoundError(f"temporary artifact directory does not exist: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        backup = None
        if target.exists():
            backup = target.parent / f".{target.name}.{uuid4().hex}.bak"
            target.replace(backup)
        try:
            source.replace(target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                backup.replace(target)
            raise
        if backup is not None and backup.exists():
            shutil.rmtree(backup)
        return target

    def enrich_manifest(self, manifest: dict[str, Any], artifact_dir: str | Path) -> dict[str, Any]:
        enriched = dict(manifest)
        enriched.setdefault("schema_version", MANIFEST_SCHEMA_VERSION)
        enriched.setdefault("version", MANIFEST_SCHEMA_VERSION)
        enriched.setdefault("created_at", _now())
        enriched.setdefault("artifact_type", _artifact_type_from_manifest(enriched))
        files = checksum_files(Path(artifact_dir))
        enriched["artifact_files"] = files
        enriched["size_bytes"] = sum(item["size_bytes"] for item in files)
        enriched["file_count"] = len(files)
        enriched["checksum_algorithm"] = "sha256"
        enriched["has_optimizer"] = bool(enriched.get("optimizer_path"))
        if "checkpoint" in enriched and isinstance(enriched["checkpoint"], dict):
            enriched["checkpoint"] = {
                **enriched["checkpoint"],
                "size_bytes": enriched["size_bytes"],
                "public": bool(enriched["checkpoint"].get("public", False)),
                "expires_at": enriched["checkpoint"].get("expires_at"),
            }
        return enriched

    def validate_manifest_files(self, manifest: dict[str, Any], artifact_dir: str | Path | None = None) -> None:
        artifact_dir = Path(artifact_dir) if artifact_dir is not None else self._artifact_dir_from_manifest(manifest)
        validate_manifest_files(manifest, artifact_dir)

    def raise_if_manifest_expired(self, manifest: dict[str, Any]) -> None:
        if manifest_expired(manifest):
            from toy_modal.errors import CheckpointNotFoundError

            raise CheckpointNotFoundError(manifest.get("path", "expired checkpoint"))

    def read_artifact_manifest(
        self,
        project_id: str,
        run_id: str,
        artifact_type: str,
        name: str,
    ) -> dict[str, Any]:
        return self.read_json(
            self.layout.artifact_manifest_path(project_id, run_id, artifact_type, name)
        )

    def list_artifact_manifests(
        self,
        project_id: str,
        run_id: str,
        artifact_type: str | None = None,
    ) -> list[dict[str, Any]]:
        base = self.layout.run_dir(project_id, run_id)
        roots: Iterable[Path]
        if artifact_type is None:
            roots = (item for item in base.iterdir() if item.is_dir()) if base.exists() else ()
        else:
            roots = [base / artifact_type]
        manifests: list[dict[str, Any]] = []
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*/manifest.json")):
                manifests.append(self.read_json(path))
        return manifests

    def append_jsonl(self, path: str | Path, record: dict[str, Any]) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return target

    @staticmethod
    def read_json(path: str | Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def write_json(path: str | Path, data: dict[str, Any]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(target)

    def _artifact_dir_from_manifest(self, manifest: dict[str, Any]) -> Path:
        from toy_modal.paths import parse_toy_path

        parsed = parse_toy_path(manifest["path"])
        return self.layout.artifact_dir(
            parsed.project_id,
            parsed.run_id,
            parsed.artifact_type,
            parsed.name,
        )


def checksum_files(artifact_dir: Path) -> list[dict[str, Any]]:
    if not artifact_dir.exists():
        return []
    files: list[dict[str, Any]] = []
    for path in sorted(artifact_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"unsafe symlink in artifact directory: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append(
            {
                "path": str(path.relative_to(artifact_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return files


def validate_manifest_files(manifest: dict[str, Any], artifact_dir: Path) -> None:
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact directory missing: {artifact_dir}")
    for item in manifest.get("artifact_files", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact path in manifest: {item['path']!r}")
        path = artifact_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"artifact file missing: {path}")
        data = path.read_bytes()
        if len(data) != int(item["size_bytes"]):
            raise ValueError(f"artifact file size mismatch: {path}")
        digest = hashlib.sha256(data).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"artifact file checksum mismatch: {path}")


def file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_expired(manifest: dict[str, Any]) -> bool:
    expires_at = manifest_expiration(manifest)
    return expires_at is not None and expires_at <= datetime.now(timezone.utc)


def manifest_expiration(manifest: dict[str, Any]) -> datetime | None:
    expires_at = (manifest.get("checkpoint") or {}).get("expires_at")
    if not expires_at:
        return None
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at


def _artifact_type_from_manifest(manifest: dict[str, Any]) -> str | None:
    path = manifest.get("path")
    if not path or "://" not in str(path):
        return None
    try:
        from toy_modal.paths import parse_toy_path

        return parse_toy_path(str(path)).artifact_type
    except Exception:
        return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
