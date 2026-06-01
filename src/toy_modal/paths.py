"""Helpers for canonical toy-modal artifact paths."""

from __future__ import annotations

from toy_modal import types


def build_toy_path(project_id: str, run_id: str, artifact_type: str, name: str) -> str:
    return f"toy-modal://{project_id}/{run_id}/{artifact_type}/{name}"


def parse_toy_path(path: str, *, accept_tinker_paths: bool = False) -> types.ParsedToyPath:
    scheme, sep, rest = path.partition("://")
    if not sep:
        raise ValueError(f"Invalid toy_modal path: {path!r}")
    was_tinker_path = scheme == "tinker"
    if was_tinker_path and accept_tinker_paths:
        scheme = "toy-modal"
    if scheme != "toy-modal":
        raise ValueError(f"Unsupported path scheme: {scheme!r}")

    parts = [part for part in rest.split("/") if part]
    if was_tinker_path and len(parts) == 3:
        run_id, artifact_type, name = parts
        if artifact_type == "weights":
            artifact_type = "checkpoints"
        if artifact_type == "sampler":
            artifact_type = "sampler_weights"
        return types.ParsedToyPath(
            scheme="toy-modal",
            project_id="default",
            run_id=run_id,
            artifact_type=artifact_type,
            name=name,
        )
    if len(parts) < 4:
        raise ValueError(f"Invalid toy_modal path: {path!r}")
    project_id, run_id, artifact_type = parts[:3]
    if was_tinker_path and artifact_type == "weights":
        artifact_type = "checkpoints"
    if was_tinker_path and artifact_type == "sampler":
        artifact_type = "sampler_weights"
    name = "/".join(parts[3:])
    return types.ParsedToyPath(
        scheme="toy-modal",
        project_id=project_id,
        run_id=run_id,
        artifact_type=artifact_type,
        name=name,
    )
