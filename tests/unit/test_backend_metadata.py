from toy_modal import types
from toy_modal.backend.storage import ArtifactStore
from toy_modal.backend.metadata import handle_metadata_route
from toy_modal.backend.registry import create_run
from toy_modal.backend.trainer_worker import TrainerEngine
from datetime import datetime, timedelta, timezone


def test_volume_backed_metadata_routes_cover_artifact_lifecycle(tmp_path) -> None:
    registry = {}
    run_root = str(tmp_path / "runs")
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "meta",
                "base_model": "local-model",
                "lora_config": types.LoraConfig(rank=4),
                "user_metadata": {"suite": "metadata"},
            },
            registry=registry,
            run_root=run_root,
        )
    )
    engine = TrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=run_root,
    )
    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": response.training_run_id, "name": "step"})
    )
    sampler_weights = types.SaveWeightsForSamplerResponse.model_validate(
        engine.save_weights_for_sampler(
            {"training_run_id": response.training_run_id, "name": "sampler"}
        )
    )

    run = _route("rest.get_training_run", {"training_run_id": response.training_run_id}, registry, run_root)
    assert run["training_run_id"] == response.training_run_id

    checkpoints = types.CheckpointsListResponse.model_validate(
        _route("rest.list_checkpoints", {"training_run_id": response.training_run_id}, registry, run_root)
    )
    assert {item.checkpoint_type for item in checkpoints.checkpoints} == {"training", "sampler"}

    _route(
        "rest.set_checkpoint_ttl_from_toy_path",
        {"toy_path": checkpoint.path, "ttl_seconds": 60},
        registry,
        run_root,
    )
    _route(
        "rest.set_checkpoint_public",
        {"toy_path": checkpoint.path, "public": True},
        registry,
        run_root,
    )
    updated = types.CheckpointsListResponse.model_validate(
        _route("rest.list_checkpoints", {"training_run_id": response.training_run_id}, registry, run_root)
    ).checkpoints
    published = next(item for item in updated if item.checkpoint_id == "step")
    assert published.public is True
    assert published.expires_at is not None

    archive = types.CheckpointArchiveUrlResponse.model_validate(
        _route(
            "rest.get_checkpoint_archive_url_from_toy_path",
            {"toy_path": checkpoint.path},
            registry,
            run_root,
        )
    )
    assert archive.url.startswith("modal-volume://toy-modal-test-runs/")
    assert archive.url.endswith(".tar.gz")
    store = ArtifactStore.from_runs_root(run_root)
    archive_path = store.layout.archives_dir / "meta" / response.training_run_id / "step.tar.gz"
    assert archive_path.exists()
    archive_metadata_path = store.layout.archive_metadata_path("meta", response.training_run_id, "step")
    assert archive_metadata_path.exists()
    archive_metadata = store.read_json(archive_metadata_path)
    assert archive_metadata["source_toy_path"] == checkpoint.path
    assert archive_metadata["sha256"]

    inspection = _route(
        "rest.inspect_checkpoint_artifact_from_toy_path",
        {"toy_path": checkpoint.path},
        registry,
        run_root,
    )
    assert inspection["manifest"]["schema_version"] == 1
    assert inspection["manifest"]["artifact_type"] == "checkpoints"
    assert inspection["archive_metadata"]["source_toy_path"] == checkpoint.path

    session_ids = types.ListSessionsResponse.model_validate(
        _route("rest.list_sessions", {}, registry, run_root)
    ).sessions
    assert session_ids
    session = types.GetSessionResponse.model_validate(
        _route("rest.get_session", {"session_id": session_ids[0]}, registry, run_root)
    )
    assert response.training_run_id in session.training_run_ids
    assert session.sampler_ids
    sampler = types.GetSamplerResponse.model_validate(
        _route("rest.get_sampler", {"sampler_id": session.sampler_ids[0]}, registry, run_root)
    )
    assert sampler.model_path == sampler_weights.path

    _route(
        "rest.delete_checkpoint_from_toy_path",
        {"toy_path": checkpoint.path},
        registry,
        run_root,
    )
    remaining = types.CheckpointsListResponse.model_validate(
        _route("rest.list_checkpoints", {"training_run_id": response.training_run_id}, registry, run_root)
    ).checkpoints
    assert all(item.checkpoint_id != "step" for item in remaining)
    assert not archive_path.exists()
    assert not archive_metadata_path.exists()


def test_metadata_routes_cleanup_expired_checkpoints(tmp_path) -> None:
    registry = {}
    run_root = str(tmp_path / "runs")
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "meta",
                "base_model": "local-model",
                "lora_config": types.LoraConfig(rank=4),
            },
            registry=registry,
            run_root=run_root,
        )
    )
    engine = TrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=run_root,
    )
    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": response.training_run_id, "name": "expired"})
    )
    store = ArtifactStore.from_runs_root(run_root)
    manifest_path = store.layout.artifact_manifest_path(
        "meta",
        response.training_run_id,
        "checkpoints",
        "expired",
    )
    manifest = store.read_json(manifest_path)
    manifest["checkpoint"]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(seconds=1)
    ).isoformat()
    store.write_json(manifest_path, manifest)

    remaining = types.CheckpointsListResponse.model_validate(
        _route("rest.list_checkpoints", {"training_run_id": response.training_run_id}, registry, run_root)
    ).checkpoints

    assert all(item.checkpoint_id != "expired" for item in remaining)


def _route(route: str, payload: dict, registry, run_root: str):
    return handle_metadata_route(
        route,
        payload,
        registry=registry,
        run_root=run_root,
        archive_url_prefix="modal-volume://toy-modal-test-runs",
    )
