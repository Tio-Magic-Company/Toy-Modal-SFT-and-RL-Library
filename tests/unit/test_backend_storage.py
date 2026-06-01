from toy_modal.backend.storage import ArtifactStore, VolumeLayout


def test_volume_layout_paths_are_stable(tmp_path) -> None:
    layout = VolumeLayout.from_root(tmp_path)

    assert layout.models_dir == tmp_path / "models"
    assert layout.runs_dir == tmp_path / "runs"
    assert layout.logs_dir == tmp_path / "logs"
    assert layout.archives_dir == tmp_path / "archives"
    assert (
        layout.artifact_manifest_path("project", "run_1", "checkpoints", "step-1")
        == tmp_path / "runs" / "project" / "run_1" / "checkpoints" / "step-1" / "manifest.json"
    )


def test_artifact_store_writes_run_and_manifest_metadata(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    store.ensure_layout()

    store.write_run_metadata("project", "run_1", {"training_run_id": "run_1", "model_seq_id": 2})
    store.write_artifact_manifest(
        "project",
        "run_1",
        "checkpoints",
        "step-2",
        {"path": "toy-modal://project/run_1/checkpoints/step-2", "model_seq_id": 2},
    )
    store.append_jsonl(tmp_path / "logs" / "project" / "run_1" / "events.jsonl", {"event": "saved"})

    assert store.read_run_metadata("project", "run_1")["model_seq_id"] == 2
    assert store.find_run_metadata("run_1")["training_run_id"] == "run_1"
    manifests = store.list_artifact_manifests("project", "run_1", "checkpoints")
    assert manifests[0]["path"] == "toy-modal://project/run_1/checkpoints/step-2"
    assert manifests[0]["model_seq_id"] == 2
    assert manifests[0]["artifact_files"] == []
    assert manifests[0]["size_bytes"] == 0
    assert (tmp_path / "logs" / "project" / "run_1" / "events.jsonl").read_text()


def test_artifact_store_can_use_runs_volume_root(tmp_path) -> None:
    runs_root = tmp_path / "runs-volume"
    store = ArtifactStore.from_runs_root(runs_root)

    store.write_run_metadata("project", "run_1", {"training_run_id": "run_1"})
    store.write_artifact_manifest(
        "project",
        "run_1",
        "sampler_weights",
        "sample-1",
        {"path": "toy-modal://project/run_1/sampler_weights/sample-1"},
    )

    assert (runs_root / "project" / "run_1" / "metadata.json").exists()
    assert (
        runs_root
        / "project"
        / "run_1"
        / "sampler_weights"
        / "sample-1"
        / "manifest.json"
    ).exists()


def test_artifact_store_manifest_checksums_detect_corruption(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    artifact_dir = store.layout.artifact_dir("project", "run_1", "checkpoints", "step")
    artifact_dir.mkdir(parents=True)
    weights = artifact_dir / "adapter.safetensors"
    weights.write_bytes(b"valid weights")

    manifest = store.write_artifact_manifest(
        "project",
        "run_1",
        "checkpoints",
        "step",
        {"path": "toy-modal://project/run_1/checkpoints/step"},
    )
    payload = store.read_json(manifest)
    assert payload["artifact_files"][0]["sha256"]
    store.validate_manifest_files(payload)

    weights.write_bytes(b"bad weights!!")
    try:
        store.validate_manifest_files(payload)
    except ValueError as exc:
        assert "checksum mismatch" in str(exc)
    else:
        raise AssertionError("corrupted artifact file was accepted")


def test_artifact_store_promotes_temp_artifact_directory(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    temp_dir = store.temp_artifact_dir("project", "run_1", "checkpoints", "step")
    temp_dir.mkdir(parents=True)
    (temp_dir / "payload.txt").write_text("ok", encoding="utf-8")

    final_dir = store.promote_artifact_dir(temp_dir, "project", "run_1", "checkpoints", "step")

    assert final_dir.exists()
    assert not temp_dir.exists()
    assert (final_dir / "payload.txt").read_text(encoding="utf-8") == "ok"
