# Storage and Operations

Canonical state belongs in the run Volume. Modal Dict is a metadata/cache layer.

## Layout

```text
/runs/<project_id>/<training_run_id>/metadata.json
/runs/<project_id>/<training_run_id>/checkpoints/<name>/manifest.json
/runs/<project_id>/<training_run_id>/checkpoints/<name>/adapter/
/runs/<project_id>/<training_run_id>/checkpoints/<name>/optimizer.pt
/runs/<project_id>/<training_run_id>/sampler_weights/<name>/manifest.json
/runs/<project_id>/<training_run_id>/sampler_weights/<name>/adapter/
/runs/<project_id>/<training_run_id>/logs/
/logs/<project_id>/<training_run_id>/events.jsonl
/archives/<project_id>/<training_run_id>/<name>.json
/models/
```

The helper class for this layout is `toy_modal.backend.storage.ArtifactStore`.
Use `ArtifactStore.from_runs_root("/runs")` inside workers that receive the run
Volume mounted at `/runs`. Use `ArtifactStore("/state")` in tests or tools that
want a parent state root with `runs/`, `logs/`, and `archives/` subdirectories.

## Write Semantics

- Checkpoint manifests are written to a temporary file and then atomically
  replaced.
- PEFT training checkpoints contain adapter files, optimizer state, and a
  manifest. Sampler weights contain adapter files and a manifest.
- Run metadata is written to `metadata.json` using the same atomic replacement
  helper.
- Recipe and operational logs append JSONL records.
- Workers commit Volume writes after optimizer or checkpoint updates.
- Workers reload Volumes before reading artifacts written by another worker.
- Deployed metadata routes rebuild from Volume JSON and manifests when Modal
  Dict cache entries are missing or expired.
- Archive URL responses currently point at `modal-volume://...` metadata records
  for operator retrieval rather than public signed object-store URLs.

## Recovery

- Reconstruct metadata from `metadata.json` and checkpoint manifests when cache
  state is missing.
- Treat stale model sequence errors as user-visible conflicts.
- Retry only idempotent submit/retrieve operations.
