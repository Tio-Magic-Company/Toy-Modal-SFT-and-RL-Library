# Modal State Model

In deployed mode, GPU memory is a cache. It can be warm, fast, and useful, but
it is not canonical state.

Canonical state belongs in Modal Volumes plus durable metadata files:

```text
/runs/<project_id>/<training_run_id>/metadata.json
/runs/<project_id>/<training_run_id>/checkpoints/<name>/manifest.json
/runs/<project_id>/<training_run_id>/checkpoints/<name>/adapter/
/runs/<project_id>/<training_run_id>/checkpoints/<name>/optimizer.pt
/runs/<project_id>/<training_run_id>/sampler_weights/<name>/manifest.json
/runs/<project_id>/<training_run_id>/sampler_weights/<name>/adapter/
/logs/<project_id>/<training_run_id>/events.jsonl
/archives/<project_id>/<training_run_id>/<name>.json
/models/
```

Modal Volume writes are not automatically visible to every running container.
Writers commit after changes. Readers reload before consuming artifacts written
by another worker.

Modal Dict mirrors small metadata for speed. It is not the source of truth.
Backend metadata routes should rebuild from Volume JSON and manifests when Dict
entries are missing or expired.

Modal Queues are coordination-only. They should not store checkpoints, rollout
history, optimizer state, or any other durable state.
