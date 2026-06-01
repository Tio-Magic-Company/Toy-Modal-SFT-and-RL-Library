# Paths And Artifacts

`toy_modal` uses `toy-modal://` as the canonical artifact path scheme:

```text
toy-modal://<project_id>/<run_id>/<artifact_type>/<name>
```

Artifact types currently include:

- `checkpoints` for training state.
- `sampler_weights` for weights exported for sampling.

Examples:

```text
toy-modal://recipe/run_123/checkpoints/chat-sft
toy-modal://recipe/run_123/sampler_weights/chat-sft-sampler
```

`tinker://` paths are not accepted by default. They are compatibility input
only and require explicit opt-in:

```python
service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
    accept_tinker_paths=True,
)
```

Use `toy-modal://` in new code, docs, logs, and metadata.

Training checkpoints may include adapter weights, optimizer state, and a
manifest. Sampler weights are intended for inference and normally do not include
optimizer state.
