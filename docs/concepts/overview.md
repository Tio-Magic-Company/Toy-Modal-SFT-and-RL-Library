# Overview

`toy_modal` provides a clean-room, Tinker-style SDK surface backed by a
user-owned Modal deployment.

```python
import toy_modal as tinker
from toy_modal import types

service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
training = service.create_lora_training_client(
    tinker.DEFAULT_BASE_MODEL,
    rank=4,
)
sampling = service.create_sampling_client(
    base_model=tinker.DEFAULT_BASE_MODEL,
)
```

Core concepts:

- `ServiceClient` is the entry point.
- `TrainingClient` owns a LoRA training run.
- `SamplingClient` samples from a base model or saved sampler weights.
- `RestClient` reads metadata and artifact information.
- Heavy calls return `APIFuture` handles.
- Artifact paths prefer `toy-modal://`.

Runtime transports:

| Transport | Purpose |
| --- | --- |
| `modal-direct` | Default. Python client calls deployed Modal functions/classes directly. |
| `http` | Calls a deployed HTTP gateway URL. |

The public local runtime transport has been removed. Fast tests use test-only
fakes; user workflows target Modal.
