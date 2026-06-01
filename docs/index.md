# toy_modal Documentation

`toy_modal` is a clean-room Python SDK scaffold for Tinker-style training loops
backed by a user-owned Modal deployment. It is not affiliated with Tinker or
Thinking Machines Lab.

Use `toy_modal` when you want to write normal Python loops around a small client
surface:

- `ServiceClient` creates training, sampling, and REST metadata clients.
- `TrainingClient` owns a LoRA training run.
- `SamplingClient` samples from a base model or saved sampler weights.
- Heavy calls return `APIFuture` handles so local code can submit work and wait
  later.

The supported compatibility import pattern is:

```python
import toy_modal as tinker
from toy_modal import types
```

No top-level `tinker` package is shipped.

## What Requires Modal

Runtime workflows use `transport="modal-direct"` by default. This calls a
deployed Modal app with the Modal Python client and requires Modal
authentication.

Real Hugging Face model loading, Unsloth or PEFT LoRA training, Unsloth or
Transformers sampling from saved adapters, Modal Volumes, Modal Dict metadata
acceleration, and the HTTP gateway require deployed Modal infrastructure. Those
runs are user-owned and cost-bearing.

The current checked-in evidence includes no-credential unit tests and one
tiny-model `modal-direct` PEFT/Transformers validation report:
`dev_notes/validation_reports/modal-peft-20260527T163656Z/modal_parity_20260527T163730Z.json`
with 19 passes and 0 failures. Treat that as a baseline for PEFT training,
saved-adapter sampling, tokenizer access, stale rollout rejection, and core REST
metadata. It is not evidence for deployed HTTP gateway behavior, large models,
throughput, production archive downloads, or full cookbook parity.

## Choose Your Path

| Goal | Start here |
| --- | --- |
| Deploy and smoke-check Modal | [`getting_started/quickstart.md`](getting_started/quickstart.md) |
| Understand SDK concepts | [`concepts/overview.md`](concepts/overview.md) |
| Run the tutorial series | [`tutorials/index.md`](tutorials/index.md) |
| Use RL losses | [`guides/reinforcement_learning.md`](guides/reinforcement_learning.md) |
| Run cookbook recipes | [`guides/cookbook_recipes.md`](guides/cookbook_recipes.md) |
| Deploy backend | [`guides/deployment_to_modal.md`](guides/deployment_to_modal.md) |
| Inspect Unsloth backend | [`technical/unsloth_backend.md`](technical/unsloth_backend.md) |
| Debug failures | [`troubleshooting.md`](troubleshooting.md) |

Operators should read
[`guides/operations_and_cleanup.md`](guides/operations_and_cleanup.md) before
long runs. It covers logs, checkpoints, resume, cleanup, Volume commit/reload
semantics, preemption, and stale rollout guards.

## Intentionally Not Supported Yet

- A top-level package named `tinker`.
- Public local runtime transports.
- Default acceptance of `tinker://` paths. Use `toy-modal://`; `tinker://` is
  accepted only when explicit compatibility opt-in is enabled.
- Arbitrary client-supplied Python loss callables in the backend.
- General use of `forward_backward_custom`.
- Default setup that deploys Modal apps or starts GPU jobs.
- Claims of performance, benchmark, or feature parity without checked-in
  validation evidence.
- Production-grade DPO, distillation, tool-use, multi-agent, VLM, Harbor/agent
  RL, SDFT, or True Thinking Score workflows. These are documented as current
  smoke scaffolds or planning categories where relevant.
