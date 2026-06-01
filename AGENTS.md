# AGENTS.md

## Working Context

This repository is the early scaffold for `toy_modal`: a clean-room,
Tinker-style Python SDK with a user-owned Modal backend. The current internal
state report is `dev_notes/README.md`.

The documentation corpus is organized into three top-level folders:

- `docs/`: user documentation, Marimo notebooks, local examples, and cookbook
  recipe scripts for `toy_modal`.
- `reference_docs/`: copied reference material for Modal and the original
  Tinker docs/tutorials. Use these as source references, not as user-facing
  `toy_modal` docs.
- `dev_notes/`: private state-of-repo report, validation scripts, and
  validation reports.

Before changing SDK or backend behavior, read the relevant checked-in reference
docs instead of relying on memory:

- Tinker API shape: `reference_docs/tinker/api/serviceclient.md`,
  `trainingclient.md`, `samplingclient.md`, `restclient.md`, `apifuture.md`,
  `types.md`, and `exceptions.md`.
- Modal project layout and deployment:
  `reference_docs/modal/other-topics/project-structure.md`,
  `reference_docs/modal/deployment/apps.md`, and
  `reference_docs/modal/deployment/trigger-deployed-functions.md`.
- Modal state and storage:
  `reference_docs/modal/data-sharing-and-storage/volumes.md`,
  `dicts.md`, `queues.md`, `model-weights.md`, and
  `cloud-bucket-mounts.md`.
- Modal compute and scaling:
  `reference_docs/modal/gpus-and-resources/gpu.md`, `resources.md`,
  `reference_docs/modal/scaling-out/concurrent-inputs.md`,
  `dynamic-batching.md`, `job-queue.md`, and `batch-processing.md`.
- Modal web/API access:
  `reference_docs/modal/web-endpoints/webhooks.md`, `webhook-urls.md`,
  `webhook-timeouts.md`, and `webhook-proxy-auth.md`.
- Modal operations:
  `reference_docs/modal/secrets-and-env-vars/secrets.md`,
  `environment_variables.md`,
  `reference_docs/modal/reliability-and-robustness/retries.md`,
  `timeouts.md`, `preemption.md`, and `troubleshooting.md`.

## Implementation Rules

- Keep compatibility clean-room. Use public Tinker-style method names and
  behavior described in local references, but do not imply affiliation with
  Tinker or Thinking Machines Lab.
- Preserve `import toy_modal as tinker` as a supported user pattern. Do not add
  or publish a top-level package named `tinker` without legal review.
- Prefer `toy-modal://` for new paths. Accept `tinker://` only through an
  explicit compatibility option.
- Heavy calls should return `APIFuture`-style handles. Do not make web
  endpoints block on long GPU work.
- Treat Modal GPU memory as cache only. Canonical run state belongs in Modal
  Volumes plus durable metadata files; Modal Dict is a fast metadata/cache layer
  with documented size and inactivity limits.
- Commit Volume writes after checkpoints or optimizer state changes, and reload
  Volumes before reading artifacts from another worker.
- Do not use Modal Queues for persistent state; their TTL and item limits make
  them coordination-only.
- Keep arbitrary client-supplied Python loss callables disabled by default.
  They are remote code execution.
- Keep `AdamParams.weight_decay` defaulting to `0.0`; this is a documented
  compatibility difference from PyTorch AdamW defaults.
- Keep `SamplingClient` picklable. Do not serialize active process, network, or
  sidecar handles.

## Project Structure

- `src/toy_modal/types.py`: Pydantic public type models.
- `src/toy_modal/futures.py`: shared future abstraction.
- `src/toy_modal/clients/`: public SDK clients.
- `src/toy_modal/transport/`: transport abstraction plus `local-mock`,
  `modal-direct`, and HTTP transports.
- `src/toy_modal/backend/`: Modal app, worker, metadata, storage, and loss
  helpers.
- `docs/examples/`: runnable local-mock workflows that mirror the target SDK
  shape.
- `docs/recipes/`: clean-room cookbook recipe scripts and workflow examples.
- `docs/tutorials/notebooks/`: Marimo tutorial notebooks stored as `.py` files.
- `tests/unit/`: fast tests that must not require Modal credentials or GPUs.

## Validation

Run fast local checks before handing off:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
PYTHONDONTWRITEBYTECODE=1 python docs/examples/sft_minimal.py
```

For documentation-only edits, run:

```bash
git diff --check
```

Modal deployment validation is a separate integration step:

```bash
toy-modal backend deploy
```

Use `dev_notes/README.md` and `python dev_notes/validation/run_modal_validation.py --help`
for cost-bearing Modal validation. The current green baseline is the 2026-05-27
PEFT/Transformers `modal-direct` report with 19 passes and 0 failures. Do not
claim HTTP gateway, large-model, performance, or cookbook parity from that
report alone.

When reviewing Modal logs, a `StaleModelSequenceError` is expected only for the
`training.stale_old_logprobs_guard` validation probe when the report marks it as
an expected failure. Treat the same error in normal recipe training or sampling
as a real bug.

Do not run Modal deploys or GPU jobs unless the user asks for them or approves
the cost-bearing action.
