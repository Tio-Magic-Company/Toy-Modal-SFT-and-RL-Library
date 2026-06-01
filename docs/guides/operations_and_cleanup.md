# Operations And Cleanup

Use this page before long or deployed runs.

## Artifacts And Logs

Recipe logs commonly include:

- `metrics.jsonl`: loss, optimizer step, sample tokens/text, rewards, pass rate,
  or recipe-specific metrics.
- `checkpoints.jsonl`: training run ID, checkpoint path, sampler path, and
  resume pointers.
- `trajectories.jsonl`: grouped rollout records for RL recipes.

Backend state is conceptually stored under:

```text
/runs/<project_id>/<training_run_id>/metadata.json
/runs/<project_id>/<training_run_id>/checkpoints/<name>/
/runs/<project_id>/<training_run_id>/sampler_weights/<name>/
/logs/<project_id>/<training_run_id>/events.jsonl
/archives/<project_id>/<training_run_id>/
/models/
```

## Resume

Use `create_training_client_from_state(path)` for weights-only resume. Use
`create_training_client_from_state_with_optimizer(path)` to restore optimizer
state and continue the same schedule.

Promoted recipes that accept `--resume` pass that path into the optimizer-state
resume helper.

## Volume Semantics

Modal Volumes require explicit commit/reload discipline:

- Commit after metadata, checkpoints, sampler weights, and optimizer state
  changes.
- Reload before a worker reads artifacts written by another worker.
- Avoid reloading a Volume while files under that mount are open.

## Preemption And Retries

Modal GPU functions can be preempted. Treat heavy operations as retryable only
when the operation is idempotent or checkpointed. Keep checkpoint writes atomic
and prefer resuming from the last committed checkpoint instead of relying on
warm GPU memory.

HTTP gateway mode should submit long-running work and poll/retrieve the result
instead of blocking a web request.

## Stale Rollout Guards

RL data can include `old_logprobs_model_seq_id`. When provided, the backend can
reject training data collected from a stale model sequence instead of training
against the wrong policy revision.

If you see a stale model sequence conflict, collect fresh rollouts from the
current sampler weights or resume the matching checkpoint.

## Cleanup

Local recipe cleanup is usually deleting the selected `--log-path`.

For deployed runs, cleanup can include:

- Delete unneeded checkpoints through `RestClient` or CLI helpers.
- Remove stale Modal Volumes only when no deployed app or running function uses
  them.
- Remove unused Modal Secrets through Modal tooling.
- Stop or replace deployed apps that are no longer needed.

Do not use Modal Queues as cleanup targets for durable state; they are not where
canonical state should live.
