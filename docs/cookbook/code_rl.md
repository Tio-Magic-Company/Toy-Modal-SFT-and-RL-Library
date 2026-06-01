# Code RL

## Purpose

RL-shaped code training where candidate code is evaluated by user-owned local
execution and only structured reward/loss inputs are sent to the backend.

## What The Recipe Does

The recipe evaluates two tiny Python candidates with `python -I -c`, applies a
timeout, builds trajectories with pass/fail rewards, converts them to structured
RL datums, trains with `importance_sampling`, `ppo`, or `cispo`, and saves a
checkpoint.

## Inputs

Arguments include:

- `--loss-fn importance_sampling|ppo|cispo`
- `--timeout-seconds`
- `--resume`
- `--log-path`

## Command

```bash
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/code-rl
```

## Modal Command

Cost-bearing after backend deployment:

```bash
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/code-rl-modal
```

## Outputs

Stdout JSON plus `metrics.jsonl` and `checkpoints.jsonl` when `--log-path` is
set. Metrics include pass rate and sandbox records.

## Resume Behavior

Pass `--resume toy-modal://.../checkpoints/<name>` to restore weights and
optimizer state.

## Cost Notes

Deployed training is cost-bearing. Candidate code still
runs locally in this recipe; Modal Sandbox integration is not part of the
current workflow.

## Cleanup

Delete local logs and unneeded deployed artifacts.

## Known Limitations

This is a tiny scaffold, not a secure multi-tenant code execution system. Do not
run untrusted code without your own sandboxing controls.
