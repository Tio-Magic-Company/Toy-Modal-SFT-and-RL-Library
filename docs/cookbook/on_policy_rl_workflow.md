# On Policy RL Workflow

## Purpose

Minimal on-policy RL flow using sampler rollout logprobs and the implemented
`importance_sampling` loss.

## What The Recipe Does

The recipe exports rollout sampler weights, samples one completion, uses the
generated logprobs or `compute_logprobs`, computes a simple reward in user code,
adds `old_logprobs_model_seq_id`, trains with `importance_sampling`, and saves a
checkpoint.

## Inputs

Arguments include `--prompt`, `--answer`, `--resume`, `--eval-output`, and
`--log-path`.

## Command

```bash
python docs/recipes/on_policy_rl_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/on-policy-rl
```

## Modal Command

Cost-bearing after backend deployment:

```bash
python docs/recipes/on_policy_rl_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/on-policy-rl-modal
```

## Outputs

Stdout JSON, optional eval JSON, `metrics.jsonl`, and `checkpoints.jsonl`.

## Resume Behavior

Pass `--resume toy-modal://.../checkpoints/<name>` to restore weights and
optimizer state.

## Cost Notes

Modal mode can allocate trainer and sampler GPUs.

## Cleanup

Delete local logs and unneeded deployed checkpoints/sampler weights.

## Known Limitations

This is a minimal one-sample workflow. Broader on-policy scheduling,
environment integration, and benchmark validation are future work.
