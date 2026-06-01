# Math RL

## Purpose

Grouped rollout RL for verifiable math prompts. Rewards run in user code, and
the backend receives structured RL tensors.

## What The Recipe Does

The recipe creates a training run, exports rollout sampler weights, samples a
small group, scores completions with a numeric-answer reward, computes
group-relative advantages, skips degenerate groups, trains with
`importance_sampling`, `ppo`, or `cispo`, and saves a checkpoint.

## Inputs

Arguments include:

- `--prompt`
- `--answer`
- `--group-size`
- `--loss-fn importance_sampling|ppo|cispo`
- `--resume`
- `--log-path`

## Command

```bash
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/math-rl
```

## Modal Command

Cost-bearing after backend deployment:

```bash
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/math-rl-modal
```

## Outputs

Stdout JSON plus `metrics.jsonl`, `checkpoints.jsonl`, and
`trajectories.jsonl` when `--log-path` is set. Records include rewards, loss
name, checkpoint, rollout model path, datum count, and optimizer step.

## Resume Behavior

Pass `--resume toy-modal://.../checkpoints/<name>` to restore weights and
optimizer state.

## Cost Notes

Modal runs may allocate both trainer and sampler GPU workers.

## Cleanup

Delete local logs and unneeded deployed checkpoints/sampler weights. Keep a
checkpoint path if you need resume.

## Known Limitations

The default prompt and reward are tiny validation scaffolds. This is not a
benchmark claim. GRPO is implemented as a cookbook helper pattern, not a backend
loss name.
