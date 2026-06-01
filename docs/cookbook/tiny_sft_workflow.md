# Tiny SFT Workflow

## Purpose

A small supervised workflow with optional dataset input, checkpointing, sampler
export, eval output, and resume hooks.

## What The Recipe Does

The recipe loads two default prompt/completion rows or a user JSONL dataset,
runs one `cross_entropy` step, saves a checkpoint, exports sampler weights, and
samples from the saved weights.

## Inputs

Optional JSONL rows:

```json
{"prompt":"Question: 2+2? Answer:","completion":" 4"}
```

Arguments include `--dataset`, `--resume`, `--eval-output`, and `--log-path`.

## Command

```bash
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/tiny-sft
```

## Modal Command

Cost-bearing after backend deployment:

```bash
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/tiny-sft-modal
```

## Outputs

Stdout JSON, optional eval JSON, `metrics.jsonl`, and `checkpoints.jsonl`.

## Resume Behavior

Pass `--resume toy-modal://.../checkpoints/<name>` to restore weights and
optimizer state.

## Cost Notes

Modal runs can allocate GPUs.

## Cleanup

Delete local log/eval files and unneeded deployed artifacts.

## Known Limitations

The default dataset is tiny and intended for workflow validation only.
