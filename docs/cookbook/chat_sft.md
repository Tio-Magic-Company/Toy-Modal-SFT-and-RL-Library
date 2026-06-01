# Chat SFT

## Purpose

Supervised finetuning for conversational examples using a small clean-room chat
renderer.

## What The Recipe Does

The recipe loads a tiny default JSONL conversation when `--dataset` is omitted,
renders it into prompt plus assistant target tokens, runs one `cross_entropy`
step, saves a checkpoint, exports sampler weights, and samples.

## Inputs

Optional JSONL with either:

```json
{"messages":[{"role":"user","content":"Say hello."},{"role":"assistant","content":"Hello."}]}
```

or:

```json
{"prompt":"Say hello.","completion":"Hello."}
```

## Command

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/chat-sft
```

## Modal Command

Cost-bearing after backend deployment:

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/chat-sft-modal
```

## Outputs

Stdout JSON plus `metrics.jsonl` and `checkpoints.jsonl` when `--log-path` is
set. Records include recipe name, training run ID, checkpoint, sampler model
path, loss, optimizer step, sample tokens, and datum count.

## Resume Behavior

Pass `--resume toy-modal://.../checkpoints/<name>` to load weights and optimizer
state before continuing.

## Cost Notes

`modal-direct` can allocate GPUs and download model
weights in your Modal workspace.

## Cleanup

Delete local `runs/chat-sft` when done. For deployed runs, delete unneeded
checkpoints and sampler artifacts through metadata helpers or workspace cleanup
processes.

## Known Limitations

The renderer is deliberately simple and tokenizer-agnostic. Model-family
chat-template parity is future work.
