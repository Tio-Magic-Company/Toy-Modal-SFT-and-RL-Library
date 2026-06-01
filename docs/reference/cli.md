# CLI Reference

The console script is `toy-modal`.

Global options include `--project-id`, `--transport`, `--base-url`,
`--app-name`, `--environment-name`, and `--api-key`.

## Backend

```bash
toy-modal backend deploy
toy-modal backend check --app-name toy-modal-backend
toy-modal backend prefetch-model unsloth/tinyllama-bnb-4bit --dry-run --backend unsloth
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --dry-run --backend unsloth
```

`backend deploy` and real prefetch commands are cost-bearing because they
contact Modal. `backend check` verifies deployed app connectivity and creates a
metadata-only training run without running a training step.
`backend prefetch-model --backend auto` follows the configured engine defaults;
with the current backend defaults that means Unsloth for model prefetch and
Transformers for tokenizer-only prefetch.

## Smoke Test

```bash
toy-modal smoke-test --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit
```

This uses `modal-direct` by default and may allocate Modal resources.

## Runs

```bash
toy-modal run list
toy-modal run info <run_id>
toy-modal run stop <run_id>
```

`run stop` currently prints transport-specific cancellation guidance.

## Checkpoints

```bash
toy-modal checkpoint list <run_id>
toy-modal checkpoint download <path> <destination>
toy-modal checkpoint delete <path>
```

Use `toy-modal://` paths unless explicit compatibility opt-in is configured in
client code.

## Cookbook

```bash
toy-modal cookbook list
toy-modal cookbook smoke sl_loop --app-name toy-modal-backend --log-path runs/sl-loop
toy-modal cookbook smoke --all --app-name toy-modal-backend --log-path runs/all-recipes
```
