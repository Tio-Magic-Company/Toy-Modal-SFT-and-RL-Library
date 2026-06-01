# Setup The Modal Backend

The setup tutorial is now a Marimo notebook:

```bash
marimo edit docs/tutorials/notebooks/000_setup_tutorial.py
```

It teaches the real Toy Modal shape: your Python notebook prepares data, owns
the training loop, and decides when to run expensive cells; your Modal
deployment handles model loading, GPU work, Volumes, checkpoints, and metadata.

## What Gets Configured

The notebook exposes UI controls for the deployment settings that become
`TOY_MODAL_*` environment variables:

```text
TOY_MODAL_APP_NAME
TOY_MODAL_TRAIN_GPU
TOY_MODAL_SAMPLE_GPU
TOY_MODAL_PREFETCH_GPU
TOY_MODAL_TRAINER_ENGINE
TOY_MODAL_SAMPLER_ENGINE
TOY_MODAL_MODEL_VOLUME
TOY_MODAL_RUN_VOLUME
TOY_MODAL_REGISTRY_DICT
TOY_MODAL_SAMPLE_MAX_CONTAINERS
TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS
TOY_MODAL_UNSLOTH_LOAD_IN_4BIT
TOY_MODAL_UNSLOTH_MAX_SEQ_LENGTH
```

Use the public tiny GPT-2 test model for the first backend wiring validation.
The tutorial series itself defaults to the original tutorial model targets such
as `Qwen/Qwen3.5-4B`, `Qwen/Qwen3-4B-Instruct-2507`, and selected Llama models.
Llama and other gated Hugging Face models require a Modal Secret:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
export TOY_MODAL_HF_SECRET_NAME=huggingface-token
```

Before loading a real tutorial model, preflight the cache request:

```bash
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --dry-run --backend unsloth
```

For an actual Modal Volume prefetch, remove `--dry-run` only after confirming
the app, model Volume, GPU selection, and token access:

```bash
toy-modal backend prefetch-model Qwen/Qwen3.5-4B --backend unsloth --app-name toy-modal-backend
```

The default deployment engines are `unsloth-peft` for training and `unsloth`
for sampling. They preserve the Toy Modal client API while using Unsloth Core
for model loading, quantized LoRA setup, and inference patching. Switch to
`peft` and `transformers` for the plain baseline, or `tiny` and `tiny` for
deterministic route checks.

## Deploy

Open the notebook, review the generated exports, then enable the deployment
checkbox only after acknowledging costs. The notebook runs the same deployment
entrypoint as:

```bash
toy-modal backend deploy
```

The CLI delegates to Modal:

```bash
python -m modal deploy -m toy_modal.backend.app
```

## Smoke Check

After deployment, enable the smoke-check checkbox in the setup notebook. The
smoke check creates a `ServiceClient` with `transport="modal-direct"`, reads
server capabilities, creates a sampler for the tiny model, and asks the remote
tokenizer to encode a short prompt.

## Run The Tutorial Series

Once setup passes, start with:

```bash
marimo edit docs/tutorials/notebooks/101_hello_toy_modal.py
```

Each later notebook has its own explanatory markdown cells, code cells, UI
guards, and limitations section.

## Limits

The setup notebook proves that a configured Modal app can deploy and respond to
the selected smoke check. It does not prove large-model quality, production
throughput, deployed HTTP gateway behavior, or Hugging Face publishing. Those
are separate validation tiers covered by `dev_notes/README.md`, the validation
wrapper help, and export tutorials.
