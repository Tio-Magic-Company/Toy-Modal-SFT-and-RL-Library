# Deployment Guide

Local tests do not require Modal. Deployed validation is a separate manual step.

## Backend Deploy

```bash
toy-modal backend deploy
```

The command delegates to:

```bash
modal deploy -m toy_modal.backend.app
```

## Configuration

Use environment variables to control Modal resources:

```text
TOY_MODAL_APP_NAME=toy-modal-backend
TOY_MODAL_TRAIN_GPU=A100
TOY_MODAL_SAMPLE_GPU=L40S
TOY_MODAL_PREFETCH_GPU=L40S
TOY_MODAL_TRAINER_ENGINE=unsloth-peft
TOY_MODAL_SAMPLER_ENGINE=unsloth
TOY_MODAL_MODEL_VOLUME=toy-modal-model-cache
TOY_MODAL_RUN_VOLUME=toy-modal-runs
TOY_MODAL_REGISTRY_DICT=toy-modal-registry
TOY_MODAL_SAMPLE_MAX_CONTAINERS=2
TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS=0
TOY_MODAL_SUPPORTED_MODELS="unsloth/tinyllama-bnb-4bit unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
```

These values are captured into the deployed Modal function/class runtime
environment at app definition time. Set them before `toy-modal backend deploy`,
especially `TOY_MODAL_TRAINER_ENGINE` and `TOY_MODAL_SAMPLER_ENGINE`.

The default engine pair is Unsloth-backed. Use `peft` plus `transformers` for
the plain baseline, or `tiny` plus `tiny` for deterministic route checks that
avoid model loading.
Full Unsloth model prefetches need GPU-backed model loading, so unset
`TOY_MODAL_PREFETCH_GPU` resolves to `TOY_MODAL_SAMPLE_GPU` for Unsloth
deployments. Set `TOY_MODAL_PREFETCH_GPU` explicitly to use a different GPU.
`TOY_MODAL_SUPPORTED_MODELS` is optional advisory capability metadata; it does
not restrict which compatible model IDs users can request. It is propagated to
the deployed function environment so capability checks see the same advertised
catalog that was configured at deploy time.

## Secrets

The default validation model is public and does not require a Hugging Face
secret. For private Hugging Face models, create a user-owned Modal Secret and
tell the backend which secret name to attach:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
export TOY_MODAL_HF_SECRET_NAME=huggingface-token
```

Leave `TOY_MODAL_HF_SECRET_NAME` unset for public models. The backend only
attaches a Hugging Face secret when this variable is set.

## Cost Controls

- Defaults use `min_containers=0`.
- Sampler containers are bounded by `TOY_MODAL_SAMPLE_MAX_CONTAINERS`.
- Do not run deploys or GPU jobs from automation unless explicitly approved.

For the Unsloth validation path, use
[`../../dev_notes/README.md`](../../dev_notes/README.md) and
`python dev_notes/validation/run_modal_validation.py --help`. The wrapper
defaults to `unsloth-peft`/`unsloth`; pass `--trainer-engine peft
--sampler-engine transformers` to reproduce the historical PEFT/Transformers
baseline. The Unsloth validation default is `unsloth/tinyllama-bnb-4bit`;
the historical baseline default remains `hf-internal-testing/tiny-random-gpt2`.
Use `--supported-models` to set `TOY_MODAL_SUPPORTED_MODELS` through the
wrapper, or `--skip-supported-model-check` when intentionally validating a model
outside the advisory catalog.

## Current Validation Status

The latest inspected tiny-model `modal-direct` run passed on 2026-05-27:

```text
dev_notes/validation_reports/modal-peft-20260527T163656Z/modal_parity_20260527T163730Z.json
summary: 19 pass, 0 fail, 0 skipped
```

That evidence covers PEFT training, checkpoint save/load, saved-adapter
sampling, base-model sampling, tokenizer access, stale rollout rejection, and
core REST metadata routes. It does not cover deployed HTTP gateway behavior,
deployed cookbook recipes, larger models, throughput, or production archive
downloads.
