# Deployment To Modal

Modal deployment is optional, user-owned, and cost-bearing. Do not run these
commands as part of local setup.

## Prerequisites

- A Modal account and workspace.
- Modal CLI installed and authenticated.
- `toy_modal` installed with Modal/backend dependencies.
- A Hugging Face token in a Modal Secret if private models are used.
- A deliberate GPU and cost plan.

No-credential validation before deploy:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
```

Deploy:

```bash
toy-modal backend deploy
```

The CLI delegates to:

```bash
modal deploy -m toy_modal.backend.app
```

After deployment, check Modal connectivity without running a training step:

```bash
toy-modal backend check --app-name toy-modal-backend
```

## Configuration

The backend reads environment variables:

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

Set these before deployment. The Modal app stamps the resolved values into each
deployed function/class runtime environment, so the remote workers use the same
trainer and sampler engines as the deploying shell.

The default trainer and sampler engines are `unsloth-peft` and `unsloth`.
They keep the Tinker-style Toy Modal API but delegate model loading, quantized
LoRA setup, and inference patching to Unsloth Core. Set
`TOY_MODAL_TRAINER_ENGINE=peft` and `TOY_MODAL_SAMPLER_ENGINE=transformers` for
the plain PEFT/Transformers baseline, or set both engines to `tiny` for
deterministic route validation.

`TOY_MODAL_PREFETCH_GPU` controls the remote model-prefetch function. When it is
unset, Unsloth deployments use `TOY_MODAL_SAMPLE_GPU` for full model prefetches;
plain PEFT/Transformers deployments keep prefetch CPU-only unless you set it.

`TOY_MODAL_SUPPORTED_MODELS` controls only the advisory model list returned by
`get_server_capabilities()`. It is not an allow-list; clients can still request
other compatible Hugging Face model IDs.

The default validation model is public and does not require a Hugging Face
secret. For private models, create a user-owned Modal Secret and set the backend
environment variable before deployment:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
export TOY_MODAL_HF_SECRET_NAME=huggingface-token
```

When using the validation wrapper, pass `--hf-secret-name huggingface-token`
instead. Leave the variable unset for public models.

## Cost Controls

- Defaults use `min_containers=0`.
- Trainer workers are capped at one container in the current app scaffold.
- Sampler workers are bounded by `TOY_MODAL_SAMPLE_MAX_CONTAINERS`.
- Use smaller GPU types and tiny models for validation before moving to larger
  models.
- Avoid shell loops or automation that repeatedly submits training or sampling
  jobs.
- Stop or delete unused deployments, runs, checkpoints, and Volumes according
  to your workspace policy.

## How Deployed Recipes Differ

Recipes can trigger real model loading, GPU allocation, PEFT adapter writes,
and Modal Volume state.

Example after deployment:

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/chat-sft-modal
```

Review cost and validation notes before running this command.

## Validation Status

Checked-in tests validate SDK shapes, fake Modal client workflows,
recipe smoke behavior, and backend helper logic.

The latest inspected tiny-model `modal-direct` PEFT/Transformers validation
passed on 2026-05-27:

```text
dev_notes/validation_reports/modal-peft-20260527T163656Z/modal_parity_20260527T163730Z.json
summary: 19 pass, 0 fail, 0 skipped
```

That evidence covers PEFT training, checkpoint save/load, saved-adapter
sampling, base-model sampling, tokenizer access, stale rollout rejection, and
core REST metadata routes. It does not cover deployed HTTP gateway behavior,
deployed cookbook recipes, larger models, throughput, or production archive
downloads.

For cost-bearing Modal validation guidance, current evidence, remaining parity
work, command logs, JSON reports, optional HTTP checks, optional deployed recipe
smoke, and cleanup context, use
[`../../dev_notes/README.md`](../../dev_notes/README.md).

The walk-away wrapper command is:

```bash
python dev_notes/validation/run_modal_validation.py --install --full-modal --i-understand-costs
```

It writes a report directory under `dev_notes/validation_reports/` for later
review. The wrapper defaults to the Unsloth engine pair; add `--trainer-engine
peft --sampler-engine transformers --base-model
hf-internal-testing/tiny-random-gpt2` when reproducing the historical baseline.
For custom advertised catalogs, pass `--supported-models "model/a model/b"`.
If the validation base model is intentionally outside the deployed advisory
catalog, add `--skip-supported-model-check`.

After the Modal parity report passes, the next recommended validation tier is
the deployed HTTP gateway probe plus promoted cookbook workflows: chat SFT, math
RL with `ppo`, math RL with `cispo`, on-policy RL, tiny SFT, and code RL with
`ppo`, `cispo`, and `importance_sampling`.
