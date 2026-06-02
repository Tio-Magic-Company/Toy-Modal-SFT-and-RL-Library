# toy_modal

`toy_modal` is a clean-room, Tinker-style Python SDK scaffold backed by a
user-owned Modal deployment. Users write ordinary Python training and sampling
loops around a small client surface, while model work runs in their own Modal
workspace.

This project is independent. It is not affiliated with Tinker, Thinking
Machines Lab, or Modal. The goal is compatibility with the public workflow
shape, not reuse of proprietary service behavior.

## Status

The SDK surface, Modal `modal-direct` transport, deployed HTTP transport,
PEFT trainer path, Transformers sampler path, HTTP gateway contract, cookbook
framework, examples, and Marimo tutorial series are in place.


## Quickstart

Install the package, deploy the backend when you are ready for Modal spend, then
run a connectivity check:

```bash
python -m pip install -e '.[backend,dev]'
toy-modal backend deploy
toy-modal backend check --app-name toy-modal-backend
```

Minimal SDK usage:

```python
import toy_modal as tinker
from toy_modal import types

service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
training = service.create_lora_training_client(
    base_model=tinker.DEFAULT_BASE_MODEL,
    rank=8,
    train_unembed=False,
)

tokenizer = training.get_tokenizer()
datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Question: 2+2? Answer: 4")),
    loss_fn_inputs={"target_tokens": [4], "weights": [1]},
)

loss = training.forward_backward([datum], "cross_entropy").result()
step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()

print(loss.loss)
print(step.optimizer_step)
```

Cookbook commands now target Modal by default:

```bash
toy-modal cookbook list
toy-modal cookbook smoke sl_loop --app-name toy-modal-backend --log-path runs/sl_loop
```

## Repository Layout

The documentation and tutorial material is split into three top-level folders:

| Path | Purpose |
| --- | --- |
| `docs/` | User documentation, Modal-backed examples, Marimo notebooks, and cookbook recipe scripts for `toy_modal`. |


Runtime code remains under `src/toy_modal/`:

```text
src/toy_modal/
  clients/       Public SDK clients
  transport/     modal-direct and deployed HTTP transports
  backend/       Modal app, workers, storage, metadata, and loss helpers
  cookbook.py    Reusable cookbook workflow helpers
  types.py       Pydantic request/response models
  futures.py     APIFuture abstraction
```

Key user-facing entrypoints:

- `docs/index.md`
- `docs/getting_started/quickstart.md`
- `docs/tutorials/notebooks/README.md`
- `docs/examples/README.md`
- `docs/recipes/README.md`

## Modal Backend

Modal usage is user-owned and cost-bearing. Deploy only when you intend to run
the backend:

```bash
toy-modal backend deploy
```

The backend reads resource settings from environment variables at deploy time:

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
TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS=0
```

The default deployment engines use Unsloth Core for LoRA training and sampling.
Set `TOY_MODAL_TRAINER_ENGINE=peft` and
`TOY_MODAL_SAMPLER_ENGINE=transformers` for the plain PEFT/Transformers
baseline, or `TOY_MODAL_TRAINER_ENGINE=tiny` and
`TOY_MODAL_SAMPLER_ENGINE=tiny` for deterministic scaffold validation.

For private Hugging Face model downloads, create a user-owned Modal Secret and
attach it by name:

```bash
modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"
export TOY_MODAL_HF_SECRET_NAME=huggingface-token
```

Use `dev_notes/State_of_Repo.md` and
`python dev_notes/validation/run_modal_validation.py --help` before running any
cost-bearing validation.

## Development Checks

Run fast no-credential checks before handing off changes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
python dev_notes/validation/run_modal_validation.py --help
git diff --check
```

Do not run Modal deploys or GPU jobs unless the user explicitly requests or
approves that cost-bearing action.
