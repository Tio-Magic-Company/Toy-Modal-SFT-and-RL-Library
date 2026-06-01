import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import asyncio
    import json
    import os
    import time
    import warnings

    warnings.filterwarnings("ignore", message="IProgress not found")

    import marimo as mo
    import toy_modal as tinker
    from toy_modal import completers, cookbook, renderers, weights

    return asyncio, completers, cookbook, json, mo, os, renderers, time, tinker, weights

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
# Tutorial 000: Set up the Toy Modal backend

Prepare a user-owned Modal deployment before running the rest of the notebooks.

Toy Modal is a framework for remotely fine-tuning and reinforcement-learning language models on infrastructure you own in Modal. The SDK keeps your training loop, data preparation, rewards, and evaluation logic in Python while Modal workers handle model loading, forward/backward passes, optimizer steps, sampling, checkpoint materialization, and model cache Volumes.

> Please note: This setup notebook can deploy and smoke-check a configured Modal app, but it does not prove large-model quality, production throughput, deployed HTTP gateway behavior, or Hub publishing.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Understand which work stays in your local Python process and which work runs on Modal.
2. Choose app, GPU, storage, and Hugging Face secret settings before deployment.
3. Run a smoke check that creates a ServiceClient through transport="modal-direct".
4. Keep deployment and remote checks behind explicit cost acknowledgement.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/000_setup_tutorial.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding Toy Modal backend setup</summary>

Toy Modal separates your local control loop from the remote compute plane. Your notebook decides what to train, what to sample, and when to checkpoint; the Modal app owns GPU workers, model caches, persistent Volumes, and deployment settings.

In Toy Modal, setup means choosing the Modal app name, GPUs, storage names, sampler/trainer engines, and optional Hugging Face Secret before any tutorial does real work. Once the app is deployed, every later notebook creates a `ServiceClient` with `transport="modal-direct"` and points at that app.

Further reading:

- [Modal apps and deployments](https://modal.com/docs/guide/apps)
- [Modal Secrets](https://modal.com/docs/guide/secrets)
- [Modal Volumes](https://modal.com/docs/guide/volumes)

</details>
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Configure this notebook

Set the Modal app, environment, and base model once. The default model follows the default Unsloth backend and is meant for low-cost validation before choosing larger public models. These are real remote model choices: prefetch first, expect Modal GPU cost, and use HF credentials for gated models such as Llama.

Checking the acknowledgement box does not run anything by itself. The separate remote-execution checkbox makes the expensive cells easy to spot.

Before running expensive cells, preflight the selected model cache from a terminal:

```bash
toy-modal backend prefetch-model <model-id> --dry-run --backend unsloth
```

For a real Modal prefetch, remove `--dry-run` only after confirming the app, Volume, GPU, and any required Hugging Face token setup.
    """)
    return

@app.cell
def _(mo, tinker):
    app_name = mo.ui.text(value="toy-modal-backend", label="Modal app name")
    environment_name = mo.ui.text(value="", label="Modal environment name, optional")
    base_model = mo.ui.text(value=tinker.DEFAULT_BASE_MODEL, label="Base model")
    max_tokens = mo.ui.number(start=1, stop=256, value=16, label="Max generated tokens")
    cost_ack = mo.ui.checkbox(value=False, label="I understand this notebook may allocate Modal resources")
    run_remote = mo.ui.checkbox(value=False, label="Run remote modal-direct cells")
    train_gpu = mo.ui.text(value="T4", label="Training GPU")
    sample_gpu = mo.ui.text(value="T4", label="Sampling GPU")
    prefetch_gpu = mo.ui.text(value="T4", label="Prefetch GPU")
    trainer_engine = mo.ui.dropdown(options=["unsloth-peft", "peft", "tiny"], value="unsloth-peft", label="Trainer engine")
    sampler_engine = mo.ui.dropdown(options=["unsloth", "transformers", "tiny"], value="unsloth", label="Sampler engine")
    model_volume = mo.ui.text(value="toy-modal-model-cache", label="Model cache Volume")
    run_volume = mo.ui.text(value="toy-modal-runs", label="Run/checkpoint Volume")
    registry_dict = mo.ui.text(value="toy-modal-registry", label="Registry Dict")
    sample_max_containers = mo.ui.number(start=1, stop=8, value=1, label="Max sampler containers")
    hf_secret_name = mo.ui.text(value="", label="HF Modal Secret name, optional")
    deploy_now = mo.ui.checkbox(value=False, label="Deploy now, equivalent to --deploy --i-understand-costs")
    smoke_check = mo.ui.checkbox(value=False, label="Smoke-check modal-direct connection")
    controls = [app_name, environment_name, base_model, max_tokens, cost_ack, run_remote]
    if True:
        controls.extend([train_gpu, sample_gpu, prefetch_gpu, trainer_engine, sampler_engine, model_volume, run_volume, registry_dict, sample_max_containers, hf_secret_name, deploy_now, smoke_check])
    mo.vstack(controls)
    return app_name, base_model, cost_ack, environment_name, max_tokens, run_remote, train_gpu, sample_gpu, prefetch_gpu, trainer_engine, sampler_engine, model_volume, run_volume, registry_dict, sample_max_containers, hf_secret_name, deploy_now, smoke_check

@app.cell
def _(app_name, base_model, environment_name, json, max_tokens, os):
    repo_root = os.getcwd()
    settings = {
        "project_id": "tutorial-000",
        "transport": "modal-direct",
        "app_name": app_name.value,
        "environment_name": environment_name.value or None,
        "base_model": base_model.value,
        "max_tokens": int(max_tokens.value),
    }
    print(json.dumps(settings, indent=2, sort_keys=True))
    return repo_root, settings,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Notebook shape

This notebook is the setup desk for the rest of the series. It does not assume a global service. Instead, it teaches the concrete environment variables and commands that stamp a Modal app with your chosen GPUs, engines, Volumes, and optional secrets.

The code cells are intentionally explicit. The goal is to make each moving part visible: client construction, renderer or data construction, remote call boundaries, futures, result inspection, and limitations.
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Build the local inputs

This cell is intentionally local. It prepares prompts, examples, configs, comparisons, or export plans before any remote worker is contacted. That separation is the main Toy Modal programming model.
    """)
    return

@app.cell
def _(app_name, base_model, environment_name, hf_secret_name, json, model_volume, prefetch_gpu, registry_dict, run_volume, sample_gpu, sample_max_containers, sampler_engine, train_gpu, trainer_engine):
    backend_env = {
        "TOY_MODAL_APP_NAME": app_name.value,
        "TOY_MODAL_TRAIN_GPU": train_gpu.value,
        "TOY_MODAL_SAMPLE_GPU": sample_gpu.value,
        "TOY_MODAL_PREFETCH_GPU": prefetch_gpu.value,
        "TOY_MODAL_TRAINER_ENGINE": trainer_engine.value,
        "TOY_MODAL_SAMPLER_ENGINE": sampler_engine.value,
        "TOY_MODAL_MODEL_VOLUME": model_volume.value,
        "TOY_MODAL_RUN_VOLUME": run_volume.value,
        "TOY_MODAL_REGISTRY_DICT": registry_dict.value,
        "TOY_MODAL_SAMPLE_MAX_CONTAINERS": str(sample_max_containers.value),
        "TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS": "0",
    }
    if hf_secret_name.value:
        backend_env["TOY_MODAL_HF_SECRET_NAME"] = hf_secret_name.value

    print("Review these exports before deployment:\n")
    for key, value in backend_env.items():
        print(f"export {key}={json.dumps(value)}")
    print("\nInstall command: python -m pip install -e '.[backend,tutorials]'")
    print("Local Unsloth engine tests: python -m pip install -e '.[backend,unsloth,tutorials]'")
    print("Modal auth command: modal setup")
    print('Optional private-model secret: modal secret create huggingface-token HF_TOKEN="$HF_TOKEN"')
    print("Deploy command: toy-modal backend deploy")
    return backend_env,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, backend_env, base_model, cost_ack, deploy_now, environment_name, mo, os, repo_root, smoke_check, tinker):
    mo.stop(not cost_ack.value, "Check the cost acknowledgement box before deploying or smoke-checking Modal.")

    if deploy_now.value:
        import subprocess
        import sys

        print("Running deployment through the Toy Modal CLI. This delegates to Modal.")
        subprocess.run(
            [sys.executable, "-m", "toy_modal.cli", "backend", "deploy"],
            cwd=repo_root,
            env={**os.environ, **backend_env},
            check=True,
        )
    else:
        print("Deployment checkbox is off. This cell is showing the command only:")
        print("toy-modal backend deploy")

    if smoke_check.value:
        service_client = tinker.ServiceClient(
            project_id="tutorial-000",
            transport="modal-direct",
            app_name=app_name.value,
            environment_name=environment_name.value or None,
        )
        capabilities = await service_client.get_server_capabilities_async()
        sampler = await service_client.create_sampling_client_async(base_model=base_model.value)
        tokenizer = sampler.get_tokenizer()
        print("Connected to Modal app:", app_name.value)
        print("Supported model names:", capabilities.supported_model_names)
        print("Smoke prompt tokens:", tokenizer.encode("Toy Modal setup check"))
    else:
        print("Smoke-check checkbox is off. After deployment, enable it to verify modal-direct connectivity.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(app_name, base_model, deploy_now, json, smoke_check):
    summary = {
        "app_name": app_name.value,
        "base_model": base_model.value,
        "deploy_checkbox": deploy_now.value,
        "smoke_check_checkbox": smoke_check.value,
        "transport": "modal-direct",
        "next_command": "marimo edit docs/tutorials/notebooks/101_hello_toy_modal.py",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Open 101 to inspect the client hierarchy.
- Run a tiny-model SFT notebook before moving to larger GPUs.
- Use `dev_notes/README.md` and the validation wrapper help before cost-bearing integration reports.
    """)
    return


if __name__ == "__main__":
    app.run()
