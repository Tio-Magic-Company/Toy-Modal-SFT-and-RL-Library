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
# Tutorial 204: Weights and checkpoints

Save state, save sampler weights, inspect REST metadata, and discuss artifact download.

Weights are durable artifacts in Modal Volumes plus metadata. GPU memory is cache; checkpoints and sampler weights are the recoverable state.

> Please note: Production archive download behavior is deployment-specific and should be validated for your Modal storage policy.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Save optimizer/training state.
2. Create sampler weights.
3. List checkpoints.
4. Understand toy-modal:// paths.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/204_weights.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding weights, checkpoints, and artifacts</summary>

Training produces state. Some state is for resuming optimization, such as optimizer moments and model sequence IDs. Other state is for inference, such as saved LoRA adapter weights that can be loaded by a sampler.

Toy Modal treats Modal GPU memory as cache. Durable state belongs in Modal Volumes and metadata records. Checkpoints use `toy-modal://` paths so notebooks can pass artifacts between training, sampling, export, and publishing flows.

Further reading:

- [Modal Volumes](https://modal.com/docs/guide/volumes)
- [Hugging Face PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Transformers save/load pretrained models](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)

</details>
    """)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Configure this notebook

Set the Modal app, environment, and base model once. The default model matches the original tutorial's model target where practical. These are real remote model choices: prefetch first, expect Modal GPU cost, and use HF credentials for gated models such as Llama.

Checking the acknowledgement box does not run anything by itself. The separate remote-execution checkbox makes the expensive cells easy to spot.

Before running expensive cells, preflight the selected model cache from a terminal:

```bash
toy-modal backend prefetch-model <model-id> --dry-run
```

For a real Modal prefetch, remove `--dry-run` only after confirming the app, Volume, GPU, and any required Hugging Face token setup.
    """)
    return

@app.cell
def _(mo):
    app_name = mo.ui.text(value="toy-modal-backend", label="Modal app name")
    environment_name = mo.ui.text(value="", label="Modal environment name, optional")
    base_model = mo.ui.text(value="Qwen/Qwen3-4B-Instruct-2507", label="Base model")
    max_tokens = mo.ui.number(start=1, stop=256, value=16, label="Max generated tokens")
    cost_ack = mo.ui.checkbox(value=False, label="I understand this notebook may allocate Modal resources")
    run_remote = mo.ui.checkbox(value=False, label="Run remote modal-direct cells")

    controls = [app_name, environment_name, base_model, max_tokens, cost_ack, run_remote]
    if False:
        controls.extend([])
    mo.vstack(controls)
    return app_name, base_model, cost_ack, environment_name, max_tokens, run_remote

@app.cell
def _(app_name, base_model, environment_name, json, max_tokens, os):
    repo_root = os.getcwd()
    settings = {
        "project_id": "tutorial-204",
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

This mirrors the weights lifecycle tutorial while keeping downloads and archive access explicit.

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
def _():
    checkpoint_names = ["before-step", "after-step", "sampler"]
    print("Lifecycle checkpoints:", checkpoint_names)
    return checkpoint_names,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, mo, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before saving weights.")
    service_client = tinker.ServiceClient(project_id="tutorial-204", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer()
    datum = tinker.Datum(model_input=tinker.ModelInput.from_ints(tokenizer.encode("Toy Modal checkpoint tutorial")), loss_fn_inputs={"target_tokens": tokenizer.encode(" ok"), "weights": [1.0]})
    initial_state = await training_client.save_state_async("before-step")
    fwdbwd = await training_client.forward_backward_async([datum], "cross_entropy")
    optim = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    await fwdbwd.result_async(); await optim.result_async()
    after_state = await training_client.save_state_async("after-step")
    sampler = await training_client.save_weights_and_get_sampling_client_async("sampler")
    rest = service_client.create_rest_client()
    checkpoints = await rest.list_checkpoints_async(training_client.training_run_id)
    print("Initial state:", initial_state.result().path if hasattr(initial_state, "result") else initial_state.path)
    print("Sampler path:", sampler.model_path)
    print("Checkpoint count:", len(checkpoints.checkpoints))
    return after_state, checkpoints, rest, sampler, training_client,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Downloads use toy_modal.weights.download(rest, toy_path, destination) once you choose an artifact path.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 501 for HF export.
- Use operations docs for Volume cleanup.
- Use 303/304 for periodic checkpointing in loops.
    """)
    return


if __name__ == "__main__":
    app.run()
