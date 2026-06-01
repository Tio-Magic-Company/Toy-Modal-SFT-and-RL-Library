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
# Tutorial 501: Export a Hugging Face-compatible model

Download or materialize artifacts, merge adapters when dependencies are available, and write an HF model folder.

Export moves from toy-modal:// artifacts to Hugging Face-shaped local directories.

> Please note: Real merge/export requires compatible checkpoint artifacts,
> model downloads, and either Unsloth export dependencies or the explicit
> Transformers/PEFT fallback.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Locate a Toy Modal checkpoint.
2. Build an HF model directory.
3. Understand optional Unsloth or PEFT/Transformers imports.
4. Avoid network calls by default.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,export,unsloth,tutorials]'
marimo edit docs/tutorials/notebooks/501_export_hf.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding Hugging Face export</summary>

Export turns Toy Modal artifacts into a directory layout that standard Hugging Face tools can understand. Depending on the workflow, you may publish a merged model or a base-model-plus-adapter setup.

Toy Modal export helpers are local/offline by default. They download or
materialize artifact metadata, use Unsloth lazily by default, keep an explicit
PEFT/Transformers fallback, and write manifests when a real merge cannot be
completed in the current environment.

Further reading:

- [Transformers save_pretrained](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)
- [PEFT checkpoint format](https://huggingface.co/docs/peft/developer_guides/checkpoint)
- [Hugging Face Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)

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
    base_model = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model")
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
        "project_id": "tutorial-501",
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

This mirrors the export notebook with dry-run defaults and lazy optional dependencies.

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
    adapter_dir = "artifacts/tutorial-adapter"
    output_dir = "artifacts/tutorial-hf-model"
    export_plan = {"merge": False, "local_files_only": True, "adapter_dir": adapter_dir, "output_dir": output_dir}
    print(export_plan)
    return adapter_dir, export_plan, output_dir,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, mo, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before creating export source weights.")
    service_client = tinker.ServiceClient(project_id="tutorial-501", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    sampler = await training.save_weights_and_get_sampling_client_async("export-source")
    print("Sampler model path:", sampler.model_path)
    print("Use weights.download(rest_client, toy_path, destination) when you choose the artifact to export.")
    return sampler,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(adapter_dir, base_model, output_dir, weights):
    model_dir = weights.build_hf_model(
        base_model=base_model.value,
        adapter_dir=adapter_dir,
        output_dir=output_dir,
        backend="unsloth",
        merge=False,
        local_files_only=True,
    )
    print("HF-shaped directory:", model_dir)
    return model_dir,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 502 for adapter-only packaging.
- Use 503 for guarded Hub publishing.
- Validate exported models before promotion.
    """)
    return


if __name__ == "__main__":
    app.run()
