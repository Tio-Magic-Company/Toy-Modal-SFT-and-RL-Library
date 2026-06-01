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
# Tutorial 303: SFT with config

Move from hand-written loops to config objects and dataset builders.

Config objects make tutorial loops repeatable without hiding the underlying client calls.

> Please note: The config helper is intentionally small and does not replace a production trainer.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Define a dataset builder.
2. Create SupervisedTrainConfig.
3. Run a config-driven loop.
4. Record metrics and checkpoints.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/303_sft_with_config.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding config-driven SFT</summary>

A config-driven trainer makes repeated experiments easier to compare. Instead of scattering model names, LoRA ranks, learning rates, checkpoint intervals, and dataset construction throughout a notebook, a config object gathers them in one place.

Toy Modal config helpers are deliberately small. They still create ordinary clients and datums, but they make the experiment boundary explicit: dataset builder, model, transport, learning rate, max steps, and checkpoint behavior.

Further reading:

- [Hugging Face Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Hydra configuration framework](https://hydra.cc/docs/intro/)

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
        "project_id": "tutorial-303",
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

This mirrors the SFT config tutorial: builder, config, train entrypoint, result.

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
def _(cookbook):
    class TinyDatasetBuilder:
        common_config = cookbook.ChatDatasetBuilderCommonConfig(model_name_for_tokenizer="configured-by-notebook")
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
        def __call__(self):
            conversations = [[cookbook.Message("user", "Say Toy Modal"), cookbook.Message("assistant", "Toy Modal")]]
            datums = cookbook.render_conversation_datums(self.tokenizer, conversations)
            return cookbook.InMemorySupervisedDataset(datums), None
    return TinyDatasetBuilder,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(TinyDatasetBuilder, app_name, base_model, cookbook, cost_ack, environment_name, mo, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before config SFT.")
    service_client = tinker.ServiceClient(project_id="tutorial-303", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    builder = TinyDatasetBuilder(training_client.get_tokenizer())
    train_dataset, _ = builder()
    result = cookbook.run_supervised_train_loop(training_client, next(iter(train_dataset)), cookbook.TrainLoopConfig(steps=1, checkpoint_prefix="sft-config"))
    print(result)
    return result,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Config loops should still expose model, transport, checkpoint, and logging choices.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 102 to understand the raw loop.
- Use 304 for RL config.
- Use recipes for longer-running variants.
    """)
    return


if __name__ == "__main__":
    app.run()
