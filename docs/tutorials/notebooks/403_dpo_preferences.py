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
# Tutorial 403: DPO and preference learning

Represent chosen/rejected comparisons and build DPO-shaped datums.

Preference learning starts with structured data: prompt, chosen completion, rejected completion, and an explicit renderer.

> Please note: Structured preference data exists, but production DPO training needs selected backend validation.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Create comparisons.
2. Render chosen/rejected answers.
3. Attach preference labels.
4. Discuss safe DPO support.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/403_dpo_preferences.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding DPO and preference data</summary>

Preference learning starts from comparisons: for a prompt, one completion is preferred over another. Direct Preference Optimization uses those pairs to update the policy without first training a separate reward model.

Toy Modal represents comparisons as structured data and renders chosen/rejected responses into datums. The tutorial keeps this safe and explicit: preference data shape is supported, while arbitrary custom Python loss execution remains disabled.

Further reading:

- [DPO paper](https://arxiv.org/abs/2305.18290)
- [Hugging Face TRL DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [RLHF overview](https://huggingface.co/blog/rlhf)

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
        "project_id": "tutorial-403",
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

This mirrors the preference tutorial while keeping arbitrary custom losses disabled.

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
    comparisons = [cookbook.Comparison(prompt="What is Toy Modal?", chosen="A Modal-backed post-training framework.", rejected="A toy unrelated to ML."), cookbook.Comparison(prompt="Where do GPUs run?", chosen="In your Modal deployment.", rejected="Inside this notebook process.")]
    return comparisons,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, comparisons, cookbook, cost_ack, environment_name, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before building preference datums with a tokenizer.")
    service_client = tinker.ServiceClient(project_id="tutorial-403", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    dpo_datums = cookbook.build_dpo_datums(tokenizer, comparisons, renderer)
    print("DPO-shaped datums:", len(dpo_datums))
    print("First loss inputs:", sorted(dpo_datums[0].loss_fn_inputs))
    return dpo_datums, renderer,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("DPO parity is data/modeling parity here; backend custom Python loss execution stays disabled.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 407 for RLHF composition.
- Use 202 for loss safety.
- Use 501-503 after training artifacts exist.
    """)
    return


if __name__ == "__main__":
    app.run()
