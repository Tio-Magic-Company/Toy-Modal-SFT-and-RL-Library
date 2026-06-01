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
# Tutorial 407: Full RLHF pipeline

Compose SFT, preference data, and RL into one staged workflow.

RLHF is a pipeline, not one call: supervised behavior, preference signal, and policy optimization each need separate data and metrics.

> Please note: This is pipeline composition, not a production reward-model training system.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Run SFT stage.
2. Build preference datums.
3. Define reward model shape.
4. Run RL stage.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/407_rlhf_pipeline.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding RLHF pipeline composition</summary>

RLHF is a staged workflow. A common shape is: start with SFT, collect or generate preferences, train or apply a preference signal, then optimize the policy with RL while evaluating each stage.

Toy Modal keeps those stages composable. The same client hierarchy handles SFT, preference-shaped datums, sampler checkpoints, and RL updates; your notebook owns stage boundaries, metrics, promotion decisions, and rollback points.

Further reading:

- [InstructGPT paper](https://arxiv.org/abs/2203.02155)
- [Hugging Face RLHF blog](https://huggingface.co/blog/rlhf)
- [DPO paper](https://arxiv.org/abs/2305.18290)

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
    base_model = mo.ui.text(value="meta-llama/Llama-3.2-3B", label="Base model")
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
        "project_id": "tutorial-407",
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

This mirrors the full pipeline tutorial while keeping each stage tutorial-scale.

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
    conversations = [[{"role": "user", "content": "What is Toy Modal?"}, {"role": "assistant", "content": "A Modal-backed post-training framework."}]]
    comparisons = [cookbook.Comparison("What is Toy Modal?", "A Modal-backed post-training framework.", "A board game.")]
    return comparisons, conversations,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, comparisons, conversations, cookbook, cost_ack, environment_name, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before RLHF pipeline composition.")
    service_client = tinker.ServiceClient(project_id="tutorial-407", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    sft_datums = [renderer.conversation_to_datum(conv) for conv in conversations]
    sft_result = cookbook.run_supervised_train_loop(training, sft_datums, cookbook.TrainLoopConfig(steps=1, checkpoint_prefix="rlhf-sft"))
    pref_datums = cookbook.build_dpo_datums(tokenizer, comparisons, renderer)
    print("SFT:", sft_result)
    print("Preference datums:", len(pref_datums))
    return pref_datums, renderer, sft_result,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("A production RLHF pipeline needs stage-specific evals, artifact promotion, and rollback points.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 403 for preference data.
- Use 406 for distillation.
- Use 501-503 for exporting promoted artifacts.
    """)
    return


if __name__ == "__main__":
    app.run()
