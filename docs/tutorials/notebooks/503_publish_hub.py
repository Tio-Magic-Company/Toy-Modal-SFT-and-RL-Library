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
# Tutorial 503: Publish to the Hugging Face Hub

Generate a model card and publish only when explicit flags and credentials are present.

Publishing mutates an external service, so this notebook keeps dry-run as the normal path.

> Please note: Publishing is never automatic. Real upload requires an explicit checkbox, HF_TOKEN, network access, and validated local artifacts.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Create a ModelCardConfig.
2. Write model card text.
3. Dry-run Hub publishing.
4. Require HF_TOKEN for real upload.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/503_publish_hub.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding publishing model artifacts</summary>

Publishing is a release action, not a training step. A good publish workflow verifies local artifacts, writes a clear model card, checks license and intended use, then uploads only after credentials and repository settings are explicit.

Toy Modal keeps publishing dry-run by default. The notebook can generate model-card text and package metadata without network calls; a real Hugging Face upload requires an explicit UI opt-in and `HF_TOKEN`.

Further reading:

- [Hugging Face Hub upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Hugging Face model cards](https://huggingface.co/docs/hub/model-cards)
- [Hugging Face Hub repositories](https://huggingface.co/docs/hub/repositories)

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
        "project_id": "tutorial-503",
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

This mirrors the Hub publish notebook with guardrails around credentials and upload.

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
def _(base_model, weights):
    repo_id = "your-name/toy-modal-tutorial-model"
    card = weights.ModelCardConfig(model_name="Toy Modal Tutorial Model", base_model=base_model.value, description="Tutorial artifact produced by Toy Modal notebooks.", metrics={"tutorial_score": 1.0})
    print(weights.generate_model_card(card))
    return card, repo_id,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
def _(mo):
    publish = mo.ui.checkbox(value=False, label="Publish to Hugging Face Hub, requires HF_TOKEN and validated artifacts")
    folder_path = mo.ui.text(value="artifacts/tutorial-hf-model", label="Folder to upload")
    mo.vstack([publish, folder_path])
    return folder_path, publish,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(cost_ack, folder_path, mo, os, publish, repo_id, weights):
    mo.stop(not cost_ack.value, "Check the cost acknowledgement box before publishing externally.")
    mo.stop(not publish.value, "Dry run only. Enable publish after validating artifacts and HF_TOKEN.")
    mo.stop("HF_TOKEN" not in os.environ, "Set HF_TOKEN before publishing.")
    result = weights.publish_to_hf_hub(folder_path.value, repo_id=repo_id, dry_run=False)
    print(result)
    return result,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 501 or 502 to produce artifacts first.
- Review the generated model card.
- Keep dry runs in CI.
    """)
    return


if __name__ == "__main__":
    app.run()
