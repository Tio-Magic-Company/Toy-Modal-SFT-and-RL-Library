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
# Tutorial 201: Rendering conversations into model inputs

Study renderers, train-on-what choices, stop sequences, and response parsing.

Renderers bridge chat messages and model tokens. Toy Modal ships clean-room role-colon renderers rather than vendor template clones.

> Please note: Renderer names are compatibility conveniences. They do not claim exact vendor chat-template equivalence.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Build generation prompts.
2. Build supervised datums.
3. Inspect token weights.
4. Understand clean-room template differences.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/201_rendering.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding rendering and chat templates</summary>

A renderer is the translation layer between human-readable messages and token-level training data. It decides how roles are formatted, where generation should start, which stop sequences matter, and which tokens receive training weight.

Toy Modal renderers are clean-room and tokenizer-agnostic by default. The `role_colon` renderer uses simple `Role: content` lines so tutorials can teach the mechanics without claiming exact vendor chat-template equivalence.

Further reading:

- [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
- [PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)

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

This renderer tutorial also includes a VLM prompt-construction section for
`Qwen/Qwen3-VL-235B-A22B-Instruct`. That model is intentionally not loaded by
default because it is much larger and requires separate cost approval.
    """)
    return

@app.cell
def _(mo):
    app_name = mo.ui.text(value="toy-modal-backend", label="Modal app name")
    environment_name = mo.ui.text(value="", label="Modal environment name, optional")
    base_model = mo.ui.text(value="Qwen/Qwen3-30B-A3B", label="Base model")
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
        "project_id": "tutorial-201",
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

This mirrors the rendering tutorial: messages go in, token sequences and loss weights come out.

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
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Define renderer."}, {"role": "assistant", "content": "A renderer maps messages to tokens and training weights."}]
    print(messages)
    return messages,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, messages, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before fetching the tokenizer.")
    service_client = tinker.ServiceClient(project_id="tutorial-201", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    sampling_client = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)
    prompt = renderer.build_generation_prompt(messages[:-1])
    datum = renderer.conversation_to_datum(messages)
    print("Prompt length:", prompt.length())
    print("Datum length:", datum.model_input.length())
    print("Trainable weights:", sum(datum.loss_fn_inputs["weights"]))
    return datum, prompt, renderer, tokenizer,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(datum):
    print("Loss inputs keys:", sorted(datum.loss_fn_inputs))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 102 for SFT datums.
- Use 203 for message completers.
- Use 404 for multi-turn message environments.
    """)
    return


if __name__ == "__main__":
    app.run()
