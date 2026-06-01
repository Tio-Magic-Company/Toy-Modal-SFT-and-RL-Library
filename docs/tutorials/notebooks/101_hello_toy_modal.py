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
# Tutorial 101: Hello Toy Modal

Take the first read-only tour through ServiceClient, SamplingClient, tokenization, and sampling.

Toy Modal is a framework for running LLM training and inference workflows against your own Modal deployment. You keep Python control locally; Modal workers run the expensive GPU-side operations.

> Please note: This notebook reflects the models and metadata exposed by your deployed Modal app, not a global hosted service.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Create the root ServiceClient for a deployed Modal app.
2. Inspect server capabilities.
3. Create a SamplingClient for a base model.
4. Encode a prompt, sample completions, and inspect response fields.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/101_hello_toy_modal.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding remote clients and sampling</summary>

A remote training SDK is mostly a small set of client objects. The root client carries project and transport settings, while specialized clients expose sampling, training, and metadata operations.

In Toy Modal, `ServiceClient` is the root. It creates `SamplingClient` objects for inference and `TrainingClient` objects for LoRA training. Sampling starts with text, tokenizes it for the selected model, sends tokens to the Modal sampler worker, then decodes completion tokens back into text.

Further reading:

- [Modal client and app concepts](https://modal.com/docs/guide/apps)
- [Hugging Face tokenizer overview](https://huggingface.co/docs/transformers/main/en/tokenizer_summary)
- [marimo notebooks are Python files](https://docs.marimo.io/)

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
        "project_id": "tutorial-101",
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

The original first tutorial introduces a remote GPU service and a small client hierarchy. This notebook teaches the same shape for Toy Modal: ServiceClient creates specialized clients, and those clients return futures or async responses for remote work.

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
    prompt_text = "The three most important things to know about Toy Modal are"
    prompt_variants = [
        prompt_text,
        "Toy Modal helps me fine-tune models by",
        "A user-owned Modal backend is useful because",
    ]
    print("Prompts prepared for sampling:")
    for item in prompt_variants:
        print("-", item)
    return prompt_text, prompt_variants,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, max_tokens, mo, prompt_text, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable cost acknowledgement and remote execution before contacting Modal.")
    service_client = tinker.ServiceClient(
        project_id="tutorial-101",
        transport="modal-direct",
        app_name=app_name.value,
        environment_name=environment_name.value or None,
    )
    capabilities = await service_client.get_server_capabilities_async()
    print("Available model names:")
    for model_name in capabilities.supported_model_names:
        print("  -", model_name)

    sampling_client = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampling_client.get_tokenizer()
    prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
    params = tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.7, stop=["\n"])
    result = await sampling_client.sample_async(prompt=prompt, sampling_params=params, num_samples=3)
    for index, sequence in enumerate(result.sequences):
        completion = sequence.tokens[prompt.length():]
        print(f"Sample {index} stop={sequence.stop_reason} text={tokenizer.decode(completion)!r}")
    return capabilities, prompt, result, sampling_client, tokenizer,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(prompt, result):
    sequence = result.sequences[0]
    print("Stop reason:", sequence.stop_reason)
    print("Generated token count:", max(0, len(sequence.tokens) - prompt.length()))
    print("First generated token ids:", sequence.tokens[prompt.length(): prompt.length() + 10])
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Open 102 to update LoRA weights with supervised fine-tuning.
- Use 103 when you need concurrent sampling patterns.
- Use 204 when you are ready to inspect saved weights.
    """)
    return


if __name__ == "__main__":
    app.run()
