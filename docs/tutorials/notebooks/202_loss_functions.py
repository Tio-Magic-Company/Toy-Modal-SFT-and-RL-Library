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
# Tutorial 202: Loss functions

Compare cross-entropy with RL-shaped losses and show why custom Python losses remain disabled by default.

Loss functions define how token logprobs become gradients. Toy Modal validates structured loss inputs and keeps arbitrary client Python disabled by default.

> Please note: Custom Python loss callables remain disabled by default because they are remote code execution.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Inspect required loss inputs.
2. Run cross-entropy on supervised data.
3. Build PPO/CISPO-shaped fields.
4. Understand custom loss safety.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/202_loss_functions.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding language-model losses</summary>

A loss function turns model predictions into a scalar training signal. In SFT, cross-entropy rewards high probability on target tokens. In RL-style losses, token logprobs are combined with old logprobs, advantages, masks, and sometimes KL penalties.

Toy Modal keeps this structured. Built-in losses accept explicit `loss_fn_inputs`; arbitrary client-supplied Python losses stay disabled by default because running user code on remote workers is a security boundary.

Further reading:

- [PyTorch CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Hugging Face TRL trainer docs](https://huggingface.co/docs/trl/main/en/index)

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
        "project_id": "tutorial-202",
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

This follows the original losses notebook but frames custom loss execution as an explicit future trusted-execution problem.

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
    supervised_tokens = ["Toy", "Modal", "runs", "on", "Modal"]
    rl_example = {"target_tokens": [1, 2], "old_logprobs": [-0.2, -0.3], "advantages": [1.0, -1.0], "weights": [1.0, 1.0]}
    print("RL loss input example:", rl_example)
    return rl_example, supervised_tokens,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, mo, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before running losses on Modal.")
    service_client = tinker.ServiceClient(project_id="tutorial-202", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer()
    prompt_tokens = tokenizer.encode("Toy Modal loss tutorial:")
    target_tokens = tokenizer.encode(" structured inputs")
    datum = tinker.Datum(model_input=tinker.ModelInput.from_ints([*prompt_tokens, *target_tokens]), loss_fn_inputs={"target_tokens": target_tokens, "weights": [0.0] * len(prompt_tokens) + [1.0] * len(target_tokens)})
    result_future = await training_client.forward_backward_async([datum], "cross_entropy")
    result = await result_future.result_async()
    print("Cross-entropy loss:", result.loss)
    return datum, result, training_client,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Custom Python losses are not enabled by default; use built-in structured losses instead.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 104 for an RL loss in context.
- Use 402 for KL and PPO/CISPO knobs.
- Use backend docs before designing trusted custom execution.
    """)
    return


if __name__ == "__main__":
    app.run()
