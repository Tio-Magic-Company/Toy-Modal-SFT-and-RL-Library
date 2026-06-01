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
# Tutorial 203: Completers

Wrap SamplingClient in token and message completers, then use the same shape for judge-style scoring.

Completers are small adapters over SamplingClient. They keep rollout code focused on prompts, messages, and rewards instead of repeated sampling boilerplate.

> Please note: Judge quality depends entirely on the configured model and prompt, and should be validated before using rewards for training.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Create a token completer.
2. Create a message completer with a renderer.
3. Parse responses into messages.
4. Sketch LLM-as-judge rewards.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/203_completers.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding completers and judge calls</summary>

A completer is a small callable wrapper around sampling. It hides tokenization and sampling parameters so rollout code can say “complete this prompt” or “answer these messages” without repeating boilerplate.

Toy Modal provides token and message completers over `SamplingClient`. The same pattern can power LLM-as-judge workflows: build a judging prompt, sample a score, parse it conservatively, and turn it into a reward or metric.

Further reading:

- [Hugging Face generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)
- [OpenAI Evals repository](https://github.com/openai/evals)
- [Inspect AI documentation](https://inspect.aisi.org.uk/)

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
        "project_id": "tutorial-203",
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

This mirrors the completer tutorial: token completions first, message completions second, judging last.

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
    judge_prompt = "Rate this answer from 1 to 5: Toy Modal keeps Python loops local. Score:"
    messages = [{"role": "user", "content": "What does Toy Modal run remotely?"}]
    return judge_prompt, messages,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, completers, cost_ack, environment_name, max_tokens, messages, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before creating completers.")
    service_client = tinker.ServiceClient(project_id="tutorial-203", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    sampling_client = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)
    token_completer = completers.ToyModalTokenCompleter(sampling_client, max_tokens=int(max_tokens.value), temperature=0.4)
    message_completer = completers.ToyModalMessageCompleter(sampling_client, renderer, max_tokens=int(max_tokens.value), temperature=0.4)
    prompt = renderer.build_generation_prompt(messages)
    token_result = await token_completer(prompt, stop=renderer.get_stop_sequences())
    message = await message_completer(messages)
    print("Token text:", tokenizer.decode(token_result.tokens))
    print("Message:", message)
    return message, message_completer, token_result,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Use completers anywhere a rollout policy or judge wants a simple callable interface.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 301 for rollout policies.
- Use 405 for multi-agent scoring.
- Use 205 for evaluator wrappers.
    """)
    return


if __name__ == "__main__":
    app.run()
