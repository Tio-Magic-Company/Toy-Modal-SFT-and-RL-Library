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
# Tutorial 405: Multi-agent self-play with MessageEnv

Score assistant messages in a small multi-agent or self-play loop.

Multi-agent RL is still message generation plus scoring; the important part is explicit state and reward bookkeeping.

> Please note: This uses deterministic scoring rather than a validated multi-agent judge model.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Represent agents as message policies.
2. Score responses with a judge function.
3. Build group rewards.
4. Keep self-play auditable.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/405_multi_agent.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding multi-agent scoring</summary>

Multi-agent learning extends the environment idea to multiple roles or policies. Agents may cooperate, critique, debate, play games, or provide judging signals for each other.

Toy Modal keeps the mechanics message-based: each agent response is a sampled message, the transcript is explicit, and rewards are local Python scores. This notebook uses deterministic scoring so the shape is clear before adding model judges.

Further reading:

- [PettingZoo multi-agent environments](https://pettingzoo.farama.org/)
- [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [OpenAI Evals repository](https://github.com/openai/evals)

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
        "project_id": "tutorial-405",
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

This mirrors the multi-agent notebook at tutorial scale.

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
    roles = ["solver", "critic"]
    def score_message(message):
        content = str(message.get("content", ""))
        return 1.0 if "because" in content.lower() or "Modal" in content else 0.0
    seed_messages = [{"role": "user", "content": "Explain why remote workers help training."}]
    return roles, score_message, seed_messages,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, completers, cost_ack, environment_name, max_tokens, mo, renderers, roles, run_remote, score_message, seed_messages, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before multi-agent sampling.")
    service_client = tinker.ServiceClient(project_id="tutorial-405", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    sampler = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampler.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    message_completer = completers.ToyModalMessageCompleter(sampler, renderer, max_tokens=int(max_tokens.value), temperature=0.7)
    messages = []
    for role in roles:
        msg = await message_completer(seed_messages + [{"role": "system", "content": f"You are the {role}."}])
        msg["agent"] = role
        msg["reward"] = score_message(msg)
        messages.append(msg)
    print(messages)
    return messages,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Self-play needs saved transcripts, judge calibration, and safeguards against reward hacking.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 203 for completers.
- Use 404 for sequence extension.
- Use 407 for pipeline composition.
    """)
    return


if __name__ == "__main__":
    app.run()
