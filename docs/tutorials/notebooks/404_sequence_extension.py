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
# Tutorial 404: Sequence extension in multi-turn RL

Extend conversations across turns while preserving masks and reward attribution.

Sequence extension means the prompt is not fixed forever; each assistant action can become part of the next observation.

> Please note: This notebook does not stress-test context overflow or long-horizon memory.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Represent a multi-turn message state.
2. Convert MessageEnv into a token environment.
3. Reward the assistant turn.
4. Prepare sequence-extension metadata.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/404_sequence_extension.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding sequence extension and multi-turn state</summary>

In single-turn RL, the prompt is fixed and the model produces one answer. In sequence extension, generated messages become part of the next context, so the environment evolves over turns.

Toy Modal models this with message environments and renderers. The environment owns conversation history; the renderer turns the current messages into a generation prompt; the reward function scores the new assistant message and can decide whether the episode continues.

Further reading:

- [Hugging Face chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [PettingZoo multi-agent API](https://pettingzoo.farama.org/api/)

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
        "project_id": "tutorial-404",
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

This mirrors the multi-turn sequence tutorial with MessageEnv and EnvFromMessageEnv.

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
    history = [{"role": "system", "content": "Answer briefly."}, {"role": "user", "content": "Ask me a follow-up about Modal."}]
    message_env = cookbook.MessageEnv(history, reward_fn=lambda msg: 1.0 if "Modal" in str(msg.get("content", "")) else 0.0)
    return history, message_env,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cookbook, cost_ack, environment_name, max_tokens, message_env, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before sequence-extension rollout.")
    service_client = tinker.ServiceClient(project_id="tutorial-404", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    sampler = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampler.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    env = cookbook.EnvFromMessageEnv(message_env, renderer)
    observation, stop = await env.initial_observation()
    response = await sampler.sample_async(observation, 1, tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.6, stop=[s for s in stop if isinstance(s, str)]))
    step = await env.step(response.sequences[0].tokens[observation.length():])
    print("Reward:", step.reward, "Info:", step.info)
    return env, step,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("For long contexts, track which tokens are prompt history and which tokens receive action loss.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 405 for multi-agent message scoring.
- Use 301 for environment basics.
- Use 402 for reward degeneracy.
    """)
    return


if __name__ == "__main__":
    app.run()
