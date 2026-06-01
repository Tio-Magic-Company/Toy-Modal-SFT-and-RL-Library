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
# Tutorial 302: Build a custom RL environment

Subclass ProblemEnv for a new task and define answer and format rewards.

A custom environment is the boundary between your task and the training algorithm. It should be deterministic, debuggable, and explicit about rewards.

> Please note: This environment checks a toy fact; real tasks need evaluation against held-out data and adversarial parse cases.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Define task state.
2. Override answer checks.
3. Add formatting penalties.
4. Run one grouped rollout.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/302_custom_environment.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding custom RL environments</summary>

A custom environment defines what the model sees, what action format is valid, and how reward is computed. Good environments are explicit about parse failures and easy to debug before training.

In Toy Modal, a `ProblemEnv` builds the generation prompt with a renderer, receives sampled tokens or text, checks answer/format conditions locally, and returns a `StepResult`. That reward becomes the signal used to assemble RL datums.

Further reading:

- [Gymnasium custom environment tutorial](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Spinning Up: RL intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [DeepSeekMath / GRPO paper](https://arxiv.org/abs/2402.03300)

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
        "project_id": "tutorial-302",
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

This mirrors the custom environment notebook using Toy Modal cookbook primitives.

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
    class CapitalCityEnv(cookbook.ProblemEnv):
        def get_question(self):
            return "Answer with only the city: what is the capital of France?"
        def check_answer(self, text):
            return "paris" in text.lower()
        def check_format(self, text):
            return len(text.split()) <= 4
    return CapitalCityEnv,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(CapitalCityEnv, app_name, base_model, completers, cookbook, cost_ack, environment_name, max_tokens, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before running the custom env.")
    service_client = tinker.ServiceClient(project_id="tutorial-302", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    sampler = await training_client.save_weights_and_get_sampling_client_async("capital-policy")
    policy = completers.ToyModalTokenCompleter(sampler, max_tokens=int(max_tokens.value), temperature=0.5)
    builder = cookbook.ProblemGroupBuilder(lambda: CapitalCityEnv("", "Paris", renderer), num_envs=2)
    group = await cookbook.do_group_rollout(builder, policy)
    print("Rollout rewards:", group.get_total_rewards())
    return group,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Before training, inspect failures by reading trajectory.metadata for parse and reward details.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 301 for base abstractions.
- Use 402 for reward degeneracy handling.
- Use 304 for config-driven training.
    """)
    return


if __name__ == "__main__":
    app.run()
