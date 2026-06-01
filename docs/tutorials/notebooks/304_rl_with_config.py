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
# Tutorial 304: RL with config

Configure prompts, group size, loss function, KL reference, and training loop parameters.

RL config keeps the moving parts visible: prompts, sampling params, reward function, group size, loss, KL, checkpointing.

> Please note: The helper demonstrates wiring, not full on-policy production scheduling or large-model KL validation.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Define RLTrainConfig.
2. Collect rollouts.
3. Assemble RL datums.
4. Run a config-driven update.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/304_rl_with_config.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding config-driven RL</summary>

RL has more moving parts than SFT: prompts, sampling parameters, group size, reward functions, loss family, KL settings, checkpointing, and rollout storage. A config object makes those choices visible before remote work starts.

Toy Modal `RLTrainConfig` wires those choices into a small loop: collect grouped rollouts, compute rewards and advantages, build RL datums, and run a configured optimizer step on Modal.

Further reading:

- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [Hugging Face TRL PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)
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
        "project_id": "tutorial-304",
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

This mirrors the RL config tutorial using Toy Modal config helpers.

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
def _(base_model, cookbook):
    prompts = ["Return the digit four.", "Return the digit five."]
    def reward_fn(prompt, completion):
        expected = "4" if "four" in prompt else "5"
        return 1.0 if expected in completion else 0.0
    kl_reference = cookbook.KLReferenceConfig(coef=0.02, reference_model=base_model.value)
    return kl_reference, prompts, reward_fn,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cookbook, cost_ack, environment_name, kl_reference, max_tokens, mo, prompts, reward_fn, run_remote):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before config RL.")
    config = cookbook.RLTrainConfig(
        prompts=prompts,
        model_name=base_model.value,
        transport="modal-direct",
        project_id="tutorial-304",
        app_name=app_name.value,
        environment_name=environment_name.value or None,
        group_size=2,
        max_steps=1,
        max_tokens=int(max_tokens.value),
        kl_reference=kl_reference,
    )
    result = cookbook.run_rl_config(config, reward_fn=reward_fn)
    print(result)
    return config, result,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(kl_reference):
    print("KL config:", kl_reference)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 104 for raw RL mechanics.
- Use 402 for hyperparameter effects.
- Use 407 for pipeline composition.
    """)
    return


if __name__ == "__main__":
    app.run()
