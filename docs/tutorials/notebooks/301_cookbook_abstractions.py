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
# Tutorial 301: Cookbook RL abstractions

Replace raw rollout code with ProblemEnv, ProblemGroupBuilder, trajectories, and GRPO datum assembly.

Cookbook abstractions keep RL code organized: environments own task state, group builders own grouped rollout creation, and datum builders translate trajectories into losses.

> Please note: The example environment is tiny and single-turn; production tasks need robust parsing and cleanup.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Create a ProblemEnv.
2. Build a group of environments.
3. Use a completer policy.
4. Turn trajectories into training datums.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/301_cookbook_abstractions.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding RL environments and trajectories</summary>

Raw RL loops become repetitive: build prompts, sample, parse completions, score rewards, compute advantages, and assemble training data. Environment abstractions package that task-specific logic behind a small interface.

Toy Modal cookbook helpers use `ProblemEnv`, group builders, trajectories, and datum assembly. The environment owns task state and reward checks; the policy samples through a completer; the framework converts completed trajectories into RL loss inputs.

Further reading:

- [Gymnasium environment creation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)
- [Spinning Up: RL intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [PettingZoo API docs](https://pettingzoo.farama.org/api/)

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
        "project_id": "tutorial-301",
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

This mirrors the abstraction tutorial after the raw RL loop in 104.

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
    class ArithmeticEnv(cookbook.ProblemEnv):
        def check_answer(self, text):
            return self.answer in text

    def build_env(renderer):
        return ArithmeticEnv("What is 4 + 4?", "8", renderer)
    return ArithmeticEnv, build_env,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, build_env, completers, cookbook, cost_ack, environment_name, max_tokens, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before cookbook rollouts.")
    service_client = tinker.ServiceClient(project_id="tutorial-301", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    sampler = await training_client.save_weights_and_get_sampling_client_async("cookbook-policy")
    policy = completers.ToyModalTokenCompleter(sampler, max_tokens=int(max_tokens.value), temperature=0.7)
    builder = cookbook.ProblemGroupBuilder(lambda: build_env(renderer), num_envs=3)
    group = await cookbook.do_group_rollout(builder, policy)
    advs = cookbook.compute_advantages([group])
    datums, metadata = cookbook.assemble_training_data([group], advs, model_seq_id=training_client.model_seq_id)
    print("Rewards:", group.get_total_rewards())
    print("Datums:", len(datums), "Metadata:", metadata)
    return datums, group, metadata,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Abstractions make the raw 104 loop reusable across tasks.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 302 to build a custom environment.
- Use 304 to connect environments to config training.
- Use 405 for multi-agent MessageEnv variants.
    """)
    return


if __name__ == "__main__":
    app.run()
