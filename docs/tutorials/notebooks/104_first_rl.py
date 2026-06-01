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
# Tutorial 104: Reinforcement learning with verifiable rewards

Build the raw RL loop: sample groups, score completions, compute advantages, and train with an RL loss.

Toy Modal lets reward code stay local while Modal workers do sampling and gradient work. That separation is the core RL shape.

> Please note: The task is intentionally synthetic. Real RL needs stronger graders, more rollouts, checkpointing, and stale-logprob safeguards.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Define a small verifiable reward function.
2. Collect multiple completions per prompt.
3. Build GRPO-style training datums.
4. Run an RL forward/backward and optimizer step.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/104_first_rl.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding RL with verifiable rewards</summary>

Reinforcement learning trains a policy from rewards rather than fixed target answers. For language models, the policy is the model, the action is generated text, and the reward is a score assigned by code, a grader, or another model.

This notebook uses verifiable rewards: local Python checks whether a completion contains the expected answer. Toy Modal samples groups of completions remotely, keeps reward logic local, converts rewards into advantages, then sends RL-shaped datums back to Modal for an update.

Further reading:

- [Spinning Up: key concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
- [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
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
    base_model = mo.ui.text(value="meta-llama/Llama-3.1-8B", label="Base model")
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
        "project_id": "tutorial-104",
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

This follows the original first-RL notebook: a small task, grouped completions, rewards, group-relative advantages, and one policy update.

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
    math_prompts = ["What is 2 + 2?", "What is 3 + 5?", "What is 7 - 4?"]
    answers = {"What is 2 + 2?": "4", "What is 3 + 5?": "8", "What is 7 - 4?": "3"}

    def reward_fn(prompt, completion):
        return 1.0 if answers[prompt] in completion else 0.0

    print("Reward table:", answers)
    return answers, math_prompts, reward_fn,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cookbook, cost_ack, environment_name, math_prompts, max_tokens, mo, reward_fn, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable cost acknowledgement and remote execution before RL sampling/training.")
    service_client = tinker.ServiceClient(project_id="tutorial-104", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer()
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="rl-initial-policy")
    groups = cookbook.collect_grouped_rollouts(
        sampler=sampling_client,
        tokenizer=tokenizer,
        prompts=math_prompts,
        group_size=3,
        sampling_params=tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.8),
        reward_fn=reward_fn,
    )
    for group in groups:
        print(group.prompt, group.get_total_rewards())
    datums = cookbook.grpo_datums_from_trajectory_groups(groups, model_seq_id=training_client.model_seq_id, skip_degenerate=False)
    print("Training datums:", len(datums))
    fwdbwd_future = await training_client.forward_backward_async(datums, "importance_sampling")
    optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=1e-4))
    fwdbwd_result = await fwdbwd_future.result_async()
    optim_result = await optim_future.result_async()
    print("RL loss:", fwdbwd_result.loss)
    print("Optimizer step:", optim_result.optimizer_step)
    return datums, groups, training_client,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(cookbook, groups):
    for group in groups:
        print(group.prompt, "rewards", group.get_total_rewards(), "advantages", cookbook.group_relative_advantages(group.get_total_rewards(), skip_degenerate=False))
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 301 to replace the raw loop with environment abstractions.
- Use 402 for KL and group-size knobs.
- Use 304 for config-driven RL.
    """)
    return


if __name__ == "__main__":
    app.run()
