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
# Tutorial 103: Efficient sampling with futures

Compare sequential sampling with concurrent futures for remote Modal workers.

Every modal-direct call has scheduling and GPU time. Toy Modal exposes async methods so a notebook can keep many requests in flight while Modal batches and pipelines work.

> Please note: Timing depends on your Modal deployment, current cold starts, model size, and sampler concurrency limits.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Understand why remote sampling benefits from concurrency.
2. Submit many sample requests before awaiting results.
3. Use num_samples for grouped completions.
4. Measure wall-clock timing without changing model weights.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/103_async_patterns.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding async sampling and futures</summary>

Remote calls have latency: request scheduling, network round trips, tokenizer work, and GPU generation time. If you submit one request and wait before sending the next, your local process spends most of its time idle.

Toy Modal exposes async methods and future-style handles so you can submit several sampling or training operations before awaiting results. This is especially useful for RL, where one prompt may need a group of completions before rewards and advantages can be computed.

Further reading:

- [Python asyncio tasks](https://docs.python.org/3/library/asyncio-task.html)
- [Modal concurrent inputs](https://modal.com/docs/guide/concurrent-inputs)
- [Modal dynamic batching](https://modal.com/docs/guide/dynamic-batching)

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
        "project_id": "tutorial-103",
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

The original async tutorial contrasts send-wait loops with concurrent requests. This notebook keeps that learning arc and uses Toy Modal sampling clients.

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
    prompts = [
        "Explain Modal Volumes in one sentence.",
        "Name two reasons to use LoRA.",
        "What does a renderer do?",
        "Why use futures for sampling?",
        "What should stay out of arbitrary custom losses?",
        "How do checkpoints relate to sampler weights?",
    ]
    print(f"Prepared {len(prompts)} prompts")
    return prompts,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, asyncio, base_model, cost_ack, environment_name, max_tokens, mo, prompts, run_remote, time, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable cost acknowledgement and remote execution before sampling on Modal.")
    service_client = tinker.ServiceClient(project_id="tutorial-103", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    sampling_client = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = sampling_client.get_tokenizer()
    params = tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.7, stop=["\n"])

    start = time.time()
    sequential_text = []
    for prompt_text in prompts[:3]:
        prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
        response = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        sequential_text.append(tokenizer.decode(response.sequences[0].tokens[prompt.length():]))
    sequential_time = time.time() - start

    async def sample_one(prompt_text):
        prompt = tinker.ModelInput.from_ints(tokenizer.encode(prompt_text))
        response = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        return tokenizer.decode(response.sequences[0].tokens[prompt.length():])

    start = time.time()
    concurrent_text = await asyncio.gather(*[sample_one(prompt_text) for prompt_text in prompts])
    concurrent_time = time.time() - start
    print("Sequential seconds:", round(sequential_time, 2))
    print("Concurrent seconds:", round(concurrent_time, 2))
    for question, answer in zip(prompts, concurrent_text):
        print("Q:", question)
        print("A:", answer[:160], "\n")
    return concurrent_text, concurrent_time, sequential_text, sequential_time,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(concurrent_time, sequential_time):
    print("Speedup estimate:", sequential_time / concurrent_time if concurrent_time else None)
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use this pattern inside RL rollouts in 104.
- Use completers in 203 to hide token plumbing.
- Tune sampler max containers before large sweeps.
    """)
    return


if __name__ == "__main__":
    app.run()
