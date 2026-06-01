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
# Tutorial 205: Evaluations

Build NLL and sampling evaluators that can run beside training loops.

Evaluations are ordinary Python code around the clients. Keeping them outside the backend makes metrics easy to customize.

> Please note: The included evaluators are tutorial-scale helpers, not a full benchmark harness.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Construct held-out datums.
2. Run an NLL evaluator.
3. Build a sampling evaluator.
4. Log metrics separately from training.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/205_evaluations.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding evaluation loops</summary>

Evaluation is the discipline of measuring model behavior separately from training. You can evaluate likelihood on held-out examples, grade sampled answers, run rubric checks, or call an external harness.

Toy Modal keeps evaluation in Python around the same clients. A training client can score NLL-style datums; a sampling client can generate answers for a grader. The important habit is recording which checkpoint or model sequence each metric describes.

Further reading:

- [Inspect AI documentation](https://inspect.aisi.org.uk/)
- [OpenAI Evals repository](https://github.com/openai/evals)
- [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/index)

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
        "project_id": "tutorial-205",
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

This follows the original evaluations notebook: small eval sets, one NLL metric, one generation metric.

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
    eval_conversations = [[{"role": "user", "content": "What is Toy Modal?"}, {"role": "assistant", "content": "A Modal-backed post-training framework."}]]
    def grader(prompt, completion):
        return 1.0 if "Modal" in completion else 0.0
    return eval_conversations, grader,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cookbook, cost_ack, environment_name, eval_conversations, grader, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before running evaluators.")
    service_client = tinker.ServiceClient(project_id="tutorial-205", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)
    eval_datums = [renderer.conversation_to_datum(conv) for conv in eval_conversations]
    nll = cookbook.NLLEvaluator("toy_modal_eval", eval_datums).evaluate(training_client)
    sampler = await training_client.save_weights_and_get_sampling_client_async("eval-sampler")
    sampling_metric = cookbook.SamplingEvaluator("toy_modal_sampling", ["Say Modal"], grader).evaluate(sampler, tokenizer)
    print({**nll, **sampling_metric})
    return eval_datums, nll, sampling_metric,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Evaluation metrics should be logged with the model sequence id and checkpoint path they describe.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use Inspect AI or your own harness for production evals.
- Use 406 for teacher/student eval loops.
- Use 407 for RLHF stage metrics.
    """)
    return


if __name__ == "__main__":
    app.run()
