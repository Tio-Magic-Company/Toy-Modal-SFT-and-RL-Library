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
# Tutorial 406: Prompt distillation

Use a teacher prompt to create student SFT examples.

Prompt distillation turns expensive prompt behavior into training data for a student adapter.

> Please note: Teacher quality requires a capable configured model; tiny validation models only prove wiring.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Define teacher and student prompts.
2. Collect teacher completions.
3. Render student training datums.
4. Train a smaller prompt-free behavior.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/406_prompt_distillation.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding prompt distillation</summary>

Prompt distillation turns an expensive or verbose teacher behavior into supervised data for a student. The teacher may use a long system prompt, stronger model, or more context; the student is trained to imitate the resulting answers with a shorter prompt or cheaper adapter.

Toy Modal uses the same sampling and SFT primitives: sample teacher answers with a `SamplingClient`, build student conversations, render them into datums, and fine-tune the student adapter remotely.

Further reading:

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- [DistilBERT paper](https://arxiv.org/abs/1910.01108)
- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)

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
        "project_id": "tutorial-406",
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

This mirrors the distillation notebook: teacher generation, dataset construction, student update, comparison.

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
    teacher_system = "You are a careful Toy Modal expert. Answer with one concrete implementation detail."
    student_system = "You are a concise assistant."
    questions = ["What is modal-direct?", "What do Volumes store?", "Why use LoRA?"]
    return questions, student_system, teacher_system,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(app_name, base_model, cost_ack, environment_name, max_tokens, mo, questions, renderers, run_remote, student_system, teacher_system, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable remote execution before teacher/student distillation.")
    service_client = tinker.ServiceClient(project_id="tutorial-406", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    teacher = await service_client.create_sampling_client_async(base_model=base_model.value)
    tokenizer = teacher.get_tokenizer(); renderer = renderers.get_renderer("role_colon", tokenizer)
    teacher_answers = []
    for question in questions:
        prompt = renderer.build_generation_prompt([{"role": "system", "content": teacher_system}, {"role": "user", "content": question}])
        response = await teacher.sample_async(prompt, 1, tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.3, stop=renderer.get_stop_sequences()))
        message, _ = renderer.parse_response(response.sequences[0].tokens[prompt.length():])
        teacher_answers.append(renderers.get_text_content(message))
    conversations = [[{"role": "system", "content": student_system}, {"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in zip(questions, teacher_answers)]
    training = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    datums = [renderer.conversation_to_datum(conv) for conv in conversations]
    result = await (await training.forward_backward_async(datums, "cross_entropy")).result_async()
    print("Teacher answers:", teacher_answers)
    print("Student SFT loss:", result.loss)
    return conversations, teacher_answers,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _():
    print("Distillation quality is capped by teacher quality and prompt coverage.")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Use 102 for SFT mechanics.
- Use 205 for evaluation.
- Use 407 for full RLHF stage composition.
    """)
    return


if __name__ == "__main__":
    app.run()
