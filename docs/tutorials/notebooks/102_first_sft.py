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
# Tutorial 102: Your first supervised fine-tuning run

Teach a Toy Modal Tutor persona with supervised data, LoRA training, optimizer steps, and post-training sampling.

Toy Modal is a framework using Modal to remotely fine-tune and reinforcement-learn language models. Your notebook owns the examples, rewards, and loop; Modal owns GPU-heavy forward passes, backpropagation, optimizer steps, and sampling.

> Please note: This is a tiny mechanics run. It teaches the SFT loop but does not claim convergence, quality, or throughput for larger models.

```
Your machine (Python control)          Your Modal app (remote compute)
+------------------------------+       +--------------------------------+
| data prep, rewards, evals    | ----> | forward/backward, sampling     |
| notebook decisions           | <---- | optimizer steps, checkpoints   |
+------------------------------+       +--------------------------------+
```

By the end of this notebook you will:

1. Build training data from chat messages with a renderer.
2. Create a LoRA TrainingClient on Modal.
3. Run forward_backward and optim_step repeatedly.
4. Save trained weights and sample from them.

Run it with Marimo:

```bash
python -m pip install -e '.[backend,tutorials]'
marimo edit docs/tutorials/notebooks/102_first_sft.py
```

Remote cells are Modal-first and stay gated behind the in-notebook cost acknowledgement, the same intent as passing `--i-understand-costs` in the old script tutorials.
    """)
    return



@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
<details>
<summary>Understanding SFT and LoRA</summary>

Supervised fine-tuning, or SFT, trains a model to imitate target answers for given prompts. For chat models, examples are usually conversations: system instructions, user messages, and assistant responses. The loss is applied mainly to the assistant tokens so the model learns what to say, not to memorize the prompt.

Toy Modal expresses this as `Datum` objects. A renderer converts messages into model tokens and loss weights; `forward_backward` computes gradients remotely; `optim_step` updates LoRA adapter weights on Modal. LoRA is parameter-efficient: it trains small adapter matrices instead of every base-model parameter.

Further reading:

- [Hugging Face TRL SFTTrainer](https://huggingface.co/docs/trl/main/en/sft_trainer)
- [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/developer_guides/lora)
- [LoRA paper](https://arxiv.org/abs/2106.09685)

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
        "project_id": "tutorial-102",
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

This mirrors the first SFT notebook structure: build a persona dataset, render it into Datum objects, train for a few steps, plot or print the loss curve, then ask the trained adapter questions.

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
    SYSTEM_PROMPT = (
        "You are Toy Modal Tutor, a concise assistant for a framework that uses Modal "
        "to fine-tune and reinforcement-learn language models. Explain that users own "
        "the Modal app, keep Python control loops locally, and send GPU-heavy work to workers."
    )
    conversations = [
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "What is Toy Modal?"}, {"role": "assistant", "content": "Toy Modal is a clean-room SDK for running post-training loops against your own Modal backend."}],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Where does training run?"}, {"role": "assistant", "content": "Your Python loop runs locally, while Modal workers run forward, backward, optimizer, checkpoint, and sampling work."}],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "What is a Datum?"}, {"role": "assistant", "content": "A Datum packages model_input tokens together with loss_fn_inputs such as target tokens, weights, logprobs, or advantages."}],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "What is a renderer?"}, {"role": "assistant", "content": "A renderer converts messages into model tokens and decides which tokens receive training weight."}],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "How do I train?"}, {"role": "assistant", "content": "Create a TrainingClient, build datums, call forward_backward, call optim_step, then save weights for sampling."}],
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am Toy Modal Tutor, a small assistant trained to explain the Toy Modal framework."}],
    ]
    print(f"Prepared {len(conversations)} Toy Modal Tutor conversations")
    return SYSTEM_PROMPT, conversations,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Run the remote workflow

The next cell is where Modal may be contacted. It uses `transport="modal-direct"`, which calls your deployed Modal app with the Modal Python client. Keep this gate closed until your app name, base model, and cost expectations are correct.
    """)
    return

@app.cell
async def _(SYSTEM_PROMPT, app_name, base_model, conversations, cost_ack, environment_name, max_tokens, mo, renderers, run_remote, tinker):
    mo.stop(not cost_ack.value or not run_remote.value, "Enable cost acknowledgement and remote execution before training on Modal.")
    service_client = tinker.ServiceClient(project_id="tutorial-102", transport="modal-direct", app_name=app_name.value, environment_name=environment_name.value or None)
    training_client = await service_client.create_lora_training_client_async(base_model=base_model.value, rank=4)
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)
    training_data = [renderer.conversation_to_datum(conv) for conv in conversations]
    losses = []
    for step in range(3):
        fwdbwd_future = await training_client.forward_backward_async(training_data, "cross_entropy")
        optim_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=0.0002))
        fwdbwd_result = await fwdbwd_future.result_async()
        optim_result = await optim_future.result_async()
        losses.append(fwdbwd_result.loss)
        print(f"Step {step}: loss={fwdbwd_result.loss} optimizer_step={optim_result.optimizer_step}")

    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="toy-modal-tutor-sft")
    prompt = renderer.build_generation_prompt([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "What is Toy Modal?"}])
    result = await sampling_client.sample_async(prompt=prompt, num_samples=1, sampling_params=tinker.SamplingParams(max_tokens=int(max_tokens.value), temperature=0.3, stop=renderer.get_stop_sequences()))
    message, termination = renderer.parse_response(result.sequences[0].tokens[prompt.length():])
    print("Sampled answer:", renderers.get_text_content(message))
    print("Termination:", termination.value)
    return losses, renderer, sampling_client, tokenizer, training_client,

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Inspect and reason about the result

The point of each tutorial is not just to call an API. Inspect the output shape, connect it back to the training loop, and decide what you would log or validate before scaling the run.
    """)
    return

@app.cell
def _(losses):
    print("Losses:", losses)
    print("Loss decreased?", losses[-1] <= losses[0] if len(losses) > 1 and None not in losses else "not measured")
    return

@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
## Next steps

- Open 103 for efficient sampling futures.
- Open 201 for renderer details.
- Move to config-driven SFT in 303 once the raw loop is clear.
    """)
    return


if __name__ == "__main__":
    app.run()
