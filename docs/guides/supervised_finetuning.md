# Supervised Finetuning

Supervised finetuning uses the `cross_entropy` loss with prompt plus completion
tokens in `Datum.model_input`.

```python
prompt = tokenizer.encode("Question: 2+2? Answer:")
completion = tokenizer.encode(" 4")

datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt + completion),
    loss_fn_inputs={
        "target_tokens": completion,
        "weights": [0.0] * len(prompt) + [1.0] * len(completion),
    },
)

loss = training.forward_backward([datum], "cross_entropy").result()
step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

`target_tokens` must align with the suffix of `model_input`. `weights` can be
one value per model-input token or one value per target token.

Local command:

```bash
toy-modal backend check --app-name toy-modal-backend
```

For a fuller promoted workflow:

```bash
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/tiny-sft
```
