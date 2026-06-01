# Quickstart

`toy_modal` now runs through a user-owned Modal backend. The SDK default is
`transport="modal-direct"`, which uses the Modal Python client to call deployed
functions and classes.

Install with backend dependencies:

```bash
python -m pip install -e '.[backend,dev]'
```

Deploy when you are ready for Modal spend:

```bash
toy-modal backend deploy
```

Check the deployed app without running a training step:

```bash
toy-modal backend check --app-name toy-modal-backend
```

Minimal Python workflow:

```python
import toy_modal as tinker
from toy_modal import types

service = tinker.ServiceClient(
    project_id="demo",
    transport="modal-direct",
    app_name="toy-modal-backend",
)
training = service.create_lora_training_client(
    tinker.DEFAULT_BASE_MODEL,
    rank=4,
)
tokenizer = training.get_tokenizer()

datum = types.Datum(
    model_input=types.ModelInput.from_ints(tokenizer.encode("Question: 2+2? Answer: 4")),
    loss_fn_inputs={"target_tokens": [4], "weights": [1]},
)

loss = training.forward_backward([datum], "cross_entropy").result()
step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
sampler = training.save_weights_and_get_sampling_client("step-1")
```

For deployed recipe examples:

```bash
python docs/recipes/tiny_sft_workflow.py \
  --transport modal-direct \
  --app-name toy-modal-backend \
  --base-model unsloth/tinyllama-bnb-4bit \
  --log-path runs/tiny-sft-modal
```
