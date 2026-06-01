# Examples

Examples run from a checkout against a deployed backend. Set
`TOY_MODAL_APP_NAME` and `TOY_MODAL_ENVIRONMENT` as needed for `modal-direct`.
For the deployed HTTP gateway, pass `--transport http --base-url "$TOY_MODAL_HTTP_BASE_URL"`
and `--api-key "$TOY_MODAL_HTTP_API_KEY"` when the gateway requires a token.

## Minimal SFT

```bash
python docs/examples/sft_minimal.py
```

Shows the core workflow:

- create a LoRA training client
- tokenize prompt/answer text
- call `forward_backward`
- call `optim_step`
- export sampler weights
- sample from the saved weights

## RL-Shaped Loop

```bash
python docs/examples/rl_minimal.py
```

Shows the SDK shape for a reward-driven loop using the
`importance_sampling` loss name. Real reward modeling and rollout collection
are backend/cookbook responsibilities.

## Batch Sampling

```bash
python docs/examples/sampling_batch.py
```

Creates a base-model `SamplingClient` and requests multiple samples.

## Sampling Logprobs

```bash
python docs/examples/logprob_sampling.py
```

Requests generated-token logprobs, prompt logprobs, and top-k prompt logprobs.
This is the smallest example to consult when building RL or preference-learning
loops that need scoring.

## Checkpoint Resume

```bash
python docs/examples/checkpoint_resume.py
```

Saves a checkpoint and resumes from it with optimizer state.

Use `create_training_client_from_state` when the optimizer should reset, and
`create_training_client_from_state_with_optimizer` when resuming the same run
schedule.

## REST Metadata

```bash
python docs/examples/rest_metadata.py
```

Exercises run lookup, checkpoint archive URLs, session listing, and sampler
metadata lookup.

## Cookbook Helper

```bash
python docs/examples/cookbook_smoke.py
```

Runs `toy_modal.cookbook.run_smoke_recipe` directly and prints a JSON record
with the recipe name, loss, optimizer step, model path, and sample output.

## HTTP Gateway

The old in-process HTTP example has been removed. HTTP transport examples use a
deployed gateway URL:

```bash
python docs/examples/sft_minimal.py \
  --transport http \
  --base-url "$TOY_MODAL_HTTP_BASE_URL" \
  --api-key "$TOY_MODAL_HTTP_API_KEY"
```
