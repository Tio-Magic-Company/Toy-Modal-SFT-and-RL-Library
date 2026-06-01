# Reinforcement Learning

`toy_modal` supports a structured RL path. Rewards run in user code. The
backend receives tensors only and never executes reward functions or arbitrary
Python loss callables by default.

## Built-In Losses

Structured RL loss names:

- `importance_sampling`
- `ppo`
- `cispo`

`ppo` and `cispo` accept:

- `clip_low_threshold`
- `clip_high_threshold`

`importance_sampling` also accepts optional `kl_coef` when `ref_logprobs` are
provided.

## Required Datum Fields

Each RL `Datum` uses prompt plus completion tokens in `model_input`.

Required `Datum.loss_fn_inputs`:

- `target_tokens`: completion tokens, aligned to the suffix of `model_input`.
- `old_logprobs` or `logprobs`: rollout-policy logprobs for completion tokens.
- `advantages`: one value per completion token.

Optional fields:

- `weights`: one value per completion token, or one value per `model_input`
  token.
- `masks`: one value per completion token.
- `ref_logprobs`: reference-policy logprobs for sampled KL.
- `old_logprobs_model_seq_id`: rollout guard. If present, it must match the
  training client's current `model_seq_id`.

## Minimal Datum

```python
datum = types.Datum(
    model_input=types.ModelInput.from_ints(prompt_tokens + completion_tokens),
    loss_fn_inputs={
        "target_tokens": completion_tokens,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "weights": [1.0] * len(completion_tokens),
        "old_logprobs_model_seq_id": training.model_seq_id,
    },
)

loss = training.forward_backward(
    [datum],
    "ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    },
).result()
step = training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
```

## GRPO-Style Cookbook Pattern

GRPO is a cookbook helper pattern, not a backend loss name. The helper collects
grouped rollouts, computes group-relative advantages in user code, skips
degenerate groups by default, and emits structured `Datum` objects for one of
the supported backend losses.

```python
from toy_modal import types
from toy_modal.cookbook import (
    collect_grouped_rollouts,
    grpo_datums_from_trajectory_groups,
)

groups = collect_grouped_rollouts(
    sampler=rollout_sampler,
    tokenizer=tokenizer,
    prompts=["Question: 2+2? Answer with a single number:"],
    group_size=3,
    sampling_params=types.SamplingParams(max_tokens=4, temperature=0.7, seed=23),
    reward_fn=lambda _prompt, completion: 1.0 if "4" in completion else -1.0,
)

datums = grpo_datums_from_trajectory_groups(
    groups,
    loss_fn="ppo",
    model_seq_id=training.model_seq_id,
)
```

Degenerate groups are groups where all rewards are effectively the same. They
produce no useful group-relative advantage signal, so the helper skips them by
default.

## Local Recipe

```bash
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/math-rl
```

The recipe writes metrics and checkpoint records under `--log-path`, plus
`trajectories.jsonl` for grouped rollout records.
