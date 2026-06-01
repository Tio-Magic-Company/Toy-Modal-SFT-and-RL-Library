# Sampling and RL Guide

Sampling supports base-model and saved-weight workflows through
`SamplingClient.sample` and `compute_logprobs`.

Deployed Modal defaults to `TOY_MODAL_SAMPLER_ENGINE=unsloth` for Unsloth-backed
generation and scoring. Set `TOY_MODAL_SAMPLER_ENGINE=transformers` for the
plain Transformers sampler. Test fakes and tiny engines return deterministic
token/logprob shapes:

- generated-token logprobs on `SampledSequence.logprobs`
- prompt logprobs when `include_prompt_logprobs=True`
- top-k prompt logprobs when `topk_prompt_logprobs > 0`
- seeded generation for reproducible smoke tests

The Unsloth and Transformers samplers support base-model sampling and PEFT
adapter loading from `toy-modal://.../sampler_weights/...` manifests written by
`save_weights_for_sampler`.

The tiny-model deployed baseline on 2026-05-27 validated base-model sampling,
saved-adapter sampling, logprob scoring, tokenizer access, and stale rollout
rejection through `modal-direct`. Larger models, throughput, deployed HTTP, and
deployed cookbook recipe coverage remain separate validation tiers.

RL recipes keep rewards in user code. The backend receives structured loss
inputs such as target tokens, weights, old logprobs, and advantages. It never
executes reward functions or arbitrary Python callables by default.

## Built-in RL Losses

`PeftTrainerEngine` implements three structured RL losses over completion
tokens only:

- `importance_sampling`: unbounded policy-gradient importance weighting.
- `ppo`: clipped PPO surrogate with configurable ratio thresholds.
- `cispo`: clipped importance ratio detached and used as a logprob weight.

Each `Datum` must provide:

- `model_input`: prompt plus completion tokens.
- `target_tokens`: completion tokens, aligned to the suffix of `model_input`.
- `old_logprobs`: completion-token logprobs from the rollout policy. The
  Tinker-style `logprobs` key is accepted as a compatibility alias.
- `advantages`: one value per completion token.
- `weights`: optional, either one value per completion token or one value per
  `model_input` token.
- `masks`: optional, one value per completion token.
- `ref_logprobs`: optional reference-policy logprobs for sampled KL.
- `old_logprobs_model_seq_id`: optional model sequence guard. If present, it
  must match the current training client's `model_seq_id`.

Accepted `loss_fn_config` keys for `importance_sampling`:

- `kl_coef`: optional coefficient for sampled `new_logprobs - ref_logprobs`.

Accepted `loss_fn_config` keys for `ppo` and `cispo`:

- `clip_low_threshold`: default `0.8`.
- `clip_high_threshold`: default `1.2`.

The losses use:

```text
ratio = exp(new_logprobs - old_logprobs)
importance_sampling = -sum(weights * masks * ratio * advantages)
ppo = -sum(weights * masks * min(ratio * advantages, clipped_ratio * advantages))
cispo = -sum(weights * masks * detach(clipped_ratio) * new_logprobs * advantages)
```

Shape validation runs before backend submission in examples and cookbook smoke
helpers so arrays fail early when they do not align.

GRPO is intentionally not a backend loss name. The cookbook layer provides a
GRPO-style helper that collects grouped rollouts, computes group-relative
advantages in user code, skips degenerate groups, and submits structured
`Datum` objects through `importance_sampling`, `ppo`, or `cispo`.
