# Losses Reference

Supported structured loss names:

- `cross_entropy`
- `importance_sampling`
- `ppo`
- `cispo`

Arbitrary Python loss callables are disabled by default. Do not use
`forward_backward_custom` as a general feature.

## cross_entropy

Required:

- `target_tokens`

Optional:

- `weights`

`target_tokens` must be a non-empty suffix of `model_input`. `weights` may be
one value per model-input token or one value per target token.

## importance_sampling

Required:

- `target_tokens`
- `old_logprobs` or `logprobs`
- `advantages`

Optional:

- `weights`
- `masks`
- `ref_logprobs`
- `old_logprobs_model_seq_id`

Optional config:

- `kl_coef`

## ppo

Uses the same datum fields as `importance_sampling`.

Config:

- `clip_low_threshold`
- `clip_high_threshold`

## cispo

Uses the same datum fields as `importance_sampling`.

Config:

- `clip_low_threshold`
- `clip_high_threshold`

## GRPO

GRPO is not a backend loss name. Use
`toy_modal.cookbook.grpo_datums_from_trajectory_groups` to build structured
datums, then train with `importance_sampling`, `ppo`, or `cispo`.
