# PEFT Trainer Backend

`PeftTrainerEngine` is an optional backend implementation behind the same worker
methods as the deterministic `TrainerEngine`. It is selected in a Modal
deployment with:

```text
TOY_MODAL_TRAINER_ENGINE=peft
```

The SDK surface remains `types.LoraConfig`; PEFT stays a backend dependency.

## Validation Evidence

The tiny-model `modal-direct` PEFT/Transformers baseline passed on 2026-05-27:

```text
dev_notes/validation_reports/modal-peft-20260527T163656Z/modal_parity_20260527T163730Z.json
summary: 19 pass, 0 fail, 0 skipped
```

That report covers PEFT training, checkpoint save/load, sampler-weight export,
saved-adapter sampling/logprobs/tokenizer access, base-model sampling, stale
rollout rejection, and core REST metadata routes. It does not cover HTTP gateway
validation, large models, throughput, production archive download, or deployed
cookbook parity.

## Public-To-PEFT Mapping

| `types.LoraConfig` field | PEFT mapping |
| --- | --- |
| `rank` | `LoraConfig(r=rank)` |
| `alpha` | `LoraConfig(lora_alpha=alpha)`; defaults to `rank` when unset |
| `dropout` | `LoraConfig(lora_dropout=dropout)` |
| `target_modules` | Direct override of PEFT target modules |
| `train_attn` | Adds family-specific attention projection names |
| `train_mlp` | Adds family-specific MLP projection names |
| `train_unembed` | Adds the family-specific unembedding module |

All PEFT configs use `task_type="CAUSAL_LM"` and `bias="none"`.

## Model-Family Targets

| Model family marker | Attention targets | MLP targets | Unembed target |
| --- | --- | --- | --- |
| `qwen`, `llama`, `mistral`, `mixtral`, `gemma`, `yi`, `deepseek`, `phi` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `gate_proj`, `up_proj`, `down_proj` | `lm_head` |
| `gpt2`, `gpt-2`, `distilgpt2` | `c_attn`, `c_proj` | `c_fc`, `c_proj` | `lm_head` |
| `gpt_neox`, `gpt-neox`, `pythia`, `dolly` | `query_key_value`, `dense` | `dense_h_to_4h`, `dense_4h_to_h` | `embed_out` |
| `falcon` | `query_key_value`, `dense` | `dense_h_to_4h`, `dense_4h_to_h` | `lm_head` |
| `opt-`, `facebook/opt`, `/opt` | `q_proj`, `k_proj`, `v_proj`, `out_proj` | `fc1`, `fc2` | `lm_head` |
| default causal LM | `q_proj`, `k_proj`, `v_proj`, `o_proj` | `gate_proj`, `up_proj`, `down_proj` | `lm_head` |

GPT-2-style `c_proj` is a suffix used in both attention and MLP modules. Use
`target_modules` when a model family needs more precise targeting than suffix
matching can provide.

## Training Behavior

SFT uses shifted causal-LM logits. Labels are built from `target_tokens` aligned
to the suffix of `model_input`; prompt and zero-weight positions are assigned
`-100`. `weights` may match either the full `model_input` length or the
completion `target_tokens` length.

`importance_sampling` is intentionally narrow: it trains only over completion
tokens and requires completion-aligned `old_logprobs`, `advantages`, optional
`weights`, optional `masks`, and optional `ref_logprobs`.
