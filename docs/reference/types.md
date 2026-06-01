# Types Reference

Common public models live in `toy_modal.types`.

## ModelInput

```python
types.ModelInput.from_ints([1, 2, 3])
```

`to_ints()` returns token IDs and raises if the input contains non-token chunks.

## Datum

```python
types.Datum(
    model_input=types.ModelInput.from_ints(tokens),
    loss_fn_inputs={"target_tokens": target_tokens},
)
```

`loss_fn_inputs` accepts plain lists and tensor-like values converted to
structured data where supported.

## AdamParams

Fields include `learning_rate`, `beta1`, `beta2`, `eps`, `weight_decay`, and
`grad_clip_norm`. `weight_decay` defaults to `0.0`.

## SamplingParams

Fields include `max_tokens`, `seed`, `stop`, `temperature`, `top_k`, and
`top_p`.

## Outputs

`ForwardBackwardOutput` includes loss fields, metrics, token counts, optional
gradient ID, and `model_seq_id`.

`LoadWeightsResponse` includes the loaded path, optional training run ID,
`model_seq_id`, `optimizer_step`, and LoRA config.

`OptimStepResponse` includes `model_seq_id`, `optimizer_step`, and metrics.

`SampleResponse` includes `sequences`, plus optional prompt and top-k prompt
logprobs. `samples` is a compatibility alias for `sequences`.
