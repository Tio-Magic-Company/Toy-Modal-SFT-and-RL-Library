# Sampling

Create a sampler from a base model:

```python
sampler = service.create_sampling_client(base_model=tinker.DEFAULT_BASE_MODEL)
```

Or export training weights and sample from them:

```python
sampler = training.save_weights_and_get_sampling_client("step-1")
```

Sample:

```python
response = sampler.sample(
    types.ModelInput.from_ints(tokenizer.encode("Question: 3+5? Answer:")),
    num_samples=1,
    sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0),
    include_prompt_logprobs=True,
    topk_prompt_logprobs=2,
).result()
```

Use `response.samples` or `response.sequences` for generated sequences. Each
sequence has `tokens`, `stop_reason`, and optional generated-token `logprobs`.

Compute prompt logprobs:

```python
logprobs = sampler.compute_logprobs(
    types.ModelInput.from_ints(tokenizer.encode("Hello world"))
).result()
```

Test fakes return deterministic token/logprob shapes. Deployed Unsloth or
Transformers sampling requires Modal infrastructure and has separate validation
requirements.
