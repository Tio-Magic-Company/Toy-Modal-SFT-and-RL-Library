# SDK Reference

Concise public surface reference. See guides for tutorials.

## Import

```python
import toy_modal as tinker
from toy_modal import types
```

## ServiceClient

- `get_server_capabilities()`
- `create_lora_training_client(base_model, rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True, user_metadata=None)`
- `create_training_client_from_state(path, user_metadata=None, weights_access_token=None)`
- `create_training_client_from_state_with_optimizer(path, user_metadata=None, weights_access_token=None)`
- `create_sampling_client(model_path=None, base_model=None, retry_config=None)`
- `create_rest_client()`

## TrainingClient

- `forward(data, loss_fn, loss_fn_config=None) -> APIFuture`
- `forward_backward(data, loss_fn, loss_fn_config=None) -> APIFuture`
- `optim_step(adam_params) -> APIFuture`
- `save_state(name, ttl_seconds=None) -> APIFuture`
- `load_state(path, weights_access_token=None) -> APIFuture`
- `load_state_with_optimizer(path, weights_access_token=None) -> APIFuture`
- `save_weights_for_sampler(name, ttl_seconds=None) -> APIFuture`
- `save_weights_and_get_sampling_client(name=None, retry_config=None)`
- `get_info()`
- `get_tokenizer()`

`forward_backward_custom` exists but is intentionally disabled by default.

## SamplingClient

- `sample(prompt, num_samples, sampling_params, include_prompt_logprobs=False, topk_prompt_logprobs=0) -> APIFuture`
- `compute_logprobs(prompt) -> APIFuture`
- `get_tokenizer()`
- `get_base_model()`

`SamplingClient` is picklable.

## RestClient

Includes run lookup, checkpoint listing/deletion, TTL, publish/unpublish,
archive metadata, session listing, and sampler lookup helpers. Prefer
`*_toy_path` methods and `toy-modal://` paths.

## Deviations

- Canonical paths use `toy-modal://`.
- `tinker://` requires explicit compatibility opt-in.
- `AdamParams.weight_decay` defaults to `0.0`.
- Custom Python loss callables are disabled by default.
