# Glossary

`APIFuture`
: Awaitable handle returned by heavy SDK calls.

`modal-direct`
: Transport that calls a deployed Modal backend through the Modal Python client.

`HTTP gateway`
: Modal-hosted FastAPI gateway for submit/retrieve style access.

`toy-modal://`
: Canonical artifact path scheme.

`TrainingClient`
: Client for one LoRA training run.

`SamplingClient`
: Client for sampling from a base model or saved sampler weights.

`RestClient`
: Client for metadata, checkpoint, session, and sampler operations.

`Modal Volume`
: Durable filesystem used for canonical deployed state.

`Modal Dict`
: Fast metadata/cache layer, not canonical state.

`Modal Queue`
: Coordination primitive, not durable state.

`GRPO`
: Cookbook pattern for grouped rollouts and group-relative advantages. Not a
backend loss name in `toy_modal`.
