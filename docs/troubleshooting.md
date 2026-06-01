# Troubleshooting

## Import Fails

Install from the checkout:

```bash
python -m pip install -e .
```

Or run commands from the repository root, where tests add `src` to the Python
path.

## `base_url is required for http transport`

HTTP transport needs a deployed gateway URL:

```python
tinker.ServiceClient(
    project_id="demo-http",
    transport="http",
    base_url="https://<your-deployed-gateway>",
)
```

Use `modal-direct` for the default Python client path.

## `Unsupported path scheme: 'tinker'`

Use `toy-modal://` paths. `tinker://` is compatibility input only and requires:

```python
tinker.ServiceClient(..., accept_tinker_paths=True)
```

## Empty Or Invalid RL Batch

For RL losses, each datum needs prompt plus completion tokens, non-empty
`target_tokens`, rollout logprobs, and advantages. Lengths must match completion
tokens unless the field explicitly allows full-sequence weights.

## Stale Rollout Logprobs

If `old_logprobs_model_seq_id` is set and the model has advanced, collect new
rollouts or resume the matching checkpoint.

## `forward_backward_custom` Is Disabled

This is intentional. Use `cross_entropy`, `importance_sampling`, `ppo`, or
`cispo`.

## Modal Authentication Or Deployment Fails

Confirm Modal CLI authentication and workspace access. Local quickstart does not
deploy Modal automatically. Deployed commands are cost-bearing and should be run only after
reviewing [`guides/deployment_to_modal.md`](guides/deployment_to_modal.md).

## Deployed Worker Cannot See A Checkpoint

This is usually a Volume visibility issue. Writers must commit after saving.
Readers must reload before reading artifacts created by another worker.

## Web Gateway Times Out

Long GPU work should not block web requests. HTTP gateway mode should submit
work asynchronously and retrieve results later.

## Recipe Output Is Different From Docs

The docs describe output shape, not exact values. Values may change as the
backend evolves or model dependencies update.
