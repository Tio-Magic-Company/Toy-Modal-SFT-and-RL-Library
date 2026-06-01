# Safety Model

`toy_modal` keeps executable user logic outside the backend by default.

## Rewards

Rewards run in user code. RL recipes collect samples, compute rewards locally,
and send structured tensors to the backend:

- completion tokens
- old rollout logprobs
- advantages
- weights and masks
- optional reference logprobs

## Custom Losses

`forward_backward_custom` is intentionally disabled in the scaffold. Arbitrary
client-supplied Python loss callables are remote code execution when sent to a
remote worker.

Only use built-in structured loss names unless a future trusted direct mode is
designed and explicitly enabled.

## Code Execution

`docs/recipes/code_rl.py` runs candidate code in a user-owned local Python subprocess
with a timeout. The backend receives structured reward/loss inputs, not code to
execute.

## Secrets

Private model tokens belong in user-owned Modal Secrets for deployed runs. Do
not log secret values, API keys, raw credentials, or private dataset rows.
