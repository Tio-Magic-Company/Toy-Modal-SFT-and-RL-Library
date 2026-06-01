# Security Guide

## Custom Losses

`forward_backward_custom` is disabled by default. Enabling remote execution of
client-provided Python would be remote code execution and requires a separate
trusted-execution design.

If a trusted direct mode is added later, it must require all of:

- explicit `allow_unsafe_custom_loss=True`
- direct user-owned Modal deployment
- no HTTP transport exposure
- clear owner-only documentation

## Auth and Secrets

- Use user-owned Modal Secrets for private model tokens.
- HTTP gateway deployments should require a project-owned bearer token or Modal
  proxy authentication.
- Do not log secret values, API keys, raw credentials, or private dataset rows.

