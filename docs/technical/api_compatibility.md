# API Compatibility

`toy_modal` keeps a clean-room Tinker-style public method shape while targeting
user-owned Modal infrastructure.

Supported:

- `import toy_modal as tinker`
- `ServiceClient`, `TrainingClient`, `SamplingClient`, `RestClient`
- `APIFuture` sync/async result handling
- `toy-modal://` artifact paths
- Explicit `tinker://` path compatibility when opted in
- `modal-direct` and deployed HTTP transports

Not supported:

- A top-level package named `tinker`
- Public local runtime transports
- In-process HTTP gateway shortcuts
- Arbitrary backend-executed Python loss callables by default

No-credential validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
python dev_notes/validation/run_modal_validation.py --help
git diff --check
```
