# Cookbook

Cookbook pages describe Modal-backed recipes under `docs/recipes/`. Commands
use `modal-direct` and may allocate Modal resources after the backend is
deployed.

Use the tiny public model for short validation runs:

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/chat-sft-modal
```

Review [`../guides/deployment_to_modal.md`](../guides/deployment_to_modal.md)
and [`../guides/operations_and_cleanup.md`](../guides/operations_and_cleanup.md)
before long runs.
