# Cookbook Technical Notes

Recipes live under `docs/recipes/` and default to `modal-direct`.

```bash
python docs/recipes/sl_loop.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/sl_loop
```

The recipe framework validates SDK request shapes, logging, artifact paths, and
loss input construction. Full claims require deployed validation evidence for
the specific recipe and model class.

Promoted tiny-model checks should use:

```bash
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/tiny-sft
python docs/recipes/on_policy_rl_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/on-policy-rl
```
