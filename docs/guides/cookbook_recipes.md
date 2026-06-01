# Cookbook Recipes

Recipes live under `docs/recipes/` and default to `modal-direct` with the
Unsloth tiny validation model. Deployed runs are user-owned and cost-bearing.

List recipe names:

```bash
toy-modal cookbook list
```

Run a smoke recipe:

```bash
toy-modal cookbook smoke sl_loop --log-path runs/sl-loop
```

Promoted workflow scripts:

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/chat-sft
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/math-rl
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --loss-fn ppo --log-path runs/code-rl
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/tiny-sft
python docs/recipes/on_policy_rl_workflow.py --transport modal-direct --app-name toy-modal-backend --base-model unsloth/tinyllama-bnb-4bit --log-path runs/on-policy-rl
```

Promoted recipe pages:

- [`../cookbook/chat_sft.md`](../cookbook/chat_sft.md)
- [`../cookbook/math_rl.md`](../cookbook/math_rl.md)
- [`../cookbook/code_rl.md`](../cookbook/code_rl.md)
- [`../cookbook/tiny_sft_workflow.md`](../cookbook/tiny_sft_workflow.md)
- [`../cookbook/on_policy_rl_workflow.md`](../cookbook/on_policy_rl_workflow.md)

Other recipe families are tracked in
[`../cookbook/recipe_status.md`](../cookbook/recipe_status.md).

Deployed commands use the same scripts with `--transport modal-direct` after
the user deploys the backend. Those commands are user-owned and cost-bearing.

The next deployed recipe validation tier should archive logs for:

- `docs/recipes/chat_sft.py --transport modal-direct`
- `docs/recipes/math_rl.py --transport modal-direct --loss-fn ppo`
- `docs/recipes/math_rl.py --transport modal-direct --loss-fn cispo`
- `docs/recipes/on_policy_rl_workflow.py --transport modal-direct`

Run this only after the Modal Unsloth validation report passes. Use
[`../../dev_notes/README.md`](../../dev_notes/README.md) for the current
remaining-parity and validation status.
