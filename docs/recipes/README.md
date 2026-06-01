# Recipe Scripts

Recipes under `docs/recipes/` are clean-room workflow examples for the
Modal-backed SDK. They default to `transport="modal-direct"` and
`unsloth/tinyllama-bnb-4bit`, matching the default Unsloth backend profile.
Set `--app-name` and `--environment-name` for your deployed backend.

Run one recipe after deployment:

```bash
python docs/recipes/sl_loop.py \
  --transport modal-direct \
  --app-name toy-modal-backend \
  --base-model unsloth/tinyllama-bnb-4bit \
  --log-path runs/sl_loop
```

Historical promoted workflows have checked-in PEFT/Transformers tiny-model
Modal validation evidence. Use these exact commands only when reproducing that
baseline:

```bash
python docs/recipes/chat_sft.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --log-path runs/chat-sft-modal
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --loss-fn ppo --log-path runs/math-rl-ppo-modal
python docs/recipes/math_rl.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --loss-fn cispo --log-path runs/math-rl-cispo-modal
python docs/recipes/on_policy_rl_workflow.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --log-path runs/on-policy-rl-modal
python docs/recipes/tiny_sft_workflow.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --log-path runs/tiny-sft-modal
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --loss-fn ppo --log-path runs/code-rl-ppo-modal
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --loss-fn cispo --log-path runs/code-rl-cispo-modal
python docs/recipes/code_rl.py --transport modal-direct --app-name toy-modal-peft-validation --base-model hf-internal-testing/tiny-random-gpt2 --loss-fn importance_sampling --log-path runs/code-rl-is-modal
```

Remaining recipe families may still be scaffolds for data shaping, logging, or
workflow structure. Do not claim production parity until a deployed validation
report covers that recipe.
