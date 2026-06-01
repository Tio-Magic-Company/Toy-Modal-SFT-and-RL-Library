# Cookbook Helpers Reference

Reusable helpers live in `toy_modal.cookbook`.

## Smoke Helpers

- `RecipeConfig`
- `RecipeResult`
- `list_recipes()`
- `run_smoke_recipe(config)`
- `run_many_smoke_recipes(...)`

## Chat Helpers

- `Message`
- `ChatTemplateRenderer`
- `load_conversation_jsonl(path)`
- `render_conversation_datums(tokenizer, conversations)`

## Training Loop Helpers

- `TrainLoopConfig`
- `run_supervised_train_loop(training, datums, config)`
- `run_rl_train_loop(training, datums, config, loss_fn_config=None)`

## RL Helpers

- `Trajectory`
- `TrajectoryGroup`
- `RolloutStore`
- `collect_grouped_rollouts(...)`
- `group_relative_advantages(...)`
- `grpo_datums_from_trajectory_groups(...)`

These helpers keep rewards and environment logic in user code and submit only
structured tensors to the backend.
