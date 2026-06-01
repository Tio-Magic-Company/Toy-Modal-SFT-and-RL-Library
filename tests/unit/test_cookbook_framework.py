import json

import toy_modal as tinker
from toy_modal import types
from toy_modal.cookbook import (
    ChatTemplateRenderer,
    LocalBenchmark,
    Message,
    RolloutStore,
    RLTrainConfig,
    SupervisedTrainConfig,
    Trajectory,
    TrajectoryGroup,
    TrainLoopConfig,
    RECIPE_NAMES,
    RecipeConfig,
    build_service,
    collect_grouped_rollouts,
    grpo_datums_from_trajectory_groups,
    group_relative_advantages,
    load_conversation_jsonl,
    render_conversation_datums,
    run_many_smoke_recipes,
    run_rl_train_loop,
    run_smoke_recipe,
)
from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal


def test_cookbook_framework_runs_one_recipe(tmp_path, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    result = run_smoke_recipe(RecipeConfig(name="sl_loop", log_path=tmp_path))

    assert result.recipe == "sl_loop"
    assert result.optimizer_step == 1
    assert result.model_path.startswith("toy-modal://")
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "checkpoints.jsonl").exists()


def test_cookbook_recipe_config_uses_modal_app_env(monkeypatch) -> None:
    monkeypatch.setenv("TOY_MODAL_APP_NAME", "toy-modal-peft-validation")
    monkeypatch.setenv("TOY_MODAL_ENVIRONMENT", "dev")

    service = build_service(RecipeConfig(name="chat_sft", transport="modal-direct"))

    assert service.app_name == "toy-modal-peft-validation"
    assert service.environment_name == "dev"


def test_training_configs_carry_modal_app_env(monkeypatch) -> None:
    monkeypatch.setenv("TOY_MODAL_APP_NAME", "toy-modal-notebooks")
    monkeypatch.setenv("TOY_MODAL_ENVIRONMENT", "tutorials")

    class EmptyDatasetBuilder:
        common_config = None

        def __call__(self):
            return [], None

    sft_config = SupervisedTrainConfig(dataset_builder=EmptyDatasetBuilder())
    rl_config = RLTrainConfig(prompts=["prompt"])

    assert sft_config.app_name == "toy-modal-notebooks"
    assert sft_config.environment_name == "tutorials"
    assert rl_config.app_name == "toy-modal-notebooks"
    assert rl_config.environment_name == "tutorials"


def test_cookbook_framework_runs_selected_recipes(tmp_path, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    results = run_many_smoke_recipes(["sl_loop", "rl_loop"], log_path=tmp_path)

    assert [result.recipe for result in results] == ["sl_loop", "rl_loop"]
    assert all(result.optimizer_step == 1 for result in results)
    assert (tmp_path / "sl_loop" / "metrics.jsonl").exists()
    assert (tmp_path / "rl_loop" / "metrics.jsonl").exists()
    assert "chat_sft" in RECIPE_NAMES


def test_chat_renderer_loads_jsonl_and_builds_structured_datums(tmp_path, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    dataset = tmp_path / "chat.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Say hi"},
                    {"role": "assistant", "content": "Hi"},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    tokenizer = service.create_lora_training_client(DEFAULT_BASE_MODEL).get_tokenizer()

    conversations = load_conversation_jsonl(dataset)
    datums = render_conversation_datums(tokenizer, conversations, renderer=ChatTemplateRenderer())

    assert conversations == [
        [Message(role="user", content="Say hi"), Message(role="assistant", content="Hi")]
    ]
    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["target_tokens"]
    assert sum(datums[0].loss_fn_inputs["weights"]) == len(datums[0].loss_fn_inputs["target_tokens"])


def test_grpo_datums_skip_degenerate_groups_and_support_rl_losses() -> None:
    group = TrajectoryGroup(
        prompt="2+2",
        trajectories=[
            Trajectory("2+2", [1, 2], [3], [-0.1], 1.0, text="4"),
            Trajectory("2+2", [1, 2], [4], [-0.2], -1.0, text="5"),
        ],
    )
    degenerate = TrajectoryGroup(
        prompt="same",
        trajectories=[
            Trajectory("same", [1], [2], [-0.1], 0.0),
            Trajectory("same", [1], [3], [-0.1], 0.0),
        ],
    )

    assert group_relative_advantages([1.0, -1.0]) == [1.0, -1.0]
    assert group_relative_advantages([0.0, 0.0]) is None
    for loss_fn in ("importance_sampling", "ppo", "cispo"):
        datums = grpo_datums_from_trajectory_groups(
            [group, degenerate],
            loss_fn=loss_fn,
            model_seq_id=0,
        )
        assert len(datums) == 2
        assert datums[0].loss_fn_inputs["old_logprobs_model_seq_id"] == 0
        assert datums[0].loss_fn_inputs["advantages"] == [1.0]


def test_grouped_rollouts_store_and_rl_train_loop(tmp_path, monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    tokenizer = training.get_tokenizer()
    sampler = training.save_weights_and_get_sampling_client("rollout")
    store = RolloutStore(tmp_path / "trajectories.jsonl")

    groups = collect_grouped_rollouts(
        sampler=sampler,
        tokenizer=tokenizer,
        prompts=["Question: 2+2?"],
        group_size=2,
        sampling_params=types.SamplingParams(max_tokens=1, seed=5),
        reward_fn=lambda _prompt, completion: 1.0 if completion else -1.0,
        store=store,
    )
    datums = grpo_datums_from_trajectory_groups(
        [
            TrajectoryGroup(
                prompt=groups[0].prompt,
                trajectories=[
                    groups[0].trajectories[0],
                    Trajectory(
                        groups[0].prompt,
                        groups[0].trajectories[1].prompt_tokens,
                        groups[0].trajectories[1].completion_tokens,
                        groups[0].trajectories[1].old_logprobs,
                        -1.0,
                    ),
                ],
            )
        ],
        loss_fn="ppo",
        model_seq_id=training.model_seq_id,
    )
    result = run_rl_train_loop(training, datums, TrainLoopConfig(loss_fn="ppo"))

    assert store.read_groups()
    assert result.optimizer_step == 1
    assert result.checkpoints


def test_local_benchmark_scaffold_runs_against_sampler(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    sampler = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL)
    tokenizer = sampler.get_tokenizer()
    benchmark = LocalBenchmark("tiny", ["prompt"], lambda _prompt, completion: 1.0 if completion else 0.0)

    assert benchmark.run(sampler, tokenizer)["tiny/score"] == 1.0
