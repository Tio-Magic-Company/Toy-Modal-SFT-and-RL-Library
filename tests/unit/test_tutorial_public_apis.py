from pathlib import Path

import toy_modal as tinker
from toy_modal import completers, renderers, weights
from toy_modal.cookbook import (
    Comparison,
    KLReferenceConfig,
    MessageEnv,
    PreferenceModel,
    build_dpo_datums,
    estimate_lora_parameters,
    get_lr,
    grid_sweep,
    linear_lr,
    preference_reward,
)
from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal


def test_hyperparam_and_preference_helpers() -> None:
    assert get_lr(DEFAULT_BASE_MODEL) == 1e-4
    assert linear_lr(1, 3, 1e-4) == 5e-5
    assert grid_sweep({"rank": [2, 4], "lr": [1e-4]}) == [
        {"rank": 2, "lr": 1e-4},
        {"rank": 4, "lr": 1e-4},
    ]
    assert estimate_lora_parameters(hidden_size=8, num_layers=2, rank=4) == 512
    assert KLReferenceConfig(coef=0.1).coef == 0.1
    assert preference_reward(PreferenceModel(), "q", "long answer") > 0
    assert completers.ToyModalTokenCompleter is completers.TinkerTokenCompleter
    assert completers.ToyModalMessageCompleter is completers.TinkerMessageCompleter


def test_dpo_datums_and_message_env(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    tokenizer = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL).get_tokenizer()
    renderer = renderers.get_renderer("role_colon", tokenizer)

    datums = build_dpo_datums(
        tokenizer,
        [Comparison(prompt="2+2?", chosen="4", rejected="5")],
        renderer,
    )
    env = MessageEnv([{"role": "user", "content": "Say OK"}])
    step = env.step_message({"role": "assistant", "content": "OK"})

    assert len(datums) == 1
    assert datums[0].loss_fn_inputs["preference_label"] == 1
    assert datums[0].loss_fn_inputs["rejected_tokens"]
    assert datums[0].loss_fn_inputs["rejected_target_tokens"]
    assert step.reward == 1.0


def test_chat_template_renderer_prefers_tokenizer_template() -> None:
    class TemplateTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is True
            text = "|".join(f"{item['role']}:{item['content']}" for item in messages)
            if add_generation_prompt:
                text += "|assistant:"
            return [ord(char) % 101 for char in text]

        def encode(self, text):
            return [ord(char) % 101 for char in text]

        def decode(self, tokens):
            return "".join(chr(token) for token in tokens)

    renderer = renderers.get_renderer("qwen3", TemplateTokenizer())
    model_input = renderer.build_generation_prompt([{"role": "user", "content": "hi"}])
    supervised, weights = renderer.build_supervised_example(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )

    assert isinstance(renderer, renderers.ChatTemplateRenderer)
    assert model_input.length() > 0
    assert supervised.length() == len(weights)
    assert sum(weights) > 0


def test_weight_helpers_offline(tmp_path) -> None:
    adapter = weights.build_lora_adapter(tmp_path / "checkpoint", tmp_path / "adapter", dry_run=True)
    model = weights.build_hf_model(
        base_model=DEFAULT_BASE_MODEL,
        adapter_dir=adapter,
        output_dir=tmp_path / "hf",
        merge=True,
        local_files_only=True,
        dry_run=True,
    )
    card = weights.generate_model_card(
        weights.ModelCardConfig(
            model_name="toy",
            base_model="base",
            metrics={"score": 1.0},
        )
    )
    publish = weights.publish_to_hf_hub(model, repo_id="user/repo", dry_run=True)

    assert (adapter / "adapter_config.json").exists()
    assert any(Path(model).iterdir())
    assert "# toy" in card
    assert publish["dry_run"] is True
