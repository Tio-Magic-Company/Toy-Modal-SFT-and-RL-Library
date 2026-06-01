import pytest

from toy_modal import types
from toy_modal.backend.lora_mapping import build_peft_lora_kwargs, resolve_lora_target_modules
from toy_modal.errors import BadRequestError


def test_qwen_lora_mapping_uses_attention_mlp_and_unembed_targets() -> None:
    kwargs = build_peft_lora_kwargs(
        "Qwen/Qwen3-1.7B",
        types.LoraConfig(rank=8, alpha=16, dropout=0.05),
    )

    assert kwargs["r"] == 8
    assert kwargs["lora_alpha"] == 16
    assert kwargs["lora_dropout"] == 0.05
    assert kwargs["task_type"] == "CAUSAL_LM"
    assert kwargs["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]


def test_lora_mapping_respects_train_switches_and_target_override() -> None:
    config = types.LoraConfig(
        rank=4,
        train_attn=True,
        train_mlp=False,
        train_unembed=False,
    )
    assert resolve_lora_target_modules("meta-llama/Llama-3.2-1B", config) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]

    override = types.LoraConfig(
        rank=4,
        target_modules=["custom_proj", "custom_proj", "other_proj"],
        train_attn=False,
        train_mlp=False,
        train_unembed=False,
    )
    assert resolve_lora_target_modules("unknown/model", override) == [
        "custom_proj",
        "other_proj",
    ]


def test_gpt2_mapping_documents_c_proj_suffix_limitation() -> None:
    kwargs = build_peft_lora_kwargs(
        "gpt2",
        types.LoraConfig(rank=2, train_unembed=False),
    )

    assert kwargs["target_modules"] == ["c_attn", "c_proj", "c_fc"]
    assert kwargs["lora_alpha"] == 2


def test_lora_mapping_rejects_empty_targets() -> None:
    with pytest.raises(BadRequestError):
        resolve_lora_target_modules(
            "unknown/model",
            types.LoraConfig(
                train_attn=False,
                train_mlp=False,
                train_unembed=False,
            ),
        )
