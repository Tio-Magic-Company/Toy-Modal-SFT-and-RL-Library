import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from toy_modal import types
from toy_modal.backend.losses import token_logprobs_from_logits
from toy_modal.backend.peft_trainer import PeftTrainerEngine
from toy_modal.backend.registry import create_run


def test_peft_trainable_parameter_count_is_adapter_only(tmp_path) -> None:
    engine, _ = _engine(tmp_path)

    trainable, total = engine.trainable_parameter_count()
    trainable_names = [
        name
        for name, parameter in engine.model.named_parameters()
        if parameter.requires_grad
    ]

    assert 0 < trainable < total
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)


def test_peft_overfits_one_sft_batch(tmp_path) -> None:
    engine, run_id = _engine(tmp_path)
    datum = _sft_datum()

    first = types.ForwardBackwardOutput.model_validate(
        engine.forward({"training_run_id": run_id, "data": [_payload(datum)], "loss_fn": "cross_entropy"})
    ).loss
    for _ in range(12):
        engine.forward_backward(
            {
                "training_run_id": run_id,
                "data": [_payload(datum)],
                "loss_fn": "cross_entropy",
                "expected_model_seq_id": engine._record()["model_seq_id"],
            }
        )
        engine.optim_step(
            {
                "training_run_id": run_id,
                "adam_params": types.AdamParams(
                    learning_rate=0.05,
                    grad_clip_norm=1.0,
                ).model_dump(mode="json"),
                "expected_model_seq_id": engine._record()["model_seq_id"],
            }
        )
    last = types.ForwardBackwardOutput.model_validate(
        engine.forward(
            {
                "training_run_id": run_id,
                "data": [_payload(datum)],
                "loss_fn": "cross_entropy",
                "expected_model_seq_id": engine._record()["model_seq_id"],
            }
        )
    ).loss

    assert last < first


def test_peft_checkpoint_writes_adapter_optimizer_and_manifest(tmp_path) -> None:
    engine, run_id = _engine(tmp_path)
    _train_one_step(engine, run_id)

    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": run_id, "name": "step-1"})
    )
    manifest = _manifest_path(tmp_path, checkpoint.path)

    assert manifest.exists()
    assert (manifest.parent / "adapter" / "adapter_config.json").exists()
    assert (
        (manifest.parent / "adapter" / "adapter_model.safetensors").exists()
        or (manifest.parent / "adapter" / "adapter_model.bin").exists()
    )
    assert (manifest.parent / "optimizer.pt").exists()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert {item["path"] for item in payload["adapter_files"]} >= {"adapter_config.json"}


def test_peft_load_state_with_optimizer_restores_optimizer_state(tmp_path) -> None:
    engine, run_id = _engine(tmp_path)
    _train_one_step(engine, run_id)
    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": run_id, "name": "resume"})
    )

    engine.load_state({"path": checkpoint.path, "optimizer": False})
    assert engine._record()["optimizer_step"] == 0

    engine.load_state({"path": checkpoint.path, "optimizer": True})
    assert engine._record()["optimizer_step"] == 1
    assert engine.optimizer is not None
    assert engine.optimizer.state_dict()["state"]


def test_peft_completion_logprobs_match_manual_logits(tmp_path) -> None:
    engine, _ = _engine(tmp_path)
    datum = _sft_datum()
    engine._ensure_model_loaded()
    items = engine._tensorize_supervised(
        __import__("toy_modal.backend.loss_inputs", fromlist=["prepare_supervised_batch_items"])
        .prepare_supervised_batch_items([datum])
    )

    engine.model.eval()
    with torch.no_grad():
        logits = engine.model(
            input_ids=items["input_ids"],
            attention_mask=items["attention_mask"],
        ).logits
    gathered, mask = token_logprobs_from_logits(logits, items["labels"])
    engine_rows = engine.completion_logprobs([datum])

    assert torch.allclose(
        torch.tensor(engine_rows[0], dtype=gathered.dtype),
        gathered[0][mask[0]].detach().cpu(),
    )


def _engine(tmp_path: Path) -> tuple[PeftTrainerEngine, str]:
    model_dir = _tiny_model_dir(tmp_path)
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "peft",
                "base_model": str(model_dir),
                "lora_config": types.LoraConfig(
                    rank=2,
                    alpha=4,
                    train_attn=True,
                    train_mlp=False,
                    train_unembed=False,
                    target_modules=["c_attn"],
                    seed=123,
                ),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )
    engine = PeftTrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    return engine, response.training_run_id


def _tiny_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "tiny-gpt2"
    if model_dir.exists():
        return model_dir

    vocab = {str(index): index for index in range(64)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="0"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="0",
        pad_token="1",
        eos_token="2",
        bos_token="2",
    )
    config = GPT2Config(
        vocab_size=64,
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(model_dir)
    fast.save_pretrained(model_dir)
    return model_dir


def _sft_datum() -> types.Datum:
    return types.Datum(
        model_input=types.ModelInput.from_ints([3, 4, 5, 6, 7]),
        loss_fn_inputs={
            "target_tokens": [5, 6, 7],
            "weights": [0.0, 0.0, 1.0, 1.0, 1.0],
        },
    )


def _payload(datum: types.Datum) -> dict:
    return datum.model_dump(mode="json")


def _train_one_step(engine: PeftTrainerEngine, run_id: str) -> None:
    datum = _sft_datum()
    engine.forward_backward(
        {
            "training_run_id": run_id,
            "data": [_payload(datum)],
            "loss_fn": "cross_entropy",
            "expected_model_seq_id": engine._record()["model_seq_id"],
        }
    )
    engine.optim_step(
        {
            "training_run_id": run_id,
            "adam_params": types.AdamParams(learning_rate=0.01).model_dump(mode="json"),
            "expected_model_seq_id": engine._record()["model_seq_id"],
        }
    )


def _manifest_path(tmp_path: Path, toy_path: str) -> Path:
    _, _, rest = toy_path.partition("://")
    project_id, run_id, artifact_type, name = rest.split("/", 3)
    return tmp_path / "runs" / project_id / run_id / artifact_type / name / "manifest.json"
