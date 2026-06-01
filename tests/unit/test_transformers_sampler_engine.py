from pathlib import Path
import base64

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from toy_modal import types
from toy_modal.backend.peft_trainer import PeftTrainerEngine
from toy_modal.backend.registry import create_run
from toy_modal.backend.sampler_worker import TransformersSamplerEngine, _resolve_model_reference
from toy_modal.errors import CheckpointNotFoundError
from toy_modal.paths import build_toy_path


def test_transformers_sampler_scores_and_samples_base_model(tmp_path) -> None:
    model_dir = _tiny_model_dir(tmp_path)
    sampler = TransformersSamplerEngine.load(
        base_model=str(model_dir),
        model_path=None,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    prompt = types.ModelInput.from_ints([3, 4, 5]).model_dump(mode="json")

    response = types.SampleResponse.model_validate(
        sampler.sample(
            {
                "prompt": prompt,
                "num_samples": 2,
                "sampling_params": types.SamplingParams(
                    max_tokens=2,
                    seed=123,
                    temperature=0.7,
                    top_k=5,
                    top_p=0.9,
                ).model_dump(mode="json"),
                "include_prompt_logprobs": True,
                "topk_prompt_logprobs": 3,
            }
        )
    )
    logprobs = sampler.compute_logprobs({"prompt": prompt})

    assert len(response.samples) == 2
    assert len(response.samples[0].logprobs) == 2
    assert response.prompt_logprobs == logprobs
    assert response.topk_prompt_logprobs[0] is None
    assert len(response.topk_prompt_logprobs[1]) == 3


def test_transformers_sampler_loads_saved_peft_adapter(tmp_path) -> None:
    model_dir = _tiny_model_dir(tmp_path)
    registry = {}
    run = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "sampler",
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
    trainer = PeftTrainerEngine.load_or_initialize(
        run.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    sampler_weights = types.SaveWeightsForSamplerResponse.model_validate(
        trainer.save_weights_for_sampler({"training_run_id": run.training_run_id, "name": "adapter"})
    )
    sampler = TransformersSamplerEngine.load(
        base_model=None,
        model_path=sampler_weights.path,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )

    response = types.SampleResponse.model_validate(
        sampler.sample(
            {
                "prompt": types.ModelInput.from_ints([3, 4]).model_dump(mode="json"),
                "num_samples": 1,
                "sampling_params": types.SamplingParams(max_tokens=1, seed=1).model_dump(mode="json"),
            }
        )
    )

    assert response.samples[0].tokens


def test_sampler_reference_resolves_adapter_saved_at_artifact_root(tmp_path) -> None:
    run_root = tmp_path / "runs"
    artifact_dir = run_root / "project" / "run_1" / "sampler_weights" / "weights"
    artifact_dir.mkdir(parents=True)
    toy_path = build_toy_path("project", "run_1", "sampler_weights", "weights")
    (artifact_dir / "manifest.json").write_text(
        '{"base_model": "model", "adapter_path": "adapter"}',
        encoding="utf-8",
    )
    (artifact_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

    base_model, adapter_dir, manifest = _resolve_model_reference(
        base_model=None,
        model_path=toy_path,
        run_root=run_root,
    )

    assert base_model == "model"
    assert adapter_dir == artifact_dir
    assert manifest == {"base_model": "model", "adapter_path": "adapter"}


def test_sampler_reference_missing_adapter_error_includes_visible_entries(tmp_path) -> None:
    run_root = tmp_path / "runs"
    artifact_dir = run_root / "project" / "run_1" / "sampler_weights" / "weights"
    artifact_dir.mkdir(parents=True)
    toy_path = build_toy_path("project", "run_1", "sampler_weights", "weights")
    (artifact_dir / "manifest.json").write_text(
        '{"base_model": "model", "adapter_path": "adapter"}',
        encoding="utf-8",
    )

    with pytest.raises(CheckpointNotFoundError) as exc_info:
        _resolve_model_reference(
            base_model=None,
            model_path=toy_path,
            run_root=run_root,
        )

    message = str(exc_info.value)
    assert "adapter directory missing" in message
    assert "manifest.json" in message


def test_sampler_reference_materializes_embedded_adapter_files(tmp_path) -> None:
    run_root = tmp_path / "runs"
    artifact_dir = run_root / "project" / "run_1" / "sampler_weights" / "weights"
    artifact_dir.mkdir(parents=True)
    toy_path = build_toy_path("project", "run_1", "sampler_weights", "weights")
    (artifact_dir / "manifest.json").write_text(
        json_manifest(
            {
                "base_model": "model",
                "adapter_path": "adapter",
                "adapter_files": [
                    {
                        "path": "adapter_config.json",
                        "encoding": "base64",
                        "data": base64.b64encode(b"{}").decode("ascii"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _, adapter_dir, _ = _resolve_model_reference(
        base_model=None,
        model_path=toy_path,
        run_root=run_root,
    )

    assert adapter_dir is not None
    assert adapter_dir.is_dir()
    assert (adapter_dir / "adapter_config.json").exists()


def json_manifest(payload: dict) -> str:
    import json

    return json.dumps(payload)


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
