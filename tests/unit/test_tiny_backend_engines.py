from toy_modal import types
from toy_modal.backend.registry import create_run
from toy_modal.backend.sampler_worker import SamplerEngine
from toy_modal.backend.trainer_worker import TrainerEngine


def test_tiny_trainer_checkpoint_and_sampler_roundtrip(tmp_path) -> None:
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "tiny",
                "base_model": "hf-internal-testing/tiny-random-gpt2",
                "lora_config": types.LoraConfig(rank=4),
                "user_metadata": {"purpose": "test"},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )
    engine = TrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={"target_tokens": [2, 3, 4], "weights": [1, 1, 1]},
    )

    forward = types.ForwardBackwardOutput.model_validate(
        engine.forward_backward(
            {
                "training_run_id": response.training_run_id,
                "data": [datum.model_dump(mode="json")],
                "loss_fn": "cross_entropy",
                "loss_fn_config": {},
                "expected_model_seq_id": 0,
            }
        )
    )
    assert forward.gradient_id
    step = types.OptimStepResponse.model_validate(
        engine.optim_step(
            {
                "training_run_id": response.training_run_id,
                "adam_params": types.AdamParams(learning_rate=1e-4).model_dump(mode="json"),
                "expected_model_seq_id": 0,
            }
        )
    )
    assert step.optimizer_step == 1

    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": response.training_run_id, "name": "step-1"})
    )
    sampler_weights = types.SaveWeightsForSamplerResponse.model_validate(
        engine.save_weights_for_sampler({"training_run_id": response.training_run_id, "name": "sampler-1"})
    )
    assert checkpoint.path.startswith("toy-modal://")
    assert sampler_weights.path.startswith("toy-modal://")

    sampler = SamplerEngine.load(
        base_model=None,
        model_path=sampler_weights.path,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    sample = types.SampleResponse.model_validate(
        sampler.sample(
            {
                "prompt": types.ModelInput.from_ints([65]).model_dump(mode="json"),
                "num_samples": 1,
                "sampling_params": types.SamplingParams(max_tokens=3, seed=1).model_dump(mode="json"),
                "include_prompt_logprobs": True,
                "topk_prompt_logprobs": 2,
            }
        )
    )
    assert sample.samples[0].tokens
    assert sample.prompt_logprobs
    assert sample.topk_prompt_logprobs


def test_tiny_trainer_load_state_with_and_without_optimizer(tmp_path) -> None:
    registry = {}
    response = types.CreateTrainingRunResponse.model_validate(
        create_run(
            {
                "project_id": "tiny",
                "base_model": "hf-internal-testing/tiny-random-gpt2",
                "lora_config": types.LoraConfig(rank=4),
                "user_metadata": {},
            },
            registry=registry,
            run_root=str(tmp_path / "runs"),
        )
    )
    engine = TrainerEngine.load_or_initialize(
        response.training_run_id,
        registry=registry,
        model_root=str(tmp_path / "models"),
        run_root=str(tmp_path / "runs"),
    )
    engine.forward_backward(
        {
            "training_run_id": response.training_run_id,
            "data": [
                types.Datum(
                    model_input=types.ModelInput.from_ints([1, 2]),
                    loss_fn_inputs={"target_tokens": [2]},
                ).model_dump(mode="json")
            ],
            "loss_fn": "cross_entropy",
            "loss_fn_config": {},
            "expected_model_seq_id": 0,
        }
    )
    engine.optim_step(
        {
            "training_run_id": response.training_run_id,
            "adam_params": types.AdamParams(learning_rate=1e-4).model_dump(mode="json"),
            "expected_model_seq_id": 0,
        }
    )
    checkpoint = types.SaveWeightsResponse.model_validate(
        engine.save_state({"training_run_id": response.training_run_id, "name": "resume"})
    )

    engine.load_state({"path": checkpoint.path, "optimizer": False})
    assert registry[f"run:{response.training_run_id}"]["optimizer_step"] == 0
    engine.load_state({"path": checkpoint.path, "optimizer": True})
    assert registry[f"run:{response.training_run_id}"]["optimizer_step"] == 1
