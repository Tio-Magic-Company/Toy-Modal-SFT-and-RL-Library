import pickle

import pytest

import toy_modal as tinker
from toy_modal import types
from toy_modal.errors import StaleModelSequenceError
from toy_modal.paths import parse_toy_path

from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal


def _service() -> tinker.ServiceClient:
    return tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")


def test_modal_direct_fake_sft_workflow(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL, rank=4)
    tokenizer = training.get_tokenizer()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokenizer.encode("abc")),
        loss_fn_inputs={"target_tokens": [1], "weights": [1]},
    )

    fwdbwd = training.forward_backward([datum], "cross_entropy")
    optim = training.optim_step(types.AdamParams(learning_rate=1e-4))

    assert fwdbwd.result().gradient_id
    assert optim.result().optimizer_step == 1
    assert training.get_info().model_seq_id == 1

    sampler = training.save_weights_and_get_sampling_client("step-1")
    sample = sampler.sample(
        types.ModelInput.from_ints(tokenizer.encode("x")),
        1,
        types.SamplingParams(max_tokens=2),
    ).result()
    assert sample.sequences[0].tokens


def test_paths_and_rest_metadata(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    checkpoint = training.save_state("initial").result()

    parsed = parse_toy_path(checkpoint.path)
    assert parsed.project_id == "test"
    assert parsed.artifact_type == "checkpoints"

    rest = service.create_rest_client()
    run = rest.get_training_run_by_toy_path(checkpoint.path).result()
    assert run.training_run_id == training.training_run_id
    checkpoints = rest.list_checkpoints(training.training_run_id).result().checkpoints
    assert checkpoints
    assert checkpoints[0].tinker_path == checkpoint.path
    assert rest.get_weights_info_by_toy_path(checkpoint.path).result().base_model == DEFAULT_BASE_MODEL
    assert rest.list_user_checkpoints().result().checkpoints


def test_checkpoint_resume_optimizer_semantics(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    training = service.create_lora_training_client(
        DEFAULT_BASE_MODEL,
        rank=8,
        seed=123,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
    )
    tokenizer = training.get_tokenizer()
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokenizer.encode("abc")),
        loss_fn_inputs={"target_tokens": [1], "weights": [1]},
    )
    training.forward_backward([datum], "cross_entropy").result()
    training.optim_step(types.AdamParams(learning_rate=1e-4)).result()
    checkpoint = training.save_state("after-step").result()

    training.load_state(checkpoint.path).result()
    assert training.optimizer_step == 0
    training.load_state_with_optimizer(checkpoint.path).result()
    assert training.optimizer_step == 1

    weights_only = service.create_training_client_from_state(checkpoint.path)
    with_optimizer = service.create_training_client_from_state_with_optimizer(checkpoint.path)

    assert weights_only.training_run_id != training.training_run_id
    assert weights_only.model_seq_id == training.model_seq_id
    assert weights_only.optimizer_step == 0
    assert weights_only.lora_config == training.lora_config
    assert with_optimizer.training_run_id != training.training_run_id
    assert with_optimizer.model_seq_id == training.model_seq_id
    assert with_optimizer.optimizer_step == training.optimizer_step
    assert with_optimizer.lora_config == training.lora_config


def test_stale_old_logprobs_model_seq_id_conflicts(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL, rank=8)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": [3],
            "old_logprobs": [-1.0],
            "advantages": [1.0],
            "old_logprobs_model_seq_id": 1,
        },
    )

    with pytest.raises(StaleModelSequenceError):
        training.forward_backward([datum], "importance_sampling").result()

    probe = training._transport.submit(
        "training.validate_old_logprobs_sequence",
        {
            "training_run_id": training.training_run_id,
            "data": [datum.model_dump(mode="json")],
            "loss_fn": "importance_sampling",
        },
        result_type=dict,
    ).result()
    assert probe["accepted"] is False
    assert probe["error_type"] == "StaleModelSequenceError"


def test_tinker_path_aliases_when_enabled(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(
        project_id="test",
        transport="modal-direct",
        app_name="toy-modal-test",
        accept_tinker_paths=True,
    )
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    checkpoint = training.save_state("initial").result()
    tinker_path = checkpoint.path.replace("toy-modal://", "tinker://", 1)

    run = service.create_rest_client().get_training_run_by_tinker_path(tinker_path).result()
    assert run.training_run_id == training.training_run_id


def test_sampling_client_pickles_without_transport_handles(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    sampler = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL)
    rebuild, args = sampler.__reduce__()
    state = args[0]

    restored = pickle.loads(pickle.dumps(sampler))
    assert rebuild.__name__ == "_rebuild_sampling_client"
    assert not hasattr(state, "_transport")
    assert not hasattr(state, "_sampling_client_sidecar_handle")
    assert all(not hasattr(value, "submit") for value in state.client_config.values())
    assert restored.get_base_model() == DEFAULT_BASE_MODEL
    assert restored._client_config["transport"] == "modal-direct"


def test_sampling_client_from_training_preserves_connection_config_when_pickled(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="project-a", transport="modal-direct", app_name="toy-modal-test")
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    sampler = training.create_sampling_client(model_path="toy-modal://project-a/run/sampler_weights/latest")

    restored = pickle.loads(pickle.dumps(sampler))

    assert restored._client_config["project_id"] == "project-a"
    assert restored._client_config["transport"] == "modal-direct"
    assert restored._client_config["app_name"] == "toy-modal-test"


def test_sample_response_samples_alias(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = _service()
    sampler = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL)
    tokenizer = sampler.get_tokenizer()
    response = sampler.sample(
        types.ModelInput.from_ints(tokenizer.encode("x")),
        1,
        types.SamplingParams(max_tokens=1),
    ).result()

    assert response.samples == response.sequences
