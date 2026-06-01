import asyncio
import inspect
import importlib.util
import json
from pathlib import Path

import pytest

import toy_modal as tinker
from toy_modal import types
from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal
from toy_modal.errors import APIError, TinkerError, ToyModalError


FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "api_parity_fixture.json"


def test_public_exports_and_exception_aliases() -> None:
    for name in [
        "APIFuture",
        "RestClient",
        "SamplingClient",
        "ServiceClient",
        "TrainingClient",
        "TinkerError",
        "APIError",
        "BadRequestError",
        "NotFoundError",
        "RequestFailedError",
    ]:
        assert hasattr(tinker, name)

    assert tinker.TinkerError is ToyModalError
    assert issubclass(APIError, TinkerError)


def test_documented_client_methods_are_present() -> None:
    expected = {
        tinker.ServiceClient: {
            "get_server_capabilities",
            "get_server_capabilities_async",
            "create_lora_training_client",
            "create_lora_training_client_async",
            "create_training_client_from_state",
            "create_training_client_from_state_async",
            "create_training_client_from_state_with_optimizer",
            "create_training_client_from_state_with_optimizer_async",
            "create_sampling_client",
            "create_sampling_client_async",
            "create_rest_client",
        },
        tinker.TrainingClient: {
            "forward",
            "forward_async",
            "forward_backward",
            "forward_backward_async",
            "forward_backward_custom",
            "forward_backward_custom_async",
            "optim_step",
            "optim_step_async",
            "save_state",
            "save_state_async",
            "load_state",
            "load_state_async",
            "load_state_with_optimizer",
            "load_state_with_optimizer_async",
            "save_weights_for_sampler",
            "save_weights_for_sampler_async",
            "get_info",
            "get_info_async",
            "get_tokenizer",
            "create_sampling_client",
            "create_sampling_client_async",
            "save_weights_and_get_sampling_client",
            "save_weights_and_get_sampling_client_async",
        },
        tinker.SamplingClient: {
            "sample",
            "sample_async",
            "compute_logprobs",
            "compute_logprobs_async",
            "get_tokenizer",
            "get_base_model",
            "get_base_model_async",
            "__reduce__",
        },
        tinker.RestClient: {
            "get_training_run",
            "get_training_run_async",
            "get_training_run_by_tinker_path",
            "get_training_run_by_tinker_path_async",
            "get_weights_info_by_tinker_path",
            "list_training_runs",
            "list_training_runs_async",
            "list_checkpoints",
            "list_checkpoints_async",
            "get_checkpoint_archive_url",
            "get_checkpoint_archive_url_async",
            "delete_checkpoint",
            "delete_checkpoint_async",
            "delete_checkpoint_from_tinker_path",
            "delete_checkpoint_from_tinker_path_async",
            "get_checkpoint_archive_url_from_tinker_path",
            "get_checkpoint_archive_url_from_tinker_path_async",
            "publish_checkpoint_from_tinker_path",
            "publish_checkpoint_from_tinker_path_async",
            "unpublish_checkpoint_from_tinker_path",
            "unpublish_checkpoint_from_tinker_path_async",
            "set_checkpoint_ttl_from_tinker_path",
            "set_checkpoint_ttl_from_tinker_path_async",
            "list_user_checkpoints",
            "list_user_checkpoints_async",
            "get_session",
            "get_session_async",
            "list_sessions",
            "list_sessions_async",
            "get_sampler",
            "get_sampler_async",
        },
    }

    for cls, method_names in expected.items():
        missing = {name for name in method_names if not hasattr(cls, name)}
        assert not missing, f"{cls.__name__} missing {sorted(missing)}"


def test_generated_api_fixture_matches_public_surface() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    class_map = {
        "ServiceClient": tinker.ServiceClient,
        "TrainingClient": tinker.TrainingClient,
        "SamplingClient": tinker.SamplingClient,
        "RestClient": tinker.RestClient,
    }

    for exported_name in fixture["exports"]:
        assert hasattr(tinker, exported_name)

    for class_name, class_spec in fixture["classes"].items():
        cls = class_map[class_name]
        for method_name, expected_params in class_spec["methods"].items():
            method = getattr(cls, method_name)
            actual_params = [
                name
                for name in inspect.signature(method).parameters
                if name != "self"
            ]
            assert actual_params == expected_params, f"{class_name}.{method_name}"

    for type_name, expected_fields in fixture["types"].items():
        model = getattr(types, type_name)
        assert list(model.model_fields) == expected_fields


def test_api_fixture_generator_matches_checked_in_fixture() -> None:
    generator_path = FIXTURE_PATH.with_name("generate_api_parity_fixture.py")
    spec = importlib.util.spec_from_file_location("generate_api_parity_fixture", generator_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.build_fixture() == json.loads(FIXTURE_PATH.read_text())
    assert module.dump_fixture(module.build_fixture()) == FIXTURE_PATH.read_text()


def test_documented_async_signatures_use_named_parameters() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text())
    class_map = {
        "ServiceClient": tinker.ServiceClient,
        "TrainingClient": tinker.TrainingClient,
        "SamplingClient": tinker.SamplingClient,
        "RestClient": tinker.RestClient,
    }

    for class_name, class_spec in fixture["classes"].items():
        for method_name in class_spec["methods"]:
            if not method_name.endswith("_async"):
                continue
            signature = inspect.signature(getattr(class_map[class_name], method_name))
            varargs = {
                parameter.name
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }
            }
            assert not varargs, f"{class_name}.{method_name} exposes generic {sorted(varargs)}"


def test_docs_use_toy_modal_alias_without_top_level_tinker_package() -> None:
    repo_root = Path(__file__).parents[2]
    assert not (repo_root / "src" / "tinker").exists()

    doc_roots = [
        repo_root / "docs" / "examples",
        repo_root / "docs" / "recipes",
        repo_root / "docs" / "tutorials" / "notebooks",
    ]
    for root in doc_roots:
        for path in root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "import tinker" not in source, str(path)
            assert "from tinker" not in source, str(path)

    tutorial_parity = (repo_root / "docs" / "examples" / "tutorial_parity.py").read_text(
        encoding="utf-8"
    )
    assert "import toy_modal as tinker" in tutorial_parity


def test_documented_type_models_and_fields_are_present() -> None:
    expected_type_names = [
        "LoadWeightsResponse",
        "WeightsInfoResponse",
        "LoadWeightsRequest",
        "CreateModelRequest",
        "UnhandledExceptionEvent",
        "Datum",
        "Checkpoint",
        "ParsedCheckpointTinkerPath",
        "SamplingParams",
        "SaveWeightsForSamplerRequest",
        "ModelInput",
        "SessionEndEvent",
        "CreateSamplingSessionResponse",
        "CheckpointsListResponse",
        "SampleResponse",
        "FutureRetrieveRequest",
        "ForwardBackwardOutput",
        "ModelData",
        "GetInfoResponse",
        "SaveWeightsResponse",
        "LoraConfig",
        "SaveWeightsForSamplerResponseInternal",
        "SaveWeightsForSamplerResponse",
        "CreateSamplingSessionRequest",
        "OptimStepResponse",
        "SampleRequest",
        "TrainingRun",
        "TelemetrySendRequest",
        "CheckpointArchiveUrlResponse",
        "SupportedModel",
        "GetServerCapabilitiesResponse",
        "SessionStartEvent",
        "GenericEvent",
        "TryAgainResponse",
        "TrainingRunsResponse",
        "ForwardBackwardInput",
        "ImageAssetPointerChunk",
        "TelemetryBatch",
        "TensorData",
        "EncodedTextChunk",
        "AdamParams",
        "ImageChunk",
        "SampledSequence",
        "Cursor",
        "SaveWeightsRequest",
        "GetSessionResponse",
        "ListSessionsResponse",
        "GetSamplerResponse",
    ]
    missing = [name for name in expected_type_names if not hasattr(types, name)]
    assert not missing

    assert "grad_clip_norm" in types.AdamParams.model_fields
    assert types.AdamParams(learning_rate=1e-4).weight_decay == 0.0
    assert types.AdamParams(learning_rate=1e-4).grad_clip_norm == 0.0

    capabilities = types.GetServerCapabilitiesResponse(supported_models=[DEFAULT_BASE_MODEL])
    assert capabilities.supported_models[0].model_name == DEFAULT_BASE_MODEL
    assert capabilities.supported_model_names == [DEFAULT_BASE_MODEL]
    assert capabilities.trainer_engine is None
    assert capabilities.sampler_engine is None
    assert capabilities.backend_profile == {}

    output = types.ForwardBackwardOutput(
        loss_fn_output_type="ArrayRecord",
        loss_fn_outputs={"loss": types.TensorData(data=[1.0])},
        metrics={"loss": 1.0},
    )
    assert output.loss_fn_outputs["loss"].shape == (1,)

    checkpoint = types.Checkpoint(
        checkpoint_id="step",
        checkpoint_type="training",
        tinker_path="toy-modal://default/run/checkpoints/step",
    )
    assert checkpoint.toy_path == checkpoint.tinker_path

    archive = types.CheckpointArchiveUrlResponse(url="modal-volume://archive", expires=1)
    assert archive.expires_at is not None


def test_tinker_paths_require_explicit_compatibility_opt_in() -> None:
    from toy_modal.paths import build_toy_path, parse_toy_path

    toy_path = build_toy_path("project", "run", "checkpoints", "step")
    tinker_path = toy_path.replace("toy-modal://", "tinker://", 1)

    assert toy_path == "toy-modal://project/run/checkpoints/step"
    with pytest.raises(ValueError, match="Unsupported path scheme"):
        parse_toy_path(tinker_path)

    parsed = parse_toy_path(tinker_path, accept_tinker_paths=True)
    assert parsed.scheme == "toy-modal"
    assert parsed.project_id == "project"
    assert parsed.artifact_type == "checkpoints"


def test_stage0_default_safety_invariants(monkeypatch) -> None:
    from toy_modal.backend.config import load_config

    monkeypatch.delenv("TOY_MODAL_ALLOW_UNSAFE_CUSTOM_LOSS", raising=False)
    assert load_config().allow_unsafe_custom_loss is False
    assert types.AdamParams(learning_rate=1e-4).weight_decay == 0.0

    backend_dir = Path(__file__).parents[2] / "src" / "toy_modal" / "backend"
    backend_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in backend_dir.glob("*.py")
        if path.name != "__init__.py"
    )
    assert "modal.Queue" not in backend_sources
    assert "Queue.from_name" not in backend_sources


def test_image_chunks_and_tensor_conversion() -> None:
    image = types.ImageChunk(data=b"image-bytes", format="png", expected_tokens=256)
    dumped = image.model_dump(mode="json")
    assert dumped["data"] == "aW1hZ2UtYnl0ZXM="
    assert types.ImageChunk.model_validate(dumped).data == b"image-bytes"

    pointer = types.ImageAssetPointerChunk(uri="s3://bucket/image.png", format="png")
    assert pointer.location == "s3://bucket/image.png"
    assert pointer.uri == pointer.location

    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1]),
        loss_fn_inputs={"targets": types.TensorData(data=[1, 2, 3])},
    )
    assert datum.loss_fn_inputs["targets"].shape == (3,)


def test_modal_direct_rest_path_helpers_sessions_and_samplers(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    rest = service.create_rest_client()

    checkpoint = training.save_state("initial").result()
    assert rest.get_checkpoint_archive_url_from_toy_path(checkpoint.path).result().url

    rest.publish_checkpoint_from_toy_path(checkpoint.path).result()
    published = rest.list_checkpoints(training.training_run_id).result().checkpoints[0]
    assert published.public is True

    rest.set_checkpoint_ttl_from_toy_path(checkpoint.path, 60).result()
    ttl_checkpoint = rest.list_checkpoints(training.training_run_id).result().checkpoints[0]
    assert ttl_checkpoint.expires_at is not None

    rest.unpublish_checkpoint_from_toy_path(checkpoint.path).result()
    unpublished = rest.list_checkpoints(training.training_run_id).result().checkpoints[0]
    assert unpublished.public is False

    sampler_weights = training.save_weights_for_sampler("sampler").result()
    sessions = rest.list_sessions().result()
    assert sessions.sessions
    session = rest.get_session(sessions.sessions[0]).result()
    assert training.training_run_id in session.training_run_ids
    assert session.sampler_ids
    sampler = rest.get_sampler(session.sampler_ids[0]).result()
    assert sampler.model_path == sampler_weights.path

    rest.delete_checkpoint_from_toy_path(checkpoint.path).result()
    remaining = rest.list_checkpoints(training.training_run_id).result().checkpoints
    assert all(item.checkpoint_id != "initial" for item in remaining)


def test_async_methods_return_documented_shapes(monkeypatch) -> None:
    backend = install_fake_modal(monkeypatch)

    async def scenario() -> None:
        service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
        training = await service.create_lora_training_client_async(DEFAULT_BASE_MODEL)
        info = await training.get_info_async()
        assert info.model_data.model_name == DEFAULT_BASE_MODEL

        save_future = await training.save_state_async("async")
        checkpoint = await save_future.result_async()
        assert checkpoint.path.startswith("toy-modal://")

        rest = service.create_rest_client()
        run = await rest.get_training_run_by_toy_path_async(checkpoint.path)
        assert run.training_run_id == training.training_run_id

        sampler = await training.save_weights_and_get_sampling_client_async("async-sampler")
        response = await sampler.sample_async(
            types.ModelInput.from_ints([1]),
            1,
            types.SamplingParams(max_tokens=1),
        )
        assert response.samples

    asyncio.run(scenario())
    assert backend.async_function_spawns == 2
    assert backend.async_method_spawns == 3
    assert backend.async_gets == 5


def test_sampling_logprob_shapes(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    sampler = service.create_sampling_client(base_model=DEFAULT_BASE_MODEL)
    response = sampler.sample(
        types.ModelInput.from_ints([10, 11, 12]),
        2,
        types.SamplingParams(max_tokens=3, seed=123),
        include_prompt_logprobs=True,
        topk_prompt_logprobs=2,
    ).result()

    assert len(response.samples) == 2
    assert len(response.samples[0].logprobs) == 3
    assert response.prompt_logprobs == [None, -1.0, -1.0]
    assert response.topk_prompt_logprobs[0] is None
    assert len(response.topk_prompt_logprobs[1]) == 2


def test_unsupported_custom_loss_fails_clearly(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = tinker.ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)

    with pytest.raises(NotImplementedError, match="security design"):
        training.forward_backward_custom([], lambda data, logprobs: (0.0, {}))


def test_documented_async_methods_are_coroutines() -> None:
    assert inspect.iscoroutinefunction(tinker.ServiceClient.get_server_capabilities_async)
    assert inspect.iscoroutinefunction(tinker.TrainingClient.save_state_async)
    assert inspect.iscoroutinefunction(tinker.SamplingClient.sample_async)
    assert inspect.iscoroutinefunction(tinker.RestClient.get_sampler_async)
