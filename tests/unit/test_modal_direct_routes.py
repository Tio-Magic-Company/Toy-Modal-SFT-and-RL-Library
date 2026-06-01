import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from toy_modal.transport.modal_direct import ModalDirectTransport


def test_modal_direct_routes_rest_and_state_creation_to_functions(monkeypatch) -> None:
    fake_modal = _FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake_modal.module)
    transport = ModalDirectTransport(app_name="toy-modal-test", environment_name="dev")

    rest = transport.submit("rest.list_training_runs", {"limit": 1}, result_type=object)
    assert rest.result() == {"ok": "metadata_route"}
    assert fake_modal.function_calls[-1] == (
        "toy-modal-test",
        "metadata_route",
        "dev",
        {"route": "rest.list_training_runs", "payload": {"limit": 1}},
    )

    load = transport.submit("training.load_state", {"path": "toy-modal://p/r/checkpoints/c"}, result_type=object)
    assert load.result() == {"ok": "create_training_run_from_state"}
    assert fake_modal.function_calls[-1][1] == "create_training_run_from_state"

    tokenizer = transport.submit(
        "tokenizer.encode",
        {"base_model": "model", "text": "hello"},
        result_type=object,
    )
    assert tokenizer.result() == {"ok": "tokenizer_route"}
    assert fake_modal.function_calls[-1][3]["operation"] == "encode"


def test_modal_direct_routes_existing_training_calls_to_worker(monkeypatch) -> None:
    fake_modal = _FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake_modal.module)
    transport = ModalDirectTransport(app_name="toy-modal-test")

    future = transport.submit(
        "training.optim_step",
        {"training_run_id": "run_1", "adam_params": {"learning_rate": 1e-4}},
        result_type=object,
    )

    assert future.result() == {"ok": "optim_step"}
    assert fake_modal.class_calls[-1] == (
        "toy-modal-test",
        "TrainerWorker",
        None,
        {"run_id": "run_1"},
        "optim_step",
    )

    probe = transport.submit(
        "training.validate_old_logprobs_sequence",
        {"training_run_id": "run_1", "data": []},
        result_type=object,
    )

    assert probe.result() == {"ok": "validate_old_logprobs_sequence"}
    assert fake_modal.class_calls[-1] == (
        "toy-modal-test",
        "TrainerWorker",
        None,
        {"run_id": "run_1"},
        "validate_old_logprobs_sequence",
    )


def test_modal_direct_routes_sampler_uses_supported_modal_parameter_values(monkeypatch) -> None:
    fake_modal = _FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake_modal.module)
    transport = ModalDirectTransport(app_name="toy-modal-test")

    future = transport.submit(
        "sampling.sample",
        {"model_path": "toy-modal://p/r/sampler_weights/w", "base_model": None},
        result_type=object,
    )

    assert future.result() == {"ok": "sample"}
    assert fake_modal.class_calls[-1] == (
        "toy-modal-test",
        "SamplerWorker",
        None,
        {"model_path": "toy-modal://p/r/sampler_weights/w", "base_model": ""},
        "sample",
    )


def test_modal_direct_async_routes_use_modal_aio_interfaces(monkeypatch) -> None:
    fake_modal = _FakeModal()
    monkeypatch.setitem(sys.modules, "modal", fake_modal.module)
    transport = ModalDirectTransport(app_name="toy-modal-test", environment_name="dev")

    async def run():
        rest = await transport.submit_async(
            "rest.list_training_runs",
            {"limit": 1},
            result_type=object,
        )
        trainer = await transport.submit_async(
            "training.forward",
            {"training_run_id": "run_1", "data": []},
            result_type=object,
        )
        sampler = await transport.submit_async(
            "sampling.compute_logprobs",
            {"model_path": "toy-modal://p/r/sampler_weights/w"},
            result_type=object,
        )
        return (
            await rest.result_async(),
            await trainer.result_async(),
            await sampler.result_async(),
        )

    assert asyncio.run(run()) == (
        {"ok": "metadata_route"},
        {"ok": "forward"},
        {"ok": "compute_logprobs"},
    )
    assert fake_modal.async_function_spawns == 1
    assert fake_modal.async_method_spawns == 2
    assert fake_modal.async_gets == 3


def test_modal_app_declares_volume_reload_and_commit_boundaries() -> None:
    app_source = (Path(__file__).parents[2] / "src" / "toy_modal" / "backend" / "app.py").read_text()

    for method_name in [
        "forward_backward",
        "optim_step",
        "save_state",
        "load_state",
        "save_weights_for_sampler",
        "validate_old_logprobs_sequence",
    ]:
        assert f"def {method_name}(self, payload: dict) -> dict:" in app_source
    assert app_source.count("run_volume.reload()") >= 8
    assert app_source.count("run_volume.commit()") >= 6
    assert "modal.FunctionCall.from_id(dependency).get()" in app_source
    assert "modal.parameter(default=None)" not in app_source
    assert "base_model=self.base_model or None" in app_source
    assert '"TOY_MODAL_TRAINER_ENGINE": config.trainer_engine' in app_source
    assert '"TOY_MODAL_SAMPLER_ENGINE": config.sampler_engine' in app_source
    assert '"TOY_MODAL_PREFETCH_GPU"] = config.resolved_prefetch_gpu' in app_source
    assert '"TOY_MODAL_SUPPORTED_MODELS"] = " ".join(config.supported_models)' in app_source
    assert 'prefetch_function_options["gpu"] = config.resolved_prefetch_gpu' in app_source
    assert "image_packages.extend(config.unsloth_pip_packages)" in app_source
    assert '"TOY_MODAL_UNSLOTH_PACKAGE": config.unsloth_package' in app_source
    assert '"trainer_engine": config.trainer_engine' in app_source
    assert '"sampler_engine": config.sampler_engine' in app_source
    assert '"pip_packages": list(config.unsloth_pip_packages)' in app_source
    assert '"supported_models": payload.get("configured_models") or list(config.resolved_supported_models)' in app_source
    assert '"model_families": list(DEFAULT_UNSLOTH_MODEL_FAMILIES)' in app_source
    assert app_source.count("env=runtime_env") >= 7
    assert "ModalDirectTransport(app_name=config.app_name)" in app_source


class _AsyncCallable:
    def __init__(self, callback) -> None:
        self._callback = callback

    def __call__(self, *args, **kwargs):
        return self._callback(*args, **kwargs)

    async def aio(self, *args, **kwargs):
        return self._callback(*args, async_call=True, **kwargs)


class _FakeCall:
    def __init__(self, result, object_id="call_1") -> None:
        self.object_id = object_id
        self._result = result
        self.get = _AsyncCallable(self._get)

    def _get(self, timeout=None, *, async_call=False):
        if async_call:
            _FakeModal.active.async_gets += 1
        return self._result

    def done(self):
        return True

    def cancel(self):
        return None


class _FakeFunctionHandle:
    def __init__(self, fake_modal, app_name, function_name, environment_name) -> None:
        self.fake_modal = fake_modal
        self.app_name = app_name
        self.function_name = function_name
        self.environment_name = environment_name
        self.spawn = _AsyncCallable(self._spawn)

    def _spawn(self, payload, *, async_call=False):
        if async_call:
            self.fake_modal.async_function_spawns += 1
        self.fake_modal.function_calls.append(
            (self.app_name, self.function_name, self.environment_name, payload)
        )
        return _FakeCall({"ok": self.function_name})


class _FakeWorker:
    def __init__(self, fake_modal, app_name, class_name, environment_name, params) -> None:
        self.fake_modal = fake_modal
        self.app_name = app_name
        self.class_name = class_name
        self.environment_name = environment_name
        self.params = params

    def __getattr__(self, name):
        fake_modal = self.fake_modal
        app_name = self.app_name
        class_name = self.class_name
        environment_name = self.environment_name
        params = self.params

        class _Method:
            def __init__(self) -> None:
                self.spawn = _AsyncCallable(self._spawn)

            def _spawn(self, payload, *, async_call=False):
                if async_call:
                    fake_modal.async_method_spawns += 1
                fake_modal.class_calls.append((app_name, class_name, environment_name, params, name))
                return _FakeCall({"ok": name})

        return _Method()


class _FakeClassHandle:
    def __init__(self, fake_modal, app_name, class_name, environment_name) -> None:
        self.fake_modal = fake_modal
        self.app_name = app_name
        self.class_name = class_name
        self.environment_name = environment_name

    def __call__(self, **params):
        return _FakeWorker(
            self.fake_modal,
            self.app_name,
            self.class_name,
            self.environment_name,
            params,
        )


class _FakeModal:
    def __init__(self) -> None:
        _FakeModal.active = self
        self.function_calls = []
        self.class_calls = []
        self.async_function_spawns = 0
        self.async_method_spawns = 0
        self.async_gets = 0

        fake_modal = self

        class Function:
            @staticmethod
            def from_name(app_name, function_name, environment_name=None):
                return _FakeFunctionHandle(fake_modal, app_name, function_name, environment_name)

        class Cls:
            @staticmethod
            def from_name(app_name, class_name, environment_name=None):
                return _FakeClassHandle(fake_modal, app_name, class_name, environment_name)

        self.module = SimpleNamespace(Function=Function, Cls=Cls)
