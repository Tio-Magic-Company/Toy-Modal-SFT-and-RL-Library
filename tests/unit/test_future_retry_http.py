import asyncio
from concurrent.futures import Future
import sys
from types import SimpleNamespace

import pytest

from toy_modal import DEFAULT_UNSLOTH_BASE_MODEL, ServiceClient, types
from toy_modal.backend import http_gateway as gateway
from toy_modal.backend.http_gateway import InMemoryHTTPJobStore, _result_response
from toy_modal.futures import APIFuture
from toy_modal.transport.base import JobRef
from toy_modal.transport.http_gateway import HTTPGatewayTransport
from toy_modal.transport.retry import RetryingTransport

from fake_modal_backend import DEFAULT_BASE_MODEL, install_fake_modal


class _ManualTransport:
    name = "manual"

    def __init__(self) -> None:
        self.future: Future[str] = Future()

    def submit(self, route, payload, *, result_type):
        return APIFuture(JobRef(job_id="manual-job", route=route, native_ref=self.future), self, result_type)

    async def submit_async(self, route, payload, *, result_type):
        return self.submit(route, payload, result_type=result_type)

    def get_result(self, job_ref, timeout=None):
        return job_ref.native_ref.result(timeout=timeout)

    async def get_result_async(self, job_ref, timeout=None):
        return await asyncio.wait_for(asyncio.wrap_future(job_ref.native_ref), timeout=timeout)

    def job_id(self, job_ref):
        return job_ref.job_id

    def done(self, job_ref):
        return job_ref.native_ref.done()

    def cancel(self, job_ref):
        return job_ref.native_ref.cancel()


class _FlakyTransport:
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, route, payload, *, result_type):
        self.calls += 1
        future: Future[str] = Future()
        if self.calls == 1:
            future.set_exception(RuntimeError("temporary"))
        else:
            future.set_result("ok")
        return APIFuture(JobRef(job_id=f"job-{self.calls}", route=route, native_ref=future), self, result_type)

    async def submit_async(self, route, payload, *, result_type):
        return self.submit(route, payload, result_type=result_type)

    def get_result(self, job_ref, timeout=None):
        return job_ref.native_ref.result(timeout=timeout)

    async def get_result_async(self, job_ref, timeout=None):
        return await asyncio.wait_for(asyncio.wrap_future(job_ref.native_ref), timeout=timeout)

    def job_id(self, job_ref):
        return job_ref.job_id

    def done(self, job_ref):
        return job_ref.native_ref.done()

    def cancel(self, job_ref):
        return job_ref.native_ref.cancel()


class _AsyncFlakyTransport:
    name = "async-flaky"

    def __init__(self) -> None:
        self.async_calls = 0

    def submit(self, route, payload, *, result_type):
        raise AssertionError("async retry path should not use sync submit")

    async def submit_async(self, route, payload, *, result_type):
        self.async_calls += 1
        future: Future[str] = Future()
        if self.async_calls == 1:
            future.set_exception(RuntimeError("temporary"))
        else:
            future.set_result("ok")
        return APIFuture(JobRef(job_id=f"async-job-{self.async_calls}", route=route, native_ref=future), self, result_type)

    def get_result(self, job_ref, timeout=None):
        raise AssertionError("async retry path should not use sync result retrieval")

    async def get_result_async(self, job_ref, timeout=None):
        return await asyncio.wait_for(asyncio.wrap_future(job_ref.native_ref), timeout=timeout)

    def job_id(self, job_ref):
        return job_ref.job_id

    def done(self, job_ref):
        return job_ref.native_ref.done()

    def cancel(self, job_ref):
        return job_ref.native_ref.cancel()


class _TokenizerTransport(_FlakyTransport):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer_kwargs = None

    def get_tokenizer(self, *, base_model=None, model_path=None):
        self.tokenizer_kwargs = {"base_model": base_model, "model_path": model_path}
        return "tokenizer"


class _GatewayTransport:
    name = "gateway-fake"

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, route, payload, *, result_type):
        self.calls += 1
        future: Future[dict] = Future()
        future.set_result({"transport": "modal-direct", "route": route, "payload": payload})
        return APIFuture(JobRef(job_id=f"gateway-{self.calls}", route=route, native_ref=future), self, result_type)

    async def submit_async(self, route, payload, *, result_type):
        return self.submit(route, payload, result_type=result_type)

    def get_result(self, job_ref, timeout=None):
        return job_ref.native_ref.result(timeout=timeout)

    async def get_result_async(self, job_ref, timeout=None):
        return await asyncio.wait_for(asyncio.wrap_future(job_ref.native_ref), timeout=timeout)

    def job_id(self, job_ref):
        return job_ref.job_id

    def done(self, job_ref):
        return job_ref.native_ref.done()

    def cancel(self, job_ref):
        return job_ref.native_ref.cancel()


class _AsyncOnlyGatewayTransport(_GatewayTransport):
    def __init__(self) -> None:
        super().__init__()
        self.async_submits = 0
        self.async_results = 0

    def submit(self, route, payload, *, result_type):
        raise AssertionError("HTTP gateway handlers should use submit_async")

    async def submit_async(self, route, payload, *, result_type):
        self.async_submits += 1
        if route == "not.a.route":
            from toy_modal.errors import BadRequestError

            raise BadRequestError("unsupported route")
        future: Future[dict] = Future()
        future.set_result({"transport": "async-gateway", "route": route, "payload": payload})
        return APIFuture(JobRef(job_id=f"async-gateway-{self.async_submits}", route=route, native_ref=future), self, result_type)

    def get_result(self, job_ref, timeout=None):
        raise AssertionError("HTTP gateway handlers should use result_async")

    async def get_result_async(self, job_ref, timeout=None):
        self.async_results += 1
        return await asyncio.wait_for(asyncio.wrap_future(job_ref.native_ref), timeout=timeout)


class _HTTPFutureTransport(HTTPGatewayTransport):
    def __init__(self) -> None:
        super().__init__(base_url="http://gateway.test")
        self.job_id_value = "http-job"
        self.result_payload = None
        self.cancelled = False

    def complete(self, payload) -> None:
        self.result_payload = payload

    def _request(self, method, path, body, *, timeout=None):
        if method == "POST" and path == "/submit":
            assert body == {"route": "server.capabilities", "payload": {"project_id": "test"}}
            return {"job_id": self.job_id_value}
        if method == "GET" and path == f"/status/{self.job_id_value}":
            return {"done": self.result_payload is not None or self.cancelled}
        if method == "POST" and path == f"/cancel/{self.job_id_value}":
            self.cancelled = True
            return {"cancelled": True}
        if method == "GET" and path.startswith(f"/retrieve/{self.job_id_value}"):
            if self.result_payload is None:
                return {"status": "pending"}
            return {"status": "done", "result": self.result_payload}
        raise AssertionError(f"unexpected request: {method} {path} {body}")


def test_api_future_timeout_cancel_future_and_await() -> None:
    transport = _ManualTransport()
    api_future = transport.submit("manual", {}, result_type=str)

    assert api_future.future() is transport.future
    assert api_future.done is False
    with pytest.raises(TimeoutError):
        api_future.result(timeout=0.001)

    transport.future.set_result("done")
    assert api_future.result(timeout=0.001) == "done"
    assert asyncio.run(api_future.result_async(timeout=0.001)) == "done"
    assert asyncio.run(_await_future(api_future)) == "done"

    cancellable = _ManualTransport().submit("manual", {}, result_type=str)
    assert cancellable.cancel() is True


async def _await_future(api_future):
    return await api_future


def test_modal_direct_future_result_async_await_and_cancel(monkeypatch) -> None:
    backend = install_fake_modal(monkeypatch)
    service = ServiceClient(project_id="test", transport="modal-direct", app_name="toy-modal-test")
    future = service._transport.submit(
        "server.capabilities",
        {"project_id": "test"},
        result_type=types.GetServerCapabilitiesResponse,
    )

    assert future.done is False
    assert asyncio.run(future.result_async(timeout=1.0)).transport == "modal-direct"
    assert future.done is True
    assert backend.async_gets == 1

    async_capabilities = asyncio.run(service.get_server_capabilities_async())
    assert async_capabilities.transport == "modal-direct"
    assert backend.async_function_spawns == 1
    assert backend.async_gets == 2

    awaited = service._transport.submit(
        "server.capabilities",
        {"project_id": "test"},
        result_type=types.GetServerCapabilitiesResponse,
    )
    assert DEFAULT_UNSLOTH_BASE_MODEL in asyncio.run(_await_future(awaited)).supported_model_names

    cancellable = service._transport.submit(
        "server.capabilities",
        {"project_id": "test"},
        result_type=types.GetServerCapabilitiesResponse,
    )
    assert cancellable.cancel() is True
    assert cancellable.done is True


def test_http_transport_future_timeout_result_async_await_and_cancel() -> None:
    transport = _HTTPFutureTransport()
    future = transport.submit("server.capabilities", {"project_id": "test"}, result_type=dict)

    assert future.done is False
    with pytest.raises(TimeoutError):
        future.result(timeout=0.001)

    transport.complete({"ok": True})
    assert future.done is True
    assert future.result(timeout=0.001) == {"ok": True}
    assert asyncio.run(future.result_async(timeout=0.001)) == {"ok": True}

    awaited = transport.submit("server.capabilities", {"project_id": "test"}, result_type=dict)
    transport.complete({"ok": "awaited"})
    assert asyncio.run(_await_future(awaited)) == {"ok": "awaited"}

    cancellable_transport = _HTTPFutureTransport()
    cancellable = cancellable_transport.submit(
        "server.capabilities",
        {"project_id": "test"},
        result_type=dict,
    )
    assert cancellable.cancel() is True
    assert cancellable_transport.cancelled is True


def test_retrying_transport_retries_result_retrieval() -> None:
    wrapped = _FlakyTransport()
    retrying = RetryingTransport(
        wrapped,
        types.RetryConfig(max_retries=1, initial_backoff_seconds=0.0, max_backoff_seconds=0.0),
    )
    future = retrying.submit("route", {}, result_type=str)

    assert future.result() == "ok"
    assert wrapped.calls == 2


def test_retrying_transport_async_path_uses_wrapped_async_submit() -> None:
    wrapped = _AsyncFlakyTransport()
    retrying = RetryingTransport(
        wrapped,
        types.RetryConfig(max_retries=1, initial_backoff_seconds=0.0, max_backoff_seconds=0.0),
    )

    async def scenario() -> str:
        future = await retrying.submit_async("route", {}, result_type=str)
        return await future.result_async()

    assert asyncio.run(scenario()) == "ok"
    assert wrapped.async_calls == 2


def test_retrying_transport_forwards_tokenizer_context() -> None:
    wrapped = _TokenizerTransport()
    retrying = RetryingTransport(wrapped, types.RetryConfig())

    tokenizer = retrying.get_tokenizer(base_model="base", model_path="toy-modal://p/r/s/name")

    assert tokenizer == "tokenizer"
    assert wrapped.tokenizer_kwargs == {
        "base_model": "base",
        "model_path": "toy-modal://p/r/s/name",
    }


def test_http_transport_rejects_inprocess_local_gateway() -> None:
    with pytest.raises(ValueError, match="HTTP\\(S\\) gateway URL"):
        ServiceClient(
            project_id="http",
            transport="http",
            base_url="inprocess://local-mock",
        )


def test_modal_direct_fake_contract(monkeypatch) -> None:
    install_fake_modal(monkeypatch)
    service = ServiceClient(project_id="http", transport="modal-direct", app_name="toy-modal-test")
    capabilities = service.get_server_capabilities()
    assert capabilities.transport == "modal-direct"
    assert capabilities.trainer_engine == "unsloth-peft"
    assert capabilities.sampler_engine == "unsloth"
    assert capabilities.backend_profile["uses_unsloth"] is True
    assert "unsloth" in capabilities.backend_profile
    assert "unsloth/tinyllama-bnb-4bit" in capabilities.supported_model_names
    assert "Qwen" in capabilities.backend_profile["unsloth"]["model_families"]

    training = service.create_lora_training_client(DEFAULT_BASE_MODEL)
    checkpoint = training.save_state("modal-checkpoint").result()
    assert checkpoint.path.startswith("toy-modal://")


def test_http_gateway_stages_large_results() -> None:
    store = InMemoryHTTPJobStore()

    class State:
        job_store = store

    class App:
        state = State()

    small = _result_response(App(), "job-1", {"ok": True})
    large = _result_response(App(), "job-2", {"payload": "x" * 600_000})

    assert small["result"] == {"ok": True}
    assert "result_ref" in large
    assert App.state.job_store.get_large_result(large["result_ref"]).result == {"payload": "x" * 600_000}


def test_http_gateway_requires_auth_by_default() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    client = fastapi_testclient.TestClient(gateway.create_app(transport=_GatewayTransport()))
    response = client.get("/capabilities")

    assert response.status_code == 401


def test_http_gateway_accepts_bearer_token_when_configured() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    client = fastapi_testclient.TestClient(
        gateway.create_app(transport=_GatewayTransport(), api_key="secret")
    )
    response = client.get("/capabilities", headers={"Authorization": "Bearer secret"})
    response.raise_for_status()

    assert response.json()["transport"] == "modal-direct"


def test_http_gateway_handlers_use_async_transport_and_stage_large_results() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    transport = _AsyncOnlyGatewayTransport()
    client = fastapi_testclient.TestClient(
        gateway.create_app(
            transport=transport,
            allow_unauthenticated=True,
            large_result_inline_bytes=1,
        )
    )

    capabilities = client.get("/capabilities")
    capabilities.raise_for_status()
    assert capabilities.json()["transport"] == "async-gateway"

    bad = client.post("/submit", json={"route": "not.a.route", "payload": {}})
    assert bad.status_code == 400
    assert bad.json()["detail"]["error_type"] == "BadRequestError"

    submit = client.post(
        "/submit",
        json={"route": "server.capabilities", "payload": {"project_id": "test"}},
    )
    submit.raise_for_status()
    job_id = submit.json()["job_id"]

    retrieve = client.get(f"/retrieve/{job_id}?timeout=1&route=server.capabilities")
    retrieve.raise_for_status()
    body = retrieve.json()
    assert "result_ref" in body

    large = client.get(f"/large-results/{body['result_ref']}")
    large.raise_for_status()
    assert large.json()["result"]["transport"] == "async-gateway"
    assert transport.async_submits >= 2
    assert transport.async_results >= 2


def test_http_gateway_submit_idempotency_key_deduplicates_jobs() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    client = fastapi_testclient.TestClient(
        gateway.create_app(transport=_GatewayTransport(), allow_unauthenticated=True)
    )
    payload = {
        "route": "server.capabilities",
        "payload": {},
        "idempotency_key": "same-request",
    }

    first = client.post("/submit", json=payload)
    second = client.post("/submit", json=payload)
    first.raise_for_status()
    second.raise_for_status()

    assert first.json()["job_id"] == second.json()["job_id"]
    assert second.json()["deduplicated"] is True


def test_http_gateway_retrieve_survives_app_recreation() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    transport = _GatewayTransport()
    store = InMemoryHTTPJobStore()

    submit_client = fastapi_testclient.TestClient(
        gateway.create_app(transport=transport, allow_unauthenticated=True, job_store=store)
    )
    submit = submit_client.post(
        "/submit",
        json={"route": "server.capabilities", "payload": {}},
    )
    submit.raise_for_status()
    job_id = submit.json()["job_id"]

    retrieve_client = fastapi_testclient.TestClient(
        gateway.create_app(transport=transport, allow_unauthenticated=True, job_store=store)
    )
    retrieve = retrieve_client.get(f"/retrieve/{job_id}")
    retrieve.raise_for_status()

    body = retrieve.json()
    assert body["status"] == "done"
    assert body["result"]["transport"] == "modal-direct"

    wrong_route = retrieve_client.get(f"/retrieve/{job_id}?route=training.forward")
    assert wrong_route.status_code == 404


def test_http_gateway_retrieve_can_rebuild_modal_function_call(monkeypatch) -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")

    class FakeFunctionCall:
        @staticmethod
        def from_id(job_id):
            assert job_id == "fc-1"

            return SimpleNamespace(
                done=lambda: True,
                cancel=lambda: True,
                get=lambda timeout=None: {"ok": True, "timeout": timeout},
            )

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(FunctionCall=FakeFunctionCall))

    client = fastapi_testclient.TestClient(
        gateway.create_app(transport=_GatewayTransport(), allow_unauthenticated=True)
    )
    response = client.get("/retrieve/fc-1?timeout=1")
    response.raise_for_status()

    assert response.json()["result"] == {"ok": True, "timeout": 1.0}
