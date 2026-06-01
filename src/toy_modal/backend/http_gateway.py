"""FastAPI gateway exposing the transport route contract over HTTP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import asyncio
import uuid

from toy_modal.errors import (
    AuthenticationError,
    BadRequestError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RequestFailedError,
    ToyModalError,
    UnprocessableEntityError,
)
from toy_modal.backend.config import load_config
from toy_modal.transport.modal_direct import ModalDirectTransport


LARGE_RESULT_INLINE_BYTES = 512_000
DEFAULT_LARGE_RESULT_TTL_SECONDS = 15 * 60
_DEFAULT_TRANSPORT = None
_DEFAULT_JOB_STORE = None


@dataclass
class StoredJob:
    future: Any
    route: str
    owner: str | None = None


@dataclass
class StoredLargeResult:
    result: Any
    job_id: str
    route: str
    expires_at: datetime


class InMemoryHTTPJobStore:
    """Process-local HTTP job store with explicit TTL semantics.

    Modal FunctionCall IDs remain recoverable through Modal itself. This store is
    responsible for gateway-submitted futures and staged large responses.
    """

    def __init__(self, *, large_result_ttl_seconds: int = DEFAULT_LARGE_RESULT_TTL_SECONDS) -> None:
        self.large_result_ttl_seconds = large_result_ttl_seconds
        self.jobs: dict[str, StoredJob] = {}
        self.large_results: dict[str, StoredLargeResult] = {}
        self.idempotency_keys: dict[tuple[str, str], str] = {}

    def put_job(self, job_id: str, future: Any, *, route: str, owner: str | None = None) -> None:
        self.jobs[job_id] = StoredJob(future=future, route=route, owner=owner)

    def get_job(self, job_id: str) -> StoredJob | None:
        return self.jobs.get(job_id)

    def remember_idempotency_key(self, route: str, key: str, job_id: str) -> None:
        self.idempotency_keys[(route, key)] = job_id

    def job_id_for_idempotency_key(self, route: str, key: str) -> str | None:
        return self.idempotency_keys.get((route, key))

    def put_large_result(self, *, job_id: str, route: str, result: Any) -> str:
        self.cleanup()
        result_id = uuid.uuid4().hex
        self.large_results[result_id] = StoredLargeResult(
            result=result,
            job_id=job_id,
            route=route,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self.large_result_ttl_seconds),
        )
        return result_id

    def get_large_result(self, result_id: str) -> StoredLargeResult | None:
        self.cleanup()
        return self.large_results.get(result_id)

    def cleanup(self) -> None:
        now = datetime.now(timezone.utc)
        for result_id, result in list(self.large_results.items()):
            if result.expires_at <= now:
                self.large_results.pop(result_id, None)

    def clear(self) -> None:
        self.jobs.clear()
        self.large_results.clear()
        self.idempotency_keys.clear()


def create_app(
    *,
    transport=None,
    api_key: str | None = None,
    allow_unauthenticated: bool | None = None,
    job_store: InMemoryHTTPJobStore | None = None,
    large_result_inline_bytes: int | None = None,
):
    import os

    try:
        from fastapi import FastAPI, Header, HTTPException
    except ImportError as exc:
        raise RuntimeError("HTTP gateway requires fastapi") from exc

    app = FastAPI(title="toy_modal HTTP gateway")
    app.state.transport = transport or _default_transport()
    app.state.job_store = job_store or _default_job_store()
    app.state.api_key = api_key if api_key is not None else os.getenv("TOY_MODAL_HTTP_API_KEY") or None
    app.state.allow_unauthenticated = (
        os.getenv("TOY_MODAL_HTTP_ALLOW_UNAUTHENTICATED", "0") == "1"
        if allow_unauthenticated is None
        else allow_unauthenticated
    )
    app.state.large_result_inline_bytes = (
        int(os.getenv("TOY_MODAL_HTTP_LARGE_RESULT_INLINE_BYTES", str(LARGE_RESULT_INLINE_BYTES)))
        if large_result_inline_bytes is None
        else int(large_result_inline_bytes)
    )

    def authorize(authorization: str | None) -> None:
        if app.state.api_key is None and app.state.allow_unauthenticated:
            return
        if app.state.api_key is None:
            raise HTTPException(status_code=401, detail="authorization required")
        expected = f"Bearer {app.state.api_key}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="invalid authorization")

    async def submit_route(
        route: str,
        payload: dict[str, Any],
        owner: str | None = None,
        idempotency_key: str | None = None,
    ):
        if idempotency_key:
            existing_job_id = app.state.job_store.job_id_for_idempotency_key(route, idempotency_key)
            if existing_job_id and _future_for_job(app, existing_job_id) is not None:
                return {"job_id": existing_job_id, "route": route, "deduplicated": True}
        future = await app.state.transport.submit_async(route, payload, result_type=object)
        app.state.job_store.put_job(future.job_id, future, route=route, owner=owner)
        if idempotency_key:
            app.state.job_store.remember_idempotency_key(route, idempotency_key, future.job_id)
        return {"job_id": future.job_id, "route": route}

    @app.post("/submit")
    async def submit(body: dict[str, Any], authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return await submit_route(
                body["route"],
                body.get("payload") or {},
                idempotency_key=body.get("idempotency_key"),
            )
        except Exception as exc:
            raise _http_exception(exc) from exc

    @app.get("/capabilities")
    async def capabilities(authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            future = await app.state.transport.submit_async("server.capabilities", {}, result_type=object)
            return await future.result_async()
        except Exception as exc:
            raise _http_exception(exc) from exc

    @app.post("/tokenizer/{operation}")
    async def tokenizer(operation: str, body: dict[str, Any], authorization: str | None = Header(default=None)):
        authorize(authorization)
        try:
            return await submit_route(f"tokenizer.{operation}", body)
        except Exception as exc:
            raise _http_exception(exc) from exc

    @app.post("/metadata/{route_name}")
    async def metadata(route_name: str, body: dict[str, Any], authorization: str | None = Header(default=None)):
        authorize(authorization)
        route = "rest." + route_name.replace("-", "_")
        try:
            return await submit_route(route, body)
        except Exception as exc:
            raise _http_exception(exc) from exc

    @app.get("/status/{job_id}")
    async def status(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        future = _future_for_job(app, job_id)
        if future is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, "done": await _future_done_async(future)}

    @app.post("/cancel/{job_id}")
    async def cancel(job_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        future = _future_for_job(app, job_id)
        if future is None:
            raise HTTPException(status_code=404, detail="job not found")
        return {"job_id": job_id, "cancelled": await _future_cancel_async(future)}

    @app.get("/retrieve/{job_id}")
    async def retrieve(
        job_id: str,
        timeout: float | None = None,
        route: str | None = None,
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        stored = app.state.job_store.get_job(job_id)
        future = stored.future if stored is not None else _modal_function_call(job_id)
        stored_route = stored.route if stored is not None else "<modal-function-call>"
        if future is None:
            raise HTTPException(status_code=404, detail="job not found")
        if route is not None and stored is not None and route != stored.route:
            raise HTTPException(status_code=404, detail="job not found for route")
        if not await _future_done_async(future) and timeout == 0:
            return {"job_id": job_id, "status": "pending"}
        try:
            result = await _future_result_async(future, timeout=timeout)
            return _result_response(app, job_id, result, route=stored_route)
        except TimeoutError:
            return {"job_id": job_id, "status": "pending"}
        except Exception as exc:
            raise _http_exception(exc) from exc

    @app.get("/large-results/{result_id}")
    async def large_result(result_id: str, authorization: str | None = Header(default=None)):
        authorize(authorization)
        result = app.state.job_store.get_large_result(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="large result not found or expired")
        return {
            "job_id": result.job_id,
            "route": result.route,
            "expires_at": result.expires_at.isoformat(),
            "result": result.result,
        }

    return app


def _default_transport():
    global _DEFAULT_TRANSPORT
    if _DEFAULT_TRANSPORT is None:
        config = load_config()
        _DEFAULT_TRANSPORT = ModalDirectTransport(
            app_name=config.app_name,
            environment_name=None,
        )
    return _DEFAULT_TRANSPORT


def _default_job_store() -> InMemoryHTTPJobStore:
    global _DEFAULT_JOB_STORE
    if _DEFAULT_JOB_STORE is None:
        _DEFAULT_JOB_STORE = InMemoryHTTPJobStore()
    return _DEFAULT_JOB_STORE


def _future_for_job(app, job_id: str):
    stored = app.state.job_store.get_job(job_id)
    if stored is not None:
        return stored.future
    return _modal_function_call(job_id)


def _modal_function_call(job_id: str):
    try:
        import modal
    except ImportError:
        return None
    try:
        return modal.FunctionCall.from_id(job_id)
    except Exception:
        return None


def _future_done(future) -> bool:
    done = getattr(future, "done", None)
    if callable(done):
        return bool(done())
    return bool(done)


async def _future_done_async(future) -> bool:
    transport = getattr(future, "_transport", None)
    job_ref = getattr(future, "_job_ref", None)
    done_async = getattr(transport, "done_async", None)
    if job_ref is not None and callable(done_async):
        return bool(await done_async(job_ref))
    done = getattr(future, "done", None)
    if callable(done):
        done_aio = getattr(done, "aio", None)
        if done_aio is not None:
            return bool(await done_aio())
        return await asyncio.to_thread(done)
    return bool(done)


def _future_cancel(future) -> bool:
    cancel = getattr(future, "cancel", None)
    if not callable(cancel):
        return False
    return bool(cancel())


async def _future_cancel_async(future) -> bool:
    transport = getattr(future, "_transport", None)
    job_ref = getattr(future, "_job_ref", None)
    cancel_async = getattr(transport, "cancel_async", None)
    if job_ref is not None and callable(cancel_async):
        return bool(await cancel_async(job_ref))
    cancel = getattr(future, "cancel", None)
    if not callable(cancel):
        return False
    cancel_aio = getattr(cancel, "aio", None)
    if cancel_aio is not None:
        return bool(await cancel_aio())
    return await asyncio.to_thread(cancel)


def _future_result(future, *, timeout: float | None):
    result = getattr(future, "result", None)
    if callable(result):
        return result(timeout=timeout)
    get = getattr(future, "get", None)
    if callable(get):
        return get(timeout=timeout)
    raise RuntimeError("job handle cannot retrieve a result")


async def _future_result_async(future, *, timeout: float | None):
    result_async = getattr(future, "result_async", None)
    if callable(result_async):
        return await result_async(timeout=timeout)
    get = getattr(future, "get", None)
    if callable(get):
        get_aio = getattr(get, "aio", None)
        if get_aio is not None:
            return await get_aio(timeout=timeout)
        return await asyncio.to_thread(get, timeout=timeout)
    result = getattr(future, "result", None)
    if callable(result):
        return await asyncio.to_thread(result, timeout=timeout)
    raise RuntimeError("job handle cannot retrieve a result")


def _result_response(app, job_id: str, result: Any, *, route: str = "<unknown>") -> dict[str, Any]:
    import json

    encoded = json.dumps(result, default=str).encode("utf-8")
    large_result_inline_bytes = getattr(app.state, "large_result_inline_bytes", LARGE_RESULT_INLINE_BYTES)
    if len(encoded) <= large_result_inline_bytes:
        return {"job_id": job_id, "status": "done", "result": result}
    result_id = app.state.job_store.put_large_result(job_id=job_id, route=route, result=result)
    return {"job_id": job_id, "status": "done", "result_ref": result_id}


def _http_exception(exc: Exception) -> HTTPException:
    from fastapi import HTTPException

    status = 500
    if isinstance(exc, BadRequestError):
        status = 400
    elif isinstance(exc, AuthenticationError):
        status = 401
    elif isinstance(exc, PermissionDeniedError):
        status = 403
    elif isinstance(exc, NotFoundError):
        status = 404
    elif isinstance(exc, ConflictError):
        status = 409
    elif isinstance(exc, UnprocessableEntityError):
        status = 422
    elif isinstance(exc, RateLimitError):
        status = 429
    elif isinstance(exc, (RequestFailedError, ToyModalError)):
        status = 500
    return HTTPException(
        status_code=status,
        detail={
            "error_type": type(exc).__name__,
            "error": str(exc),
        },
    )
