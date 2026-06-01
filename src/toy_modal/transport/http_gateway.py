"""HTTP gateway transport for deployed gateway endpoints."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from toy_modal.errors import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from toy_modal.futures import APIFuture
from toy_modal.serialization import to_payload
from toy_modal.transport.base import JobRef


class HTTPGatewayTransport:
    name = "http"

    def __init__(self, *, base_url: str, api_key: str | None = None, request_timeout: float | None = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.request_timeout = request_timeout
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError(
                "http transport requires a deployed HTTP(S) gateway URL; "
                "the in-process local mock gateway has been removed"
            )

    def submit(self, route: str, payload: Any, *, result_type: type[Any] | object) -> APIFuture[Any]:
        response = self._request(
            "POST",
            "/submit",
            {"route": route, "payload": to_payload(payload)},
        )
        return APIFuture(
            JobRef(job_id=response["job_id"], route=route, native_ref=response),
            self,
            result_type,
        )

    async def submit_async(
        self, route: str, payload: Any, *, result_type: type[Any] | object
    ) -> APIFuture[Any]:
        return await asyncio.to_thread(self.submit, route, payload, result_type=result_type)

    def get_result(self, job_ref: JobRef, timeout: float | None = None) -> Any:
        query_params: dict[str, str] = {"route": job_ref.route}
        if timeout is not None:
            query_params["timeout"] = str(timeout)
        query = "?" + urllib.parse.urlencode(query_params)
        response = self._request(
            "GET",
            f"/retrieve/{job_ref.job_id}{query}",
            None,
            timeout=_retrieve_request_timeout(timeout, self.request_timeout),
        )
        if response.get("status") == "pending":
            raise TimeoutError(f"request is still pending: {job_ref.job_id}")
        if response.get("status") == "failed":
            raise APIStatusError(response.get("error", "request failed"), body=response)
        if response.get("result_ref"):
            large = self._request("GET", f"/large-results/{response['result_ref']}", None)
            return large.get("result")
        return response.get("result")

    async def get_result_async(self, job_ref: JobRef, timeout: float | None = None) -> Any:
        return await asyncio.to_thread(self.get_result, job_ref, timeout)

    def job_id(self, job_ref: JobRef) -> str:
        return job_ref.job_id

    def done(self, job_ref: JobRef) -> bool:
        response = self._request("GET", f"/status/{job_ref.job_id}", None)
        return bool(response.get("done"))

    def cancel(self, job_ref: JobRef) -> bool:
        response = self._request("POST", f"/cancel/{job_ref.job_id}", {})
        return bool(response.get("cancelled"))

    def get_tokenizer(self, *, base_model: str | None = None, model_path: str | None = None):
        from toy_modal.clients.remote_tokenizer import RemoteTokenizer

        return RemoteTokenizer(transport=self, base_model=base_model, model_path=model_path)

    def _request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.request_timeout if timeout is None else timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                error_body = json.loads(exc.read().decode("utf-8"))
            except Exception:
                error_body = {"status_code": exc.code}
            error_cls = _status_error_class(exc.code)
            detail = error_body.get("detail") if isinstance(error_body, dict) else None
            message = _error_message(exc.code, detail)
            raise error_cls(message, body=error_body) from exc
        except OSError as exc:
            raise APIConnectionError("Could not reach HTTP gateway") from exc


def _retrieve_request_timeout(result_timeout: float | None, request_timeout: float | None) -> float | None:
    if result_timeout is None:
        return None
    if request_timeout is None:
        return result_timeout + 5.0
    return max(request_timeout, result_timeout + 5.0)


def _error_message(status_code: int, detail: Any) -> str:
    if isinstance(detail, dict):
        error_type = detail.get("error_type")
        error = detail.get("error")
        if error_type and error:
            return f"HTTP gateway returned {status_code}: {error_type}: {error}"
    if isinstance(detail, str):
        return f"HTTP gateway returned {status_code}: {detail}"
    return f"HTTP gateway returned {status_code}"


def _status_error_class(status_code: int):
    if status_code == 400:
        return BadRequestError
    if status_code == 401:
        return AuthenticationError
    if status_code == 403:
        return PermissionDeniedError
    if status_code == 404:
        return NotFoundError
    if status_code == 409:
        return ConflictError
    if status_code == 422:
        return UnprocessableEntityError
    if status_code == 429:
        return RateLimitError
    if status_code >= 500:
        return InternalServerError
    return APIStatusError
