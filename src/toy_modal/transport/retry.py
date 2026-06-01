"""Retry wrapper for transports that submit idempotent SDK requests."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from toy_modal import types


class RetryingTransport:
    def __init__(self, wrapped, retry_config: types.RetryConfig) -> None:
        self._wrapped = wrapped
        self.retry_config = retry_config
        self.name = getattr(wrapped, "name", "retrying")

    def submit(self, route: str, payload: Any, *, result_type: type[Any] | object):
        return RetryingFuture(
            transport=self._wrapped,
            route=route,
            payload=payload,
            result_type=result_type,
            retry_config=self.retry_config,
        )

    async def submit_async(self, route: str, payload: Any, *, result_type: type[Any] | object):
        current = await self._wrapped.submit_async(route, payload, result_type=result_type)
        return RetryingFuture(
            transport=self._wrapped,
            route=route,
            payload=payload,
            result_type=result_type,
            retry_config=self.retry_config,
            current=current,
        )

    def get_tokenizer(self, *, base_model: str | None = None, model_path: str | None = None):
        if hasattr(self._wrapped, "get_tokenizer"):
            return self._wrapped.get_tokenizer(base_model=base_model, model_path=model_path)
        raise AttributeError("wrapped transport does not expose get_tokenizer")


class RetryingFuture:
    def __init__(
        self,
        *,
        transport,
        route: str,
        payload: Any,
        result_type: type[Any] | object,
        retry_config: types.RetryConfig,
        current=None,
    ) -> None:
        self._transport = transport
        self._route = route
        self._payload = payload
        self._result_type = result_type
        self._retry_config = retry_config
        self._current = current or self._transport.submit(route, payload, result_type=result_type)

    @property
    def job_id(self) -> str:
        return self._current.job_id

    @property
    def done(self) -> bool:
        return self._current.done

    def result(self, timeout: float | None = None):
        delay = self._retry_config.initial_backoff_seconds
        last_exc: BaseException | None = None
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                return self._current.result(timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_config.max_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, self._retry_config.max_backoff_seconds)
                self._current = self._transport.submit(
                    self._route,
                    self._payload,
                    result_type=self._result_type,
                )
        assert last_exc is not None
        raise last_exc

    async def result_async(self, timeout: float | None = None):
        delay = self._retry_config.initial_backoff_seconds
        last_exc: BaseException | None = None
        for attempt in range(self._retry_config.max_retries + 1):
            try:
                return await self._current.result_async(timeout=timeout)
            except Exception as exc:
                last_exc = exc
                if attempt >= self._retry_config.max_retries:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._retry_config.max_backoff_seconds)
                self._current = await self._transport.submit_async(
                    self._route,
                    self._payload,
                    result_type=self._result_type,
                )
        assert last_exc is not None
        raise last_exc

    def cancel(self) -> bool:
        return self._current.cancel()

    def future(self):
        return self._current.future()

    def __await__(self):
        return self.result_async().__await__()
