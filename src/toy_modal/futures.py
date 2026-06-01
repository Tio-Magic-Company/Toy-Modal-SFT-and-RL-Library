"""Future abstraction shared by direct Modal and HTTP transports."""

from __future__ import annotations

from typing import Generic, TypeVar

from toy_modal.serialization import decode_result

T = TypeVar("T")


class APIFuture(Generic[T]):
    """A small awaitable future that normalizes transport-specific job handles."""

    def __init__(self, job_ref, transport, result_type: type[T] | object):
        self._job_ref = job_ref
        self._transport = transport
        self._result_type = result_type

    @property
    def job_id(self) -> str:
        return self._transport.job_id(self._job_ref)

    @property
    def done(self) -> bool:
        return self._transport.done(self._job_ref)

    def result(self, timeout: float | None = None) -> T:
        payload = self._transport.get_result(self._job_ref, timeout=timeout)
        return decode_result(payload, self._result_type)

    async def result_async(self, timeout: float | None = None) -> T:
        payload = await self._transport.get_result_async(self._job_ref, timeout=timeout)
        return decode_result(payload, self._result_type)

    def cancel(self) -> bool:
        return self._transport.cancel(self._job_ref)

    def future(self):
        return self._job_ref.native_ref

    def __await__(self):
        return self.result_async().__await__()
