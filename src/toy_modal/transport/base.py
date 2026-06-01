"""Transport protocol for backend invocations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from toy_modal.futures import APIFuture

T = TypeVar("T")


@dataclass(frozen=True)
class JobRef:
    job_id: str
    route: str
    native_ref: Any = None


class Transport(Protocol):
    name: str

    def submit(self, route: str, payload: Any, *, result_type: type[T] | object) -> APIFuture[T]: ...

    async def submit_async(
        self, route: str, payload: Any, *, result_type: type[T] | object
    ) -> APIFuture[T]: ...

    def get_result(self, job_ref: JobRef, timeout: float | None = None) -> Any: ...

    async def get_result_async(self, job_ref: JobRef, timeout: float | None = None) -> Any: ...

    def job_id(self, job_ref: JobRef) -> str: ...

    def done(self, job_ref: JobRef) -> bool: ...

    def cancel(self, job_ref: JobRef) -> bool: ...
