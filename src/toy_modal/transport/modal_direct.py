"""Direct Modal Python SDK transport."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from toy_modal.errors import BackendUnavailableError, BadRequestError
from toy_modal.futures import APIFuture
from toy_modal.transport.base import JobRef


class ModalDirectTransport:
    name = "modal-direct"

    FUNCTION_ROUTES = {
        "server.capabilities": "get_server_capabilities",
        "training.create_lora": "create_lora_training_run",
    }
    TRAINER_METHOD_ROUTES = {
        "training.forward": "forward",
        "training.forward_backward": "forward_backward",
        "training.optim_step": "optim_step",
        "training.save_state": "save_state",
        "training.load_state": "load_state",
        "training.save_weights_for_sampler": "save_weights_for_sampler",
        "training.validate_old_logprobs_sequence": "validate_old_logprobs_sequence",
    }
    SAMPLER_METHOD_ROUTES = {
        "sampling.sample": "sample",
        "sampling.compute_logprobs": "compute_logprobs",
    }
    REST_ROUTES = {
        "rest.get_training_run",
        "rest.get_training_run_by_toy_path",
        "rest.get_weights_info_by_toy_path",
        "rest.list_training_runs",
        "rest.list_checkpoints",
        "rest.list_user_checkpoints",
        "rest.get_checkpoint_archive_url",
        "rest.get_checkpoint_archive_url_from_toy_path",
        "rest.inspect_checkpoint_artifact_from_toy_path",
        "rest.delete_checkpoint",
        "rest.delete_checkpoint_from_toy_path",
        "rest.set_checkpoint_ttl_from_toy_path",
        "rest.set_checkpoint_public",
        "rest.get_session",
        "rest.list_sessions",
        "rest.get_sampler",
    }
    TOKENIZER_ROUTES = {
        "tokenizer.encode": "encode",
        "tokenizer.decode": "decode",
    }

    def __init__(self, *, app_name: str, environment_name: str | None = None) -> None:
        self.app_name = app_name
        self.environment_name = environment_name

    def submit(self, route: str, payload: Any, *, result_type: type[Any] | object) -> APIFuture[Any]:
        try:
            import modal
        except ImportError as exc:
            raise BackendUnavailableError(
                "modal-direct transport requires the optional 'modal' dependency"
            ) from exc

        if route in self.FUNCTION_ROUTES:
            return self._submit_function(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route == "training.load_state" and not payload.get("training_run_id"):
            return self._submit_named_function(
                modal=modal,
                function_name="create_training_run_from_state",
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.TRAINER_METHOD_ROUTES:
            return self._submit_trainer_method(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.REST_ROUTES:
            return self._submit_named_function(
                modal=modal,
                function_name="metadata_route",
                route=route,
                payload={"route": route, "payload": payload},
                result_type=result_type,
            )
        if route in self.SAMPLER_METHOD_ROUTES:
            return self._submit_sampler_method(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.TOKENIZER_ROUTES:
            return self._submit_named_function(
                modal=modal,
                function_name="tokenizer_route",
                route=route,
                payload={**payload, "operation": self.TOKENIZER_ROUTES[route]},
                result_type=result_type,
            )
        raise BadRequestError(f"Route {route!r} is not wired for modal-direct yet")

    def _submit_function(
        self,
        *,
        modal,
        route: str,
        payload: Any,
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        function_name = self.FUNCTION_ROUTES[route]
        return self._submit_named_function(
            modal=modal,
            function_name=function_name,
            route=route,
            payload=payload,
            result_type=result_type,
        )

    def _submit_named_function(
        self,
        *,
        modal,
        function_name: str,
        route: str,
        payload: Any,
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        function = modal.Function.from_name(
            self.app_name,
            function_name,
            environment_name=self.environment_name,
        )
        call = function.spawn(payload)
        return self._future(route, call, result_type)

    def _submit_trainer_method(
        self,
        *,
        modal,
        route: str,
        payload: dict[str, Any],
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        run_id = payload.get("training_run_id")
        if not run_id:
            raise BadRequestError(f"{route} requires training_run_id")
        cls = modal.Cls.from_name(
            self.app_name,
            "TrainerWorker",
            environment_name=self.environment_name,
        )
        worker = cls(run_id=run_id)
        method = getattr(worker, self.TRAINER_METHOD_ROUTES[route])
        call = method.spawn(payload)
        return self._future(route, call, result_type)

    def _submit_sampler_method(
        self,
        *,
        modal,
        route: str,
        payload: dict[str, Any],
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        cls = modal.Cls.from_name(
            self.app_name,
            "SamplerWorker",
            environment_name=self.environment_name,
        )
        worker = cls(
            model_path=payload.get("model_path") or "",
            base_model=payload.get("base_model") or "",
        )
        method = getattr(worker, self.SAMPLER_METHOD_ROUTES[route])
        call = method.spawn(payload)
        return self._future(route, call, result_type)

    def _future(self, route: str, call: Any, result_type: type[Any] | object) -> APIFuture[Any]:
        return APIFuture(
            JobRef(job_id=getattr(call, "object_id", str(uuid4())), route=route, native_ref=call),
            self,
            result_type,
        )

    async def submit_async(
        self, route: str, payload: Any, *, result_type: type[Any] | object
    ) -> APIFuture[Any]:
        try:
            import modal
        except ImportError as exc:
            raise BackendUnavailableError(
                "modal-direct transport requires the optional 'modal' dependency"
            ) from exc

        if route in self.FUNCTION_ROUTES:
            return await self._submit_function_async(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route == "training.load_state" and not payload.get("training_run_id"):
            return await self._submit_named_function_async(
                modal=modal,
                function_name="create_training_run_from_state",
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.TRAINER_METHOD_ROUTES:
            return await self._submit_trainer_method_async(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.REST_ROUTES:
            return await self._submit_named_function_async(
                modal=modal,
                function_name="metadata_route",
                route=route,
                payload={"route": route, "payload": payload},
                result_type=result_type,
            )
        if route in self.SAMPLER_METHOD_ROUTES:
            return await self._submit_sampler_method_async(
                modal=modal,
                route=route,
                payload=payload,
                result_type=result_type,
            )
        if route in self.TOKENIZER_ROUTES:
            return await self._submit_named_function_async(
                modal=modal,
                function_name="tokenizer_route",
                route=route,
                payload={**payload, "operation": self.TOKENIZER_ROUTES[route]},
                result_type=result_type,
            )
        raise BadRequestError(f"Route {route!r} is not wired for modal-direct yet")

    async def _submit_function_async(
        self,
        *,
        modal,
        route: str,
        payload: Any,
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        function_name = self.FUNCTION_ROUTES[route]
        return await self._submit_named_function_async(
            modal=modal,
            function_name=function_name,
            route=route,
            payload=payload,
            result_type=result_type,
        )

    async def _submit_named_function_async(
        self,
        *,
        modal,
        function_name: str,
        route: str,
        payload: Any,
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        function = modal.Function.from_name(
            self.app_name,
            function_name,
            environment_name=self.environment_name,
        )
        call = await self._spawn_async(function, payload)
        return self._future(route, call, result_type)

    async def _submit_trainer_method_async(
        self,
        *,
        modal,
        route: str,
        payload: dict[str, Any],
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        run_id = payload.get("training_run_id")
        if not run_id:
            raise BadRequestError(f"{route} requires training_run_id")
        cls = modal.Cls.from_name(
            self.app_name,
            "TrainerWorker",
            environment_name=self.environment_name,
        )
        worker = cls(run_id=run_id)
        method = getattr(worker, self.TRAINER_METHOD_ROUTES[route])
        call = await self._spawn_async(method, payload)
        return self._future(route, call, result_type)

    async def _submit_sampler_method_async(
        self,
        *,
        modal,
        route: str,
        payload: dict[str, Any],
        result_type: type[Any] | object,
    ) -> APIFuture[Any]:
        cls = modal.Cls.from_name(
            self.app_name,
            "SamplerWorker",
            environment_name=self.environment_name,
        )
        worker = cls(
            model_path=payload.get("model_path") or "",
            base_model=payload.get("base_model") or "",
        )
        method = getattr(worker, self.SAMPLER_METHOD_ROUTES[route])
        call = await self._spawn_async(method, payload)
        return self._future(route, call, result_type)

    @staticmethod
    async def _spawn_async(handle: Any, payload: Any) -> Any:
        spawn = handle.spawn
        spawn_aio = getattr(spawn, "aio", None)
        if spawn_aio is not None:
            return await spawn_aio(payload)
        return await asyncio.to_thread(spawn, payload)

    def get_result(self, job_ref: JobRef, timeout: float | None = None) -> Any:
        return job_ref.native_ref.get(timeout=timeout)

    async def get_result_async(self, job_ref: JobRef, timeout: float | None = None) -> Any:
        get = job_ref.native_ref.get
        get_aio = getattr(get, "aio", None)
        if get_aio is not None:
            return await get_aio(timeout=timeout)
        return await asyncio.to_thread(get, timeout=timeout)

    def job_id(self, job_ref: JobRef) -> str:
        return job_ref.job_id

    def done(self, job_ref: JobRef) -> bool:
        try:
            return bool(job_ref.native_ref.done())
        except AttributeError:
            return False

    async def done_async(self, job_ref: JobRef) -> bool:
        try:
            done = job_ref.native_ref.done
        except AttributeError:
            return False
        done_aio = getattr(done, "aio", None)
        if done_aio is not None:
            return bool(await done_aio())
        return bool(await asyncio.to_thread(done))

    def cancel(self, job_ref: JobRef) -> bool:
        try:
            job_ref.native_ref.cancel()
            return True
        except AttributeError:
            return False

    async def cancel_async(self, job_ref: JobRef) -> bool:
        try:
            cancel = job_ref.native_ref.cancel
        except AttributeError:
            return False
        cancel_aio = getattr(cancel, "aio", None)
        if cancel_aio is not None:
            await cancel_aio()
        else:
            await asyncio.to_thread(cancel)
        return True

    def get_tokenizer(self, *, base_model: str | None = None, model_path: str | None = None):
        from toy_modal.clients.remote_tokenizer import RemoteTokenizer

        return RemoteTokenizer(transport=self, base_model=base_model, model_path=model_path)
