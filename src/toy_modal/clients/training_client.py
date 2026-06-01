"""Training client for Tinker-style local loops."""

from __future__ import annotations

from typing import Callable

from toy_modal import types
from toy_modal.clients.sampling_client import SamplingClient
from toy_modal.futures import APIFuture
from toy_modal.serialization import to_payload


class TrainingClient:
    def __init__(
        self,
        *,
        transport,
        training_run_id: str,
        project_id: str | None,
        base_model: str,
        lora_config: types.LoraConfig,
        model_seq_id: int = 0,
        optimizer_step: int = 0,
        user_metadata: dict[str, str] | None = None,
        accept_tinker_paths: bool = False,
        client_config: dict[str, object] | None = None,
    ) -> None:
        self._transport = transport
        self.training_run_id = training_run_id
        self.project_id = project_id
        self.base_model = base_model
        self.lora_config = lora_config
        self.model_seq_id = model_seq_id
        self.optimizer_step = optimizer_step
        self.user_metadata = user_metadata or {}
        self.accept_tinker_paths = accept_tinker_paths
        self._client_config = dict(client_config or {})
        self._last_forward_backward_job_id: str | None = None
        self._last_forward_backward_model_seq_id: int | None = None

    def forward(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self._submit_training_call(
            "training.forward",
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )

    async def forward_async(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._submit_training_call_async(
            "training.forward",
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )

    def forward_backward(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        future = self._submit_training_call(
            "training.forward_backward",
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        self._last_forward_backward_job_id = future.job_id
        self._last_forward_backward_model_seq_id = self.model_seq_id
        return future

    async def forward_backward_async(
        self,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None = None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        future = await self._submit_training_call_async(
            "training.forward_backward",
            data=data,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
        self._last_forward_backward_job_id = future.job_id
        self._last_forward_backward_model_seq_id = self.model_seq_id
        return future

    def forward_backward_custom(
        self,
        data: list[types.Datum],
        loss_fn: Callable,
        *,
        loss_type_input: str = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        raise NotImplementedError(
            "forward_backward_custom is intentionally disabled in the scaffold; "
            "trusted direct-mode callable shipping needs an explicit security design."
        )

    async def forward_backward_custom_async(
        self,
        data: list[types.Datum],
        loss_fn: Callable,
        *,
        loss_type_input: str = "logprobs",
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self.forward_backward_custom(data, loss_fn, loss_type_input=loss_type_input)

    def optim_step(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        future = self._transport.submit(
            "training.optim_step",
            {
                "training_run_id": self.training_run_id,
                "adam_params": adam_params,
                "depends_on": self._last_forward_backward_job_id,
                "expected_model_seq_id": self._last_forward_backward_model_seq_id,
            },
            result_type=types.OptimStepResponse,
        )
        return _UpdatingFuture(future, self._record_optim_step)

    async def optim_step_async(self, adam_params: types.AdamParams) -> APIFuture[types.OptimStepResponse]:
        future = await self._transport.submit_async(
            "training.optim_step",
            {
                "training_run_id": self.training_run_id,
                "adam_params": adam_params,
                "depends_on": self._last_forward_backward_job_id,
                "expected_model_seq_id": self._last_forward_backward_model_seq_id,
            },
            result_type=types.OptimStepResponse,
        )
        return _UpdatingFuture(future, self._record_optim_step)

    def save_state(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        return self._transport.submit(
            "training.save_state",
            {
                "training_run_id": self.training_run_id,
                "name": name,
                "ttl_seconds": ttl_seconds,
            },
            result_type=types.SaveWeightsResponse,
        )

    async def save_state_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> APIFuture[types.SaveWeightsResponse]:
        return await self._transport.submit_async(
            "training.save_state",
            {
                "training_run_id": self.training_run_id,
                "name": name,
                "ttl_seconds": ttl_seconds,
            },
            result_type=types.SaveWeightsResponse,
        )

    def load_state(
        self,
        path: str,
        weights_access_token: str | None = None,
    ) -> APIFuture[types.LoadWeightsResponse]:
        future = self._transport.submit(
            "training.load_state",
            {
                "path": path,
                "training_run_id": self.training_run_id,
                "optimizer": False,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=types.LoadWeightsResponse,
        )
        return _UpdatingFuture(future, self._record_load_state)

    async def load_state_async(
        self,
        path: str,
        weights_access_token: str | None = None,
    ) -> APIFuture[types.LoadWeightsResponse]:
        future = await self._transport.submit_async(
            "training.load_state",
            {
                "path": path,
                "training_run_id": self.training_run_id,
                "optimizer": False,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=types.LoadWeightsResponse,
        )
        return _UpdatingFuture(future, self._record_load_state)

    def load_state_with_optimizer(
        self,
        path: str,
        weights_access_token: str | None = None,
    ) -> APIFuture[types.LoadWeightsResponse]:
        future = self._transport.submit(
            "training.load_state",
            {
                "path": path,
                "training_run_id": self.training_run_id,
                "optimizer": True,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=types.LoadWeightsResponse,
        )
        return _UpdatingFuture(future, self._record_load_state)

    async def load_state_with_optimizer_async(
        self,
        path: str,
        weights_access_token: str | None = None,
    ) -> APIFuture[types.LoadWeightsResponse]:
        future = await self._transport.submit_async(
            "training.load_state",
            {
                "path": path,
                "training_run_id": self.training_run_id,
                "optimizer": True,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
            },
            result_type=types.LoadWeightsResponse,
        )
        return _UpdatingFuture(future, self._record_load_state)

    def save_weights_for_sampler(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return self._transport.submit(
            "training.save_weights_for_sampler",
            {
                "training_run_id": self.training_run_id,
                "name": name,
                "ttl_seconds": ttl_seconds,
            },
            result_type=types.SaveWeightsForSamplerResponse,
        )

    async def save_weights_for_sampler_async(
        self,
        name: str,
        ttl_seconds: int | None = None,
    ) -> APIFuture[types.SaveWeightsForSamplerResponse]:
        return await self._transport.submit_async(
            "training.save_weights_for_sampler",
            {
                "training_run_id": self.training_run_id,
                "name": name,
                "ttl_seconds": ttl_seconds,
            },
            result_type=types.SaveWeightsForSamplerResponse,
        )

    def get_info(self) -> types.GetInfoResponse:
        return types.GetInfoResponse(
            training_run_id=self.training_run_id,
            base_model=self.base_model,
            is_lora=True,
            lora_rank=self.lora_config.rank,
            model_seq_id=self.model_seq_id,
            optimizer_step=self.optimizer_step,
            user_metadata=self.user_metadata,
        )

    async def get_info_async(self) -> types.GetInfoResponse:
        return self.get_info()

    def get_tokenizer(self):
        if hasattr(self._transport, "get_tokenizer"):
            try:
                return self._transport.get_tokenizer(base_model=self.base_model, model_path=None)
            except TypeError:
                return self._transport.get_tokenizer()
        raise NotImplementedError("Remote tokenizer lookup is not implemented yet")

    def create_sampling_client(
        self,
        model_path: str,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        return SamplingClient(
            transport=self._transport,
            model_path=model_path,
            base_model=self.base_model,
            retry_config=retry_config,
            client_config={
                **self._client_config,
                "transport": self._client_config.get("transport", getattr(self._transport, "name", "unknown")),
            },
        )

    async def create_sampling_client_async(
        self,
        model_path: str,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        return self.create_sampling_client(model_path, retry_config=retry_config)

    def save_weights_and_get_sampling_client(
        self,
        name: str | None = None,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        save = self.save_weights_for_sampler(name or f"seq-{self.model_seq_id}").result()
        return self.create_sampling_client(save.path, retry_config=retry_config)

    async def save_weights_and_get_sampling_client_async(
        self,
        name: str | None = None,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        save_future = await self.save_weights_for_sampler_async(name or f"seq-{self.model_seq_id}")
        save = await save_future.result_async()
        return self.create_sampling_client(save.path, retry_config=retry_config)

    def _submit_training_call(
        self,
        route: str,
        *,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return self._transport.submit(
            route,
            {
                "training_run_id": self.training_run_id,
                "data": to_payload(data),
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config or {},
                "expected_model_seq_id": self.model_seq_id,
            },
            result_type=types.ForwardBackwardOutput,
        )

    async def _submit_training_call_async(
        self,
        route: str,
        *,
        data: list[types.Datum],
        loss_fn: types.LossFnType,
        loss_fn_config: dict[str, float] | None,
    ) -> APIFuture[types.ForwardBackwardOutput]:
        return await self._transport.submit_async(
            route,
            {
                "training_run_id": self.training_run_id,
                "data": to_payload(data),
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config or {},
                "expected_model_seq_id": self.model_seq_id,
            },
            result_type=types.ForwardBackwardOutput,
        )

    def _record_optim_step(self, response: types.OptimStepResponse) -> None:
        self.model_seq_id = response.model_seq_id
        self.optimizer_step = response.optimizer_step
        self._last_forward_backward_job_id = None
        self._last_forward_backward_model_seq_id = None

    def _record_load_state(self, response: types.LoadWeightsResponse) -> None:
        if response.training_run_id is not None:
            self.training_run_id = response.training_run_id
        if response.model_seq_id is not None:
            self.model_seq_id = response.model_seq_id
        if response.optimizer_step is not None:
            self.optimizer_step = response.optimizer_step
        self._last_forward_backward_job_id = None
        self._last_forward_backward_model_seq_id = None


class _UpdatingFuture(APIFuture):
    def __init__(self, wrapped: APIFuture, callback):
        self._wrapped = wrapped
        self._callback = callback

    @property
    def job_id(self) -> str:
        return self._wrapped.job_id

    @property
    def done(self) -> bool:
        return self._wrapped.done

    def result(self, timeout: float | None = None):
        result = self._wrapped.result(timeout=timeout)
        self._callback(result)
        return result

    async def result_async(self, timeout: float | None = None):
        result = await self._wrapped.result_async(timeout=timeout)
        self._callback(result)
        return result

    def cancel(self) -> bool:
        return self._wrapped.cancel()

    def future(self):
        return self._wrapped.future()
