"""Sampling client for base models or saved adapter weights."""

from __future__ import annotations

from dataclasses import dataclass

from toy_modal import types
from toy_modal.futures import APIFuture
from toy_modal.transport.retry import RetryingTransport


@dataclass(frozen=True)
class _SamplingClientPickleState:
    model_path: str | None
    base_model: str | None
    retry_config: types.RetryConfig | None
    client_config: dict[str, object]


def _rebuild_sampling_client(state: _SamplingClientPickleState) -> "SamplingClient":
    from toy_modal.clients.service_client import ServiceClient

    service = ServiceClient(
        project_id=state.client_config.get("project_id"),
        app_name=state.client_config.get("app_name", "toy-modal-backend"),
        environment_name=state.client_config.get("environment_name"),
        transport=state.client_config.get("transport", "modal-direct"),
        base_url=state.client_config.get("base_url"),
        api_key=state.client_config.get("api_key"),
    )
    return service.create_sampling_client(
        model_path=state.model_path,
        base_model=state.base_model,
        retry_config=state.retry_config,
    )


class SamplingClient:
    def __init__(
        self,
        *,
        transport,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: types.RetryConfig | None = None,
        client_config: dict[str, object] | None = None,
    ) -> None:
        self._transport = transport
        if retry_config is not None:
            self._transport = RetryingTransport(self._transport, retry_config)
        self.model_path = model_path
        self.base_model = base_model
        self.retry_config = retry_config
        self._client_config = client_config or {}

    def sample(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> APIFuture[types.SampleResponse]:
        return self._transport.submit(
            "sampling.sample",
            {
                "model_path": self.model_path,
                "base_model": self.base_model,
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
                "include_prompt_logprobs": include_prompt_logprobs,
                "topk_prompt_logprobs": topk_prompt_logprobs,
            },
            result_type=types.SampleResponse,
        )

    async def sample_async(
        self,
        prompt: types.ModelInput,
        num_samples: int,
        sampling_params: types.SamplingParams,
        include_prompt_logprobs: bool = False,
        topk_prompt_logprobs: int = 0,
    ) -> types.SampleResponse:
        future = await self._transport.submit_async(
            "sampling.sample",
            {
                "model_path": self.model_path,
                "base_model": self.base_model,
                "prompt": prompt,
                "num_samples": num_samples,
                "sampling_params": sampling_params,
                "include_prompt_logprobs": include_prompt_logprobs,
                "topk_prompt_logprobs": topk_prompt_logprobs,
            },
            result_type=types.SampleResponse,
        )
        return await future.result_async()

    def compute_logprobs(self, prompt: types.ModelInput) -> APIFuture[list[float | None]]:
        return self._transport.submit(
            "sampling.compute_logprobs",
            {
                "model_path": self.model_path,
                "base_model": self.base_model,
                "prompt": prompt,
            },
            result_type=list[float | None],
        )

    async def compute_logprobs_async(self, prompt: types.ModelInput) -> list[float | None]:
        future = await self._transport.submit_async(
            "sampling.compute_logprobs",
            {
                "model_path": self.model_path,
                "base_model": self.base_model,
                "prompt": prompt,
            },
            result_type=list[float | None],
        )
        return await future.result_async()

    def get_tokenizer(self):
        if hasattr(self._transport, "get_tokenizer"):
            try:
                return self._transport.get_tokenizer(
                    base_model=self.base_model,
                    model_path=self.model_path,
                )
            except TypeError:
                return self._transport.get_tokenizer()
        raise NotImplementedError("Remote tokenizer lookup is not implemented yet")

    def get_base_model(self) -> str:
        if self.base_model:
            return self.base_model
        if self.model_path:
            return self.model_path
        raise ValueError("SamplingClient has neither base_model nor model_path")

    async def get_base_model_async(self) -> str:
        return self.get_base_model()

    def __reduce__(self):
        return (
            _rebuild_sampling_client,
            (
                _SamplingClientPickleState(
                    model_path=self.model_path,
                    base_model=self.base_model,
                    retry_config=self.retry_config,
                    client_config=self._client_config,
                ),
            ),
        )
