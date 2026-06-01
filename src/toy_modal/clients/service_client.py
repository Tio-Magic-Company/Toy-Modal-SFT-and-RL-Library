"""Service entry point for creating training, sampling, and REST clients."""

from __future__ import annotations

from toy_modal import types
from toy_modal.clients.rest_client import RestClient
from toy_modal.clients.sampling_client import SamplingClient
from toy_modal.clients.training_client import TrainingClient
from toy_modal.transport.http_gateway import HTTPGatewayTransport
from toy_modal.transport.modal_direct import ModalDirectTransport


class ServiceClient:
    def __init__(
        self,
        user_metadata: dict[str, str] | None = None,
        project_id: str | None = None,
        *,
        backend: str = "modal",
        app_name: str = "toy-modal-backend",
        environment_name: str | None = None,
        transport: str = "modal-direct",
        base_url: str | None = None,
        api_key: str | None = None,
        modal_profile: str | None = None,
        accept_tinker_paths: bool = False,
    ) -> None:
        self.user_metadata = user_metadata or {}
        self.project_id = project_id
        self.backend = backend
        self.app_name = app_name
        self.environment_name = environment_name
        self.modal_profile = modal_profile
        self.accept_tinker_paths = accept_tinker_paths
        self._transport_name = transport
        self.base_url = base_url
        self.api_key = api_key
        self._transport = self._build_transport(transport, base_url=base_url, api_key=api_key)

    def _build_transport(self, transport: str, *, base_url: str | None, api_key: str | None):
        if transport == "local-mock":
            raise ValueError(
                "transport='local-mock' has been removed. Deploy the Modal backend "
                "and use transport='modal-direct', or use transport='http' with a "
                "deployed gateway URL."
            )
        if transport == "modal-direct":
            return ModalDirectTransport(
                app_name=self.app_name,
                environment_name=self.environment_name,
            )
        if transport == "http":
            if not base_url:
                raise ValueError("base_url is required for http transport")
            return HTTPGatewayTransport(base_url=base_url, api_key=api_key)
        raise ValueError(f"Unsupported transport: {transport}")

    def get_server_capabilities(self) -> types.GetServerCapabilitiesResponse:
        return self._transport.submit(
            "server.capabilities",
            {"project_id": self.project_id},
            result_type=types.GetServerCapabilitiesResponse,
        ).result()

    async def get_server_capabilities_async(self) -> types.GetServerCapabilitiesResponse:
        future = await self._transport.submit_async(
            "server.capabilities",
            {"project_id": self.project_id},
            result_type=types.GetServerCapabilitiesResponse,
        )
        return await future.result_async()

    def create_lora_training_client(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        metadata = {**self.user_metadata, **(user_metadata or {})}
        lora_config = types.LoraConfig(
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
        response = self._transport.submit(
            "training.create_lora",
            {
                "base_model": base_model,
                "project_id": self.project_id,
                "seed": seed,
                "lora_config": lora_config,
                "user_metadata": metadata,
            },
            result_type=types.CreateTrainingRunResponse,
        ).result()
        return self._training_client_from_create_response(response)

    def _training_client_from_create_response(
        self,
        response: types.CreateTrainingRunResponse,
    ) -> TrainingClient:
        return TrainingClient(
            transport=self._transport,
            training_run_id=response.training_run_id,
            project_id=response.project_id,
            base_model=response.base_model,
            lora_config=response.lora_config,
            model_seq_id=response.model_seq_id,
            optimizer_step=response.optimizer_step,
            user_metadata=response.user_metadata,
            accept_tinker_paths=self.accept_tinker_paths,
            client_config=self._client_config(),
        )

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int = 32,
        seed: int | None = None,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
        user_metadata: dict[str, str] | None = None,
    ) -> TrainingClient:
        metadata = {**self.user_metadata, **(user_metadata or {})}
        lora_config = types.LoraConfig(
            rank=rank,
            seed=seed,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
        future = await self._transport.submit_async(
            "training.create_lora",
            {
                "base_model": base_model,
                "project_id": self.project_id,
                "seed": seed,
                "lora_config": lora_config,
                "user_metadata": metadata,
            },
            result_type=types.CreateTrainingRunResponse,
        )
        response = await future.result_async()
        return self._training_client_from_create_response(response)

    def create_training_client_from_state(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return self._create_training_client_from_state(
            path,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=False,
        )

    def _create_training_client_from_state(
        self,
        path: str,
        *,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
        optimizer: bool,
    ) -> TrainingClient:
        load = self._transport.submit(
            "training.load_state",
            {
                "path": path,
                "optimizer": optimizer,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
                "project_id": self.project_id,
                "user_metadata": {**self.user_metadata, **(user_metadata or {})},
            },
            result_type=types.LoadWeightsResponse,
        ).result()
        if load.training_run_id is None:
            raise ValueError("Backend load_state response did not include training_run_id")
        run = self.create_rest_client().get_training_run(load.training_run_id).result()
        return TrainingClient(
            transport=self._transport,
            training_run_id=run.training_run_id,
            project_id=run.project_id,
            base_model=run.base_model,
            lora_config=load.lora_config or types.LoraConfig(rank=run.lora_rank or 32),
            model_seq_id=load.model_seq_id,
            optimizer_step=run.optimizer_step,
            user_metadata=run.user_metadata,
            accept_tinker_paths=self.accept_tinker_paths,
            client_config=self._client_config(),
        )

    async def _create_training_client_from_state_async(
        self,
        path: str,
        *,
        user_metadata: dict[str, str] | None,
        weights_access_token: str | None,
        optimizer: bool,
    ) -> TrainingClient:
        future = await self._transport.submit_async(
            "training.load_state",
            {
                "path": path,
                "optimizer": optimizer,
                "weights_access_token": weights_access_token,
                "accept_tinker_paths": self.accept_tinker_paths,
                "project_id": self.project_id,
                "user_metadata": {**self.user_metadata, **(user_metadata or {})},
            },
            result_type=types.LoadWeightsResponse,
        )
        load = await future.result_async()
        if load.training_run_id is None:
            raise ValueError("Backend load_state response did not include training_run_id")
        run = await self.create_rest_client().get_training_run_async(load.training_run_id)
        return TrainingClient(
            transport=self._transport,
            training_run_id=run.training_run_id,
            project_id=run.project_id,
            base_model=run.base_model,
            lora_config=load.lora_config or types.LoraConfig(rank=run.lora_rank or 32),
            model_seq_id=load.model_seq_id,
            optimizer_step=run.optimizer_step,
            user_metadata=run.user_metadata,
            accept_tinker_paths=self.accept_tinker_paths,
            client_config=self._client_config(),
        )

    async def create_training_client_from_state_async(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return await self._create_training_client_from_state_async(
            path,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=False,
        )

    def create_training_client_from_state_with_optimizer(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return self._create_training_client_from_state(
            path,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=True,
        )

    async def create_training_client_from_state_with_optimizer_async(
        self,
        path: str,
        user_metadata: dict[str, str] | None = None,
        weights_access_token: str | None = None,
    ) -> TrainingClient:
        return await self._create_training_client_from_state_async(
            path,
            user_metadata=user_metadata,
            weights_access_token=weights_access_token,
            optimizer=True,
        )

    def create_sampling_client(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        if not model_path and not base_model:
            raise ValueError("Either model_path or base_model is required")
        return SamplingClient(
            transport=self._transport,
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
            client_config=self._client_config(),
        )

    async def create_sampling_client_async(
        self,
        model_path: str | None = None,
        base_model: str | None = None,
        retry_config: types.RetryConfig | None = None,
    ) -> SamplingClient:
        return self.create_sampling_client(
            model_path=model_path,
            base_model=base_model,
            retry_config=retry_config,
        )

    def create_rest_client(self) -> RestClient:
        return RestClient(
            transport=self._transport,
            accept_tinker_paths=self.accept_tinker_paths,
        )

    def _client_config(self) -> dict[str, object]:
        config: dict[str, object] = {
            "project_id": self.project_id,
            "app_name": self.app_name,
            "environment_name": self.environment_name,
            "transport": self._transport_name,
            "base_url": self.base_url,
            "api_key": self.api_key,
        }
        return {key: value for key, value in config.items() if value is not None}
