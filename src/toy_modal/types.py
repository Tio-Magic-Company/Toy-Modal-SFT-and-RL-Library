"""Pydantic models for the Tinker-style compatibility surface."""

from __future__ import annotations

import base64
import binascii
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

LossFnType = Literal["cross_entropy", "importance_sampling", "ppo", "cispo", "dpo"]
RunStatus = Literal["initializing", "ready", "training", "saving", "failed", "deleted"]
CheckpointType = Literal["training", "sampler"]
ModelID = str


class StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class EncodedTextChunk(StrictBase):
    type: Literal["tokens"] = "tokens"
    tokens: list[int]


class ImageChunk(StrictBase):
    type: Literal["image"] = "image"
    data: bytes
    format: str
    expected_tokens: int | None = None

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, value: bytes | str) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            try:
                return base64.b64decode(value.encode("ascii"), validate=True)
            except (binascii.Error, UnicodeEncodeError) as exc:
                raise ValueError("ImageChunk.data must be bytes or a base64-encoded string") from exc
        raise TypeError("ImageChunk.data must be bytes or a base64-encoded string")

    @field_serializer("data", when_used="json")
    def serialize_data(self, value: bytes) -> str:
        return base64.b64encode(value).decode("ascii")


class ImageAssetPointerChunk(StrictBase):
    type: Literal["image_asset_pointer"] = "image_asset_pointer"
    location: str = Field(validation_alias=AliasChoices("location", "uri"))
    format: str
    expected_tokens: int | None = None

    @property
    def uri(self) -> str:
        return self.location


ModelInputChunk = EncodedTextChunk | ImageChunk | ImageAssetPointerChunk


class ModelInput(StrictBase):
    chunks: list[ModelInputChunk]

    @classmethod
    def from_ints(cls, tokens: list[int]) -> "ModelInput":
        return cls(chunks=[EncodedTextChunk(tokens=tokens)])

    @classmethod
    def empty(cls) -> "ModelInput":
        return cls(chunks=[])

    def to_ints(self) -> list[int]:
        tokens: list[int] = []
        for chunk in self.chunks:
            if not isinstance(chunk, EncodedTextChunk):
                raise ValueError("ModelInput contains non-token chunks")
            tokens.extend(chunk.tokens)
        return tokens

    def length(self) -> int:
        return len(self.to_ints())

    def append(self, chunk: ModelInputChunk) -> "ModelInput":
        return self.model_copy(update={"chunks": [*self.chunks, chunk]})

    def append_int(self, token: int) -> "ModelInput":
        chunks = list(self.chunks)
        if chunks and isinstance(chunks[-1], EncodedTextChunk):
            last = chunks[-1].model_copy(update={"tokens": [*chunks[-1].tokens, token]})
            chunks[-1] = last
        else:
            chunks.append(EncodedTextChunk(tokens=[token]))
        return self.model_copy(update={"chunks": chunks})


class TensorData(StrictBase):
    data: list[int] | list[float]
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    sparse_crow_indices: list[int] | None = None
    sparse_col_indices: list[int] | None = None

    @classmethod
    def from_list(
        cls,
        data: list[int] | list[float],
        *,
        dtype: str | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> "TensorData":
        return cls(data=data, dtype=dtype, shape=shape)

    @model_validator(mode="after")
    def infer_shape(self) -> "TensorData":
        if self.shape is None:
            self.shape = (len(self.data),)
        return self

    def to_numpy(self):
        import numpy as np

        array = np.array(self.data, dtype=self.dtype)
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array

    def to_torch(self):
        import torch

        dtype = _torch_dtype(self.dtype)
        tensor = torch.tensor(self.data, dtype=dtype) if dtype is not None else torch.tensor(self.data)
        if self.shape is not None:
            tensor = tensor.reshape(self.shape)
        return tensor


class Datum(StrictBase):
    model_input: ModelInput
    loss_fn_inputs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("loss_fn_inputs", mode="before")
    @classmethod
    def convert_tensors(cls, data: Any) -> Any:
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise TypeError("loss_fn_inputs must be a dictionary")
        return {key: _convert_tensor_like(value) for key, value in data.items()}


class AdamParams(StrictBase):
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.0


class LoraConfig(StrictBase):
    rank: int = 32
    seed: int | None = None
    alpha: int | None = None
    dropout: float = 0.0
    train_mlp: bool = True
    train_attn: bool = True
    train_unembed: bool = True
    target_modules: list[str] | None = None


class RetryConfig(StrictBase):
    max_retries: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0


class ForwardBackwardOutput(StrictBase):
    loss_fn_output_type: str | None = None
    loss_fn_outputs: dict[str, Any] = Field(default_factory=dict)
    loss: float | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    num_tokens: int | None = None
    gradient_id: str | None = None
    model_seq_id: int = 0


class OptimStepResponse(StrictBase):
    model_seq_id: int = 0
    optimizer_step: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)


class SamplingParams(StrictBase):
    max_tokens: int
    seed: int | None = None
    stop: list[str] | None = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


class SampledSequence(StrictBase):
    stop_reason: str
    tokens: list[int]
    logprobs: list[float] | None = None


class SampleResponse(StrictBase):
    sequences: list[SampledSequence]
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None

    @property
    def samples(self) -> list[SampledSequence]:
        return self.sequences


class SupportedModel(StrictBase):
    model_name: str

    def __str__(self) -> str:
        return self.model_name


class GetServerCapabilitiesResponse(StrictBase):
    supported_models: list[SupportedModel] = Field(default_factory=list)
    supports_lora: bool = True
    supports_full_finetune: bool = False
    supports_sampling: bool = True
    supports_importance_sampling: bool = False
    max_batch_size: int | None = None
    transport: str | None = None
    trainer_engine: str | None = None
    sampler_engine: str | None = None
    backend_profile: dict[str, Any] = Field(default_factory=dict)

    @field_validator("supported_models", mode="before")
    @classmethod
    def convert_supported_models(cls, value: Any) -> Any:
        if value is None:
            return []
        return [
            {"model_name": item} if isinstance(item, str) else item
            for item in value
        ]

    @property
    def supported_model_names(self) -> list[str]:
        return [model.model_name for model in self.supported_models]


class ModelData(StrictBase):
    arch: str | None = None
    model_name: str
    tokenizer_id: str | None = None


class GetInfoResponse(StrictBase):
    type: str = "model_info"
    model_data: ModelData | None = None
    model_id: str | None = None
    model_name: str | None = None
    is_lora: bool = True
    lora_rank: int | None = None
    training_run_id: str | None = None
    base_model: str | None = None
    model_seq_id: int = 0
    optimizer_step: int = 0
    user_metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_model_metadata(self) -> "GetInfoResponse":
        if self.model_id is None and self.training_run_id is not None:
            self.model_id = self.training_run_id
        if self.model_name is None:
            self.model_name = self.base_model
        if self.model_data is None and self.model_name is not None:
            self.model_data = ModelData(
                arch=self.model_name,
                model_name=self.model_name,
                tokenizer_id=self.model_name,
            )
        return self


class SaveWeightsResponse(StrictBase):
    path: str
    checkpoint_id: str | None = None
    model_seq_id: int | None = None


class LoadWeightsResponse(StrictBase):
    path: str
    training_run_id: str | None = None
    model_seq_id: int | None = None
    optimizer_step: int | None = None
    lora_config: LoraConfig | None = None


class SaveStateResponse(SaveWeightsResponse):
    pass


class LoadStateResponse(LoadWeightsResponse):
    pass


class SaveWeightsForSamplerResponse(StrictBase):
    path: str
    checkpoint_id: str | None = None
    model_seq_id: int | None = None


class SaveWeightsForSamplerResponseInternal(SaveWeightsForSamplerResponse):
    sampling_session_id: str


class WeightsInfoResponse(StrictBase):
    path: str
    base_model: str
    is_lora: bool = True
    lora_rank: int | None = None
    checkpoint_type: CheckpointType | None = None


class Checkpoint(StrictBase):
    checkpoint_id: str
    checkpoint_type: CheckpointType
    time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tinker_path: str | None = None
    toy_path: str | None = None
    size_bytes: int = 0
    public: bool = False
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def normalize_paths(self) -> "Checkpoint":
        if self.tinker_path is None and self.toy_path is not None:
            self.tinker_path = self.toy_path
        if self.toy_path is None and self.tinker_path is not None:
            self.toy_path = self.tinker_path
        if self.tinker_path is None or self.toy_path is None:
            raise ValueError("Checkpoint requires tinker_path or toy_path")
        return self


class TrainingRun(StrictBase):
    training_run_id: str
    project_id: str | None = None
    base_model: str
    model_owner: str | None = None
    is_lora: bool = True
    corrupted: bool = False
    lora_rank: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_request_time: datetime | None = None
    last_checkpoint: Checkpoint | None = None
    last_sampler_checkpoint: Checkpoint | None = None
    status: RunStatus = "ready"
    model_seq_id: int = 0
    optimizer_step: int = 0
    user_metadata: dict[str, str] = Field(default_factory=dict)


class Cursor(StrictBase):
    limit: int
    offset: int
    total_count: int


class TrainingRunsResponse(StrictBase):
    training_runs: list[TrainingRun]
    cursor: Cursor


class CheckpointsListResponse(StrictBase):
    checkpoints: list[Checkpoint]
    cursor: Cursor | None = None


class CheckpointArchiveUrlResponse(StrictBase):
    url: str
    expires: int | float | None = None
    expires_at: datetime | None = None

    @model_validator(mode="after")
    def normalize_expires(self) -> "CheckpointArchiveUrlResponse":
        if self.expires is None and self.expires_at is not None:
            self.expires = int(self.expires_at.timestamp())
        if self.expires_at is None and self.expires is not None:
            self.expires_at = datetime.fromtimestamp(float(self.expires), tz=timezone.utc)
        return self


class ParsedToyPath(StrictBase):
    scheme: Literal["toy-modal"]
    project_id: str
    run_id: str
    artifact_type: Literal["checkpoints", "sampler_weights", "adapters"]
    name: str


class CreateTrainingRunResponse(StrictBase):
    training_run_id: str
    project_id: str | None = None
    base_model: str
    lora_config: LoraConfig
    model_seq_id: int = 0
    optimizer_step: int = 0
    user_metadata: dict[str, str] = Field(default_factory=dict)


class LoadWeightsRequest(StrictBase):
    path: str
    optimizer: bool = False


class CreateModelRequest(StrictBase):
    base_model: str
    user_metadata: dict[str, str] | None = None
    lora_config: LoraConfig


class SaveWeightsRequest(StrictBase):
    path: str
    ttl_seconds: int | None = None


class SaveWeightsForSamplerRequest(StrictBase):
    path: str
    ttl_seconds: int | None = None


class FutureRetrieveRequest(StrictBase):
    request_id: str
    allow_metadata_only: bool = False


class ForwardBackwardInput(StrictBase):
    data: list[Datum]
    loss_fn: LossFnType | str
    loss_fn_config: dict[str, float] | None = None


class SampleRequest(StrictBase):
    prompt: ModelInput
    sampling_params: SamplingParams
    num_samples: int
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    seq_id: int | None = None
    prompt_logprobs: bool = False
    topk_prompt_logprobs: int = 0


class CreateSamplingSessionRequest(StrictBase):
    session_id: str
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None


class CreateSamplingSessionResponse(StrictBase):
    sampling_session_id: str


class GetSessionResponse(StrictBase):
    session_id: str
    training_run_ids: list[str] = Field(default_factory=list)
    sampler_ids: list[str] = Field(default_factory=list)
    user_metadata: dict[str, str] = Field(default_factory=dict)


class ListSessionsResponse(StrictBase):
    sessions: list[str]
    cursor: Cursor | None = None


class GetSamplerResponse(StrictBase):
    sampler_id: str
    base_model: str | None = None
    model_path: str | None = None
    sampling_session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ParsedCheckpointTinkerPath(StrictBase):
    tinker_path: str
    training_run_id: str
    checkpoint_type: CheckpointType
    checkpoint_id: str

    @classmethod
    def from_tinker_path(cls, tinker_path: str) -> "ParsedCheckpointTinkerPath":
        scheme, sep, rest = tinker_path.partition("://")
        if not sep:
            raise ValueError(f"Invalid checkpoint path: {tinker_path!r}")

        parts = [part for part in rest.split("/") if part]
        if scheme == "toy-modal":
            if len(parts) < 4:
                raise ValueError(f"Invalid toy-modal checkpoint path: {tinker_path!r}")
            _, training_run_id, artifact_type = parts[:3]
            checkpoint_id = "/".join(parts[3:])
        elif scheme == "tinker":
            if len(parts) < 3:
                raise ValueError(f"Invalid tinker checkpoint path: {tinker_path!r}")
            training_run_id, artifact_type = parts[:2]
            checkpoint_id = "/".join(parts[2:])
        else:
            raise ValueError(f"Unsupported checkpoint path scheme: {scheme!r}")

        if artifact_type in {"checkpoints", "weights", "training"}:
            checkpoint_type: CheckpointType = "training"
        elif artifact_type in {"sampler_weights", "sampler"}:
            checkpoint_type = "sampler"
        else:
            raise ValueError(f"Unsupported checkpoint artifact type: {artifact_type!r}")

        return cls(
            tinker_path=tinker_path,
            training_run_id=training_run_id,
            checkpoint_type=checkpoint_type,
            checkpoint_id=checkpoint_id,
        )


class TelemetrySendRequest(StrictBase):
    platform: str
    sdk_version: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class TelemetryBatch(StrictBase):
    platform: str
    sdk_version: str
    events: list[dict[str, Any]] = Field(default_factory=list)


class SessionStartEvent(StrictBase):
    event: str = "session_start"
    severity: str = "info"


class SessionEndEvent(StrictBase):
    duration: str
    event: str = "session_end"
    severity: str = "info"


class GenericEvent(StrictBase):
    event: str = "generic"
    event_name: str
    severity: str = "info"
    event_data: dict[str, Any] = Field(default_factory=dict)


class UnhandledExceptionEvent(StrictBase):
    event: str = "unhandled_exception"
    severity: str = "error"
    traceback: str | None = None


class TryAgainResponse(StrictBase):
    request_id: str


def _convert_tensor_like(value: Any) -> Any:
    if isinstance(value, TensorData):
        return value
    if isinstance(value, dict):
        return {key: _convert_tensor_like(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_convert_tensor_like(item) for item in value]
    if isinstance(value, list):
        return [_convert_tensor_like(item) for item in value]

    module = type(value).__module__
    if module.startswith("torch") and hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return TensorData(
            data=tensor.reshape(-1).tolist(),
            dtype=str(tensor.dtype).replace("torch.", ""),
            shape=tuple(int(dim) for dim in tensor.shape),
        )

    if module.startswith("numpy") and hasattr(value, "reshape") and hasattr(value, "tolist"):
        flattened = value.reshape(-1)
        return TensorData(
            data=flattened.tolist(),
            dtype=str(getattr(value, "dtype", "")) or None,
            shape=tuple(int(dim) for dim in getattr(value, "shape", ())),
        )

    return value


def _torch_dtype(dtype: str | None):
    if dtype is None:
        return None
    import torch

    return getattr(torch, dtype, None)
