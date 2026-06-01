"""Backend loss input normalization and shape validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from toy_modal import types
from toy_modal.errors import BadRequestError


@dataclass(frozen=True)
class SupervisedBatchItem:
    tokens: list[int]
    labels: list[int]
    weights: list[float]
    prompt_length: int
    target_length: int


@dataclass(frozen=True)
class ImportanceSamplingBatchItem:
    tokens: list[int]
    labels: list[int]
    completion_tokens: list[int]
    old_logprobs: list[float]
    advantages: list[float]
    weights: list[float]
    masks: list[float]
    ref_logprobs: list[float] | None


@dataclass(frozen=True)
class DPOBatchItem:
    chosen_tokens: list[int]
    chosen_labels: list[int]
    chosen_weights: list[float]
    rejected_tokens: list[int]
    rejected_labels: list[int]
    rejected_weights: list[float]
    prompt_length: int
    beta: float
    reference_chosen_logprob: float
    reference_rejected_logprob: float


def validate_training_batch(data: list[Any], loss_fn: str) -> None:
    if loss_fn == "cross_entropy":
        prepare_supervised_batch_items(data)
        return
    if loss_fn in {"importance_sampling", "ppo", "cispo"}:
        prepare_importance_sampling_batch_items(data)
        return
    if loss_fn == "dpo":
        prepare_dpo_batch_items(data)
        return
    raise BadRequestError(f"unsupported loss_fn: {loss_fn!r}")


def prepare_supervised_batch_items(data: list[Any]) -> list[SupervisedBatchItem]:
    if not data:
        raise BadRequestError("training data must contain at least one datum")
    return [_prepare_supervised_item(datum) for datum in data]


def prepare_importance_sampling_batch_items(data: list[Any]) -> list[ImportanceSamplingBatchItem]:
    if not data:
        raise BadRequestError("training data must contain at least one datum")
    return [_prepare_importance_sampling_item(datum) for datum in data]


def prepare_dpo_batch_items(data: list[Any]) -> list[DPOBatchItem]:
    if not data:
        raise BadRequestError("training data must contain at least one datum")
    return [_prepare_dpo_item(datum) for datum in data]


def _prepare_supervised_item(raw_datum: Any) -> SupervisedBatchItem:
    datum = _datum(raw_datum)
    tokens = datum.model_input.to_ints()
    if len(tokens) < 2:
        raise BadRequestError("cross_entropy requires at least two model_input tokens")

    inputs = datum.loss_fn_inputs
    target_tokens = _int_list(inputs.get("target_tokens"), name="target_tokens", required=True)
    if not target_tokens:
        raise BadRequestError("cross_entropy requires non-empty target_tokens")
    if len(target_tokens) > len(tokens):
        raise BadRequestError(
            "target_tokens length cannot exceed model_input token length "
            f"({len(target_tokens)} > {len(tokens)})"
        )

    start = len(tokens) - len(target_tokens)
    labels = [-100] * start + target_tokens
    weights = _aligned_weights(
        inputs.get("weights"),
        sequence_length=len(tokens),
        target_length=len(target_tokens),
        start=start,
        default=[0.0] * start + [1.0] * len(target_tokens),
        name="weights",
    )
    labels = [label if weight > 0 else -100 for label, weight in zip(labels, weights)]
    return SupervisedBatchItem(
        tokens=tokens,
        labels=labels,
        weights=weights,
        prompt_length=start,
        target_length=len(target_tokens),
    )


def _prepare_importance_sampling_item(raw_datum: Any) -> ImportanceSamplingBatchItem:
    datum = _datum(raw_datum)
    tokens = datum.model_input.to_ints()
    if len(tokens) < 2:
        raise BadRequestError("importance_sampling requires at least two model_input tokens")

    inputs = datum.loss_fn_inputs
    completion_tokens = _int_list(inputs.get("target_tokens"), name="target_tokens", required=True)
    if not completion_tokens:
        raise BadRequestError("importance_sampling requires non-empty completion target_tokens")
    if len(completion_tokens) >= len(tokens):
        raise BadRequestError(
            "importance_sampling expects model_input to contain prompt tokens before "
            "completion target_tokens"
        )
    if len(completion_tokens) > len(tokens):
        raise BadRequestError(
            "target_tokens length cannot exceed model_input token length "
            f"({len(completion_tokens)} > {len(tokens)})"
        )

    start = len(tokens) - len(completion_tokens)
    labels = [-100] * start + completion_tokens
    old_logprobs_value = inputs.get("old_logprobs", inputs.get("logprobs"))
    old_logprobs = _float_list(
        old_logprobs_value,
        name="old_logprobs",
        required=True,
        expected_length=len(completion_tokens),
    )
    advantages = _float_list(
        inputs.get("advantages"),
        name="advantages",
        required=True,
        expected_length=len(completion_tokens),
    )
    weights = _completion_weights(
        inputs.get("weights"),
        sequence_length=len(tokens),
        target_length=len(completion_tokens),
        name="weights",
    )
    masks = _float_list(
        inputs.get("masks"),
        name="masks",
        required=False,
        expected_length=len(completion_tokens),
        default=[1.0] * len(completion_tokens),
    )
    ref_logprobs = None
    if inputs.get("ref_logprobs") is not None:
        ref_logprobs = _float_list(
            inputs.get("ref_logprobs"),
            name="ref_logprobs",
            required=True,
            expected_length=len(completion_tokens),
        )
    labels = [label if index >= start else -100 for index, label in enumerate(labels)]
    return ImportanceSamplingBatchItem(
        tokens=tokens,
        labels=labels,
        completion_tokens=completion_tokens,
        old_logprobs=old_logprobs,
        advantages=advantages,
        weights=weights,
        masks=masks,
        ref_logprobs=ref_logprobs,
    )


def _prepare_dpo_item(raw_datum: Any) -> DPOBatchItem:
    datum = _datum(raw_datum)
    chosen_tokens = datum.model_input.to_ints()
    if len(chosen_tokens) < 2:
        raise BadRequestError("dpo requires chosen model_input with at least two tokens")

    inputs = datum.loss_fn_inputs
    chosen_target_tokens = _int_list(inputs.get("target_tokens"), name="target_tokens", required=True)
    if not chosen_target_tokens:
        raise BadRequestError("dpo requires non-empty chosen target_tokens")
    if len(chosen_target_tokens) >= len(chosen_tokens):
        raise BadRequestError("dpo expects prompt tokens before chosen target_tokens")

    rejected_tokens = _int_list(inputs.get("rejected_tokens"), name="rejected_tokens", required=True)
    if len(rejected_tokens) < 2:
        raise BadRequestError("dpo requires rejected_tokens with at least two tokens")

    prompt_length = len(chosen_tokens) - len(chosen_target_tokens)
    rejected_target_tokens = _int_list(
        inputs.get("rejected_target_tokens"),
        name="rejected_target_tokens",
        required=False,
    )
    if not rejected_target_tokens:
        if len(rejected_tokens) <= prompt_length:
            raise BadRequestError("rejected_tokens must include completion tokens after the shared prompt")
        rejected_target_tokens = rejected_tokens[prompt_length:]
    if len(rejected_target_tokens) >= len(rejected_tokens):
        raise BadRequestError("dpo expects prompt tokens before rejected target tokens")

    chosen_start = len(chosen_tokens) - len(chosen_target_tokens)
    rejected_start = len(rejected_tokens) - len(rejected_target_tokens)
    if chosen_start != rejected_start:
        raise BadRequestError(
            "dpo chosen and rejected examples must share the same prompt length; "
            f"got {chosen_start} and {rejected_start}"
        )

    chosen_weights = _aligned_weights(
        inputs.get("chosen_weights", inputs.get("weights")),
        sequence_length=len(chosen_tokens),
        target_length=len(chosen_target_tokens),
        start=chosen_start,
        default=[0.0] * chosen_start + [1.0] * len(chosen_target_tokens),
        name="chosen_weights",
    )
    rejected_weights = _aligned_weights(
        inputs.get("rejected_weights"),
        sequence_length=len(rejected_tokens),
        target_length=len(rejected_target_tokens),
        start=rejected_start,
        default=[0.0] * rejected_start + [1.0] * len(rejected_target_tokens),
        name="rejected_weights",
    )
    return DPOBatchItem(
        chosen_tokens=chosen_tokens,
        chosen_labels=[
            token if weight > 0 else -100
            for token, weight in zip([-100] * chosen_start + chosen_target_tokens, chosen_weights)
        ],
        chosen_weights=chosen_weights,
        rejected_tokens=rejected_tokens,
        rejected_labels=[
            token if weight > 0 else -100
            for token, weight in zip([-100] * rejected_start + rejected_target_tokens, rejected_weights)
        ],
        rejected_weights=rejected_weights,
        prompt_length=chosen_start,
        beta=float(inputs.get("beta", 0.1)),
        reference_chosen_logprob=float(inputs.get("reference_chosen_logprob", 0.0)),
        reference_rejected_logprob=float(inputs.get("reference_rejected_logprob", 0.0)),
    )


def _datum(value: Any) -> types.Datum:
    if isinstance(value, types.Datum):
        return value
    return types.Datum.model_validate(value)


def _aligned_weights(
    value: Any,
    *,
    sequence_length: int,
    target_length: int,
    start: int,
    default: list[float],
    name: str,
) -> list[float]:
    if value is None:
        return list(default)
    weights = _float_list(value, name=name, required=True)
    if len(weights) == sequence_length:
        return weights
    if len(weights) == target_length:
        return [0.0] * start + weights
    raise BadRequestError(
        f"{name} length must match model_input length ({sequence_length}) or "
        f"target_tokens length ({target_length}); got {len(weights)}"
    )


def _completion_weights(
    value: Any,
    *,
    sequence_length: int,
    target_length: int,
    name: str,
) -> list[float]:
    if value is None:
        return [1.0] * target_length
    weights = _float_list(value, name=name, required=True)
    if len(weights) == target_length:
        return weights
    if len(weights) == sequence_length:
        return weights[-target_length:]
    raise BadRequestError(
        f"{name} length must match completion target_tokens length ({target_length}) "
        f"or model_input length ({sequence_length}); got {len(weights)}"
    )


def _int_list(value: Any, *, name: str, required: bool) -> list[int]:
    values = _list_value(value, name=name, required=required)
    return [int(item) for item in values]


def _float_list(
    value: Any,
    *,
    name: str,
    required: bool,
    expected_length: int | None = None,
    default: list[float] | None = None,
) -> list[float]:
    if value is None and default is not None:
        values = list(default)
    else:
        values = [float(item) for item in _list_value(value, name=name, required=required)]
    if expected_length is not None and len(values) != expected_length:
        raise BadRequestError(
            f"{name} length must equal target_tokens length ({expected_length}); "
            f"got {len(values)}"
        )
    return values


def _list_value(value: Any, *, name: str, required: bool) -> list[Any]:
    if value is None:
        if required:
            raise BadRequestError(f"{name} is required")
        return []
    if isinstance(value, types.ModelInput):
        return value.to_ints()
    if isinstance(value, types.TensorData):
        return list(value.data)
    if isinstance(value, dict):
        if "chunks" in value:
            return types.ModelInput.model_validate(value).to_ints()
        if "data" in value:
            return list(value["data"])
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    raise BadRequestError(f"{name} must be a 1D list, TensorData, or ModelInput")
