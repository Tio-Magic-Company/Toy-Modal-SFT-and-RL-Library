import pytest

from toy_modal import types
from toy_modal.backend.loss_inputs import (
    prepare_dpo_batch_items,
    prepare_importance_sampling_batch_items,
    prepare_supervised_batch_items,
    validate_training_batch,
)
from toy_modal.errors import BadRequestError


def test_supervised_labels_mask_prompt_and_zero_weight_tokens() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([10, 11, 12, 13]),
        loss_fn_inputs={
            "target_tokens": [12, 13],
            "weights": [0.0, 0.0, 1.0, 0.0],
        },
    )

    item = prepare_supervised_batch_items([datum])[0]

    assert item.labels == [-100, -100, 12, -100]
    assert item.weights == [0.0, 0.0, 1.0, 0.0]
    validate_training_batch([datum], "cross_entropy")


def test_supervised_accepts_completion_length_weights() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={"target_tokens": [3, 4], "weights": [0.25, 0.75]},
    )

    item = prepare_supervised_batch_items([datum])[0]

    assert item.labels == [-100, -100, 3, 4]
    assert item.weights == [0.0, 0.0, 0.25, 0.75]


def test_importance_sampling_requires_completion_aligned_tensors() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": [4, 5],
            "old_logprobs": [-0.5, -0.6],
            "advantages": [1.0, -0.5],
            "weights": [0.0, 0.0, 0.0, 1.0, 0.25],
            "masks": [1.0, 0.0],
            "ref_logprobs": [-0.7, -0.8],
        },
    )

    item = prepare_importance_sampling_batch_items([datum])[0]

    assert item.completion_tokens == [4, 5]
    assert item.weights == [1.0, 0.25]
    assert item.masks == [1.0, 0.0]
    validate_training_batch([datum], "importance_sampling")
    validate_training_batch([datum], "ppo")
    validate_training_batch([datum], "cispo")


def test_importance_sampling_rejects_misaligned_advantages() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3]),
        loss_fn_inputs={
            "target_tokens": [3],
            "old_logprobs": [-0.5],
            "advantages": [1.0, 2.0],
        },
    )

    with pytest.raises(BadRequestError, match="advantages length"):
        validate_training_batch([datum], "importance_sampling")


def test_importance_sampling_rejects_empty_completions() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2]),
        loss_fn_inputs={
            "target_tokens": [],
            "old_logprobs": [],
            "advantages": [],
        },
    )

    with pytest.raises(BadRequestError, match="non-empty completion"):
        validate_training_batch([datum], "importance_sampling")


def test_dpo_validates_chosen_and_rejected_completion_pairs() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={
            "target_tokens": [3, 4],
            "rejected_tokens": [1, 2, 5],
            "rejected_target_tokens": [5],
            "beta": 0.2,
        },
    )

    item = prepare_dpo_batch_items([datum])[0]

    assert item.chosen_labels == [-100, -100, 3, 4]
    assert item.rejected_labels == [-100, -100, 5]
    assert item.prompt_length == 2
    assert item.beta == 0.2
    validate_training_batch([datum], "dpo")


def test_dpo_rejects_pairs_without_shared_prompt_length() -> None:
    datum = types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4]),
        loss_fn_inputs={
            "target_tokens": [3, 4],
            "rejected_tokens": [1, 5],
            "rejected_target_tokens": [5],
        },
    )

    with pytest.raises(BadRequestError, match="same prompt length"):
        validate_training_batch([datum], "dpo")
