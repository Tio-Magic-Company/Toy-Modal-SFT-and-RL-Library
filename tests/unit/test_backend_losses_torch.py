import math

import pytest

torch = pytest.importorskip("torch")

from toy_modal.backend.losses import (
    cispo_loss,
    importance_sampling_loss,
    ppo_loss,
    token_logprobs_from_logits,
    weighted_cross_entropy_loss,
)
from toy_modal.errors import BadRequestError


def test_importance_sampling_loss_matches_fixed_tensor_math() -> None:
    new = torch.tensor([[-0.2, -0.8, -1.4]], dtype=torch.float32)
    old = torch.tensor([[-0.3, -0.6, -1.6]], dtype=torch.float32)
    advantages = torch.tensor([[1.0, -0.5, 0.25]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 0.5, 0.0]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    ref = torch.tensor([[-0.25, -0.7, -1.2]], dtype=torch.float32)

    loss, metrics = importance_sampling_loss(
        new_logprobs=new,
        old_logprobs=old,
        advantages=advantages,
        weights=weights,
        masks=masks,
        ref_logprobs=ref,
        kl_coef=0.1,
    )

    ratios = [math.exp(0.1), math.exp(-0.2), math.exp(0.2)]
    policy_loss = -((ratios[0] * 1.0 * 1.0) + (ratios[1] * -0.5 * 0.5))
    kl = (((-0.2 + 0.25) * 1.0) + ((-0.8 + 0.7) * 0.5))
    expected = policy_loss + (0.1 * kl)

    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert metrics["loss:sum"] == pytest.approx(expected, rel=1e-6)
    assert metrics["policy_loss"] == pytest.approx(policy_loss, rel=1e-6)
    assert metrics["kl"] == pytest.approx(kl, rel=1e-6, abs=1e-6)


def test_ppo_loss_matches_fixed_tensor_math() -> None:
    new = torch.tensor([[-0.2, -1.0, -0.9]], dtype=torch.float32)
    old = torch.tensor([[-0.5, -0.4, -0.9]], dtype=torch.float32)
    advantages = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 1.0, 0.5]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float32)

    loss, metrics = ppo_loss(
        new_logprobs=new,
        old_logprobs=old,
        advantages=advantages,
        weights=weights,
        masks=masks,
        clip_low_threshold=0.8,
        clip_high_threshold=1.2,
    )

    ratios = [math.exp(0.3), math.exp(-0.6), math.exp(0.0)]
    clipped = [min(max(value, 0.8), 1.2) for value in ratios]
    objectives = [
        min(ratios[0] * 1.0, clipped[0] * 1.0),
        min(ratios[1] * -2.0, clipped[1] * -2.0),
        min(ratios[2] * 0.5, clipped[2] * 0.5),
    ]
    expected = -((objectives[0] * 1.0) + (objectives[1] * 1.0))

    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert metrics["loss:sum"] == pytest.approx(expected, rel=1e-6)
    assert metrics["ppo"] == pytest.approx(expected, rel=1e-6)
    assert metrics["clip_fraction"] == pytest.approx(1.0, rel=1e-6)


def test_cispo_loss_matches_fixed_tensor_math_and_detached_ratio_gradient() -> None:
    new = torch.tensor([[-0.2, -1.0]], dtype=torch.float32, requires_grad=True)
    old = torch.tensor([[-0.5, -0.4]], dtype=torch.float32)
    advantages = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    weights = torch.tensor([[1.0, 0.5]], dtype=torch.float32)
    masks = torch.tensor([[1.0, 1.0]], dtype=torch.float32)

    loss, metrics = cispo_loss(
        new_logprobs=new,
        old_logprobs=old,
        advantages=advantages,
        weights=weights,
        masks=masks,
        clip_low_threshold=0.8,
        clip_high_threshold=1.2,
    )

    ratios = [math.exp(0.3), math.exp(-0.6)]
    clipped = [min(max(value, 0.8), 1.2) for value in ratios]
    expected = -(
        (clipped[0] * -0.2 * 1.0 * 1.0)
        + (clipped[1] * -1.0 * -2.0 * 0.5)
    )

    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert metrics["loss:sum"] == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert new.grad.tolist()[0][0] == pytest.approx(-(clipped[0] * 1.0 * 1.0), rel=1e-6)
    assert new.grad.tolist()[0][1] == pytest.approx(-(clipped[1] * -2.0 * 0.5), rel=1e-6)


def test_token_logprobs_match_manual_shifted_logits() -> None:
    logits = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 2, 0]], dtype=torch.long)

    gathered, mask = token_logprobs_from_logits(logits, labels)

    manual = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(
        -1,
        labels[:, 1:].unsqueeze(-1),
    ).squeeze(-1)
    assert torch.allclose(gathered[mask], manual[mask])


def test_cross_entropy_rejects_all_zero_weights() -> None:
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)
    weights = torch.zeros(1, 3)

    with pytest.raises(BadRequestError, match="no positive-weight"):
        weighted_cross_entropy_loss(logits, labels, weights)


def test_importance_sampling_rejects_all_zero_weights() -> None:
    values = torch.zeros(1, 2)

    with pytest.raises(BadRequestError, match="no positive-weight"):
        importance_sampling_loss(
            new_logprobs=values,
            old_logprobs=values,
            advantages=torch.ones(1, 2),
            weights=values,
            masks=torch.ones(1, 2),
        )
