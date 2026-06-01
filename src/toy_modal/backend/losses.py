"""Torch loss helpers for the PEFT trainer backend."""

from __future__ import annotations

from typing import Any

from toy_modal.errors import BadRequestError


def token_logprobs_from_logits(logits: Any, labels: Any) -> tuple[Any, Any]:
    """Gather shifted token logprobs for causal LM labels.

    Returns ``(logprobs, mask)`` with shape ``[batch, seq_len - 1]``. Positions
    with label ``-100`` are masked and have a returned logprob of zero.
    """

    import torch

    if logits.ndim != 3:
        raise BadRequestError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise BadRequestError(f"labels must have shape [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise BadRequestError(
            "logits and labels sequence shapes must match; "
            f"got logits {tuple(logits.shape)} and labels {tuple(labels.shape)}"
        )

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.clamp_min(0)
    logprobs = torch.log_softmax(shifted_logits, dim=-1)
    gathered = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return gathered.masked_fill(~mask, 0.0), mask


def weighted_cross_entropy_loss(logits: Any, labels: Any, weights: Any) -> tuple[Any, dict[str, float]]:
    selected_logprobs, mask = token_logprobs_from_logits(logits, labels)
    shifted_weights = weights[:, 1:] * mask.to(weights.dtype)
    denom = shifted_weights.sum()
    if float(denom.detach().cpu()) <= 0:
        raise BadRequestError("cross_entropy has no positive-weight target tokens")
    losses = -selected_logprobs * shifted_weights
    loss = losses.sum() / denom
    return loss, {
        "cross_entropy": float(loss.detach().cpu()),
        "num_loss_tokens": float(mask.sum().detach().cpu()),
        "weight_sum": float(denom.detach().cpu()),
    }


def importance_sampling_loss(
    *,
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    weights: Any,
    masks: Any,
    ref_logprobs: Any | None = None,
    kl_coef: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    """Unclipped importance-sampling objective over completion tokens."""

    import torch

    tensors = {
        "new_logprobs": new_logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "weights": weights,
        "masks": masks,
    }
    shape = tuple(new_logprobs.shape)
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != shape:
            raise BadRequestError(f"{name} shape {tuple(tensor.shape)} does not match {shape}")
    if ref_logprobs is not None and tuple(ref_logprobs.shape) != shape:
        raise BadRequestError(f"ref_logprobs shape {tuple(ref_logprobs.shape)} does not match {shape}")
    active_weights = weights * masks
    denom = active_weights.sum()
    if float(denom.detach().cpu()) <= 0:
        raise BadRequestError("importance_sampling has no positive-weight completion tokens")

    ratio = torch.exp(new_logprobs - old_logprobs)
    objective = ratio * advantages
    policy_loss = -(objective * active_weights).sum()

    kl_loss = new_logprobs.new_tensor(0.0)
    if ref_logprobs is not None and kl_coef:
        kl_loss = ((new_logprobs - ref_logprobs) * active_weights).sum()

    loss = policy_loss + (kl_coef * kl_loss)
    return loss, {
        "loss:sum": float(loss.detach().cpu()),
        "importance_sampling": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "kl": float(kl_loss.detach().cpu()),
        "ratio_mean": float(((ratio * active_weights).sum() / denom).detach().cpu()),
        "weight_sum": float(denom.detach().cpu()),
    }


def ppo_loss(
    *,
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    weights: Any,
    masks: Any,
    clip_low_threshold: float = 0.8,
    clip_high_threshold: float = 1.2,
) -> tuple[Any, dict[str, float]]:
    """Token-wise PPO clipped surrogate objective."""

    import torch

    active_weights, denom = _validate_rl_loss_tensors(
        name="ppo",
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        weights=weights,
        masks=masks,
    )
    _validate_clip_thresholds(clip_low_threshold, clip_high_threshold)

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(clip_low_threshold, clip_high_threshold)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    objective = torch.minimum(unclipped_objective, clipped_objective)
    loss = -(objective * active_weights).sum()
    clipped = (ratio - clipped_ratio).abs() > 1e-12
    clip_fraction = (clipped.to(active_weights.dtype) * active_weights).sum() / denom
    return loss, {
        "loss:sum": float(loss.detach().cpu()),
        "ppo": float(loss.detach().cpu()),
        "ratio_mean": float(((ratio * active_weights).sum() / denom).detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "clip_low_threshold": float(clip_low_threshold),
        "clip_high_threshold": float(clip_high_threshold),
        "weight_sum": float(denom.detach().cpu()),
    }


def cispo_loss(
    *,
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    weights: Any,
    masks: Any,
    clip_low_threshold: float = 0.8,
    clip_high_threshold: float = 1.2,
) -> tuple[Any, dict[str, float]]:
    """CISPO objective using a detached clipped ratio logprob weight."""

    active_weights, denom = _validate_rl_loss_tensors(
        name="cispo",
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        weights=weights,
        masks=masks,
    )
    _validate_clip_thresholds(clip_low_threshold, clip_high_threshold)

    ratio = (new_logprobs - old_logprobs).exp()
    clipped_ratio = ratio.clamp(clip_low_threshold, clip_high_threshold)
    objective = clipped_ratio.detach() * new_logprobs * advantages
    loss = -(objective * active_weights).sum()
    clipped = (ratio - clipped_ratio).abs() > 1e-12
    clip_fraction = (clipped.to(active_weights.dtype) * active_weights).sum() / denom
    return loss, {
        "loss:sum": float(loss.detach().cpu()),
        "cispo": float(loss.detach().cpu()),
        "ratio_mean": float(((ratio * active_weights).sum() / denom).detach().cpu()),
        "clip_fraction": float(clip_fraction.detach().cpu()),
        "clip_low_threshold": float(clip_low_threshold),
        "clip_high_threshold": float(clip_high_threshold),
        "weight_sum": float(denom.detach().cpu()),
    }


def dpo_loss(
    *,
    chosen_logprobs: Any,
    rejected_logprobs: Any,
    reference_chosen_logprobs: Any,
    reference_rejected_logprobs: Any,
    beta: Any,
) -> tuple[Any, dict[str, float]]:
    """Direct Preference Optimization loss over chosen/rejected completions."""

    import torch

    shape = tuple(chosen_logprobs.shape)
    tensors = {
        "rejected_logprobs": rejected_logprobs,
        "reference_chosen_logprobs": reference_chosen_logprobs,
        "reference_rejected_logprobs": reference_rejected_logprobs,
        "beta": beta,
    }
    for name, tensor in tensors.items():
        if tuple(tensor.shape) != shape:
            raise BadRequestError(f"{name} shape {tuple(tensor.shape)} does not match {shape}")

    policy_margin = chosen_logprobs - rejected_logprobs
    reference_margin = reference_chosen_logprobs - reference_rejected_logprobs
    logits = beta * (policy_margin - reference_margin)
    loss = -torch.nn.functional.logsigmoid(logits).mean()
    preference_accuracy = (policy_margin > reference_margin).to(chosen_logprobs.dtype).mean()
    return loss, {
        "dpo": float(loss.detach().cpu()),
        "preference_accuracy": float(preference_accuracy.detach().cpu()),
        "policy_margin": float(policy_margin.mean().detach().cpu()),
        "reference_margin": float(reference_margin.mean().detach().cpu()),
        "beta": float(beta.mean().detach().cpu()),
    }


def _validate_rl_loss_tensors(
    *,
    name: str,
    new_logprobs: Any,
    old_logprobs: Any,
    advantages: Any,
    weights: Any,
    masks: Any,
) -> tuple[Any, Any]:
    tensors = {
        "new_logprobs": new_logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "weights": weights,
        "masks": masks,
    }
    shape = tuple(new_logprobs.shape)
    for tensor_name, tensor in tensors.items():
        if tuple(tensor.shape) != shape:
            raise BadRequestError(f"{tensor_name} shape {tuple(tensor.shape)} does not match {shape}")

    active_weights = weights * masks
    denom = active_weights.sum()
    if float(denom.detach().cpu()) <= 0:
        raise BadRequestError(f"{name} has no positive-weight completion tokens")
    return active_weights, denom


def _validate_clip_thresholds(low: float, high: float) -> None:
    if low < 0:
        raise BadRequestError("clip_low_threshold must be non-negative")
    if high < low:
        raise BadRequestError("clip_high_threshold must be greater than or equal to clip_low_threshold")
