# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compact per-layer residual-stream capture and label scoring.

The capture is intentionally limited to the prediction-position residual at
each layer. Hidden vectors and the model's ordinary final-head logits are
transient inputs to scalar scoring helpers; this module provides no vector
serialization path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ResidualSlotCapture:
    """Post-final-norm residuals at the prediction position.

    Attributes:
        postnorm: Tensor shaped ``(slots, batch, hidden)``. Slot zero is the
            token embedding and slots ``1..T`` are decoder-block outputs.
        final_logits: Optional model-output logits shaped ``(batch, vocab)``.
            These are retained only transiently so the final slot uses the
            exact full-head numerical path of the ordinary scorer.
    """

    postnorm: torch.Tensor
    final_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.postnorm.ndim != 3:
            raise ValueError(
                "postnorm must have shape (slots, batch, hidden), got "
                f"{tuple(self.postnorm.shape)}"
            )
        if min(self.postnorm.shape) <= 0:
            raise ValueError("postnorm dimensions must all be nonzero")
        if self.final_logits is not None and (
            self.final_logits.ndim != 2
            or self.final_logits.shape[0] != self.postnorm.shape[1]
        ):
            raise ValueError("final_logits must have shape (batch, vocabulary)")

    @property
    def num_layers(self) -> int:
        """Return the number of decoder blocks represented by the capture."""
        return int(self.postnorm.shape[0]) - 1


@dataclass(frozen=True)
class LayerwiseLabelScores:
    """Compact label-level scalars derived from residual slots.

    Attributes:
        labels: Label order used by the last tensor dimension.
        grouped_logsumexp: Selected-logit log-sum-exp, shaped
            ``(slots, batch, labels)``.
        summed_unembedding_projection: Projection onto the sum of each
            label's unembedding rows, shaped ``(slots, batch, labels)``.
            This intentionally excludes any output-head bias.
    """

    labels: tuple[str, ...]
    grouped_logsumexp: torch.Tensor
    summed_unembedding_projection: torch.Tensor

    def __post_init__(self) -> None:
        expected_shape: tuple[int, ...] = tuple(self.grouped_logsumexp.shape)
        if len(expected_shape) != 3:
            raise ValueError("grouped_logsumexp must have three dimensions")
        if self.summed_unembedding_projection.shape != expected_shape:
            raise ValueError("score and projection tensors must have equal shapes")
        if expected_shape[-1] != len(self.labels):
            raise ValueError("label count does not match the tensor's last dimension")

    def contrast(self, positive: str, negative: str) -> torch.Tensor:
        """Return grouped-logit ``positive - negative`` at every slot.

        Args:
            positive: Name of the positive label group.
            negative: Name of the negative label group.

        Returns:
            Tensor shaped ``(slots, batch)``.

        Raises:
            ValueError: If labels are equal or either label is unavailable.
        """
        positive_index, negative_index = self._contrast_indices(positive, negative)
        return (
            self.grouped_logsumexp[..., positive_index]
            - self.grouped_logsumexp[..., negative_index]
        )

    def projection_contrast(self, positive: str, negative: str) -> torch.Tensor:
        """Return summed-unembedding ``positive - negative`` projections."""
        positive_index, negative_index = self._contrast_indices(positive, negative)
        return (
            self.summed_unembedding_projection[..., positive_index]
            - self.summed_unembedding_projection[..., negative_index]
        )

    def _contrast_indices(self, positive: str, negative: str) -> tuple[int, int]:
        if positive == negative:
            raise ValueError("positive and negative labels must differ")
        try:
            return self.labels.index(positive), self.labels.index(negative)
        except ValueError as error:
            raise ValueError(
                f"Unknown contrast label; available labels are {self.labels}"
            ) from error


def _decoder_parts(model: Any) -> tuple[Any, Any, list[Any], Any]:
    """Resolve the required Llama/Qwen2-style decoder components."""
    backbone: Any = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("model.model is required for layerwise capture")
    embedding: Any = getattr(backbone, "embed_tokens", None)
    if embedding is None:
        raise ValueError("model.model.embed_tokens is required")
    layers_value: Any = getattr(backbone, "layers", None)
    if layers_value is None:
        raise ValueError("model.model.layers is required")
    layers: list[Any] = list(layers_value)
    if not layers:
        raise ValueError("model.model.layers must contain at least one layer")
    final_norm: Any = getattr(backbone, "norm", None)
    if final_norm is None:
        raise ValueError("model.model.norm is required")
    return backbone, embedding, layers, final_norm


def _prediction_indices(
    input_ids: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """Find each batch row's last unmasked sequence position."""
    if input_ids.ndim != 2 or input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input_ids must have nonempty shape (batch, sequence)")
    if attention_mask is None:
        return torch.full(
            (input_ids.shape[0],),
            input_ids.shape[1] - 1,
            device=input_ids.device,
            dtype=torch.long,
        )
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    valid_mask: torch.Tensor = attention_mask != 0
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("every row must contain at least one unmasked token")
    positions: torch.Tensor = torch.arange(
        input_ids.shape[1], device=attention_mask.device
    ).expand_as(attention_mask)
    return positions.masked_fill(~valid_mask, -1).max(dim=1).values


def _select_prediction_position(value: Any, indices: torch.Tensor) -> torch.Tensor:
    """Reduce a hook output to its per-row prediction-position vectors."""
    tensor: Any = value[0] if isinstance(value, tuple) else value
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError(
            "hook output must begin with a (batch, sequence, hidden) tensor"
        )
    if tensor.shape[0] != indices.shape[0]:
        raise RuntimeError("hook output batch size differs from input batch size")
    row_indices: torch.Tensor = torch.arange(tensor.shape[0], device=tensor.device)
    selected: torch.Tensor = tensor[row_indices, indices.to(device=tensor.device), :]
    return selected.detach()


@torch.no_grad()
def capture_postnorm_residual_slots(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> ResidualSlotCapture:
    """Capture embedding and block residuals after applying the final norm.

    The model must use the Hugging Face Llama/Qwen2 layout: ``model.model``
    exposes ``embed_tokens``, ``layers``, and ``norm``. Only the last unmasked
    position is retained from each hook.

    Args:
        model: Loaded causal language model.
        input_ids: Integer token IDs shaped ``(batch, sequence)``.
        attention_mask: Optional mask of the same shape. Both left- and
            right-padding are supported.

    Returns:
        Post-final-norm residual slots ordered as embedding then block outputs.

    Raises:
        ValueError: If the model layout or inputs are invalid.
        RuntimeError: If a registered hook does not fire or shapes disagree.
    """
    backbone, embedding, layers, final_norm = _decoder_parts(model)
    indices: torch.Tensor = _prediction_indices(input_ids, attention_mask)
    captured: list[torch.Tensor | None] = [None] * (len(layers) + 1)
    final_norm_input: list[torch.Tensor] = []

    def make_hook(slot: int) -> Any:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if captured[slot] is not None:
                raise RuntimeError(f"residual hook for slot {slot} fired twice")
            captured[slot] = _select_prediction_position(output, indices)

        return hook

    def final_norm_pre_hook(_module: Any, args: tuple[Any, ...]) -> None:
        if final_norm_input:
            raise RuntimeError("final norm hook fired twice")
        final_norm_input.append(_select_prediction_position(args[0], indices))

    handles: list[Any] = [
        embedding.register_forward_hook(make_hook(0)),
        final_norm.register_forward_pre_hook(final_norm_pre_hook),
    ]
    handles.extend(
        layer.register_forward_hook(make_hook(index + 1))
        for index, layer in enumerate(layers)
    )
    try:
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "use_cache": False,
        }
        if attention_mask is not None:
            forward_kwargs["attention_mask"] = attention_mask
        outputs: Any = model(**forward_kwargs)
    finally:
        for handle in handles:
            handle.remove()

    missing_slots: list[int] = [
        index for index, value in enumerate(captured) if value is None
    ]
    if missing_slots:
        raise RuntimeError(f"residual hooks did not fire for slots {missing_slots}")
    if len(final_norm_input) != 1:
        raise RuntimeError("final norm pre-hook did not fire exactly once")
    residuals: list[torch.Tensor] = [value for value in captured if value is not None]
    reference_shape: tuple[int, ...] = residuals[0].shape
    if any(value.shape != reference_shape for value in residuals[1:]):
        raise RuntimeError("captured residual slots have inconsistent shapes")
    final_residual_error: float = float(
        (residuals[-1] - final_norm_input[0]).abs().max().item()
    )
    if final_residual_error > 1e-2:
        raise RuntimeError(
            "last block output differs from final-norm input by "
            f"{final_residual_error:.3e}"
        )

    norm_parameter: torch.Tensor | None = next(final_norm.parameters(), None)
    norm_device: torch.device = (
        norm_parameter.device if norm_parameter is not None else residuals[-1].device
    )
    postnorm: torch.Tensor = torch.stack(
        [final_norm(value.to(norm_device)).detach() for value in residuals], dim=0
    )
    logits: Any = getattr(outputs, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise RuntimeError("causal model output must expose rank-three logits")
    final_logits: torch.Tensor = _select_prediction_position(logits, indices)
    return ResidualSlotCapture(postnorm=postnorm, final_logits=final_logits)


def _validated_label_ids(
    label_token_ids: Mapping[str, Sequence[int]], vocab_size: int
) -> tuple[tuple[str, ...], list[list[int]]]:
    """Validate label groups and preserve their declared ordering."""
    if len(label_token_ids) < 2:
        raise ValueError("at least two label groups are required")
    labels: list[str] = []
    groups: list[list[int]] = []
    owner_by_token: dict[int, str] = {}
    for label, token_ids_value in label_token_ids.items():
        if not label:
            raise ValueError("label names must be nonempty")
        token_ids: list[int] = list(token_ids_value)
        if not token_ids:
            raise ValueError(f"label {label!r} has no token IDs")
        if len(set(token_ids)) != len(token_ids):
            raise ValueError(f"label {label!r} contains duplicate token IDs")
        for token_id in token_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("label token IDs must be integers")
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(
                    f"token ID {token_id} for label {label!r} is outside the vocabulary"
                )
            previous_owner: str | None = owner_by_token.get(token_id)
            if previous_owner is not None:
                raise ValueError(
                    f"token ID {token_id} overlaps labels {previous_owner!r} and {label!r}"
                )
            owner_by_token[token_id] = label
        labels.append(label)
        groups.append(token_ids)
    return tuple(labels), groups


@torch.no_grad()
def score_residual_slots(
    capture: ResidualSlotCapture,
    lm_head: Any,
    label_token_ids: Mapping[str, Sequence[int]],
) -> LayerwiseLabelScores:
    """Reduce transient residual slots to compact grouped-label scalars.

    Args:
        capture: Post-final-norm residual slots.
        lm_head: Output projection module exposing a rank-two ``weight`` and
            optional rank-one ``bias``.
        label_token_ids: Ordered mapping from label names to disjoint,
            nonempty token-ID sequences.

    Returns:
        Grouped selected-logit log-sum-exp scores and summed-unembedding
        projections. When the model's transient full-head output is available,
        the final slot instead aggregates its native-dtype log-softmax values.
        This exactly matches the ordinary scorer's stored log-probabilities;
        no vocabulary-sized values are serialized.

    Raises:
        ValueError: If the head or label groups are malformed.
        TypeError: If token IDs are not integers.
    """
    weight: Any = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head.weight must be a (vocabulary, hidden) tensor")
    if weight.shape[1] != capture.postnorm.shape[-1]:
        raise ValueError("lm_head hidden size does not match captured residuals")
    bias: Any = getattr(lm_head, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or bias.shape[0] != weight.shape[0]
    ):
        raise ValueError("lm_head.bias must be absent or shaped (vocabulary,)")
    labels, groups = _validated_label_ids(label_token_ids, int(weight.shape[0]))

    hidden: torch.Tensor = capture.postnorm.to(device=weight.device, dtype=weight.dtype)
    hidden_fp32: torch.Tensor = hidden.float()
    final_log_probs: torch.Tensor | None = (
        torch.log_softmax(capture.final_logits, dim=-1)
        if capture.final_logits is not None
        else None
    )
    grouped_scores: list[torch.Tensor] = []
    projections: list[torch.Tensor] = []
    for token_ids in groups:
        selected_weight: torch.Tensor = weight[token_ids]
        # Match the model's normal readout: perform the projection in the
        # checkpoint dtype, then aggregate the selected logits in fp32.
        selected_logits: torch.Tensor = (
            hidden @ selected_weight.transpose(0, 1)
        ).float()
        if bias is not None:
            selected_logits = selected_logits + bias[token_ids].float()
        grouped_score: torch.Tensor = torch.logsumexp(selected_logits, dim=-1)
        if final_log_probs is not None:
            # Preserve the full-head log-softmax's native-dtype rounding before
            # the fp32 group reduction, exactly as ordinary token TSVs do.
            grouped_score[-1] = torch.logsumexp(
                final_log_probs[:, token_ids].float(), dim=-1
            )
        grouped_scores.append(grouped_score)
        projections.append(hidden_fp32 @ selected_weight.float().sum(dim=0))
    return LayerwiseLabelScores(
        labels=labels,
        grouped_logsumexp=torch.stack(grouped_scores, dim=-1).detach(),
        summed_unembedding_projection=torch.stack(projections, dim=-1).detach(),
    )


@torch.no_grad()
def layerwise_delta_norms(
    original: ResidualSlotCapture,
    perturbed: ResidualSlotCapture,
) -> torch.Tensor:
    """Return L2 residual deltas with singleton-batch broadcasting.

    Args:
        original: Original prompt capture, normally with batch size one.
        perturbed: Capture for one or more ablated prompts.

    Returns:
        Tensor shaped ``(slots, max(original_batch, perturbed_batch))``.

    Raises:
        ValueError: If slot, hidden, device, dtype, or batch shapes disagree.
    """
    original_shape: tuple[int, ...] = tuple(original.postnorm.shape)
    perturbed_shape: tuple[int, ...] = tuple(perturbed.postnorm.shape)
    if original_shape[0] != perturbed_shape[0]:
        raise ValueError("captures have different residual-slot counts")
    if original_shape[2] != perturbed_shape[2]:
        raise ValueError("captures have different hidden sizes")
    if original.postnorm.device != perturbed.postnorm.device:
        raise ValueError("captures must be on the same device")
    if original.postnorm.dtype != perturbed.postnorm.dtype:
        raise ValueError("captures must have the same dtype")
    if original_shape[1] != perturbed_shape[1] and 1 not in (
        original_shape[1],
        perturbed_shape[1],
    ):
        raise ValueError("capture batch sizes must match or one must be singleton")
    return torch.linalg.vector_norm(
        original.postnorm - perturbed.postnorm, dim=-1
    ).detach()
