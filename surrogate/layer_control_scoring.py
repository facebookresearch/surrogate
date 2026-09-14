# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Online projection helpers for layerwise random-control captures.

This module derives only compact scalar projections from transient residual
captures. It has no hidden-vector serialization path.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from surrogate.layerwise_scoring import ResidualSlotCapture


def _validated_group_ids(
    token_id_groups: Sequence[Sequence[int]],
    vocab_size: int,
    name: str,
) -> list[list[int]]:
    """Validate a rectangular collection of nonempty token-ID groups."""
    groups: list[list[int]] = [list(group) for group in token_id_groups]
    if not groups:
        raise ValueError(f"at least one {name} token group is required")
    width: int = len(groups[0])
    if width <= 0 or any(len(group) != width for group in groups):
        raise ValueError(f"{name} token groups must be nonempty and rectangular")
    for group in groups:
        if len(set(group)) != len(group):
            raise ValueError(f"{name} token groups may not contain duplicate IDs")
        for token_id in group:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"{name} token IDs must be integers")
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"{name} token ID is outside the vocabulary")
    return groups


@torch.no_grad()
def unit_unembedding_pair_directions(
    lm_head: Any,
    token_pairs: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized one-token unembedding-difference directions.

    Args:
        lm_head: Output projection exposing a rank-two ``weight`` tensor.
        token_pairs: Ordered ``(positive, negative)`` vocabulary-ID pairs.

    Returns:
        Unit FP32 directions shaped ``(pairs, hidden)`` and their
        pre-normalization FP32 L2 norms shaped ``(pairs,)``.

    Raises:
        TypeError: If a token ID is not an integer.
        ValueError: If the head or token pairs are malformed or degenerate.
    """
    weight: Any = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head.weight must be a (vocabulary, hidden) tensor")
    pairs: list[tuple[int, int]] = list(token_pairs)
    if not pairs:
        raise ValueError("at least one unembedding token pair is required")
    if len(set(pairs)) != len(pairs):
        raise ValueError("unembedding token pairs must be unique")
    vocab_size: int = int(weight.shape[0])
    for positive, negative in pairs:
        if (
            isinstance(positive, bool)
            or isinstance(negative, bool)
            or not isinstance(positive, int)
            or not isinstance(negative, int)
        ):
            raise TypeError("unembedding token-pair IDs must be integers")
        if positive == negative:
            raise ValueError("unembedding token-pair IDs must differ")
        if not (0 <= positive < vocab_size and 0 <= negative < vocab_size):
            raise ValueError("unembedding token-pair ID is outside the vocabulary")
    positive_ids: torch.Tensor = torch.tensor(
        [pair[0] for pair in pairs], device=weight.device, dtype=torch.long
    )
    negative_ids: torch.Tensor = torch.tensor(
        [pair[1] for pair in pairs], device=weight.device, dtype=torch.long
    )
    raw: torch.Tensor = (
        weight.index_select(0, positive_ids).float()
        - weight.index_select(0, negative_ids).float()
    )
    norms: torch.Tensor = torch.linalg.vector_norm(raw, dim=1)
    if not bool(torch.isfinite(raw).all()) or not bool(torch.isfinite(norms).all()):
        raise ValueError("unembedding pair directions must be finite")
    if bool((norms <= 0).any()):
        raise ValueError("unembedding pair direction has zero norm")
    return (raw / norms[:, None]).detach(), norms.detach()


@torch.no_grad()
def seeded_isotropic_directions(
    hidden_size: int,
    seeds: Sequence[int],
) -> torch.Tensor:
    """Generate independently seeded unit Gaussian directions on CPU.

    Each seed initializes a fresh CPU ``torch.Generator``. Draw indices are
    therefore stable across batching and GPU placement. The producer records
    the torch version and every seed in its sidecar.

    Args:
        hidden_size: Dimensionality of each direction.
        seeds: Distinct nonnegative signed-64-bit seeds, one per direction.

    Returns:
        Unit FP32 directions shaped ``(len(seeds), hidden_size)`` on CPU.

    Raises:
        TypeError: If dimensions or seeds are not integers.
        ValueError: If dimensions, seeds, or generated directions are invalid.
    """
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int):
        raise TypeError("hidden_size must be an integer")
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    seed_values: list[int] = list(seeds)
    if not seed_values:
        raise ValueError("at least one isotropic seed is required")
    if len(set(seed_values)) != len(seed_values):
        raise ValueError("isotropic seeds must be distinct")
    rows: list[torch.Tensor] = []
    for seed in seed_values:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("isotropic seeds must be integers")
        if seed < 0 or seed > (2**63 - 1):
            raise ValueError("isotropic seeds must be in [0, 2**63 - 1]")
        generator: torch.Generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        row: torch.Tensor = torch.randn(
            hidden_size,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        )
        norm: torch.Tensor = torch.linalg.vector_norm(row)
        if not bool(torch.isfinite(norm)) or float(norm.item()) <= 0:
            raise ValueError("generated isotropic direction is invalid")
        rows.append(row / norm)
    return torch.stack(rows, dim=0).detach()


@torch.no_grad()
def grouped_logsumexp_contrasts(
    capture: ResidualSlotCapture,
    lm_head: Any,
    positive_token_ids: Sequence[Sequence[int]],
    negative_token_ids: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Score many grouped pseudo-label contrasts at every residual slot.

    Args:
        capture: Post-final-norm residual slots.
        lm_head: Output projection with a rank-two weight and optional bias.
        positive_token_ids: Rectangular positive token groups by draw.
        negative_token_ids: Rectangular negative token groups by draw.

    Returns:
        FP32 ``positive - negative`` grouped-logsumexp scores shaped
        ``(slots, batch, draws)``. The final slot uses the transient native-
        dtype full-head log-softmax, matching the ordinary scorer.

    Raises:
        TypeError: If token IDs are not integers.
        ValueError: If the head or groups are malformed or overlap per draw.
    """
    weight: Any = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head.weight must be a (vocabulary, hidden) tensor")
    if int(weight.shape[1]) != int(capture.postnorm.shape[-1]):
        raise ValueError("lm_head hidden size does not match captured residuals")
    bias: Any = getattr(lm_head, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.ndim != 1
        or int(bias.shape[0]) != int(weight.shape[0])
    ):
        raise ValueError("lm_head.bias must be absent or shaped (vocabulary,)")
    positive: list[list[int]] = _validated_group_ids(
        positive_token_ids, int(weight.shape[0]), "positive"
    )
    negative: list[list[int]] = _validated_group_ids(
        negative_token_ids, int(weight.shape[0]), "negative"
    )
    if len(positive) != len(negative):
        raise ValueError("positive and negative group counts must match")
    for positive_group, negative_group in zip(positive, negative):
        if set(positive_group) & set(negative_group):
            raise ValueError("positive and negative groups overlap within a draw")

    hidden: torch.Tensor = capture.postnorm.to(device=weight.device, dtype=weight.dtype)
    final_log_probs: torch.Tensor | None = (
        torch.log_softmax(capture.final_logits, dim=-1)
        if capture.final_logits is not None
        else None
    )

    def grouped_scores(groups: list[list[int]]) -> torch.Tensor:
        ids: torch.Tensor = torch.tensor(groups, device=weight.device, dtype=torch.long)
        draws: int = int(ids.shape[0])
        width: int = int(ids.shape[1])
        selected_weight: torch.Tensor = weight.index_select(0, ids.reshape(-1))
        logits: torch.Tensor = (hidden @ selected_weight.transpose(0, 1)).reshape(
            *hidden.shape[:2], draws, width
        )
        if bias is not None:
            selected_bias: torch.Tensor = bias.index_select(0, ids.reshape(-1))
            logits = logits + selected_bias.reshape(draws, width)
        scores: torch.Tensor = torch.logsumexp(logits.float(), dim=-1)
        if final_log_probs is not None:
            selected_final: torch.Tensor = final_log_probs.index_select(
                1, ids.reshape(-1)
            ).reshape(final_log_probs.shape[0], draws, width)
            scores[-1] = torch.logsumexp(selected_final.float(), dim=-1)
        return scores

    return (grouped_scores(positive) - grouped_scores(negative)).detach()


@torch.no_grad()
def summed_unembedding_group_contrast_norms(
    lm_head: Any,
    positive_token_ids: Sequence[Sequence[int]],
    negative_token_ids: Sequence[Sequence[int]],
) -> torch.Tensor:
    """Return diagnostic raw norms for summed-unembedding group contrasts."""
    weight: Any = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head.weight must be a (vocabulary, hidden) tensor")
    positive: list[list[int]] = _validated_group_ids(
        positive_token_ids, int(weight.shape[0]), "positive"
    )
    negative: list[list[int]] = _validated_group_ids(
        negative_token_ids, int(weight.shape[0]), "negative"
    )
    if len(positive) != len(negative):
        raise ValueError("positive and negative group counts must match")
    for positive_group, negative_group in zip(positive, negative):
        if set(positive_group) & set(negative_group):
            raise ValueError("positive and negative groups overlap within a draw")
    positive_ids: torch.Tensor = torch.tensor(
        positive, device=weight.device, dtype=torch.long
    )
    negative_ids: torch.Tensor = torch.tensor(
        negative, device=weight.device, dtype=torch.long
    )
    positive_sum: torch.Tensor = weight[positive_ids].float().sum(dim=1)
    negative_sum: torch.Tensor = weight[negative_ids].float().sum(dim=1)
    norms: torch.Tensor = torch.linalg.vector_norm(positive_sum - negative_sum, dim=1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 0).any()):
        raise ValueError("summed-unembedding group contrast norm is invalid")
    return norms.detach()


def _residual_delta(
    original: ResidualSlotCapture,
    perturbed: ResidualSlotCapture,
) -> torch.Tensor:
    """Validate compatible captures and return broadcast residual deltas."""
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
    return original.postnorm - perturbed.postnorm


@torch.no_grad()
def project_residual_deltas(
    original: ResidualSlotCapture,
    perturbed: ResidualSlotCapture,
    unit_directions: torch.Tensor,
) -> torch.Tensor:
    """Project post-normalization residual deltas onto unit directions.

    Args:
        original: Original prompt capture, normally with batch size one.
        perturbed: Capture for one or more segment-ablated prompts.
        unit_directions: FP32 unit directions shaped ``(directions, hidden)``.

    Returns:
        FP32 projections shaped ``(slots, batch, directions)``.

    Raises:
        ValueError: If captures or directions are incompatible or non-unit.
    """
    delta: torch.Tensor = _residual_delta(original, perturbed).float()
    if unit_directions.ndim != 2:
        raise ValueError("unit_directions must have shape (directions, hidden)")
    if unit_directions.shape[0] <= 0:
        raise ValueError("at least one unit direction is required")
    if int(unit_directions.shape[1]) != int(delta.shape[-1]):
        raise ValueError("direction hidden size differs from captured residuals")
    directions: torch.Tensor = unit_directions.to(
        device=delta.device, dtype=torch.float32
    )
    norms: torch.Tensor = torch.linalg.vector_norm(directions, dim=1)
    if not bool(torch.isfinite(directions).all()) or not bool(
        torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)
    ):
        raise ValueError("control directions must be finite and unit normalized")
    return (delta @ directions.transpose(0, 1)).detach()
