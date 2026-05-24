# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-segment attention scoring for transformers models.

Extracts attention weights from a single forward pass and aggregates them
to match the same per-segment shape as ablation-based scoring, enabling
direct comparison between attention-based and ablation-based importance.

Supports three aggregation strategies over heads and layers:
- **mean**: Average attention across all heads and layers.
- **max**: Maximum attention across heads and layers.
- **rollout**: Attention rollout (Abnar & Zuidema, 2020) — multiplicative
  propagation of attention through layers with residual connections.

Usage::

    from surrogate.attention_scoring import (
        attention_segment_scores,
        AttentionConfig,
    )

    scores = await attention_segment_scores(
        model=transformers_model,
        dialog=dialog,
        pregrouper_id="sentence",
        config=AttentionConfig(aggregation="rollout"),
    )
    # scores[i] = attention-based importance of segment i
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
from surrogate.transformers_model import TransformersModel
from surrogate.model_types import Dialog
from surrogate.utils import segment_text

logger: logging.Logger = logging.getLogger(__name__)

PregrouperID = Literal["word", "sentence"]


@dataclass
class AttentionConfig:
    """Configuration for attention weight aggregation."""

    aggregation: Literal["mean", "max", "rollout"] = "mean"


# ---------------------------------------------------------------------------
# Attention aggregation strategies
# ---------------------------------------------------------------------------


def _aggregate_mean(
    attentions: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool attention over all heads and layers.

    Args:
        attentions: (n_layers, n_heads, seq_len, seq_len)

    Returns:
        (seq_len, seq_len) averaged attention matrix.
    """
    return attentions.mean(dim=0).mean(dim=0)


def _aggregate_max(
    attentions: torch.Tensor,
) -> torch.Tensor:
    """
    Max-pool attention over all heads and layers.

    Args:
        attentions: (n_layers, n_heads, seq_len, seq_len)

    Returns:
        (seq_len, seq_len) max attention matrix.
    """
    # Flatten layers and heads, then max
    n_layers, n_heads, seq_len, _ = attentions.shape
    flat: torch.Tensor = attentions.reshape(n_layers * n_heads, seq_len, seq_len)
    return flat.max(dim=0).values


def _attention_rollout(
    attentions: torch.Tensor,
    total_layers: int | None = None,
) -> torch.Tensor:
    """
    Attention rollout (Abnar & Zuidema, 2020).

    Propagates attention multiplicatively through layers, accounting for
    residual connections. At each layer, the attention is averaged over
    heads, mixed 50/50 with the identity (residual), renormalized, and
    multiplied into the running product.

    For hybrid architectures where only a subset of layers use full
    attention, ``total_layers`` specifies the true depth.
    Layers without attention matrices are treated as identity (pure
    residual pass-through), which is the correct limit when a layer's
    mixing is non-attentive (linear attention, SSM, etc.).

    Args:
        attentions: (n_attn_layers, n_heads, seq_len, seq_len) — only
            the layers that produced attention matrices.
        total_layers: Total number of decoder layers in the model. When
            ``None`` (default), assumes every layer has attention
            (backward-compatible).

    Returns:
        (seq_len, seq_len) attention flow matrix. Entry (i, j) represents
        how much of token j's information flows to token i through the
        full network.

    Reference:
        Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in
        Transformers. ACL 2020.
    """
    n_attn_layers: int = attentions.shape[0]
    seq_len: int = attentions.shape[-1]
    device: torch.device = attentions.device

    if total_layers is None:
        total_layers = n_attn_layers

    eye: torch.Tensor = torch.eye(seq_len, device=device)
    rollout: torch.Tensor = eye.clone()

    # Residual-only mixing for non-attention layers: identity passed
    # through the same 0.5*A + 0.5*I formula gives 0.5*I + 0.5*I = I
    # (already normalized), so the rollout matrix is unchanged. We just
    # need to apply the attention layers at evenly spaced positions.

    # Average over heads at each attention layer
    attn_heads_avg: torch.Tensor = attentions.mean(
        dim=1
    )  # (n_attn_layers, seq_len, seq_len)

    # Non-attention layers (linear attention, SSM, etc.) are pure residual
    # which leaves the rollout unchanged, so we only iterate over the
    # layers that produced attention matrices.
    for layer_attn in attn_heads_avg:
        layer_with_residual: torch.Tensor = 0.5 * layer_attn + 0.5 * eye
        layer_with_residual = layer_with_residual / layer_with_residual.sum(
            dim=-1, keepdim=True
        )
        rollout = layer_with_residual @ rollout

    return rollout


# ---------------------------------------------------------------------------
# Segment-to-token mapping
# ---------------------------------------------------------------------------


def _map_segments_to_token_indices(
    full_text: str,
    segment_texts: list[str],
    offset_mapping: list[tuple[int, int]],
) -> list[list[int]]:
    """
    Map each segment's character span to overlapping token indices.

    Finds each segment text within the full prompt text, then identifies
    which tokens (by offset_mapping) overlap with that character span.

    Args:
        full_text: The full chat-templated prompt text.
        segment_texts: List of segment strings (from pre-grouper).
        offset_mapping: Per-token (start_char, end_char) from tokenizer.

    Returns:
        List of lists of token indices, one per segment.
    """
    result: list[list[int]] = []
    search_start: int = 0

    for segment in segment_texts:
        # Find segment in full text, searching forward from last match
        seg_start: int = full_text.find(segment, search_start)
        if seg_start == -1:
            # Fallback: search from beginning
            seg_start = full_text.find(segment)

        if seg_start == -1:
            logger.warning(f"Segment not found in text: {segment[:50]!r}")
            result.append([])
            continue

        seg_end: int = seg_start + len(segment)
        search_start = seg_end

        # Find overlapping tokens
        token_indices: list[int] = []
        for tok_idx, (tok_start, tok_end) in enumerate(offset_mapping):
            # Skip special tokens (offset 0,0) and check overlap
            if tok_start == tok_end == 0:
                continue
            if tok_end > seg_start and tok_start < seg_end:
                token_indices.append(tok_idx)

        result.append(token_indices)

    return result


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


@torch.no_grad()
async def attention_segment_scores(
    model: TransformersModel,
    dialog: Dialog,
    pregrouper_id: PregrouperID = "word",
    config: AttentionConfig | None = None,
) -> list[float]:
    """
    Compute per-segment attention scores from a transformers model.

    Runs a single forward pass with ``output_attentions=True``, aggregates
    attention weights according to the config, and maps token-level
    attention to segment-level scores using regex-based segmentation.

    The output has the same length as the number of segments produced by
    the segmenter, enabling direct comparison with ablation-based scores::

        ablation_scores = [logit_diff(ablated_i) for i in segments]
        attention_scores = await attention_segment_scores(model, dialog)
        agreement = spearman_r_score(ablation_scores, attention_scores)

    Args:
        model: A loaded TransformersModel.
        dialog: The Dialog to analyze.
        pregrouper_id: Segmentation granularity (default: "word").
        config: Attention aggregation configuration (default: mean-pooling).

    Returns:
        List of floats, one per segment. Higher values indicate the model
        pays more attention to that segment when making its prediction.
    """
    if config is None:
        config = AttentionConfig()

    model._ensure_loaded()

    # Step 1: Segment the user message with regex segmentation
    assert len(dialog.messages) >= 2, (
        f"Dialog must have at least 2 messages, got {len(dialog.messages)}"
    )
    user_text = dialog.messages[1].content
    segment_texts, _ = segment_text(user_text, level=pregrouper_id)

    if not segment_texts:
        return []

    # Step 2: Tokenize with offset mapping
    text: str = model.dialog_to_text(dialog)
    inputs: dict[str, Any] = model._tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True
    )
    input_ids: torch.Tensor = inputs["input_ids"].to(model._model.device)
    attention_mask: torch.Tensor | None = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model._model.device)

    offset_mapping: list[tuple[int, int]] = inputs["offset_mapping"][0].tolist()

    # Step 3: Forward pass with attention output
    outputs: Any = model._model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
    )

    # Stack attention: tuple of (1, n_heads, seq_len, seq_len) per
    # attention layer. For hybrid architectures this may be fewer than
    # the total number of decoder layers.
    raw_attentions: torch.Tensor = torch.stack(outputs.attentions).squeeze(1)
    n_attn_layers: int = raw_attentions.shape[0]
    total_layers: int = model._model.config.num_hidden_layers
    if n_attn_layers < total_layers:
        logger.info(
            f"Hybrid attention: {n_attn_layers}/{total_layers} layers "
            f"returned attention matrices"
        )

    attentions: torch.Tensor = raw_attentions

    # Step 4: Aggregate over heads and layers
    if config.aggregation == "rollout":
        attn_map: torch.Tensor = _attention_rollout(
            attentions, total_layers=total_layers
        )
    elif config.aggregation == "max":
        attn_map = _aggregate_max(attentions)
    else:
        attn_map = _aggregate_mean(attentions)

    # Step 5: Extract attention FROM the last token TO all input tokens
    last_token_attn: torch.Tensor = attn_map[-1]  # (seq_len,)

    # Step 6: Map segments to tokens and sum attention
    segment_token_indices: list[list[int]] = _map_segments_to_token_indices(
        text, segment_texts, offset_mapping
    )

    # Validate: every segment must map to at least one token. A segment
    # that maps to zero tokens means the segmentation doesn't
    # align with the tokenizer — e.g., a segment cuts across a token
    # boundary or the chat template altered the text. This should not
    # happen in practice, but if it does the attention scores would be
    # silently wrong, so we raise immediately.
    unmapped: list[int] = [
        i for i, indices in enumerate(segment_token_indices) if not indices
    ]
    if unmapped:
        bad_segments: list[str] = [segment_texts[i] for i in unmapped[:5]]
        raise ValueError(
            f"{len(unmapped)} of {len(segment_texts)} segments could not be "
            f"mapped to tokens. This likely means the "
            f"segmentation is misaligned with the tokenizer. "
            f"First unmapped segments: {bad_segments!r}"
        )

    scores: list[float] = []
    for token_indices in segment_token_indices:
        scores.append(last_token_attn[token_indices].sum().item())

    return scores
