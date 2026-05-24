# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Per-segment representation-level similarity metrics.

Quantifies how much a single-segment ablation perturbs the final-layer
hidden state at the prediction position, complementing the existing
attention-based and ablation-based per-segment signals.

Three quantities per segment:

- ``delta_norm`` — ``‖z_orig - z_ablated‖₂``. By Cauchy-Schwarz this is a
  strict upper bound (modulo ``‖w‖``) on the magnitude of the log-odds
  attribution, where ``w`` is the linear-readout direction.
- ``cossim`` — ``cos(z_orig, z_ablated)``. Direction-only similarity,
  independent of magnitude.
- ``w_dot_delta_z`` — ``w · (z_orig - z_ablated)``. With ``w`` taken as
  ``Σ W[pos] - Σ W[neg]`` from ``lm_head``, this equals the actual
  log-odds attribution (``orig_logit_diff - ablated_logit_diff``)
  *exactly* when (a) we use the post-final-norm hidden state and (b)
  there is exactly one positive and one negative token. Otherwise it is
  an approximation that we log for diagnostic purposes.

Both pre-final-norm (raw residual stream) and post-final-norm (what
``lm_head`` consumes) variants are reported.

The "exact attribution" claim:

    For Llama/Qwen2-style decoders, the next-token logits are
        logits = lm_head(model.norm(z))
    where z is the final residual-stream state. So
        logit[v] = lm_head.weight[v] · model.norm(z)
    and for single-token pos/neg sets:
        log p[pos] - log p[neg] = logit[pos] - logit[neg]
                                = (W[pos] - W[neg]) · model.norm(z)
                                = w · z_postnorm
    so an ablation that takes z_postnorm → z'_postnorm produces a
    log-odds attribution of
        w · (z_postnorm - z'_postnorm)
    exactly. The unit tests in tests/test_representation_scoring.py
    check this identity numerically.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger: logging.Logger = logging.getLogger(__name__)


def _align_direction(w: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Move a readout direction onto the hidden state's shard before math."""
    return w.to(device=z.device, dtype=z.dtype)


def delta_norm(z_orig: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
    """L2 norm of (z_orig - z_pert) along the last dimension.

    Args:
        z_orig: (..., H) tensor.
        z_pert: (..., H) tensor of the same shape.

    Returns:
        Tensor of shape (...,) with the per-row L2 norm of the difference.
    """
    return torch.linalg.vector_norm(z_orig - z_pert, dim=-1)


def cosine_similarity(z_orig: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
    """Per-row cosine similarity along the last dimension.

    Args:
        z_orig: (..., H) tensor.
        z_pert: (..., H) tensor of the same shape.

    Returns:
        Tensor of shape (...,) with values in [-1, 1].
    """
    return torch.nn.functional.cosine_similarity(z_orig, z_pert, dim=-1)


def w_dot_delta(
    w: torch.Tensor,
    z_orig: torch.Tensor,
    z_pert: torch.Tensor,
) -> torch.Tensor:
    """Inner product of w with (z_orig - z_pert).

    Under a linear readout (single-token labels, post-norm hidden state),
    this equals the change in log-odds exactly. See module docstring.

    Args:
        w: (H,) direction vector.
        z_orig: (..., H) tensor.
        z_pert: (..., H) tensor of the same shape.

    Returns:
        Tensor of shape (...,) with the inner product per row.
    """
    w = _align_direction(w, z_orig)
    return ((z_orig - z_pert) * w).sum(dim=-1)


def representation_metrics_batch(
    z_orig_prenorm: torch.Tensor,
    z_orig_postnorm: torch.Tensor,
    z_pert_prenorm: torch.Tensor,
    z_pert_postnorm: torch.Tensor,
    w_postnorm: torch.Tensor,
    w_prenorm: torch.Tensor | None = None,
) -> list[dict[str, float]]:
    """Compute per-row representation-level metrics for a batch of ablations.

    All ``z_orig_*`` tensors must broadcast against the corresponding
    ``z_pert_*`` tensors (typically ``z_orig_*`` is a single (1, H) row
    that broadcasts across N ablations of shape (N, H)).

    Args:
        z_orig_prenorm: (..., H) original residual-stream hidden state.
        z_orig_postnorm: (..., H) original post-final-norm hidden state.
        z_pert_prenorm: (N, H) perturbed residual-stream hidden states.
        z_pert_postnorm: (N, H) perturbed post-final-norm hidden states.
        w_postnorm: (H,) linear-readout direction in post-norm space.
            Typically ``Σ lm_head.weight[pos] - Σ lm_head.weight[neg]``.
        w_prenorm: (H,) optional linear direction in pre-norm space. To
            recover the LayerNorm-vs-linear decomposition exactly,
            callers should pass the gamma-scaled direction
            ``gamma .* w_postnorm`` (see
            ``TransformersModel.get_unembedding_direction_prenorm``).
            Passing ``w_postnorm`` itself is also valid as a diagnostic —
            it just won't give the clean decomposition.

    Returns:
        List of dicts (one per row in the perturbation batch) with keys:
            - delta_norm_prenorm, delta_norm_postnorm
            - cossim_prenorm, cossim_postnorm
            - w_dot_delta_z_postnorm
            - z_orig_norm_prenorm, z_pert_norm_prenorm
            - z_orig_norm_postnorm, z_pert_norm_postnorm
            - w_dot_z_orig_postnorm, w_dot_z_pert_postnorm
            - w_dot_delta_z_prenorm   (only if w_prenorm is not None)
            - w_dot_z_orig_prenorm    (only if w_prenorm is not None)
            - w_dot_z_pert_prenorm    (only if w_prenorm is not None)

    The new per-vector scalars enable the additive decomposition of the
    attribution into a "linear" component and a "LayerNorm rescaling"
    component (RMSNorm form):

        a = w · (z - z')
          = sqrt(d) * [(z_pre · w_eff) / ||z_pre||
                       - (z'_pre · w_eff) / ||z'_pre||]
          = sqrt(d) * [Δp / n + p' * (n' - n) / (n * n')]
                       \\-----/   \\--------------------/
                      "linear"     "LN rescaling"

    where ``w_eff = gamma .* w_postnorm`` is the prenorm direction the
    caller should pass as ``w_prenorm``, ``n = ||z_pre||``,
    ``n' = ||z'_pre||``, ``p = z_pre · w_eff``, ``p' = z'_pre · w_eff``,
    ``Δp = p - p'``.
    """
    w_postnorm = _align_direction(w_postnorm, z_orig_postnorm)
    if w_prenorm is not None:
        w_prenorm = _align_direction(w_prenorm, z_orig_prenorm)

    dn_pre: torch.Tensor = delta_norm(z_orig_prenorm, z_pert_prenorm)
    dn_post: torch.Tensor = delta_norm(z_orig_postnorm, z_pert_postnorm)
    cs_pre: torch.Tensor = cosine_similarity(z_orig_prenorm, z_pert_prenorm)
    cs_post: torch.Tensor = cosine_similarity(z_orig_postnorm, z_pert_postnorm)
    wdz_post: torch.Tensor = w_dot_delta(w_postnorm, z_orig_postnorm, z_pert_postnorm)
    wdz_pre: torch.Tensor | None = (
        w_dot_delta(w_prenorm, z_orig_prenorm, z_pert_prenorm)
        if w_prenorm is not None
        else None
    )

    # Per-vector norms — needed for the LN/linear decomposition. Originals
    # may be (1, H) and broadcast across N rows; we read index 0 in that
    # case.
    n_orig_pre: torch.Tensor = torch.linalg.vector_norm(z_orig_prenorm, dim=-1)
    n_pert_pre: torch.Tensor = torch.linalg.vector_norm(z_pert_prenorm, dim=-1)
    n_orig_post: torch.Tensor = torch.linalg.vector_norm(z_orig_postnorm, dim=-1)
    n_pert_post: torch.Tensor = torch.linalg.vector_norm(z_pert_postnorm, dim=-1)

    # Per-vector projections onto the readout direction.
    p_orig_post: torch.Tensor = (z_orig_postnorm * w_postnorm).sum(dim=-1)
    p_pert_post: torch.Tensor = (z_pert_postnorm * w_postnorm).sum(dim=-1)
    p_orig_pre: torch.Tensor | None = None
    p_pert_pre: torch.Tensor | None = None
    if w_prenorm is not None:
        p_orig_pre = (z_orig_prenorm * w_prenorm).sum(dim=-1)
        p_pert_pre = (z_pert_prenorm * w_prenorm).sum(dim=-1)

    def _at(t: torch.Tensor, i: int) -> float:
        return float(t[0]) if t.numel() == 1 else float(t[i])

    n: int = int(dn_pre.shape[0])
    results: list[dict[str, float]] = []
    for i in range(n):
        row: dict[str, float] = {
            "delta_norm_prenorm": float(dn_pre[i]),
            "delta_norm_postnorm": float(dn_post[i]),
            "cossim_prenorm": float(cs_pre[i]),
            "cossim_postnorm": float(cs_post[i]),
            "w_dot_delta_z_postnorm": float(wdz_post[i]),
            "z_orig_norm_prenorm": _at(n_orig_pre, i),
            "z_pert_norm_prenorm": float(n_pert_pre[i]),
            "z_orig_norm_postnorm": _at(n_orig_post, i),
            "z_pert_norm_postnorm": float(n_pert_post[i]),
            "w_dot_z_orig_postnorm": _at(p_orig_post, i),
            "w_dot_z_pert_postnorm": float(p_pert_post[i]),
        }
        if wdz_pre is not None:
            row["w_dot_delta_z_prenorm"] = float(wdz_pre[i])
        if p_orig_pre is not None and p_pert_pre is not None:
            row["w_dot_z_orig_prenorm"] = _at(p_orig_pre, i)
            row["w_dot_z_pert_prenorm"] = float(p_pert_pre[i])
        results.append(row)
    return results


def w_norm(w: torch.Tensor) -> float:
    """Convenience wrapper: ``‖w‖₂`` as a Python float.

    Log this once per model to make the Cauchy-Schwarz bound dimensional:
    ``|attribution| ≤ w_norm * delta_norm_postnorm``.
    """
    return float(torch.linalg.vector_norm(w))


def extract_repr_for_dialogs(
    model: Any,  # TransformersModel
    dialogs: list[Any],  # list[Dialog]
    batch_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model on a list of dialogs and stack last-token reps.

    Mirrors the chunking logic of ``transformers_logit_difference_batch``.

    Args:
        model: A loaded TransformersModel.
        dialogs: List of Dialog objects.
        batch_size: Forward-pass batch size.

    Returns:
        (prenorm, postnorm) tensors, each shape (len(dialogs), hidden_size),
        on the model's device.
    """
    if not dialogs:
        empty: torch.Tensor = torch.empty(0)
        return empty, empty
    texts: list[str] = [model.dialog_to_text(d) for d in dialogs]
    pre_chunks: list[torch.Tensor] = []
    post_chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunk: list[str] = texts[start : start + batch_size]
        pre, post = model.get_last_layer_repr_batch(chunk)
        pre_chunks.append(pre)
        post_chunks.append(post)
    return torch.cat(pre_chunks, dim=0), torch.cat(post_chunks, dim=0)
