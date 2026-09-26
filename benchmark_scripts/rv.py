# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Shared centered-RV statistics for multiclass fidelity analyses."""

from __future__ import annotations

import hashlib

import numpy as np

FINITE_EXTREME_MISSINGNESS_POLICY: str = (
    "model_specific_finite_extreme_preserve_all_missing"
)
FINITE_EXTREME_RELATIVE_MARGIN: float = 0.01
FINITE_EXTREME_ABSOLUTE_MARGIN: float = 0.01


def finite_extreme_offset(values: np.ndarray) -> float | None:
    """Return the model-level offset used below the minimum finite value."""
    finite: np.ndarray = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    minimum: float = float(finite.min())
    return (
        abs(minimum) * FINITE_EXTREME_RELATIVE_MARGIN + FINITE_EXTREME_ABSOLUTE_MARGIN
    )


def replace_censored_label_logprobs(
    values: np.ndarray,
    *,
    offset: float | None = None,
) -> np.ndarray:
    """Replace partially censored label scores while preserving unusable rows.

    A negative infinity denotes a label absent from a successful hosted
    top-k response. When at least one label is observed on that row, missing
    labels receive a shared value below the model's minimum finite label
    log-probability. Rows on which every label is absent contain no relative
    class information and remain unavailable as NaN.

    Args:
        values: Two-dimensional row-by-label log-probability array for one
            model and benchmark.
        offset: Optional fixed distance below the minimum finite value. When
            omitted, use the canonical one-percent-plus-0.01 rule.

    Returns:
        A copied array with partial negative-infinity censoring replaced and
        all-label-missing rows represented as NaN.

    Raises:
        ValueError: If ``values`` is not two-dimensional or ``offset`` is
            negative.
    """
    if values.ndim != 2:
        raise ValueError("Label log-probabilities must be a two-dimensional array")
    if offset is not None and offset < 0:
        raise ValueError("Finite-extreme offset must be nonnegative")
    output: np.ndarray = np.asarray(values, dtype=float).copy()
    all_labels_missing: np.ndarray = np.isneginf(output).all(axis=1)
    output[all_labels_missing] = np.nan
    finite: np.ndarray = output[np.isfinite(output)]
    if finite.size == 0:
        return output
    resolved_offset: float | None = offset
    if resolved_offset is None:
        resolved_offset = finite_extreme_offset(output)
    if resolved_offset is None:
        return output
    floor: float = float(finite.min() - resolved_offset)
    output[np.isneginf(output)] = floor
    return output


def centered_rv(x: np.ndarray, y: np.ndarray) -> float:
    """Return the RV coefficient between two centered feature matrices."""
    finite: np.ndarray = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if finite.sum() < 3:
        return float("nan")
    x_valid: np.ndarray = x[finite]
    y_valid: np.ndarray = y[finite]
    x_centered: np.ndarray = x_valid - x_valid.mean(axis=0)
    y_centered: np.ndarray = y_valid - y_valid.mean(axis=0)
    cross: np.ndarray = x_centered.T @ y_centered
    xx: np.ndarray = x_centered.T @ x_centered
    yy: np.ndarray = y_centered.T @ y_centered
    numerator: float = float(np.trace(cross @ cross.T))
    denominator: float = float(np.sqrt(np.trace(xx @ xx) * np.trace(yy @ yy)))
    return numerator / denominator if denominator > 0 else float("nan")


def aligned(
    first: np.ndarray,
    second: np.ndarray,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter aligned feature rows to pair-specific finite observations."""
    finite: np.ndarray = np.isfinite(first).all(axis=1) & np.isfinite(second).all(
        axis=1
    )
    return first[finite], second[finite], clusters[finite]


def cluster_bootstrap_rv(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Compute centered RV and a prompt-cluster percentile interval."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    point: float = centered_rv(x, y)
    unique_clusters, cluster_inverse = np.unique(clusters, return_inverse=True)
    n_clusters: int = len(unique_clusters)
    counts: np.ndarray = np.bincount(cluster_inverse, minlength=n_clusters)

    def grouped_sum(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.bincount(
                    cluster_inverse,
                    weights=values[:, column],
                    minlength=n_clusters,
                )
                for column in range(values.shape[1])
            ]
        )

    def grouped_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                grouped_sum(left * right[:, column : column + 1])
                for column in range(right.shape[1])
            ],
            axis=2,
        )

    sums_x: np.ndarray = grouped_sum(x)
    sums_y: np.ndarray = grouped_sum(y)
    sums_xx: np.ndarray = grouped_cross(x, x)
    sums_yy: np.ndarray = grouped_cross(y, y)
    sums_xy: np.ndarray = grouped_cross(x, y)
    samples: list[float] = []
    probabilities: np.ndarray = np.full(n_clusters, 1.0 / n_clusters)
    batch_size: int = 100
    for start in range(0, n_resamples, batch_size):
        size: int = min(batch_size, n_resamples - start)
        weights: np.ndarray = rng.multinomial(
            n_clusters,
            probabilities,
            size=size,
        )
        n: np.ndarray = weights @ counts
        sum_x: np.ndarray = weights @ sums_x
        sum_y: np.ndarray = weights @ sums_y
        xx: np.ndarray = np.tensordot(weights, sums_xx, axes=(1, 0)) - (
            sum_x[:, :, None] * sum_x[:, None, :] / n[:, None, None]
        )
        yy: np.ndarray = np.tensordot(weights, sums_yy, axes=(1, 0)) - (
            sum_y[:, :, None] * sum_y[:, None, :] / n[:, None, None]
        )
        xy: np.ndarray = np.tensordot(weights, sums_xy, axes=(1, 0)) - (
            sum_x[:, :, None] * sum_y[:, None, :] / n[:, None, None]
        )
        numerator: np.ndarray = np.square(xy).sum(axis=(1, 2))
        denominator: np.ndarray = np.sqrt(
            np.square(xx).sum(axis=(1, 2)) * np.square(yy).sum(axis=(1, 2))
        )
        valid: np.ndarray = denominator > 0
        samples.extend((numerator[valid] / denominator[valid]).tolist())
    alpha: float = (1.0 - confidence) / 2.0 * 100.0
    low, high = np.percentile(np.asarray(samples), [alpha, 100.0 - alpha])
    return point, float(low), float(high)


def analysis_seed(seed: int, *parts: str) -> int:
    """Derive an order-independent NumPy seed for one RV bootstrap cell."""
    digest: bytes = hashlib.sha256(
        "\x1f".join((str(seed), *parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
