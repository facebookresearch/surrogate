# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compute full-coverage and complete-case multiclass RACE fidelity.

The analysis uses a shared orthonormal centered-log-ratio (CLR) basis for the
four answer labels. Open-model analysis is strict: every prompt and ablation
in the canonical segment manifest must have four finite label scores. The
paper-cohort sensitivity analysis instead uses one global complete-case set so
that every model pair is evaluated on the same selected observations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    derived_supporting_source_paths,
    write_derived_provenance,
)

LABELS: tuple[str, ...] = ("a", "b", "c", "d")
OPEN_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
)
HOSTED_MODELS: tuple[str, ...] = (
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)
PAPER_MODELS: tuple[str, ...] = OPEN_MODELS + HOSTED_MODELS
VECTOR_METRICS: tuple[str, ...] = (
    "signed_frobenius_r",
    "direction_cosine",
    "linear_cka",
)
MAGNITUDE_METRICS: tuple[str, ...] = (
    "aitchison_magnitude_r",
    "fisher_rao_magnitude_r",
    "fisher_local_magnitude_r",
)
Aggregation = Literal["row_pooled", "prompt_equal"]
MissingnessPolicy = Literal["strict_complete", "global_complete_case_mnar"]


@dataclass(frozen=True)
class ModelSignals:
    """Finite prediction and attribution signals for one model."""

    prediction: pd.DataFrame
    attribution: pd.DataFrame
    raw_prediction_keys: frozenset[int]
    raw_attribution_keys: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class ClusterMoments:
    """Per-prompt sufficient statistics for vector and scalar metrics."""

    prompt_ids: np.ndarray
    aligned_key_sha256: str
    counts: np.ndarray
    sum_x: np.ndarray
    sum_y: np.ndarray
    cross_xx: np.ndarray
    cross_yy: np.ndarray
    cross_xy: np.ndarray
    cosine_sum: np.ndarray
    cosine_count: np.ndarray
    scalar_names: tuple[str, ...]
    scalar_sum_x: np.ndarray
    scalar_sum_y: np.ndarray
    scalar_square_x: np.ndarray
    scalar_square_y: np.ndarray
    scalar_cross: np.ndarray


def _helmert_basis(n_classes: int) -> np.ndarray:
    """Return an orthonormal basis for the zero-sum class subspace."""
    if n_classes < 2:
        raise ValueError("n_classes must be at least two")
    basis: np.ndarray = np.zeros((n_classes, n_classes - 1), dtype=np.float64)
    for column in range(n_classes - 1):
        scale: float = float(np.sqrt((column + 1) * (column + 2)))
        basis[: column + 1, column] = 1.0 / scale
        basis[column + 1, column] = -(column + 1) / scale
    return basis


def _softmax(values: np.ndarray) -> np.ndarray:
    """Return a stable row-wise softmax."""
    shifted: np.ndarray = values - np.max(values, axis=1, keepdims=True)
    exponentiated: np.ndarray = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _prediction_features(log_values: np.ndarray) -> np.ndarray:
    """Map finite class log scores to shared orthonormal CLR coordinates."""
    clr: np.ndarray = log_values - log_values.mean(axis=1, keepdims=True)
    return clr @ _helmert_basis(log_values.shape[1])


def _attribution_features(
    baseline_logs: np.ndarray,
    ablated_logs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CLR attribution vectors and their three geometric magnitudes."""
    baseline_clr: np.ndarray = baseline_logs - baseline_logs.mean(axis=1, keepdims=True)
    ablated_clr: np.ndarray = ablated_logs - ablated_logs.mean(axis=1, keepdims=True)
    delta: np.ndarray = baseline_clr - ablated_clr
    vectors: np.ndarray = delta @ _helmert_basis(delta.shape[1])

    baseline_probabilities: np.ndarray = _softmax(baseline_logs)
    ablated_probabilities: np.ndarray = _softmax(ablated_logs)
    aitchison: np.ndarray = np.linalg.norm(vectors, axis=1)
    weighted_mean: np.ndarray = np.sum(baseline_probabilities * delta, axis=1)
    weighted_second_moment: np.ndarray = np.sum(
        baseline_probabilities * delta * delta, axis=1
    )
    fisher_local: np.ndarray = np.sqrt(
        np.maximum(weighted_second_moment - weighted_mean * weighted_mean, 0.0)
    )
    bhattacharyya: np.ndarray = np.sum(
        np.sqrt(baseline_probabilities * ablated_probabilities), axis=1
    )
    fisher_rao: np.ndarray = 2.0 * np.arccos(np.clip(bhattacharyya, 0.0, 1.0))
    magnitudes: np.ndarray = np.column_stack((aitchison, fisher_rao, fisher_local))
    return vectors, magnitudes


def _required_columns() -> set[str]:
    return {
        "model",
        "prompt_idx",
        "seg_idx",
        "kind",
        "answer",
        *(f"label_lp_{label}" for label in LABELS),
    }


def _load_manifest(path: str) -> pd.DataFrame:
    """Load and validate the canonical RACE segment manifest."""
    manifest: pd.DataFrame = pd.read_csv(path, sep="\t")
    required: set[str] = {
        "prompt_idx",
        "seg_idx",
        "message_role",
        "answer",
        "n_segments",
    }
    missing: set[str] = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest omits columns {sorted(missing)}")
    manifest = manifest.copy()
    manifest["prompt_idx"] = pd.to_numeric(
        manifest["prompt_idx"], errors="raise"
    ).astype(int)
    manifest["seg_idx"] = pd.to_numeric(manifest["seg_idx"], errors="raise").astype(int)
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError("Manifest contains duplicate (prompt_idx, seg_idx) keys")
    for _, rows in manifest.groupby("prompt_idx"):
        declared: set[int] = set(rows["n_segments"].astype(int))
        actual: set[int] = set(rows["seg_idx"].astype(int))
        if len(declared) != 1 or actual != set(range(next(iter(declared)))):
            raise ValueError("Manifest has incomplete or inconsistent prompt grids")
        if rows["answer"].astype(str).nunique() != 1:
            raise ValueError("Manifest has inconsistent answers within a prompt")
    roles: set[str] = set(manifest["message_role"].dropna().astype(str))
    unexpected_roles: set[str] = roles - {"system", "user"}
    if unexpected_roles:
        raise ValueError(
            f"Manifest contains unsupported roles {sorted(unexpected_roles)}"
        )
    return manifest


def _validate_answers(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    models: tuple[str, ...],
) -> None:
    """Require every model's prompt answer to match the canonical manifest."""
    expected: pd.Series = manifest.groupby("prompt_idx")["answer"].first().astype(str)
    for model in models:
        rows: pd.DataFrame = frame[frame["model"] == model]
        observed_counts: pd.Series = rows.groupby("prompt_idx")["answer"].nunique()
        if observed_counts.gt(1).any():
            raise ValueError(
                f"Model {model!r} has inconsistent answers within a prompt"
            )
        observed: pd.Series = rows.groupby("prompt_idx")["answer"].first().astype(str)
        if set(observed.index.astype(int)) - set(expected.index.astype(int)):
            raise ValueError(f"Model {model!r} has answers outside the manifest")
        expected_for_observed: pd.Series = expected.reindex(observed.index)
        if (
            expected_for_observed.isna().any()
            or not observed.eq(expected_for_observed).all()
        ):
            raise ValueError(f"Model {model!r} answers disagree with the manifest")


def _model_signals(frame: pd.DataFrame, model: str) -> ModelSignals:
    """Build finite CLR observations for one model from a log-odds table."""
    label_columns: list[str] = [f"label_lp_{label}" for label in LABELS]
    model_rows: pd.DataFrame = frame[frame["model"] == model].copy()
    if model_rows.empty:
        raise ValueError(f"RACE log-odds omit model {model!r}")

    original: pd.DataFrame = model_rows[model_rows["kind"] == "orig"].copy()
    if original.duplicated("prompt_idx").any():
        raise ValueError(f"Model {model!r} contains duplicate original rows")
    original["prompt_idx"] = pd.to_numeric(
        original["prompt_idx"], errors="raise"
    ).astype(int)
    original_values: np.ndarray = original[label_columns].to_numpy(dtype=float)
    original_finite: np.ndarray = np.isfinite(original_values).all(axis=1)
    prediction_vectors: np.ndarray = _prediction_features(
        original_values[original_finite]
    )
    prediction: pd.DataFrame = pd.DataFrame(
        prediction_vectors,
        index=original.loc[original_finite, "prompt_idx"].to_numpy(dtype=int),
        columns=[f"vector_{index}" for index in range(prediction_vectors.shape[1])],
    )
    prediction.index.name = "prompt_idx"

    ablated: pd.DataFrame = model_rows[model_rows["kind"] == "ablated"].copy()
    if ablated["seg_idx"].isna().any():
        raise ValueError(f"Model {model!r} contains an ablated row without seg_idx")
    ablated["prompt_idx"] = pd.to_numeric(ablated["prompt_idx"], errors="raise").astype(
        int
    )
    ablated["seg_idx"] = pd.to_numeric(ablated["seg_idx"], errors="raise").astype(int)
    if ablated.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError(f"Model {model!r} contains duplicate ablated rows")
    merged: pd.DataFrame = ablated.merge(
        original[["prompt_idx", *label_columns]],
        on="prompt_idx",
        how="inner",
        suffixes=("_ablated", "_original"),
        validate="many_to_one",
    )
    baseline_values: np.ndarray = merged[
        [f"{column}_original" for column in label_columns]
    ].to_numpy(dtype=float)
    ablated_values: np.ndarray = merged[
        [f"{column}_ablated" for column in label_columns]
    ].to_numpy(dtype=float)
    finite: np.ndarray = np.isfinite(baseline_values).all(axis=1) & np.isfinite(
        ablated_values
    ).all(axis=1)
    vectors, magnitudes = _attribution_features(
        baseline_values[finite], ablated_values[finite]
    )
    attribution: pd.DataFrame = pd.DataFrame(
        np.column_stack((vectors, magnitudes)),
        index=pd.MultiIndex.from_frame(
            merged.loc[finite, ["prompt_idx", "seg_idx"]],
            names=["prompt_idx", "seg_idx"],
        ),
        columns=[
            *(f"vector_{index}" for index in range(vectors.shape[1])),
            *MAGNITUDE_METRICS,
        ],
    )
    return ModelSignals(
        prediction=prediction,
        attribution=attribution,
        raw_prediction_keys=frozenset(
            int(value) for value in original["prompt_idx"].to_numpy()
        ),
        raw_attribution_keys=frozenset(
            (int(prompt_idx), int(seg_idx))
            for prompt_idx, seg_idx in ablated[["prompt_idx", "seg_idx"]].itertuples(
                index=False, name=None
            )
        ),
    )


def _validate_signal_keys(
    signals: dict[str, ModelSignals],
    manifest: pd.DataFrame,
    strict_models: set[str],
) -> None:
    """Reject keys outside the manifest and incomplete strict-model payloads."""
    prompt_keys: set[int] = set(manifest["prompt_idx"].unique())
    segment_keys: set[tuple[int, int]] = set(
        manifest[["prompt_idx", "seg_idx"]].itertuples(index=False, name=None)
    )
    for model, values in signals.items():
        extra_raw_predictions: set[int] = set(values.raw_prediction_keys) - prompt_keys
        extra_raw_attributions: set[tuple[int, int]] = (
            set(values.raw_attribution_keys) - segment_keys
        )
        if extra_raw_predictions or extra_raw_attributions:
            raise ValueError(
                f"Model {model!r} contains raw keys outside the canonical manifest"
            )
        prediction_keys: set[int] = {int(value) for value in values.prediction.index}
        attribution_keys: set[tuple[int, int]] = {
            (int(prompt_idx), int(seg_idx))
            for prompt_idx, seg_idx in values.attribution.index
        }
        extra_predictions: set[int] = prediction_keys - prompt_keys
        extra_attributions: set[tuple[int, int]] = attribution_keys - segment_keys
        if extra_predictions or extra_attributions:
            raise ValueError(
                f"Model {model!r} contains keys outside the canonical manifest"
            )
        if model in strict_models and prediction_keys != prompt_keys:
            raise ValueError(
                f"Open model {model!r} has {len(prediction_keys)}/{len(prompt_keys)} "
                "finite prediction rows"
            )
        if model in strict_models and attribution_keys != segment_keys:
            raise ValueError(
                f"Open model {model!r} has {len(attribution_keys)}/{len(segment_keys)} "
                "finite attribution rows"
            )


def _cluster_sum(
    inverse: np.ndarray, values: np.ndarray, n_clusters: int
) -> np.ndarray:
    output: np.ndarray = np.zeros((n_clusters, *values.shape[1:]), dtype=np.float64)
    np.add.at(output, inverse, values)
    return output


def _aligned_moments(
    left: pd.DataFrame,
    right: pd.DataFrame,
    allowed_keys: set[int] | set[tuple[int, int]],
    scalar_names: tuple[str, ...],
) -> ClusterMoments:
    """Create per-prompt sufficient statistics on an explicit key set."""
    common: set[object] = set(left.index) & set(right.index) & set(allowed_keys)
    keys: list[object] = sorted(common)
    if len(keys) < 3:
        raise ValueError("At least three aligned observations are required")
    left_aligned: pd.DataFrame = left.loc[keys]
    right_aligned: pd.DataFrame = right.loc[keys]
    vector_columns: list[str] = [
        column for column in left.columns if column.startswith("vector_")
    ]
    x: np.ndarray = left_aligned[vector_columns].to_numpy(dtype=float)
    y: np.ndarray = right_aligned[vector_columns].to_numpy(dtype=float)
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Aligned vector observations must be finite and shape-matched")
    prompt_values: np.ndarray = np.asarray(
        [key[0] if isinstance(key, tuple) else key for key in keys], dtype=np.int64
    )
    prompt_ids, inverse = np.unique(prompt_values, return_inverse=True)
    n_prompts: int = len(prompt_ids)
    x_norm: np.ndarray = np.linalg.norm(x, axis=1)
    y_norm: np.ndarray = np.linalg.norm(y, axis=1)
    cosine_valid: np.ndarray = (x_norm > 0.0) & (y_norm > 0.0)
    cosines: np.ndarray = np.divide(
        np.sum(x * y, axis=1),
        x_norm * y_norm,
        out=np.zeros(len(x), dtype=np.float64),
        where=cosine_valid,
    )
    scalar_x: np.ndarray = left_aligned[list(scalar_names)].to_numpy(dtype=float)
    scalar_y: np.ndarray = right_aligned[list(scalar_names)].to_numpy(dtype=float)
    if not np.isfinite(scalar_x).all() or not np.isfinite(scalar_y).all():
        raise ValueError("Aligned scalar observations must be finite")
    digest = hashlib.sha256()
    for key in keys:
        digest.update(f"{key}\n".encode())
    return ClusterMoments(
        prompt_ids=prompt_ids,
        aligned_key_sha256=digest.hexdigest(),
        counts=_cluster_sum(inverse, np.ones((len(x), 1)), n_prompts)[:, 0],
        sum_x=_cluster_sum(inverse, x, n_prompts),
        sum_y=_cluster_sum(inverse, y, n_prompts),
        cross_xx=_cluster_sum(inverse, x[:, :, None] * x[:, None, :], n_prompts),
        cross_yy=_cluster_sum(inverse, y[:, :, None] * y[:, None, :], n_prompts),
        cross_xy=_cluster_sum(inverse, x[:, :, None] * y[:, None, :], n_prompts),
        cosine_sum=_cluster_sum(inverse, cosines[:, None], n_prompts)[:, 0],
        cosine_count=_cluster_sum(
            inverse, cosine_valid.astype(float)[:, None], n_prompts
        )[:, 0],
        scalar_names=scalar_names,
        scalar_sum_x=_cluster_sum(inverse, scalar_x, n_prompts),
        scalar_sum_y=_cluster_sum(inverse, scalar_y, n_prompts),
        scalar_square_x=_cluster_sum(inverse, scalar_x * scalar_x, n_prompts),
        scalar_square_y=_cluster_sum(inverse, scalar_y * scalar_y, n_prompts),
        scalar_cross=_cluster_sum(inverse, scalar_x * scalar_y, n_prompts),
    )


def _weighted_sum(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.tensordot(weights, values, axes=(0, 0))


def _pearson_from_sums(
    count: float,
    sum_x: float,
    sum_y: float,
    square_x: float,
    square_y: float,
    cross_xy: float,
) -> float:
    covariance: float = cross_xy - sum_x * sum_y / count
    variance_x: float = square_x - sum_x * sum_x / count
    variance_y: float = square_y - sum_y * sum_y / count
    denominator: float = float(np.sqrt(max(variance_x * variance_y, 0.0)))
    return covariance / denominator if denominator > 0.0 else float("nan")


def _evaluate(
    moments: ClusterMoments,
    cluster_multiplicities: np.ndarray,
    aggregation: Aggregation,
) -> dict[str, float]:
    """Evaluate all requested metrics for one set of prompt multiplicities."""
    base_weights: np.ndarray = (
        np.ones_like(moments.counts)
        if aggregation == "row_pooled"
        else 1.0 / moments.counts
    )
    weights: np.ndarray = base_weights * cluster_multiplicities
    count: float = float(np.dot(weights, moments.counts))
    if count <= 0.0:
        return {
            metric: float("nan") for metric in (*VECTOR_METRICS, *moments.scalar_names)
        }
    sum_x: np.ndarray = _weighted_sum(moments.sum_x, weights)
    sum_y: np.ndarray = _weighted_sum(moments.sum_y, weights)
    centered_xx: np.ndarray = (
        _weighted_sum(moments.cross_xx, weights) - np.outer(sum_x, sum_x) / count
    )
    centered_yy: np.ndarray = (
        _weighted_sum(moments.cross_yy, weights) - np.outer(sum_y, sum_y) / count
    )
    centered_xy: np.ndarray = (
        _weighted_sum(moments.cross_xy, weights) - np.outer(sum_x, sum_y) / count
    )
    centered_xx = (centered_xx + centered_xx.T) / 2.0
    centered_yy = (centered_yy + centered_yy.T) / 2.0
    signed_denominator: float = float(
        np.sqrt(max(np.trace(centered_xx) * np.trace(centered_yy), 0.0))
    )
    cka_denominator: float = float(
        np.linalg.norm(centered_xx, ord="fro") * np.linalg.norm(centered_yy, ord="fro")
    )
    metrics: dict[str, float] = {
        "signed_frobenius_r": (
            float(np.trace(centered_xy)) / signed_denominator
            if signed_denominator > 0.0
            else float("nan")
        ),
        "linear_cka": (
            float(np.sum(centered_xy * centered_xy)) / cka_denominator
            if cka_denominator > 0.0
            else float("nan")
        ),
    }
    if aggregation == "prompt_equal":
        valid_prompts: np.ndarray = moments.cosine_count > 0.0
        prompt_cosines: np.ndarray = np.divide(
            moments.cosine_sum,
            moments.cosine_count,
            out=np.zeros_like(moments.cosine_sum),
            where=valid_prompts,
        )
        valid_weights: np.ndarray = cluster_multiplicities[valid_prompts]
        metrics["direction_cosine"] = (
            float(np.dot(valid_weights, prompt_cosines[valid_prompts]))
            / float(valid_weights.sum())
            if valid_weights.sum() > 0.0
            else float("nan")
        )
    else:
        cosine_count: float = float(
            np.dot(cluster_multiplicities, moments.cosine_count)
        )
        metrics["direction_cosine"] = (
            float(np.dot(cluster_multiplicities, moments.cosine_sum)) / cosine_count
            if cosine_count > 0.0
            else float("nan")
        )
    for index, name in enumerate(moments.scalar_names):
        metrics[name] = _pearson_from_sums(
            count,
            float(np.dot(weights, moments.scalar_sum_x[:, index])),
            float(np.dot(weights, moments.scalar_sum_y[:, index])),
            float(np.dot(weights, moments.scalar_square_x[:, index])),
            float(np.dot(weights, moments.scalar_square_y[:, index])),
            float(np.dot(weights, moments.scalar_cross[:, index])),
        )
    return metrics


def _pair_seed(seed: int, model_s: str, model_t: str, scope: str) -> int:
    payload: str = f"{seed}:{model_s}:{model_t}:{scope}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def _bootstrap(
    moments: ClusterMoments,
    aggregation: Aggregation,
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float, float, int]]:
    """Return point estimate, percentile interval, and valid-draw count."""
    unit_weights: np.ndarray = np.ones(len(moments.prompt_ids), dtype=float)
    point: dict[str, float] = _evaluate(moments, unit_weights, aggregation)
    samples: dict[str, list[float]] = {metric: [] for metric in point}
    rng: np.random.Generator = np.random.default_rng(seed)
    probabilities: np.ndarray = np.full(
        len(moments.prompt_ids), 1.0 / len(moments.prompt_ids)
    )
    for _ in range(n_bootstrap):
        multiplicities: np.ndarray = rng.multinomial(
            len(moments.prompt_ids), probabilities
        )
        values: dict[str, float] = _evaluate(moments, multiplicities, aggregation)
        for metric, value in values.items():
            if np.isfinite(value):
                samples[metric].append(value)
    minimum_valid: int = max(int(0.9 * n_bootstrap), 1) if n_bootstrap else 0
    output: dict[str, tuple[float, float, float, int]] = {}
    for metric, estimate in point.items():
        values = np.asarray(samples[metric], dtype=float)
        if len(values) < minimum_valid:
            raise ValueError(
                f"{metric} has only {len(values)}/{n_bootstrap} valid bootstrap draws"
            )
        if len(values):
            low, high = np.quantile(values, [0.025, 0.975])
            interval: tuple[float, float] = (float(low), float(high))
        else:
            interval = (float("nan"), float("nan"))
        output[metric] = (estimate, interval[0], interval[1], len(values))
    return output


def _statistic(metric: str) -> str:
    if metric == "direction_cosine":
        return "mean_cosine"
    if metric == "linear_cka":
        return "cka"
    return "pearson_r"


def compute_race_multiclass(
    logodds_path: str,
    manifest_path: str,
    models: tuple[str, ...],
    missingness_policy: MissingnessPolicy,
    scopes: tuple[str, ...] = ("all", "system", "user"),
    n_bootstrap: int = 500,
    seed: int = 20260717,
) -> pd.DataFrame:
    """Compute the full RACE multiclass metric table.

    Args:
        logodds_path: Consolidated RACE log-odds TSV.
        manifest_path: Canonical segment manifest TSV or TSV.GZ.
        models: Ordered model cohort; every unordered pair is emitted.
        missingness_policy: Strict full coverage or one cohort-global finite
            intersection.
        scopes: Attribution message-role scopes to evaluate.
        n_bootstrap: Number of prompt-cluster bootstrap replicates.
        seed: Root seed used to derive deterministic pair/scope seeds.

    Returns:
        Long-form table with point estimates, confidence intervals, and
        observation coverage for every required model pair.

    Raises:
        ValueError: If inputs are malformed, models are missing, open-model
            coverage is incomplete, or too few bootstrap draws are valid.
    """
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("models must contain at least two unique names")
    if missingness_policy not in {"strict_complete", "global_complete_case_mnar"}:
        raise ValueError(f"Unsupported missingness policy {missingness_policy!r}")
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    if not scopes or len(set(scopes)) != len(scopes):
        raise ValueError("scopes must contain unique values and cannot be empty")
    unknown_scopes: set[str] = set(scopes) - {"all", "system", "user"}
    if unknown_scopes:
        raise ValueError(f"Unsupported scopes {sorted(unknown_scopes)}")
    manifest: pd.DataFrame = _load_manifest(manifest_path)
    frame: pd.DataFrame = pd.read_csv(logodds_path, sep="\t")
    missing_columns: set[str] = _required_columns() - set(frame.columns)
    if missing_columns:
        raise ValueError(f"RACE log-odds omit columns {sorted(missing_columns)}")
    _validate_answers(frame, manifest, models)
    signals: dict[str, ModelSignals] = {
        model: _model_signals(frame, model) for model in models
    }
    strict_models: set[str] = (
        set(models) if missingness_policy == "strict_complete" else set(OPEN_MODELS)
    )
    _validate_signal_keys(signals, manifest, strict_models & set(models))

    prompt_keys: set[int] = {int(value) for value in manifest["prompt_idx"].unique()}
    segment_keys: set[tuple[int, int]] = set(
        manifest[["prompt_idx", "seg_idx"]].itertuples(index=False, name=None)
    )
    if missingness_policy == "global_complete_case_mnar":
        prediction_keys: set[int] = set.intersection(
            *(
                {int(value) for value in signals[model].prediction.index}
                for model in models
            )
        )
        attribution_keys: set[tuple[int, int]] = set.intersection(
            *(
                {
                    (int(prompt_idx), int(seg_idx))
                    for prompt_idx, seg_idx in signals[model].attribution.index
                }
                for model in models
            )
        )
    else:
        prediction_keys = prompt_keys
        attribution_keys = segment_keys

    rows: list[dict[str, str | int | float]] = []
    for model_index, model_s in enumerate(models):
        for model_t in models[model_index + 1 :]:
            prediction_moments: ClusterMoments = _aligned_moments(
                signals[model_s].prediction,
                signals[model_t].prediction,
                prediction_keys,
                (),
            )
            prediction_results = _bootstrap(
                prediction_moments,
                "row_pooled",
                n_bootstrap,
                _pair_seed(seed, model_s, model_t, "prediction"),
            )
            for metric in VECTOR_METRICS:
                estimate, low, high, valid = prediction_results[metric]
                rows.append(
                    _result_row(
                        missingness_policy,
                        "open" if missingness_policy == "strict_complete" else "paper",
                        "prediction",
                        "row_pooled",
                        model_s,
                        model_t,
                        metric,
                        len(prompt_keys),
                        prediction_moments,
                        estimate,
                        low,
                        high,
                        n_bootstrap,
                        valid,
                    )
                )

            for scope in scopes:
                if scope == "all":
                    scoped_keys: set[tuple[int, int]] = attribution_keys
                    seed_scope: str = "attribution"
                else:
                    role_keys: set[tuple[int, int]] = set(
                        manifest.loc[
                            manifest["message_role"].astype(str) == scope,
                            ["prompt_idx", "seg_idx"],
                        ].itertuples(index=False, name=None)
                    )
                    scoped_keys = attribution_keys & role_keys
                    seed_scope = f"attribution:{scope}"
                expected: int = (
                    len(segment_keys)
                    if scope == "all"
                    else int((manifest["message_role"].astype(str) == scope).sum())
                )
                attribution_moments: ClusterMoments = _aligned_moments(
                    signals[model_s].attribution,
                    signals[model_t].attribution,
                    scoped_keys,
                    MAGNITUDE_METRICS,
                )
                for aggregation in ("row_pooled", "prompt_equal"):
                    attribution_results = _bootstrap(
                        attribution_moments,
                        aggregation,
                        n_bootstrap,
                        _pair_seed(seed, model_s, model_t, seed_scope),
                    )
                    for metric in (*VECTOR_METRICS, *MAGNITUDE_METRICS):
                        estimate, low, high, valid = attribution_results[metric]
                        rows.append(
                            _result_row(
                                missingness_policy,
                                (
                                    "open"
                                    if missingness_policy == "strict_complete"
                                    else "paper"
                                ),
                                scope,
                                aggregation,
                                model_s,
                                model_t,
                                metric,
                                expected,
                                attribution_moments,
                                estimate,
                                low,
                                high,
                                n_bootstrap,
                                valid,
                            )
                        )
    return pd.DataFrame(rows)


def _result_row(
    missingness_policy: MissingnessPolicy,
    cohort: str,
    scope: str,
    aggregation: Aggregation,
    model_s: str,
    model_t: str,
    metric: str,
    expected_observations: int,
    moments: ClusterMoments,
    estimate: float,
    low: float,
    high: float,
    n_bootstrap: int,
    n_bootstrap_valid: int,
) -> dict[str, str | int | float]:
    """Build one long-form result row."""
    n_observations: int = int(moments.counts.sum())
    return {
        "benchmark": "race",
        "pregrouper": "sentence",
        "cohort": cohort,
        "missingness_policy": missingness_policy,
        "scope": scope,
        "aggregation": aggregation,
        "model_s": model_s,
        "model_t": model_t,
        "metric": metric,
        "statistic": _statistic(metric),
        "expected_observations": expected_observations,
        "n_observations": n_observations,
        "n_prompts": len(moments.prompt_ids),
        "observation_coverage": n_observations / expected_observations,
        "f_point": estimate,
        "f_lo": low,
        "f_hi": high,
        "n_bootstrap": n_bootstrap,
        "n_bootstrap_valid": n_bootstrap_valid,
        "aligned_key_sha256": moments.aligned_key_sha256,
    }


def main() -> None:
    """Run the command-line RACE multiclass analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cohort", choices=("open", "paper"), default="open")
    parser.add_argument("--scopes", nargs="+", default=["all", "system", "user"])
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260717)
    args: argparse.Namespace = parser.parse_args()
    cohorts: dict[str, tuple[str, ...]] = {
        "open": OPEN_MODELS,
        "paper": PAPER_MODELS,
    }
    policies: dict[str, MissingnessPolicy] = {
        "open": "strict_complete",
        "paper": "global_complete_case_mnar",
    }
    logodds_path: str = os.path.join(args.results_dir, "race_sentence_logodds.tsv")
    manifest_path: str = os.path.join(
        args.results_dir, "race", "sentence", "segments.tsv.gz"
    )
    scopes: tuple[str, ...] = tuple(str(scope) for scope in args.scopes)
    result: pd.DataFrame = compute_race_multiclass(
        logodds_path,
        manifest_path,
        cohorts[args.cohort],
        policies[args.cohort],
        scopes,
        args.bootstrap_resamples,
        args.seed,
    )
    result.to_csv(args.output, sep="\t", index=False)
    write_derived_provenance(
        args.output,
        generator_name="benchmark_scripts.race_multiclass",
        generator_path=__file__,
        input_paths=collect_result_inputs(
            args.results_dir,
            "race",
            "sentence",
            {"logodds": logodds_path, "manifest": manifest_path},
            cohorts[args.cohort],
        ),
        parameters={
            "scopes": list(scopes),
            "coordinate_system": "helmert_ilr_a_b_c_d",
            "aggregations": ["row_pooled", "prompt_equal"],
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": 0.95,
            "seed": args.seed,
            "cohort": args.cohort,
            "requested_models": list(cohorts[args.cohort]),
            "missingness_policy": policies[args.cohort],
        },
        root_dir=args.results_dir,
        supporting_source_paths=derived_supporting_source_paths(),
    )


if __name__ == "__main__":
    main()
