# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compute RACE fidelity with centered RV coefficients.

Partially censored hosted top-k rows use a model-specific finite floor. Rows
with no observed class score remain unavailable rather than becoming an
artificial zero-margin observation.

Prediction and attribution use the six pairwise A--D margins. Representation
metrics use their scalar segment signals, and representation-to-attribution
metrics use the answer-conditioned correct-vs-rest ablation. Centered RV on
these scalar quantities is exactly Pearson r-squared, preserving the paper's
RACE convention without pretending that a scalar mechanistic signal has six
label coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd

from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    derived_supporting_source_paths,
    write_derived_provenance,
)
from benchmark_scripts.f_table import OPEN_MODELS, PAPER_MODELS
from benchmark_scripts.rv import (
    FINITE_EXTREME_ABSOLUTE_MARGIN,
    FINITE_EXTREME_MISSINGNESS_POLICY,
    FINITE_EXTREME_RELATIVE_MARGIN,
    replace_censored_label_logprobs,
)

LABELS: tuple[str, ...] = ("a", "b", "c", "d")
SCALAR_METRIC_COLUMNS: dict[str, str] = {
    "F_attn_mean_rv": "attention_mean",
    "F_attn_max_rv": "attention_max",
    "F_attn_rollout_rv": "attention_rollout",
    "F_mag_rv": "delta_norm_postnorm",
    "F_align_rv": "alignment",
}
CROSS_METRICS: dict[str, tuple[str, bool]] = {
    "F_attn_mean_to_attr_rv": ("attention_mean", False),
    "F_attn_max_to_attr_rv": ("attention_max", False),
    "F_attn_rollout_to_attr_rv": ("attention_rollout", False),
    "F_mag_to_attr_rv": ("delta_norm_postnorm", True),
    "F_align_to_attr_rv": ("alignment", False),
}


def _apply_missingness_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply finite floors and rebuild the answer-conditioned scalar score."""
    output: pd.DataFrame = frame.copy()
    label_columns: list[str] = [f"label_lp_{label}" for label in LABELS]
    for indices in output.groupby("model", sort=False).groups.values():
        output.loc[indices, label_columns] = replace_censored_label_logprobs(
            output.loc[indices, label_columns].to_numpy(dtype=float)
        )

    answer_indices: pd.Series = (
        output["answer"]
        .astype(str)
        .str.lower()
        .map({label: index for index, label in enumerate(LABELS)})
    )
    if answer_indices.isna().any():
        unknown: list[str] = sorted(
            output.loc[answer_indices.isna(), "answer"].astype(str).unique()
        )
        raise ValueError(f"Unknown RACE answers {unknown}")
    values: np.ndarray = output[label_columns].to_numpy(dtype=float)
    row_indices: np.ndarray = np.arange(len(output))
    correct_indices: np.ndarray = answer_indices.to_numpy(dtype=int)
    correct: np.ndarray = values[row_indices, correct_indices]
    alternatives: np.ndarray = values.copy()
    alternatives[row_indices, correct_indices] = -np.inf
    with np.errstate(invalid="ignore"):
        output["logodds"] = correct - np.logaddexp.reduce(alternatives, axis=1)
    return output


def _rv(x: np.ndarray, y: np.ndarray) -> float:
    """Return the RV coefficient between two centered feature matrices."""
    cross: np.ndarray = x.T @ y
    xx: np.ndarray = x.T @ x
    yy: np.ndarray = y.T @ y
    numerator: float = float(np.trace(cross @ cross.T))
    denominator: float = float(np.sqrt(np.trace(xx @ xx) * np.trace(yy @ yy)))
    return numerator / denominator if denominator > 0 else float("nan")


def _centered_rv(x: np.ndarray, y: np.ndarray) -> float:
    finite: np.ndarray = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    if finite.sum() < 3:
        return float("nan")
    x_valid: np.ndarray = x[finite]
    y_valid: np.ndarray = y[finite]
    return _rv(
        x_valid - x_valid.mean(axis=0),
        y_valid - y_valid.mean(axis=0),
    )


def _pair_vectors(frame: pd.DataFrame, representation: str) -> np.ndarray:
    labels: tuple[str, ...] = LABELS
    pairs: list[tuple[str, str]] = (
        list(itertools.combinations(labels, 2))
        if representation == "all_pairs"
        else [(labels[0], label) for label in labels[1:]]
    )
    with np.errstate(invalid="ignore"):
        return np.column_stack(
            [
                frame[f"label_lp_{first}"].to_numpy(dtype=float)
                - frame[f"label_lp_{second}"].to_numpy(dtype=float)
                for first, second in pairs
            ]
        )


def _prediction_vectors(
    frame: pd.DataFrame,
    model: str,
    representation: str,
) -> pd.DataFrame:
    rows: pd.DataFrame = frame[
        (frame["model"] == model) & (frame["kind"] == "orig")
    ].sort_values("prompt_idx")
    vectors: np.ndarray = _pair_vectors(rows, representation)
    output: pd.DataFrame = pd.DataFrame(vectors, index=rows["prompt_idx"])
    output.index.name = "prompt_idx"
    return output


def _attribution_vectors(
    frame: pd.DataFrame,
    model: str,
    representation: str,
) -> pd.DataFrame:
    rows: pd.DataFrame = frame[frame["model"] == model]
    original: pd.DataFrame = rows[rows["kind"] == "orig"]
    ablated: pd.DataFrame = rows[rows["kind"] == "ablated"]
    label_columns: list[str] = [f"label_lp_{label}" for label in LABELS]
    merged: pd.DataFrame = ablated.merge(
        original[["prompt_idx", *label_columns]],
        on="prompt_idx",
        how="inner",
        suffixes=("_ablated", "_original"),
        validate="many_to_one",
    )
    original_frame: pd.DataFrame = merged[
        [f"{column}_original" for column in label_columns]
    ].rename(columns={f"{column}_original": column for column in label_columns})
    ablated_frame: pd.DataFrame = merged[
        [f"{column}_ablated" for column in label_columns]
    ].rename(columns={f"{column}_ablated": column for column in label_columns})
    with np.errstate(invalid="ignore"):
        vectors: np.ndarray = _pair_vectors(
            original_frame, representation
        ) - _pair_vectors(ablated_frame, representation)
    index: pd.MultiIndex = pd.MultiIndex.from_frame(merged[["prompt_idx", "seg_idx"]])
    return pd.DataFrame(vectors, index=index)


def _canonical_attribution(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Return answer-conditioned correct-vs-rest ablations for one model."""
    rows: pd.DataFrame = frame[frame["model"] == model]
    original: pd.DataFrame = rows[rows["kind"] == "orig"][
        ["prompt_idx", "logodds"]
    ].rename(columns={"logodds": "original"})
    ablated: pd.DataFrame = rows[rows["kind"] == "ablated"][
        ["prompt_idx", "seg_idx", "logodds"]
    ].rename(columns={"logodds": "ablated"})
    merged: pd.DataFrame = ablated.merge(
        original,
        on="prompt_idx",
        how="inner",
        validate="many_to_one",
    )
    index: pd.MultiIndex = pd.MultiIndex.from_frame(merged[["prompt_idx", "seg_idx"]])
    with np.errstate(invalid="ignore"):
        values: np.ndarray = (
            merged["original"].to_numpy() - merged["ablated"].to_numpy()
        )
    return pd.DataFrame({"value": values}, index=index)


def _segment_signal(
    frame: pd.DataFrame,
    model: str,
    column: str,
    allowed: set[tuple[int, int]] | None,
) -> pd.DataFrame | None:
    """Return one scalar white-box signal indexed by prompt and segment."""
    rows: pd.DataFrame = frame[frame["model"] == model]
    if rows.empty:
        return None
    index: pd.MultiIndex = pd.MultiIndex.from_frame(rows[["prompt_idx", "seg_idx"]])
    if column == "alignment":
        values: pd.Series = rows["w_dot_delta_z_postnorm"] / (
            rows["w_norm"] * rows["delta_norm_postnorm"]
        )
    else:
        values = rows[column]
    signal: pd.DataFrame = pd.DataFrame(
        {"value": values.to_numpy(dtype=float)}, index=index
    )
    if allowed is not None:
        signal = signal[signal.index.isin(allowed)]
    return None if signal["value"].isna().all() else signal


def _append_rv_result(
    output_rows: list[dict[str, str | float | int]],
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    scope: str,
    representation: str,
    model_s: str,
    model_t: str,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
    absolute_target: bool = False,
    rng_scope: str | None = None,
) -> None:
    """Append one pair-specific RV estimate and prompt-cluster interval."""
    target: pd.DataFrame = second.abs() if absolute_target else second
    common: pd.Index = first.index.intersection(target.index)
    expected_clusters: np.ndarray = np.asarray(
        [index[0] if isinstance(index, tuple) else index for index in common]
    )
    x, y, clusters = _aligned(first, target)
    rng: np.random.Generator = np.random.default_rng(
        _analysis_seed(
            seed,
            representation,
            metric,
            rng_scope or scope,
            model_s,
            model_t,
        )
    )
    point, low, high = _cluster_bootstrap_rv(
        x,
        y,
        clusters,
        n_resamples,
        confidence,
        rng,
    )
    expected_observations: int = len(common)
    expected_prompts: int = len(np.unique(expected_clusters))
    n_prompts: int = len(np.unique(clusters))
    output_rows.append(
        {
            "benchmark": "race",
            "pregrouper": "sentence",
            "scope": scope,
            "representation": representation,
            "aggregation": "row_pooled",
            "model_s": model_s,
            "model_t": model_t,
            "metric": metric,
            "statistic": "rv",
            "missingness_policy": FINITE_EXTREME_MISSINGNESS_POLICY,
            "expected_observations": expected_observations,
            "n_observations": len(x),
            "n_prompts": n_prompts,
            "expected_prompts": expected_prompts,
            "prompt_coverage": (
                n_prompts / expected_prompts if expected_prompts else float("nan")
            ),
            "observation_coverage": (
                len(x) / expected_observations
                if expected_observations
                else float("nan")
            ),
            "f_point": point,
            "f_lo": low,
            "f_hi": high,
        }
    )


def _aligned(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common: pd.Index = first.index.intersection(second.index)
    x: np.ndarray = first.loc[common].to_numpy(dtype=float)
    y: np.ndarray = second.loc[common].to_numpy(dtype=float)
    finite: np.ndarray = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    clusters: np.ndarray = np.asarray(
        [index[0] if isinstance(index, tuple) else index for index in common]
    )
    return x[finite], y[finite], clusters[finite]


def _cluster_bootstrap_rv(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Compute RV and a prompt-cluster percentile interval."""
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    point: float = _centered_rv(x, y)
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


def _analysis_seed(seed: int, *parts: str) -> int:
    """Derive an order-independent NumPy seed for one bootstrap analysis."""

    digest = hashlib.sha256("\x1f".join((str(seed), *parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def compute_race_rv(
    logodds_path: str,
    manifest_path: str,
    segments_path: str,
    scopes: list[str],
    n_resamples: int,
    confidence: float,
    seed: int,
    cohort: tuple[str, ...] | None = PAPER_MODELS,
) -> pd.DataFrame:
    """Compute black-box, white-box, and cross RACE RV fidelities."""
    frame: pd.DataFrame = pd.read_csv(logodds_path, sep="\t")
    manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
    segment_columns: set[str] = {
        "model",
        "prompt_idx",
        "seg_idx",
        "attention_mean",
        "attention_max",
        "attention_rollout",
        "w_norm",
        "delta_norm_postnorm",
        "w_dot_delta_z_postnorm",
    }
    segments: pd.DataFrame = pd.read_csv(
        segments_path,
        sep="\t",
        usecols=lambda column: column in segment_columns,
    )
    missing_columns: set[str] = {f"label_lp_{label}" for label in LABELS} - set(
        frame.columns
    )
    if missing_columns:
        raise ValueError(f"Missing RACE label columns {sorted(missing_columns)}")
    frame = _apply_missingness_policy(frame)
    available: set[str] = set(frame["model"].dropna().astype(str))
    if cohort is not None:
        missing_models: set[str] = set(cohort) - available
        if missing_models:
            raise ValueError(f"RACE log-odds omit models {sorted(missing_models)}")
        models: list[str] = list(cohort)
    else:
        models = sorted(available)
    manifest_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in manifest[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }
    output_rows: list[dict[str, str | float | int]] = []

    canonical_attributions: dict[str, pd.DataFrame] = {
        model: _canonical_attribution(frame, model) for model in models
    }

    for representation in ("all_pairs", "anchor_a"):
        predictions: dict[str, pd.DataFrame] = {
            model: _prediction_vectors(frame, model, representation) for model in models
        }
        attributions: dict[str, pd.DataFrame] = {
            model: _attribution_vectors(frame, model, representation)
            for model in models
        }
        for model, values in attributions.items():
            outside: set[tuple[int, int]] = set(values.index) - manifest_keys
            if outside:
                raise ValueError(
                    f"RACE attributions for {model} include {len(outside)} keys "
                    "outside the segment manifest"
                )
        for scope in scopes:
            allowed: set[tuple[int, int]] | None = None
            if scope != "all":
                scoped: pd.DataFrame = manifest[manifest["message_role"] == scope]
                allowed = {
                    (int(prompt_idx), int(seg_idx))
                    for prompt_idx, seg_idx in scoped[
                        ["prompt_idx", "seg_idx"]
                    ].itertuples(index=False, name=None)
                }
            scoped_attrs: dict[str, pd.DataFrame] = {
                model: values if allowed is None else values[values.index.isin(allowed)]
                for model, values in attributions.items()
            }
            for metric, signals in (
                ("F_pred_rv", predictions),
                ("F_attr_rv", scoped_attrs),
            ):
                for model_index, model_a in enumerate(models):
                    for model_b in models[model_index + 1 :]:
                        bootstrap_scope: str = (
                            "prediction" if metric == "F_pred_rv" else scope
                        )
                        _append_rv_result(
                            output_rows,
                            signals[model_a],
                            signals[model_b],
                            scope=scope,
                            representation=representation,
                            model_s=model_a,
                            model_t=model_b,
                            metric=metric,
                            n_resamples=n_resamples,
                            confidence=confidence,
                            seed=seed,
                            rng_scope=bootstrap_scope,
                        )

    representation_models: list[str] = list(models)
    for scope in scopes:
        allowed = None
        if scope != "all":
            scoped = manifest[manifest["message_role"] == scope]
            allowed = {
                (int(prompt_idx), int(seg_idx))
                for prompt_idx, seg_idx in scoped[["prompt_idx", "seg_idx"]].itertuples(
                    index=False, name=None
                )
            }
        scalar_attributions: dict[str, pd.DataFrame] = {
            model: (values if allowed is None else values[values.index.isin(allowed)])
            for model, values in canonical_attributions.items()
        }
        scalar_signals: dict[str, dict[str, pd.DataFrame]] = {}
        for metric, column in SCALAR_METRIC_COLUMNS.items():
            by_model: dict[str, pd.DataFrame] = {}
            for model in representation_models:
                signal: pd.DataFrame | None = _segment_signal(
                    segments, model, column, allowed
                )
                if signal is not None:
                    by_model[model] = signal
            scalar_signals[metric] = by_model
            for model_index, model_a in enumerate(by_model):
                for model_b in list(by_model)[model_index + 1 :]:
                    _append_rv_result(
                        output_rows,
                        by_model[model_a],
                        by_model[model_b],
                        scope=scope,
                        representation="canonical_scalar",
                        model_s=model_a,
                        model_t=model_b,
                        metric=metric,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=seed,
                    )
        for metric, (column, absolute_target) in CROSS_METRICS.items():
            source_metric: str = next(
                name
                for name, candidate in SCALAR_METRIC_COLUMNS.items()
                if candidate == column
            )
            for model_s, source in scalar_signals[source_metric].items():
                for model_t, target in scalar_attributions.items():
                    if model_s == model_t:
                        continue
                    _append_rv_result(
                        output_rows,
                        source,
                        target,
                        scope=scope,
                        representation="canonical_scalar",
                        model_s=model_s,
                        model_t=model_t,
                        metric=metric,
                        n_resamples=n_resamples,
                        confidence=confidence,
                        seed=seed,
                        absolute_target=absolute_target,
                    )
    return pd.DataFrame(output_rows)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--scopes", nargs="+", default=["all"], choices=["all", "system", "user"]
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cohort",
        choices=["paper", "open", "all"],
        default="paper",
        help="Require the complete named cohort instead of silently using a subset.",
    )
    args: argparse.Namespace = parser.parse_args()
    cohorts: dict[str, tuple[str, ...] | None] = {
        "paper": PAPER_MODELS,
        "open": OPEN_MODELS,
        "all": None,
    }
    logodds_path: str = os.path.join(args.results_dir, "race_sentence_logodds.tsv")
    manifest_path: str = os.path.join(
        args.results_dir, "race", "sentence", "segments.tsv.gz"
    )
    segments_path: str = os.path.join(args.results_dir, "race_sentence_segments.tsv")
    result: pd.DataFrame = compute_race_rv(
        logodds_path,
        manifest_path,
        segments_path,
        args.scopes,
        args.bootstrap_resamples,
        args.confidence_level,
        args.seed,
        cohorts[args.cohort],
    )
    output_path: str = args.output or os.path.join(args.results_dir, "race_rv.tsv")
    result.to_csv(output_path, sep="\t", index=False)
    output_models: list[str] = sorted(
        set(result.get("model_s", pd.Series(dtype=str)).dropna().astype(str))
        | set(result.get("model_t", pd.Series(dtype=str)).dropna().astype(str))
    )
    supporting_sources: dict[str, str] = derived_supporting_source_paths()
    supporting_sources["benchmark_scripts/rv.py"] = os.path.join(
        os.path.dirname(__file__), "rv.py"
    )
    write_derived_provenance(
        output_path,
        generator_name="benchmark_scripts.race_rv",
        generator_path=__file__,
        input_paths=collect_result_inputs(
            args.results_dir,
            "race",
            "sentence",
            {
                "logodds": logodds_path,
                "manifest": manifest_path,
                "segments": segments_path,
            },
            cohorts[args.cohort],
        ),
        parameters={
            "scopes": list(args.scopes),
            "representations": ["all_pairs", "anchor_a"],
            "scalar_representation": "canonical_scalar",
            "scalar_attribution": "correct_vs_rest",
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "cohort": args.cohort,
            "requested_models": (
                list(cohorts[args.cohort]) if cohorts[args.cohort] is not None else None
            ),
            "output_models": output_models,
            "missingness_policy": FINITE_EXTREME_MISSINGNESS_POLICY,
            "finite_extreme_relative_margin": FINITE_EXTREME_RELATIVE_MARGIN,
            "finite_extreme_absolute_margin": FINITE_EXTREME_ABSOLUTE_MARGIN,
        },
        root_dir=args.results_dir,
        supporting_source_paths=supporting_sources,
    )


if __name__ == "__main__":
    main()
