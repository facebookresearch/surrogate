# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compute stated multivariate RACE fidelity with centered RV coefficients.

Hosted top-k outputs follow the paper Figure 13 implementation: each model pair
is evaluated on rows where all six pairwise margins are finite for both models.
Coverage is emitted explicitly because these complete cases are pair-specific.
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

LABELS: tuple[str, ...] = ("a", "b", "c", "d")


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
    vectors: np.ndarray = _pair_vectors(original_frame, representation) - _pair_vectors(
        ablated_frame, representation
    )
    index: pd.MultiIndex = pd.MultiIndex.from_frame(merged[["prompt_idx", "seg_idx"]])
    return pd.DataFrame(vectors, index=index)


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
    unique_clusters: np.ndarray = np.unique(clusters)
    grouped: list[
        tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    for cluster in unique_clusters:
        mask: np.ndarray = clusters == cluster
        xc: np.ndarray = x[mask]
        yc: np.ndarray = y[mask]
        grouped.append(
            (
                len(xc),
                xc.sum(axis=0),
                yc.sum(axis=0),
                xc.T @ xc,
                yc.T @ yc,
                xc.T @ yc,
            )
        )
    counts: np.ndarray = np.asarray([group[0] for group in grouped])
    sums_x: np.ndarray = np.stack([group[1] for group in grouped])
    sums_y: np.ndarray = np.stack([group[2] for group in grouped])
    sums_xx: np.ndarray = np.stack([group[3] for group in grouped])
    sums_yy: np.ndarray = np.stack([group[4] for group in grouped])
    sums_xy: np.ndarray = np.stack([group[5] for group in grouped])
    samples: list[float] = []
    for _ in range(n_resamples):
        chosen: np.ndarray = rng.integers(0, len(unique_clusters), len(unique_clusters))
        n: int = int(counts[chosen].sum())
        sum_x: np.ndarray = sums_x[chosen].sum(axis=0)
        sum_y: np.ndarray = sums_y[chosen].sum(axis=0)
        xx: np.ndarray = sums_xx[chosen].sum(axis=0) - np.outer(sum_x, sum_x) / n
        yy: np.ndarray = sums_yy[chosen].sum(axis=0) - np.outer(sum_y, sum_y) / n
        xy: np.ndarray = sums_xy[chosen].sum(axis=0) - np.outer(sum_x, sum_y) / n
        denominator: float = float(np.sqrt(np.trace(xx @ xx) * np.trace(yy @ yy)))
        if denominator > 0:
            samples.append(float(np.trace(xy @ xy.T)) / denominator)
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
    scopes: list[str],
    n_resamples: int,
    confidence: float,
    seed: int,
    cohort: tuple[str, ...] | None = PAPER_MODELS,
) -> pd.DataFrame:
    """Compute prediction and attribution RV for an explicit model cohort."""
    frame: pd.DataFrame = pd.read_csv(logodds_path, sep="\t")
    manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
    missing_columns: set[str] = {f"label_lp_{label}" for label in LABELS} - set(
        frame.columns
    )
    if missing_columns:
        raise ValueError(f"Missing RACE label columns {sorted(missing_columns)}")
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
            expected_by_metric: dict[str, int] = {
                "F_pred_rv": int(manifest["prompt_idx"].nunique()),
                "F_attr_rv": len(manifest) if allowed is None else len(allowed),
            }
            for metric, signals in (
                ("F_pred_rv", predictions),
                ("F_attr_rv", scoped_attrs),
            ):
                for model_index, model_a in enumerate(models):
                    for model_b in models[model_index + 1 :]:
                        x, y, clusters = _aligned(signals[model_a], signals[model_b])
                        bootstrap_scope: str = (
                            "prediction" if metric == "F_pred_rv" else scope
                        )
                        rng: np.random.Generator = np.random.default_rng(
                            _analysis_seed(
                                seed,
                                representation,
                                metric,
                                bootstrap_scope,
                                model_a,
                                model_b,
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
                        output_rows.append(
                            {
                                "benchmark": "race",
                                "pregrouper": "sentence",
                                "scope": scope,
                                "representation": representation,
                                "aggregation": "row_pooled",
                                "model_s": model_a,
                                "model_t": model_b,
                                "metric": metric,
                                "statistic": "rv",
                                "missingness_policy": "pair_specific_complete_case",
                                "expected_observations": expected_by_metric[metric],
                                "n_observations": len(x),
                                "n_prompts": len(np.unique(clusters)),
                                "observation_coverage": (
                                    len(x) / expected_by_metric[metric]
                                ),
                                "f_point": point,
                                "f_lo": low,
                                "f_hi": high,
                            }
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
    result: pd.DataFrame = compute_race_rv(
        logodds_path,
        manifest_path,
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
    write_derived_provenance(
        output_path,
        generator_name="benchmark_scripts.race_rv",
        generator_path=__file__,
        input_paths=collect_result_inputs(
            args.results_dir,
            "race",
            "sentence",
            {"logodds": logodds_path, "manifest": manifest_path},
            cohorts[args.cohort],
        ),
        parameters={
            "scopes": list(args.scopes),
            "representations": ["all_pairs", "anchor_a"],
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "cohort": args.cohort,
            "requested_models": (
                list(cohorts[args.cohort]) if cohorts[args.cohort] is not None else None
            ),
            "output_models": output_models,
            "missingness_policy": "pair_specific_complete_case",
        },
        root_dir=args.results_dir,
        supporting_source_paths=derived_supporting_source_paths(),
    )


if __name__ == "__main__":
    main()
