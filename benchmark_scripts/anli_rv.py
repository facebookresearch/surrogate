# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Compute ANLI fidelity with centered RV over all three label margins.

Partially censored hosted top-k rows use a model-specific finite floor. Rows
with no observed class score remain unavailable rather than becoming an
artificial zero-margin observation.
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    derived_supporting_source_paths,
    write_derived_provenance,
)
from benchmark_scripts.rv import (
    FINITE_EXTREME_ABSOLUTE_MARGIN,
    FINITE_EXTREME_MISSINGNESS_POLICY,
    FINITE_EXTREME_RELATIVE_MARGIN,
    aligned,
    analysis_seed,
    cluster_bootstrap_rv,
    replace_censored_label_logprobs,
)

LABELS: tuple[str, ...] = ("entailment", "neutral", "contradiction")
BENCHMARKS: tuple[str, ...] = ("anli_r1", "anli_r2", "anli_r3")
OPEN_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
)
PAPER_MODELS: tuple[str, ...] = OPEN_MODELS + (
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)


def _apply_missingness_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the canonical finite-extreme policy independently by model."""
    output: pd.DataFrame = frame.copy()
    label_columns: list[str] = [f"label_lp_{label}" for label in LABELS]
    for indices in output.groupby("model", sort=False).groups.values():
        output.loc[indices, label_columns] = replace_censored_label_logprobs(
            output.loc[indices, label_columns].to_numpy(dtype=float)
        )
    return output


def _pair_vectors(frame: pd.DataFrame) -> np.ndarray:
    """Return the symmetric E-N, E-C, and N-C log-odds representation."""
    with np.errstate(invalid="ignore"):
        return np.column_stack(
            [
                frame[f"label_lp_{left}"].to_numpy(dtype=float)
                - frame[f"label_lp_{right}"].to_numpy(dtype=float)
                for left, right in itertools.combinations(LABELS, 2)
            ]
        )


def _model_vectors(
    frame: pd.DataFrame, model: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return prompt prediction vectors and segment ablation vectors."""
    rows: pd.DataFrame = frame[frame["model"] == model]
    original: pd.DataFrame = rows[rows["kind"] == "orig"]
    ablated: pd.DataFrame = rows[rows["kind"] == "ablated"]
    label_columns: list[str] = [f"label_lp_{label}" for label in LABELS]
    predictions: pd.DataFrame = pd.DataFrame(
        _pair_vectors(original), index=original["prompt_idx"]
    )
    predictions.index.name = "prompt_idx"
    merged: pd.DataFrame = ablated.merge(
        original[["prompt_idx", *label_columns]],
        on="prompt_idx",
        how="inner",
        suffixes=("_ablated", "_original"),
        validate="many_to_one",
    )
    original_labels: pd.DataFrame = merged[
        [f"{column}_original" for column in label_columns]
    ].rename(columns={f"{column}_original": column for column in label_columns})
    ablated_labels: pd.DataFrame = merged[
        [f"{column}_ablated" for column in label_columns]
    ].rename(columns={f"{column}_ablated": column for column in label_columns})
    index: pd.MultiIndex = pd.MultiIndex.from_frame(merged[["prompt_idx", "seg_idx"]])
    with np.errstate(invalid="ignore"):
        attribution_values: np.ndarray = _pair_vectors(original_labels) - _pair_vectors(
            ablated_labels
        )
    attributions: pd.DataFrame = pd.DataFrame(attribution_values, index=index)
    return predictions, attributions


def _append_result(
    output: list[dict[str, str | float | int]],
    first: pd.DataFrame,
    second: pd.DataFrame,
    *,
    benchmark: str,
    scope: str,
    model_s: str,
    model_t: str,
    metric: str,
    n_resamples: int,
    confidence: float,
    seed: int,
) -> None:
    """Append one pair-specific RV estimate and prompt-cluster interval."""
    common: pd.Index = first.index.intersection(second.index)
    first_values: np.ndarray = first.loc[common].to_numpy(dtype=float)
    second_values: np.ndarray = second.loc[common].to_numpy(dtype=float)
    expected_clusters: np.ndarray = np.asarray(
        [index[0] if isinstance(index, tuple) else index for index in common]
    )
    x, y, clusters = aligned(first_values, second_values, expected_clusters)
    point, low, high = cluster_bootstrap_rv(
        x,
        y,
        clusters,
        n_resamples,
        confidence,
        np.random.default_rng(
            analysis_seed(seed, benchmark, scope, metric, model_s, model_t)
        ),
    )
    expected_prompts: int = len(np.unique(expected_clusters))
    n_prompts: int = len(np.unique(clusters))
    output.append(
        {
            "benchmark": benchmark,
            "pregrouper": "sentence",
            "scope": scope,
            "representation": "all_pairwise_log_odds",
            "aggregation": "row_pooled",
            "model_s": model_s,
            "model_t": model_t,
            "metric": metric,
            "statistic": "rv",
            "missingness_policy": FINITE_EXTREME_MISSINGNESS_POLICY,
            "expected_observations": len(common),
            "n_observations": len(x),
            "expected_prompts": expected_prompts,
            "n_prompts": n_prompts,
            "observation_coverage": len(x) / len(common) if len(common) else np.nan,
            "prompt_coverage": (
                n_prompts / expected_prompts if expected_prompts else np.nan
            ),
            "f_point": point,
            "f_lo": low,
            "f_hi": high,
        }
    )


def compute_anli_rv(
    results_dir: str,
    benchmarks: list[str],
    scopes: list[str],
    n_resamples: int,
    confidence: float,
    seed: int,
    cohort: tuple[str, ...] | None = PAPER_MODELS,
) -> pd.DataFrame:
    """Compute pairwise-complete prediction and attribution RV fidelities."""
    output: list[dict[str, str | float | int]] = []
    for benchmark in benchmarks:
        logodds_path: str = os.path.join(
            results_dir, f"{benchmark}_sentence_logodds.tsv"
        )
        manifest_base: str = os.path.join(
            results_dir, benchmark, "sentence", "segments.tsv"
        )
        manifest_path: str = next(
            path
            for path in (manifest_base + ".gz", manifest_base)
            if os.path.isfile(path)
        )
        frame: pd.DataFrame = pd.read_csv(logodds_path, sep="\t")
        manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
        required: set[str] = {f"label_lp_{label}" for label in LABELS}
        missing: set[str] = required - set(frame.columns)
        if missing:
            raise ValueError(f"{benchmark} omits label columns {sorted(missing)}")
        frame = _apply_missingness_policy(frame)
        available: set[str] = set(frame["model"].dropna().astype(str))
        models: list[str] = sorted(available) if cohort is None else list(cohort)
        if cohort is not None and (missing_models := set(cohort) - available):
            raise ValueError(f"{benchmark} omits models {sorted(missing_models)}")
        vectors: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
            model: _model_vectors(frame, model) for model in models
        }
        for scope in scopes:
            allowed: pd.MultiIndex | None = None
            if scope != "all":
                allowed = pd.MultiIndex.from_frame(
                    manifest.loc[
                        manifest["message_role"] == scope,
                        ["prompt_idx", "seg_idx"],
                    ]
                )
            for model_index, model_s in enumerate(models):
                for model_t in models[model_index + 1 :]:
                    for metric, position in (("F_pred_rv", 0), ("F_attr_rv", 1)):
                        first: pd.DataFrame = vectors[model_s][position]
                        second: pd.DataFrame = vectors[model_t][position]
                        if metric == "F_attr_rv" and allowed is not None:
                            first = first[first.index.isin(allowed)]
                            second = second[second.index.isin(allowed)]
                        _append_result(
                            output,
                            first,
                            second,
                            benchmark=benchmark,
                            scope=scope,
                            model_s=model_s,
                            model_t=model_t,
                            metric=metric,
                            n_resamples=n_resamples,
                            confidence=confidence,
                            seed=seed,
                        )
    return pd.DataFrame(output)


def main() -> None:
    """Run the public ANLI multivariate analysis."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS
    )
    parser.add_argument(
        "--scopes", nargs="+", choices=["all", "system", "user"], default=["all"]
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cohort", choices=["paper", "open", "all"], default="paper")
    args: argparse.Namespace = parser.parse_args()
    cohorts: dict[str, tuple[str, ...] | None] = {
        "paper": PAPER_MODELS,
        "open": OPEN_MODELS,
        "all": None,
    }
    result: pd.DataFrame = compute_anli_rv(
        args.results_dir,
        list(args.benchmarks),
        list(args.scopes),
        args.bootstrap_resamples,
        args.confidence_level,
        args.seed,
        cohorts[args.cohort],
    )
    output_path: str = args.output or os.path.join(args.results_dir, "anli_rv.tsv")
    result.to_csv(output_path, sep="\t", index=False)
    input_paths: dict[str, str] = {}
    for benchmark in args.benchmarks:
        input_paths.update(
            collect_result_inputs(
                args.results_dir,
                benchmark,
                "sentence",
                {
                    "logodds": os.path.join(
                        args.results_dir, f"{benchmark}_sentence_logodds.tsv"
                    )
                },
                cohorts[args.cohort],
            )
        )
    supporting_sources: dict[str, str] = derived_supporting_source_paths()
    supporting_sources["benchmark_scripts/rv.py"] = os.path.join(
        os.path.dirname(__file__), "rv.py"
    )
    write_derived_provenance(
        output_path,
        generator_name="benchmark_scripts.anli_rv",
        generator_path=__file__,
        input_paths=input_paths,
        parameters={
            "benchmarks": list(args.benchmarks),
            "scopes": list(args.scopes),
            "representation": "all_pairwise_log_odds",
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "cohort": args.cohort,
            "missingness_policy": FINITE_EXTREME_MISSINGNESS_POLICY,
            "finite_extreme_relative_margin": FINITE_EXTREME_RELATIVE_MARGIN,
            "finite_extreme_absolute_margin": FINITE_EXTREME_ABSOLUTE_MARGIN,
        },
        root_dir=args.results_dir,
        supporting_source_paths=supporting_sources,
    )


if __name__ == "__main__":
    main()
