# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Run-to-run noise ceiling for the canonical F-table signals.

Each paper-cohort model is re-run several times on one configuration with an
unchanged protocol. Two tables are derived:

``noise_ceiling.tsv``
    Same-model autocorrelations. For every model and every F-table signal, the
    signal from run ``a`` is correlated with the same signal from run ``b``
    using the F-table estimators (row-pooled Spearman, Pearson, Pearson r²) and
    prompt-cluster bootstrap. Runs are the published artifact and every
    repetition; unordered pairs are reported for symmetric signals. Cross-level
    ``*_to_attr`` rows correlate the mechanistic signal of run ``a`` with the
    attribution of run ``b`` for every ordered pair ``a != b``.

``noise_ceiling_spread.tsv``
    Run-to-run spread of every canonical F-table estimate for the
    configuration: the published point estimate next to each repetition's
    estimate, with the repetition mean, sample SD, and range. Point estimates
    do not depend on the bootstrap, so repetitions are not resampled.

The cluster bootstrap is single-threaded work; run with ``OMP_NUM_THREADS=1``
to avoid BLAS thread oversubscription.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    write_derived_provenance,
)
from benchmark_scripts.f_table import (
    _analysis_rng,
    _contrast_column,
    _contrast_metadata,
    _emit_pair_corrs,
    _emit_transfer,
    _mask_unsupported_signals,
    _per_prompt_logodds,
    _per_seg_ablation,
    _per_seg_align,
    _per_seg_segment_col,
    _process_benchmark,
    _readout_matches_contrast,
    _resolved_scope,
    _supporting_source_paths,
    _unsupported_model_components,
    OPEN_MODELS,
    PAPER_MODELS,
)

logger: logging.Logger = logging.getLogger(__name__)

PUBLISHED_RUN: str = "published"
CONTRAST: str = "canonical"
SCOPE: str = "all"
API_INFINITY_POLICY: str = "pairwise_complete"
ATTENTION_VARIANTS: tuple[str, ...] = ("rollout", "mean", "max")
SIGNAL_METRICS: tuple[str, ...] = (
    "F_pred",
    "F_attr",
    *(f"F_attn_{variant}" for variant in ATTENTION_VARIANTS),
    "F_mag",
    "F_align",
)
# Mechanistic source signal and whether the attribution target keeps its sign,
# exactly as in the F-table transfer metrics.
TRANSFER_METRICS: dict[str, tuple[str, bool]] = {
    "F_align_to_attr": ("F_align", True),
    "F_mag_to_attr": ("F_mag", False),
    **{
        f"F_attn_{variant}_to_attr": (f"F_attn_{variant}", True)
        for variant in ATTENTION_VARIANTS
    },
}
SPREAD_KEYS: list[str] = [
    "pair_population",
    "model_s",
    "model_t",
    "metric",
    "statistic",
    "aggregation",
]
OUTPUT_COLUMNS: list[str] = [
    "cohort",
    "pair_population",
    "benchmark",
    "pregrouper",
    "scope",
    "requested_scope",
    "resolved_scope",
    "contrast",
    "requested_contrast",
    "resolved_source_contrast",
    "resolved_target_contrast",
    "readout_contrast",
    "availability_status",
    "unavailable_reason",
    "api_infinity_policy",
    "aggregation",
    "model",
    "run_s",
    "run_t",
    "comparison",
    "metric",
    "statistic",
    "n_observations",
    "expected_observations",
    "observation_coverage",
    "n_prompts",
    "expected_prompts",
    "prompt_coverage",
    "f_point",
    "f_lo",
    "f_hi",
]

Signals = dict[str, dict[str, pd.Series]]


def load_run_signals(run_dir: str, benchmark: str, pregrouper: str) -> Signals:
    """
    Extract every canonical F-table signal for each paper-cohort model.

    Args:
        run_dir: Results root holding consolidated ``{benchmark}_{pregrouper}``
            segment and log-odds tables plus the raw configuration directory.
        benchmark: Classification benchmark whose stored readout matches its
            canonical contrast.
        pregrouper: Segmentation granularity.

    Returns:
        ``{metric: {model: signal}}`` for the metrics in ``SIGNAL_METRICS``;
        prompt-level signals are indexed by ``prompt_idx`` and segment-level
        signals by ``(prompt_idx, seg_idx)``.
    """
    if not _readout_matches_contrast(benchmark, CONTRAST):
        raise ValueError(f"{benchmark} alignment requires a layer readout")
    prefix: str = os.path.join(run_dir, f"{benchmark}_{pregrouper}")
    seg_df: pd.DataFrame = pd.read_csv(f"{prefix}_segments.tsv", sep="\t")
    odds_df: pd.DataFrame = pd.read_csv(f"{prefix}_logodds.tsv", sep="\t")
    missing: set[str] = set(PAPER_MODELS) - set(seg_df["model"].dropna())
    if missing:
        raise ValueError(f"{run_dir} omits required models {sorted(missing)}")
    contrast_col: str = _contrast_column(CONTRAST)
    unsupported_prediction, unsupported_attribution = _unsupported_model_components(
        run_dir, benchmark, pregrouper
    )

    def collect(extract: Callable[[str], pd.Series | None]) -> dict[str, pd.Series]:
        return {
            model: signal
            for model in PAPER_MODELS
            if (signal := extract(model)) is not None
        }

    signals: Signals = {
        "F_pred": _mask_unsupported_signals(
            collect(lambda m: _per_prompt_logodds(odds_df, m, "orig", contrast_col)),
            unsupported_prediction,
        ),
        "F_attr": _mask_unsupported_signals(
            collect(lambda m: _per_seg_ablation(odds_df, m, contrast_col)),
            unsupported_attribution,
        ),
        "F_mag": collect(
            lambda m: _per_seg_segment_col(seg_df, m, "delta_norm_postnorm")
        ),
        "F_align": collect(lambda m: _per_seg_align(seg_df, m)),
    }
    for variant in ATTENTION_VARIANTS:
        column: str = f"attention_{variant}"
        signals[f"F_attn_{variant}"] = collect(
            lambda m, column=column: _per_seg_segment_col(seg_df, m, column)
        )
    return signals


def _annotate(
    rows: list[dict[str, str | float]],
    benchmark: str,
    pregrouper: str,
    model: str,
    run_s: str,
    run_t: str,
) -> list[dict[str, str | float]]:
    for row in rows:
        metric: str = str(row["metric"])
        source_contrast, target_contrast, readout_contrast = _contrast_metadata(
            benchmark, CONTRAST, metric
        )
        point: float = float(row["f_point"])
        row.update(
            {
                "cohort": "paper",
                "pair_population": (
                    "open_open" if model in OPEN_MODELS else "hosted_hosted"
                ),
                "pregrouper": pregrouper,
                "scope": SCOPE,
                "requested_scope": SCOPE,
                "resolved_scope": _resolved_scope(metric, SCOPE),
                "contrast": CONTRAST,
                "requested_contrast": CONTRAST,
                "resolved_source_contrast": source_contrast,
                "resolved_target_contrast": target_contrast,
                "readout_contrast": readout_contrast,
                "availability_status": (
                    "available" if np.isfinite(point) else "unavailable"
                ),
                "unavailable_reason": (
                    "" if np.isfinite(point) else "insufficient_joint_observations"
                ),
                "api_infinity_policy": API_INFINITY_POLICY,
                "model": model,
                "run_s": run_s,
                "run_t": run_t,
                "comparison": (
                    "rep_vs_published"
                    if PUBLISHED_RUN in (run_s, run_t)
                    else "rep_vs_rep"
                ),
            }
        )
        row.setdefault("aggregation", "row_pooled")
    return rows


def autocorrelation_rows(
    signals_by_run: dict[str, Signals],
    benchmark: str,
    pregrouper: str,
    n_resamples: int,
    conf: float,
    seed: int,
) -> list[dict[str, str | float]]:
    """
    Correlate each model's signals with themselves across runs.

    Args:
        signals_by_run: ``{run_label: load_run_signals(...)}`` in output order.
        benchmark: Benchmark identifier written to every row.
        pregrouper: Segmentation granularity written to every row.
        n_resamples: Prompt-cluster bootstrap resamples per cell.
        conf: Two-sided bootstrap interval level.
        seed: Root seed; each cell derives its own RNG from its identity.

    Returns:
        F-table-style rows with ``model``, ``run_s``, ``run_t``, and
        ``comparison`` identifying the same-model run pair.
    """
    runs: list[str] = list(signals_by_run)
    first: Signals = signals_by_run[runs[0]]
    rows: list[dict[str, str | float]] = []

    def context(model: str) -> tuple[str, ...]:
        return (benchmark, pregrouper, "noise_ceiling", model)

    for metric in SIGNAL_METRICS:
        for model in first[metric]:
            for run_s, run_t in itertools.combinations(runs, 2):
                pair: dict[str, pd.Series] = {
                    run_s: signals_by_run[run_s][metric][model],
                    run_t: signals_by_run[run_t][metric][model],
                }
                rows.extend(
                    _annotate(
                        _emit_pair_corrs(
                            benchmark,
                            metric,
                            pair,
                            n_resamples,
                            conf,
                            np.random.default_rng(seed),
                            bootstrap_seed=seed,
                            rng_context=context(model),
                        ),
                        benchmark,
                        pregrouper,
                        model,
                        run_s,
                        run_t,
                    )
                )
    for metric, (source_metric, signed) in TRANSFER_METRICS.items():
        for model in first[source_metric]:
            for run_s, run_t in itertools.permutations(runs, 2):
                rows.extend(
                    _annotate(
                        _emit_transfer(
                            benchmark,
                            metric,
                            {run_s: signals_by_run[run_s][source_metric][model]},
                            {run_t: signals_by_run[run_t]["F_attr"][model]},
                            signed=signed,
                            n_resamples=n_resamples,
                            conf=conf,
                            rng=np.random.default_rng(seed),
                            bootstrap_seed=seed,
                            rng_context=context(model),
                        ),
                        benchmark,
                        pregrouper,
                        model,
                        run_s,
                        run_t,
                    )
                )
    return rows


def spread_table(
    published: pd.DataFrame, reps: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Summarize run-to-run spread of every F-table estimate.

    Args:
        published: Canonical F-table rows for one configuration.
        reps: ``{rep_label: F-table rows}`` recomputed on each repetition.

    Returns:
        One row per ``SPREAD_KEYS`` cell with ``f_point_published``,
        ``f_point_<rep>`` columns, and the repetition mean, sample SD, min,
        max, and minimum joint observation count.
    """
    table: pd.DataFrame = published[SPREAD_KEYS + ["f_point"]].rename(
        columns={"f_point": "f_point_published"}
    )
    observations: list[str] = []
    for label, frame in reps.items():
        keyed: pd.DataFrame = frame[SPREAD_KEYS + ["f_point", "n_observations"]]
        keyed = keyed.rename(
            columns={
                "f_point": f"f_point_{label}",
                "n_observations": f"n_observations_{label}",
            }
        )
        table = table.merge(keyed, on=SPREAD_KEYS, how="outer", validate="1:1")
        observations.append(f"n_observations_{label}")
    rep_points: pd.DataFrame = table[[f"f_point_{label}" for label in reps]]
    table["rep_mean"] = rep_points.mean(axis=1)
    table["rep_sd"] = rep_points.std(axis=1, ddof=1)
    table["rep_min"] = rep_points.min(axis=1)
    table["rep_max"] = rep_points.max(axis=1)
    table["published_minus_rep_mean"] = table["f_point_published"] - table["rep_mean"]
    table["rep_min_n_observations"] = table[observations].min(axis=1)
    return table.drop(columns=observations)


def _published_rows(results_dir: str, benchmark: str, pregrouper: str) -> pd.DataFrame:
    frame: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "f_table.tsv"), sep="\t"
    )
    return frame[
        (frame["benchmark"] == benchmark)
        & (frame["pregrouper"] == pregrouper)
        & (frame["scope"] == SCOPE)
        & (frame["contrast"] == CONTRAST)
        & (frame["api_infinity_policy"] == API_INFINITY_POLICY)
    ]


def _rep_rows(
    rep_dir: str,
    benchmark: str,
    pregrouper: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, str | float]] = _process_benchmark(
        benchmark,
        pregrouper,
        rep_dir,
        0,
        0.95,
        _analysis_rng(seed, benchmark, pregrouper),
        PAPER_MODELS,
        SCOPE,
        CONTRAST,
        API_INFINITY_POLICY,
        "row_pooled",
        seed,
    )
    frame: pd.DataFrame = pd.DataFrame(rows)
    source_open: pd.Series = frame["model_s"].isin(OPEN_MODELS)
    target_open: pd.Series = frame["model_t"].isin(OPEN_MODELS)
    frame["pair_population"] = np.select(
        [source_open & target_open, source_open | target_open],
        ["open_open", "open_hosted"],
        default="hosted_hosted",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--rep-dirs",
        nargs="+",
        required=True,
        help="Repetition results roots, labeled rep1, rep2, ... in this order.",
    )
    parser.add_argument("--benchmark", default="boolq")
    parser.add_argument(
        "--pregrouper", default="sentence", choices=["word", "sentence"]
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rep-archive-id",
        required=True,
        help=(
            "Identifier of the off-Git archive holding the repetition raw files, "
            "recorded in provenance alongside their SHA-256 digests."
        ),
    )
    # Kept outside results/, whose release inventory is validated exactly.
    parser.add_argument("--output", default="noise_ceiling/noise_ceiling.tsv")
    parser.add_argument(
        "--spread-output", default="noise_ceiling/noise_ceiling_spread.tsv"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output: str = args.output
    spread_output: str = args.spread_output
    for path in (output, spread_output):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    run_dirs: dict[str, str] = {PUBLISHED_RUN: args.results_dir}
    run_dirs.update({f"rep{i}": d for i, d in enumerate(args.rep_dirs, start=1)})

    signals_by_run: dict[str, Signals] = {
        label: load_run_signals(path, args.benchmark, args.pregrouper)
        for label, path in run_dirs.items()
    }
    autocorrelations: pd.DataFrame = pd.DataFrame(
        autocorrelation_rows(
            signals_by_run,
            args.benchmark,
            args.pregrouper,
            args.bootstrap_resamples,
            args.confidence_level,
            args.seed,
        )
    )[OUTPUT_COLUMNS]
    autocorrelations.to_csv(output, sep="\t", index=False)

    rep_frames: dict[str, pd.DataFrame] = {
        label: _rep_rows(path, args.benchmark, args.pregrouper, args.seed)
        for label, path in run_dirs.items()
        if label != PUBLISHED_RUN
    }
    spread: pd.DataFrame = spread_table(
        _published_rows(args.results_dir, args.benchmark, args.pregrouper),
        rep_frames,
    )
    spread.to_csv(spread_output, sep="\t", index=False)

    input_paths: dict[str, str] = {}
    for label, path in run_dirs.items():
        prefix: str = os.path.join(path, f"{args.benchmark}_{args.pregrouper}")
        inputs: dict[str, str] = collect_result_inputs(
            path,
            args.benchmark,
            args.pregrouper,
            {"segments": f"{prefix}_segments.tsv", "logodds": f"{prefix}_logodds.tsv"},
            PAPER_MODELS,
        )
        input_paths.update({f"{label}/{key}": value for key, value in inputs.items()})
    input_paths["published/f_table.tsv"] = os.path.join(args.results_dir, "f_table.tsv")
    parameters: dict[str, object] = {
        "benchmark": args.benchmark,
        "pregrouper": args.pregrouper,
        "runs": list(run_dirs),
        "rep_archive_id": args.rep_archive_id,
        "rep_artifact_layout": "rep{k}/{benchmark}/{pregrouper}/<raw file>",
        "scope": SCOPE,
        "contrast": CONTRAST,
        "api_infinity_policy": API_INFINITY_POLICY,
        "missingness_policy": "pair_specific_complete_case",
        "bootstrap_resamples": args.bootstrap_resamples,
        "confidence_level": args.confidence_level,
        "seed": args.seed,
        "bootstrap_rng": "sha256_cell_key_v1",
        "requested_models": list(PAPER_MODELS),
        "spread_bootstrap_resamples": 0,
    }
    for path, generator_output in (
        (output, "autocorrelation"),
        (spread_output, "spread"),
    ):
        write_derived_provenance(
            path,
            generator_name="benchmark_scripts.noise_ceiling",
            generator_path=__file__,
            input_paths=input_paths,
            parameters={**parameters, "output": generator_output},
            root_dir=args.results_dir,
            supporting_source_paths=_supporting_source_paths(),
        )
    logger.info(f"Wrote {len(autocorrelations)} rows to {output}")
    logger.info(f"Wrote {len(spread)} rows to {spread_output}")


if __name__ == "__main__":
    main()
