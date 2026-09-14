# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Postprocess token logprobs into per-label logprobs and per-pair log-odds.

Reads ``{results_dir}/{benchmark}_{pregrouper}_tokens.tsv`` and, when present,
the matching segment-status table. Using the benchmark's
``EvalConfig.label_tokens`` mapping, for each
(model, prompt_idx, seg_idx, kind) group, computes:

  - ``label_lp_<label>``: ``logsumexp`` of token logprobs whose surface
    appears in ``label_tokens[<label>]``.
  - ``logodds_<label_a>_<label_b>``: ``label_lp_<label_a> - label_lp_<label_b>``
    for every ordered pair of labels (so binary benchmarks emit two directed
    columns; ternary benchmarks like ANLI emit six).
  - ``logodds``: the canonical paper contrast, using the first two labels in
    the benchmark configuration in their declared order. For RACE, whose
    target label varies by prompt, this is instead the correct-answer log-prob
    minus ``logsumexp`` over the other three answers.

The intent is to keep aggregation choices (logsumexp vs first-token,
which token-set per label, which label-pair to subtract) downstream of
the GPU runner so they can be revisited without re-running the model.

Output: ``{results_dir}/{benchmark}_{pregrouper}_logodds.tsv``
"""

import argparse
import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from benchmark_scripts.benchmark_config import BENCHMARKS, BenchmarkSpec
from surrogate.eval_constants import EvalConfig, label_column_alias

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _logsumexp(values: Iterable[float]) -> float:
    """logsumexp over a finite iterable, ignoring NaN. Returns -inf if empty."""
    arr: np.ndarray = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("-inf")
    m: float = float(arr.max())
    return m + float(np.log(np.exp(arr - m).sum()))


def _alias_to_label(eval_config: "EvalConfig") -> dict[str, str]:
    """Build a {token_alias -> label} map from ``report_tokens``.

    The TSV ``token`` column carries report-token aliases (e.g.
    ``sp_True`` for the surface ``" True"``), not raw surfaces — so a
    direct surface match against ``label_tokens`` silently drops every
    space- and underscore-prefixed variant. Walking ``report_tokens``
    instead recovers the alias→label mapping the runner used when it
    wrote the file.
    """
    out: dict[str, str] = {}
    for label, tokens in eval_config.report_tokens.items():
        for tok in tokens:
            out[tok.alias] = label
    return out


def _mask_failed_requests(
    wide: pd.DataFrame,
    results_dir: str,
    benchmark_name: str,
    pregrouper: str,
    value_columns: list[str],
) -> pd.DataFrame:
    """Set derived values to NaN when the provider returned no valid response.

    Hosted token tables are rectangular: they contain ``-inf`` placeholders for
    every label even when a whole request failed. Request-status columns in the
    segment table distinguish that case from a successful top-k response in
    which a particular label was absent. Only the former is masked to NaN.
    """
    segment_path: str = os.path.join(
        results_dir, f"{benchmark_name}_{pregrouper}_segments.tsv"
    )
    if not os.path.exists(segment_path):
        return wide
    segments: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
    status_columns: set[str] = {
        "original_request_status",
        "segment_request_status",
    }
    present_status_columns: set[str] = status_columns & set(segments.columns)
    if not present_status_columns:
        return wide
    if present_status_columns != status_columns:
        raise ValueError(
            f"Incomplete request-status columns in {segment_path}: "
            f"{sorted(present_status_columns)}"
        )

    original_statuses: pd.DataFrame = segments.loc[
        segments["original_request_status"].notna(),
        ["model", "prompt_idx", "original_request_status"],
    ].drop_duplicates()
    if original_statuses.duplicated(["model", "prompt_idx"]).any():
        raise ValueError(f"Conflicting original request statuses in {segment_path}")
    original_statuses = original_statuses.rename(
        columns={"original_request_status": "request_status"}
    )
    original_statuses["kind"] = "orig"
    original_statuses["seg_idx_key"] = -1

    segment_statuses: pd.DataFrame = segments.loc[
        segments["segment_request_status"].notna(),
        ["model", "prompt_idx", "seg_idx", "segment_request_status"],
    ].rename(columns={"segment_request_status": "request_status"})
    segment_statuses["kind"] = "ablated"
    segment_statuses["seg_idx_key"] = pd.to_numeric(
        segment_statuses["seg_idx"], errors="raise"
    ).astype(int)
    status_rows: pd.DataFrame = pd.concat(
        [
            original_statuses[
                ["model", "prompt_idx", "seg_idx_key", "kind", "request_status"]
            ],
            segment_statuses[
                ["model", "prompt_idx", "seg_idx_key", "kind", "request_status"]
            ],
        ],
        ignore_index=True,
    )
    if status_rows.duplicated(["model", "prompt_idx", "seg_idx_key", "kind"]).any():
        raise ValueError(f"Duplicate request-status keys in {segment_path}")

    result: pd.DataFrame = wide.copy()
    result["seg_idx_key"] = result["seg_idx"].fillna(-1).astype(int)
    result = result.merge(
        status_rows,
        on=["model", "prompt_idx", "seg_idx_key", "kind"],
        how="left",
        validate="one_to_one",
    )
    failed: pd.Series = result["request_status"].notna() & result["request_status"].ne(
        "ok"
    )
    result.loc[failed, value_columns] = np.nan
    return result.drop(columns=["seg_idx_key", "request_status"])


def compute(benchmark_name: str, pregrouper: str, results_dir: str = "results") -> None:
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    spec: BenchmarkSpec = BENCHMARKS[benchmark_name]
    if spec.eval_config is None:
        logger.warning(
            f"benchmark {spec.name!r} has eval_config=None "
            f"(completion_logprob mode); skipping log-odds computation"
        )
        return
    eval_config: EvalConfig = spec.eval_config

    in_path: str = os.path.join(
        results_dir, f"{benchmark_name}_{pregrouper}_tokens.tsv"
    )
    if not os.path.exists(in_path):
        logger.error(f"Missing tokens TSV: {in_path}")
        return
    # Force string dtype on label/token; pandas auto-detects "true"/"false"
    # values as bool otherwise, which breaks the label-token surface match.
    df: pd.DataFrame = pd.read_csv(
        in_path,
        sep="\t",
        dtype={
            "label": str,
            "token": str,
            "kind": str,
            "logprob_granularity": str,
        },
        keep_default_na=False,
        na_values=[""],
    )
    logger.info(f"Loaded {in_path}: {len(df)} rows")

    alias_to_label: dict[str, str] = _alias_to_label(eval_config)
    df["label_resolved"] = df["token"].map(alias_to_label.get)
    if "logprob_granularity" in df.columns:
        aggregate_rows: pd.Series = df["logprob_granularity"].eq("label_aggregate")
        configured_labels: set[str] = set(eval_config.label_tokens)
        df.loc[aggregate_rows, "label_resolved"] = df.loc[
            aggregate_rows, "label"
        ].where(df.loc[aggregate_rows, "label"].isin(configured_labels))
    df = df.dropna(subset=["label_resolved", "logprob"])

    # Group key: (model, prompt_idx, seg_idx, kind, resolved_label) -> logsumexp
    # seg_idx is NaN for orig rows; pandas groupby drops NaN groups by default,
    # so we replace NaN with sentinel -1 for grouping then restore.
    df["seg_idx_key"] = df["seg_idx"].fillna(-1).astype(int)
    grouped: pd.DataFrame = (
        df.groupby(["model", "prompt_idx", "seg_idx_key", "kind", "label_resolved"])[
            "logprob"
        ]
        .apply(_logsumexp)
        .reset_index(name="label_lp")
    )

    # Pivot to wide on label.
    wide: pd.DataFrame = grouped.pivot_table(
        index=["model", "prompt_idx", "seg_idx_key", "kind"],
        columns="label_resolved",
        values="label_lp",
    ).reset_index()
    wide.columns = [
        (
            c
            if c in {"model", "prompt_idx", "seg_idx_key", "kind"}
            else f"label_lp_{label_column_alias(str(c))}"
        )
        for c in wide.columns
    ]
    # Restore seg_idx (NaN for orig).
    wide["seg_idx"] = wide["seg_idx_key"].where(wide["seg_idx_key"] >= 0)
    wide = wide.drop(columns=["seg_idx_key"])

    if eval_config is not None and spec.scoring_mode == "per_prompt_logit_difference":
        if "answer" not in df.columns:
            raise ValueError(
                f"{benchmark_name!r} requires an answer column for its "
                "per-prompt label contrast"
            )
        answers: pd.DataFrame = df[["model", "prompt_idx", "answer"]].drop_duplicates()
        duplicate_answers: pd.Series = answers.duplicated(
            ["model", "prompt_idx"], keep=False
        )
        if duplicate_answers.any():
            raise ValueError("Conflicting answers for the same model and prompt")
        wide = wide.merge(answers, on=["model", "prompt_idx"], how="left")

    label_aliases: list[str] = [
        label_column_alias(label) for label in eval_config.label_tokens
    ]
    # A label absent from an API's top-k response has zero represented mass,
    # hence log-probability -inf. Preserve that semantics instead of silently
    # dropping the observation during the pivot.
    for label_alias in label_aliases:
        label_col: str = f"label_lp_{label_alias}"
        if label_col not in wide.columns:
            wide[label_col] = -float("inf")
        else:
            wide[label_col] = wide[label_col].fillna(-float("inf"))

    # Per-pair log-odds for every ordered pair of labels.
    for a in label_aliases:
        col_a: str = f"label_lp_{a}"
        if col_a not in wide.columns:
            continue
        for b in label_aliases:
            if a == b:
                continue
            col_b: str = f"label_lp_{b}"
            if col_b not in wide.columns:
                continue
            wide[f"logodds_{a}_{b}"] = wide[col_a] - wide[col_b]

    if spec.scoring_mode == "per_prompt_logit_difference":

        def _target_logodds(row: pd.Series) -> float:
            answer_alias: str = label_column_alias(str(row["answer"]))
            if answer_alias not in label_aliases:
                raise ValueError(f"Answer {row['answer']!r} is not a configured label")
            positive: float = float(row[f"label_lp_{answer_alias}"])
            negative: float = _logsumexp(
                float(row[f"label_lp_{label_alias}"])
                for label_alias in label_aliases
                if label_alias != answer_alias
            )
            return positive - negative

        wide["logodds"] = wide.apply(_target_logodds, axis=1)
    else:
        if len(label_aliases) < 2:
            raise ValueError(
                f"Benchmark {benchmark_name!r} needs at least two configured labels"
            )
        canonical_a, canonical_b = label_aliases[:2]
        wide["logodds"] = (
            wide[f"label_lp_{canonical_a}"] - wide[f"label_lp_{canonical_b}"]
        )

    # Reorder columns: keys, then label_lp_*, then logodds_*.
    key_cols: list[str] = ["model", "prompt_idx", "seg_idx", "kind"]
    if "answer" in wide.columns:
        key_cols.append("answer")
    label_cols: list[str] = sorted(c for c in wide.columns if c.startswith("label_lp_"))
    odds_cols: list[str] = sorted(c for c in wide.columns if c.startswith("logodds_"))
    wide = _mask_failed_requests(
        wide,
        results_dir,
        benchmark_name,
        pregrouper,
        [*label_cols, "logodds", *odds_cols],
    )
    wide = wide[key_cols + label_cols + ["logodds"] + odds_cols]

    out_path: str = os.path.join(
        results_dir, f"{benchmark_name}_{pregrouper}_logodds.tsv"
    )
    wide.to_csv(out_path, sep="\t", index=False)
    logger.info(
        f"Wrote {out_path}: {len(wide)} rows, "
        f"{len(label_cols)} label_lp cols, {len(odds_cols)} logodds cols"
    )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARKS.keys()),
    )
    parser.add_argument(
        "--pregrouper",
        type=str,
        default="sentence",
        choices=["word", "sentence"],
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
    )
    args: argparse.Namespace = parser.parse_args()
    compute(args.benchmark, args.pregrouper, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
