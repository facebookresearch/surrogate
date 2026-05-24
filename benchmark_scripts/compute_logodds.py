# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Postprocess token logprobs into per-label logprobs and per-pair log-odds.

Reads ``{results_dir}/{benchmark}_{pregrouper}_tokens.tsv`` and the
benchmark's ``EvalConfig.label_tokens`` mapping. For each
(model, prompt_idx, seg_idx, kind) group, computes:

  - ``label_lp_<label>``: ``logsumexp`` of token logprobs whose surface
    appears in ``label_tokens[<label>]``.
  - ``logodds_<label_a>_<label_b>``: ``label_lp_<label_a> - label_lp_<label_b>``
    for every ordered pair of labels (so binary benchmarks emit one
    ``logodds_pos_neg`` and ``logodds_neg_pos``; ternary benchmarks like
    ANLI emit six pair columns).

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


def compute(
    benchmark_name: str, pregrouper: str, results_dir: str = "results"
) -> None:
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
        in_path, sep="\t", dtype={"label": str, "token": str, "kind": str}
    )
    logger.info(f"Loaded {in_path}: {len(df)} rows")

    alias_to_label: dict[str, str] = _alias_to_label(eval_config)
    df["label_resolved"] = df["token"].map(alias_to_label.get)
    df = df.dropna(subset=["label_resolved", "logprob"])

    # Group key: (model, prompt_idx, seg_idx, kind, resolved_label) -> logsumexp
    # seg_idx is NaN for orig rows; pandas groupby drops NaN groups by default,
    # so we replace NaN with sentinel -1 for grouping then restore.
    df["seg_idx_key"] = df["seg_idx"].fillna(-1).astype(int)
    grouped: pd.DataFrame = (
        df.groupby(
            ["model", "prompt_idx", "seg_idx_key", "kind", "label_resolved"]
        )["logprob"]
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
        c if c in {"model", "prompt_idx", "seg_idx_key", "kind"}
        else f"label_lp_{label_column_alias(str(c))}"
        for c in wide.columns
    ]
    # Restore seg_idx (NaN for orig).
    wide["seg_idx"] = wide["seg_idx_key"].where(wide["seg_idx_key"] >= 0)
    wide = wide.drop(columns=["seg_idx_key"])

    # Per-pair log-odds for every ordered pair of labels.
    label_aliases: list[str] = [
        label_column_alias(label) for label in eval_config.label_tokens
    ]
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

    # Reorder columns: keys, then label_lp_*, then logodds_*.
    key_cols: list[str] = ["model", "prompt_idx", "seg_idx", "kind"]
    label_cols: list[str] = sorted(
        c for c in wide.columns if c.startswith("label_lp_")
    )
    odds_cols: list[str] = sorted(c for c in wide.columns if c.startswith("logodds_"))
    wide = wide[key_cols + label_cols + odds_cols]

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
