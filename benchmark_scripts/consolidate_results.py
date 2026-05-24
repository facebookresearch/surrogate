# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Consolidate per-model TSVs into one segments TSV + one tokens TSV per benchmark.

Reads ``{results_dir}/{benchmark}/{pregrouper}/{model}_segment.tsv`` and
``{results_dir}/{benchmark}/{pregrouper}/{model}_tokens.tsv`` for every model
that has output files, concatenates with a ``model`` column, and writes:

  ``{results_dir}/{benchmark}_{pregrouper}_segments.tsv``
  ``{results_dir}/{benchmark}_{pregrouper}_tokens.tsv``

The tokens TSV is omitted if no model produced one (e.g., LAMBADA only has
segment-level completion logprobs, no per-(label, token) breakdown).

Usage:
    python -m benchmark_scripts.consolidate_results --benchmark boolq
    python -m benchmark_scripts.consolidate_results --benchmark boolq --pregrouper word
"""

import argparse
import glob
import logging
import os

import pandas as pd
from benchmark_scripts.benchmark_config import BENCHMARKS

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


def _read_per_model(pattern: str) -> tuple[list[pd.DataFrame], list[str]]:
    """Read every TSV matching pattern; return (frames, model_names) parallel lists.

    Model name is extracted from the filename via the suffix that follows
    the trailing ``_segment.tsv`` / ``_tokens.tsv`` (gzipped ``.tsv.gz``
    variants are also supported transparently — pandas auto-detects via
    the extension).
    """
    # Force string dtype on label/token/kind in tokens TSVs; pandas otherwise
    # coerces all-"true"/"false" label columns into bool. Only the tokens TSVs
    # carry these columns; segment TSVs don't, so filter dtype dict to columns
    # actually present in each file's header.
    string_cols: list[str] = ["label", "token", "kind"]
    frames: list[pd.DataFrame] = []
    models: list[str] = []
    # Pattern caller passes "*_segment.tsv"; we also pick up gzipped variants.
    paths: list[str] = sorted(set(glob.glob(pattern) + glob.glob(pattern + ".gz")))
    for path in paths:
        header: list[str] = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
        dtype: dict[str, type] = {c: str for c in string_cols if c in header}
        df: pd.DataFrame = pd.read_csv(path, sep="\t", dtype=dtype)
        # Extract model name from filename: strip _segment.tsv[.gz] or _tokens.tsv[.gz]
        base: str = os.path.basename(path)
        for suffix in (
            "_segment.tsv.gz",
            "_tokens.tsv.gz",
            "_segment.tsv",
            "_tokens.tsv",
        ):
            if base.endswith(suffix):
                models.append(base[: -len(suffix)])
                break
        else:
            models.append(os.path.splitext(base)[0])
        frames.append(df)
    return frames, models


def consolidate(
    benchmark_name: str, pregrouper: str, results_dir: str = "results"
) -> None:
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    in_dir: str = os.path.join(results_dir, benchmark_name, pregrouper)

    seg_frames, seg_models = _read_per_model(os.path.join(in_dir, "*_segment.tsv"))
    tok_frames, tok_models = _read_per_model(os.path.join(in_dir, "*_tokens.tsv"))

    if not seg_frames:
        logger.error(f"No per-model segment TSVs found in {in_dir}")
        return

    seg_df: pd.DataFrame = pd.concat(
        [df.assign(model=m) for df, m in zip(seg_frames, seg_models)],
        ignore_index=True,
    )
    seg_path: str = os.path.join(
        results_dir, f"{benchmark_name}_{pregrouper}_segments.tsv"
    )
    seg_df.to_csv(seg_path, sep="\t", index=False)
    logger.info(
        f"Wrote {seg_path}: {len(seg_df)} rows, "
        f"{len(seg_df.columns)} cols, {len(seg_models)} models ({seg_models})"
    )

    if tok_frames:
        tok_df: pd.DataFrame = pd.concat(
            [df.assign(model=m) for df, m in zip(tok_frames, tok_models)],
            ignore_index=True,
        )
        tok_path: str = os.path.join(
            results_dir, f"{benchmark_name}_{pregrouper}_tokens.tsv"
        )
        tok_df.to_csv(tok_path, sep="\t", index=False)
        logger.info(
            f"Wrote {tok_path}: {len(tok_df)} rows, "
            f"{len(tok_df.columns)} cols, {len(tok_models)} models ({tok_models})"
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
        help="Pregrouper used for scoring (determines subdir)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing per-model TSVs and where consolidated TSVs are written",
    )
    args: argparse.Namespace = parser.parse_args()
    consolidate(args.benchmark, args.pregrouper, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
