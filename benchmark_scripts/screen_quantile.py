# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
The surrogate-only screen with nothing fitted: the top third by support.

``benchmark_scripts.screen_transfer`` fits a support cut per surrogate. This
module replaces the cut with a fixed quantile, so no number is tuned, no
held-out split is needed and the same rule applies to any surrogate. A prompt
is *flagged* when the surrogate's top passage support, ``max_i Δℓ_i · sign(ℓ)``
over all of its own passage sentences, is in the top third of prompts. The
flag does not depend on the target.

Reanalysis of the published BoolQ sentence-level results. No model or API
calls; nothing is written under ``results/``. All prompts are used.

``build`` writes, under ``--output-dir``:

- ``third_transfer.csv``: surrogate → target agreement on flagged and
  unflagged prompts, also split by the surrogate's answer and by whether the
  target (and the surrogate) answered correctly.
- ``third_bins.csv``: agreement by ninth of surrogate support.
- ``third_hosted_pairs.csv``: agreement between pairs of hosted models on each
  surrogate's flagged and unflagged prompts.
- ``third_overlap.csv``: prompts flagged by both surrogates, by one, or by
  neither.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from benchmark_scripts.reanalysis_rebuttal import HOSTED_MODELS, load_signals
from benchmark_scripts.screen_transfer import (
    SURROGATES,
    flag_groups,
    pair_agreement,
    passage_attributions,
    score_subset,
    scored_prompts,
    surrogate_support,
)
from benchmark_scripts.worked_example import (
    SEED,
    _prompt_moments,
    _r2_with_ci,
    pair_segments,
    prompt_table,
)

TOP_SHARE: float = 1 / 3
N_BINS: int = 9


def top_share_flags(support: pd.Series, share: float = TOP_SHARE) -> pd.Series:
    """
    Flag the prompts whose support is in the top ``share`` of prompts.

    Args:
        support: Surrogate support per prompt.
        share: Fraction of prompts to flag.

    Returns:
        Boolean flag per prompt; prompts at the quantile are flagged.
    """
    return support >= support.quantile(1 - share)


def subset_rows(frame: pd.DataFrame, prompts: pd.DataFrame) -> list[dict[str, object]]:
    """
    Agreement on flagged and unflagged prompts of one pair, with splits.

    Args:
        frame: Shared segment table of the pair.
        prompts: Scored prompts with a boolean ``flagged`` column.

    Returns:
        Rows for all prompts, and for flagged and unflagged prompts overall,
        within each surrogate answer, and by correctness of the target.
    """
    flagged = prompts["flagged"]
    wrong = ~prompts["correct_gpt"].astype(bool)
    groups: dict[str, pd.Series] = {
        "all": pd.Series(True, index=prompts.index),
        "answer_true": prompts["pred_qwen"].eq(True),
        "answer_false": prompts["pred_qwen"].eq(False),
        "target_right": ~wrong,
        "target_wrong": wrong,
        "target_wrong_same_answer": prompts["case"].eq("both_wrong"),
        "target_wrong_surrogate_right": prompts["case"].eq("only_qwen_right"),
    }
    rows: list[dict[str, object]] = []
    for group, mask in groups.items():
        for subset, sel in (
            ("any", mask),
            ("flagged", mask & flagged),
            ("unflagged", mask & ~flagged),
        ):
            sub = prompts[sel]
            rows.append(
                {
                    "group": group,
                    "subset": subset,
                    "share_of_group": len(sub) / mask.sum() if mask.sum() else np.nan,
                    **score_subset(frame, sub),
                }
            )
    return rows


def bin_rows(
    frame: pd.DataFrame, prompts: pd.DataFrame, n_bins: int = N_BINS
) -> list[dict[str, float]]:
    """Agreement by quantile bin of the surrogate's support."""
    bins = pd.qcut(prompts["support"], n_bins, labels=False, duplicates="drop")
    moments = _prompt_moments(frame)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float]] = []
    for b, sub in prompts.groupby(bins):
        r2, lo, hi = _r2_with_ci(moments, pd.Index(sub["prompt_idx"]), rng)
        rows.append(
            {
                "bin": int(b) + 1,
                "support_median": float(sub["support"].median()),
                "n_prompts": len(sub),
                "passage_r2": r2,
                "passage_r2_lo": lo,
                "passage_r2_hi": hi,
                "top1_match": float(sub["top1"].mean()),
                "top1_overlap_baseline": float(
                    sub["overlap_top_matches_gpt"].astype(float).mean()
                ),
            }
        )
    return rows


def build(results_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    signals = load_signals(results_dir)
    targets: tuple[str, ...] = HOSTED_MODELS
    passage = passage_attributions(signals)
    transfer: list[dict[str, object]] = []
    bins: list[dict[str, object]] = []
    flags: dict[str, pd.Series] = {}
    for surrogate in SURROGATES:
        support = surrogate_support(passage, signals.prediction, surrogate)
        flags[surrogate] = top_share_flags(support)
        cut = float(support.quantile(1 - TOP_SHARE))
        for target in targets:
            frame, ell_s, ell_t, gold = pair_segments(
                results_dir, surrogate, target, signals
            )
            prompts = scored_prompts(prompt_table(frame, ell_s, ell_t, gold))
            prompts["support"] = support.reindex(prompts["prompt_idx"]).to_numpy()
            prompts = prompts[prompts["support"].notna()].copy()
            prompts["flagged"] = (
                flags[surrogate].reindex(prompts["prompt_idx"]).to_numpy(dtype=bool)
            )
            key = {"surrogate": surrogate, "target": target, "support_cut_nats": cut}
            transfer += [{**key, **row} for row in subset_rows(frame, prompts)]
            bins += [{**key, **row} for row in bin_rows(frame, prompts)]
    pd.DataFrame(transfer).to_csv(
        os.path.join(output_dir, "third_transfer.csv"), index=False
    )
    pd.DataFrame(bins).to_csv(os.path.join(output_dir, "third_bins.csv"), index=False)

    hosted: list[dict[str, object]] = []
    for surrogate in SURROGATES:
        flag = flags[surrogate]
        for model_a, model_b in itertools.combinations(targets, 2):
            for name, index in (
                ("flagged", flag[flag].index),
                ("unflagged", flag[~flag].index),
            ):
                hosted.append(
                    {
                        "screen_surrogate": surrogate,
                        "model_a": model_a,
                        "model_b": model_b,
                        "subset": name,
                        **pair_agreement(
                            passage, signals.prediction, model_a, model_b, index
                        ),
                    }
                )
    pd.DataFrame(hosted).to_csv(
        os.path.join(output_dir, "third_hosted_pairs.csv"), index=False
    )

    first, second = SURROGATES
    common = flags[first].index.intersection(flags[second].index)
    groups = flag_groups(flags[first].loc[common], flags[second].loc[common])
    overlap: list[dict[str, object]] = []
    for group, members in groups.groupby(groups):
        row: dict[str, object] = {
            "first": first,
            "second": second,
            "flagged_by": group,
            "share": len(members) / len(groups),
        }
        stats = pair_agreement(
            passage, signals.prediction, first, second, members.index
        )
        row.update({f"surrogates_{key}": value for key, value in stats.items()})
        for label, model in (("first", first), ("second", second)):
            matches = [
                pair_agreement(
                    passage, signals.prediction, model, target, members.index
                )["top1_match"]
                for target in targets
            ]
            row[f"{label}_top1_match_median"] = float(np.median(matches))
            row[f"{label}_top1_match_min"] = float(np.min(matches))
            row[f"{label}_top1_match_max"] = float(np.max(matches))
        overlap.append(row)
    pd.DataFrame(overlap).to_csv(
        os.path.join(output_dir, "third_overlap.csv"), index=False
    )
    both = (groups == "both").sum()
    either = (groups != "neither").sum()
    print(
        f"Wrote {output_dir}: top third by support; "
        f"flag Jaccard {both / either:.3f} on {len(groups)} prompts"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-dir", default=os.path.join("worked_example", "transfer")
    )
    args = parser.parse_args()
    build(args.results_dir, args.output_dir)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
