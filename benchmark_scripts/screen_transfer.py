# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Does the surrogate-only screen of the worked example transfer?

The worked example screens BoolQ prompts with one number computed from the
open surrogate alone: the support of its top passage sentence for its own
answer, ``max_i Δℓ_i · sign(ℓ)``. This module asks whether the prompts that
pass are also the ones where the surrogate matches *other* hosted targets, and
whether the same rule works for a second surrogate.

Reanalysis of the published BoolQ sentence-level results. No model or API
calls; nothing is written under ``results/``.

For each surrogate (Qwen2.5-14B, Llama-3.1-8B) the only fitted quantity is the
support cut, fit on even prompts against GPT-4o with the criterion of
:func:`benchmark_scripts.worked_example.gate_analysis`. Every reported row is
on odd prompts; targets other than GPT-4o use the frozen cut.

``build`` writes, under ``--output-dir``:

- ``screen_cuts.csv``: each surrogate's cut, the cut from the reversed split
  and the share of prompts it passes.
- ``screen_transfer.csv``: surrogate → target agreement on passing, failing
  and all held-out prompts, also split by the surrogate's answer.
- ``hosted_pairs.csv``: agreement between pairs of hosted models (no
  surrogate involved) on each surrogate's passing and failing prompts.
- ``flag_overlap.csv``: both surrogates' agreement with each target on
  prompts passed by both, by one, or by neither.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from benchmark_scripts.reanalysis_rebuttal import HOSTED_MODELS, Signals, load_signals
from benchmark_scripts.worked_example import (
    SEED,
    SUPPORT_CUTS,
    _chi2,
    _prompt_moments,
    _r2_with_ci,
    _within_r2,
    pair_segments,
    prompt_table,
)

SURROGATES: tuple[str, ...] = ("qwen2.5-14b-instruct", "llama-3.1-8b-instruct")
FIT_TARGET: str = "gpt-4o"
# The committed Qwen2.5-14B -> GPT-4o screen; build() checks it is reproduced.
REFERENCE_RULE: str = "gate_rule.csv"
REFERENCE_TOLERANCE: float = 1e-9


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------


def scored_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    """Keep the prompts the worked example scores and add a boolean ``top1``."""
    out = prompts[
        prompts["case"].ne("unscored") & prompts["top_abs_passage_match"].notna()
    ].copy()
    out["top1"] = out["top_abs_passage_match"].astype(bool)
    return out


def fit_support_cut(prompts: pd.DataFrame) -> float:
    """
    Fit the surrogate-only support cut.

    Args:
        prompts: Scored prompts with ``qwen_top_support`` and ``top1``.

    Returns:
        The cut in ``SUPPORT_CUTS`` that maximizes the χ² of top-1 passage
        match between prompts at or above the cut and prompts below it.
    """
    support = prompts["qwen_top_support"]
    return float(
        max(
            SUPPORT_CUTS,
            key=lambda c: _chi2(
                prompts.loc[support >= c, "top1"], prompts.loc[support < c, "top1"]
            ),
        )
    )


def score_subset(frame: pd.DataFrame, subset: pd.DataFrame) -> dict[str, float]:
    """
    Surrogate → target agreement on a set of prompts.

    Args:
        frame: Shared segment table of the pair.
        subset: Scored prompts to evaluate.

    Returns:
        Pooled passage r² with a prompt-cluster bootstrap interval, top-1
        passage match with its chance and word-overlap baselines, and
        within-prompt |Δℓ| r² for the surrogate and for word overlap.
    """
    index = pd.Index(subset["prompt_idx"])
    if len(index) < 3:
        return {"n_prompts": len(index)}
    r2, lo, hi = _r2_with_ci(_prompt_moments(frame), index, np.random.default_rng(SEED))
    magnitudes = frame.assign(
        abs_qwen=frame["d_qwen"].abs(), abs_gpt=frame["d_gpt"].abs()
    )
    return {
        "n_prompts": len(index),
        "passage_r2": r2,
        "passage_r2_lo": lo,
        "passage_r2_hi": hi,
        "top1_match": float(subset["top1"].mean()),
        "top1_chance": float(subset["chance_passage"].mean()),
        "top1_overlap_baseline": float(
            subset["overlap_top_matches_gpt"].astype(float).mean()
        ),
        "abs_within_r2_surrogate": _within_r2(magnitudes, index, "abs_qwen", "abs_gpt"),
        "abs_within_r2_overlap": _within_r2(
            magnitudes, index, "question_overlap", "abs_gpt"
        ),
        "share_surrogate_true": float(subset["pred_qwen"].astype(float).mean()),
    }


def screen_rows(
    frame: pd.DataFrame, prompts: pd.DataFrame, cut: float
) -> list[dict[str, object]]:
    """
    Evaluate one frozen cut on held-out (odd) prompts of one pair.

    Args:
        frame: Shared segment table of the pair.
        prompts: Output of :func:`prompt_table` for the pair.
        cut: Support cut in nats. Prompts without a finite support fail.

    Returns:
        Rows for passing, failing and all prompts, and for passing and failing
        prompts within each surrogate answer.
    """
    odd = scored_prompts(prompts)
    odd = odd[odd["prompt_idx"] % 2 == 1]
    passing = odd["qwen_top_support"] >= cut
    subsets: dict[str, pd.DataFrame] = {
        "pass": odd[passing],
        "fail": odd[~passing],
        "no_screen": odd,
    }
    for answer, name in ((True, "true"), (False, "false")):
        subsets[f"pass_answer_{name}"] = odd[passing & odd["pred_qwen"].eq(answer)]
        subsets[f"fail_answer_{name}"] = odd[~passing & odd["pred_qwen"].eq(answer)]
    return [
        {
            "subset": name,
            "coverage": len(sub) / len(odd) if len(odd) else float("nan"),
            **score_subset(frame, sub),
        }
        for name, sub in subsets.items()
    ]


# ---------------------------------------------------------------------------
# Controls that need a target-free passing set
# ---------------------------------------------------------------------------


def passage_attributions(signals: Signals) -> pd.DataFrame:
    """Every model's Δℓ on passage sentences, one column per model."""
    wide = pd.DataFrame(signals.segment["F_attr"])
    wide.index = wide.index.set_names(["prompt_idx", "seg_idx"])
    return wide[signals.segments["part"].reindex(wide.index).eq("passage").to_numpy()]


def surrogate_support(
    passage: pd.DataFrame, prediction: dict[str, pd.Series], model: str
) -> pd.Series:
    """
    A model's top passage support over all of its own passage sentences.

    Unlike ``qwen_top_support`` this does not depend on which sentences a
    target scored, so the passing set is the same for every target.

    Args:
        passage: Output of :func:`passage_attributions`.
        prediction: ``{model: log-odds}`` indexed by prompt.
        model: The surrogate.

    Returns:
        Support per prompt, for prompts with a finite non-zero log-odds.
    """
    delta = passage[model].replace([np.inf, -np.inf], np.nan).dropna()
    ell = prediction[model].reindex(delta.index.get_level_values("prompt_idx"))
    sign = pd.Series(np.sign(ell.to_numpy()), index=delta.index)
    valid = sign.notna() & sign.ne(0)
    return (delta[valid] * sign[valid]).groupby(level="prompt_idx").max()


def pair_agreement(
    passage: pd.DataFrame,
    prediction: dict[str, pd.Series],
    model_a: str,
    model_b: str,
    prompts: pd.Index,
) -> dict[str, float]:
    """
    Passage-sentence agreement of any two models on a set of prompts.

    Args:
        passage: Output of :func:`passage_attributions`.
        prediction: ``{model: log-odds}`` indexed by prompt.
        model_a: First model.
        model_b: Second model.
        prompts: Prompts to evaluate.

    Returns:
        Pooled passage r² and top-|Δℓ| sentence match with its chance rate,
        over prompts where both log-odds are finite and at least two passage
        sentences are finite for both models. Ties are broken at random.
    """
    pair = passage[[model_a, model_b]].replace([np.inf, -np.inf], np.nan).dropna()
    finite = np.isfinite(prediction[model_a]) & np.isfinite(prediction[model_b])
    keep = pd.Index(prompts).intersection(finite[finite].index)
    pair = pair[pair.index.get_level_values("prompt_idx").isin(keep)]
    sizes = pair.groupby(level="prompt_idx").size()
    pair = pair[pair.index.get_level_values("prompt_idx").isin(sizes[sizes >= 2].index)]
    if pair.index.get_level_values("prompt_idx").nunique() < 3:
        return {"n_prompts": 0}
    rng = np.random.default_rng(SEED)
    matches: list[bool] = []
    chance: list[float] = []
    for _, group in pair.groupby(level="prompt_idx", sort=True):
        a = group[model_a].abs().to_numpy()
        b = group[model_b].abs().to_numpy()
        top_a = rng.choice(np.flatnonzero(a == a.max()))
        top_b = rng.choice(np.flatnonzero(b == b.max()))
        matches.append(bool(top_a == top_b))
        chance.append(1.0 / len(group))
    return {
        "n_prompts": len(matches),
        "passage_r2": float(np.corrcoef(pair[model_a], pair[model_b])[0, 1] ** 2),
        "top1_match": float(np.mean(matches)),
        "top1_chance": float(np.mean(chance)),
    }


def flag_groups(flag_a: pd.Series, flag_b: pd.Series) -> pd.Series:
    """Label prompts as passed by both screens, by one, or by neither."""
    labels = np.select(
        [flag_a & flag_b, flag_a & ~flag_b, ~flag_a & flag_b],
        ["both", "first_only", "second_only"],
        default="neither",
    )
    return pd.Series(labels, index=flag_a.index)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _check_reference(output_dir: str, cut: float, row: dict[str, object]) -> None:
    """Stop if the Qwen2.5-14B → GPT-4o screen differs from ``gate_rule.csv``."""
    path = os.path.join(os.path.dirname(os.path.normpath(output_dir)), REFERENCE_RULE)
    if not os.path.exists(path):
        return
    rule = pd.read_csv(path)
    ref = rule[rule["subset"].eq("qwen_only_supported")].iloc[0]
    pairs = (
        (cut, ref["qwen_only_cut_nats"]),
        (row["n_prompts"], ref["n_prompts"]),
        (row["passage_r2"], ref["passage_r2"]),
        (row["top1_match"], ref["top1_match"]),
        (row["top1_overlap_baseline"], ref["top1_overlap_baseline"]),
    )
    for got, want in pairs:
        if abs(float(got) - float(want)) > REFERENCE_TOLERANCE:
            raise ValueError(f"{path}: expected {want}, recomputed {got}")


def build(results_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    signals = load_signals(results_dir)
    targets: tuple[str, ...] = HOSTED_MODELS
    cut_rows: list[dict[str, object]] = []
    transfer_rows: list[dict[str, object]] = []
    cuts: dict[str, float] = {}
    for surrogate in SURROGATES:
        tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for target in targets:
            frame, ell_s, ell_t, gold = pair_segments(
                results_dir, surrogate, target, signals
            )
            tables[target] = (frame, prompt_table(frame, ell_s, ell_t, gold))
        fit = scored_prompts(tables[FIT_TARGET][1])
        even = fit[fit["prompt_idx"] % 2 == 0]
        odd = fit[fit["prompt_idx"] % 2 == 1]
        cut = fit_support_cut(even)
        cuts[surrogate] = cut
        cut_rows.append(
            {
                "surrogate": surrogate,
                "fit_target": FIT_TARGET,
                "support_cut_nats": cut,
                "support_cut_nats_reversed_split": fit_support_cut(odd),
                "coverage_even": float((even["qwen_top_support"] >= cut).mean()),
                "coverage_odd": float((odd["qwen_top_support"] >= cut).mean()),
                "median_abs_logodds": float(fit["ell_qwen"].abs().median()),
                "median_support": float(fit["qwen_top_support"].median()),
            }
        )
        for target, (frame, prompts) in tables.items():
            for row in screen_rows(frame, prompts, cut):
                transfer_rows.append(
                    {
                        "surrogate": surrogate,
                        "target": target,
                        "fitted_pair": target == FIT_TARGET,
                        "support_cut_nats": cut,
                        **row,
                    }
                )
                if (
                    surrogate == SURROGATES[0]
                    and target == FIT_TARGET
                    and row["subset"] == "pass"
                ):
                    _check_reference(output_dir, cut, row)
    pd.DataFrame(cut_rows).to_csv(
        os.path.join(output_dir, "screen_cuts.csv"), index=False
    )
    pd.DataFrame(transfer_rows).to_csv(
        os.path.join(output_dir, "screen_transfer.csv"), index=False
    )

    passage = passage_attributions(signals)
    support = {s: surrogate_support(passage, signals.prediction, s) for s in SURROGATES}
    flags: dict[str, pd.Series] = {}
    for surrogate in SURROGATES:
        held_out = support[surrogate][support[surrogate].index % 2 == 1]
        flags[surrogate] = held_out >= cuts[surrogate]
    hosted_rows: list[dict[str, object]] = []
    for surrogate in SURROGATES:
        flag = flags[surrogate]
        for model_a, model_b in itertools.combinations(targets, 2):
            for name, index in (
                ("pass", flag[flag].index),
                ("fail", flag[~flag].index),
            ):
                hosted_rows.append(
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
    pd.DataFrame(hosted_rows).to_csv(
        os.path.join(output_dir, "hosted_pairs.csv"), index=False
    )

    first, second = SURROGATES
    common = flags[first].index.intersection(flags[second].index)
    groups = flag_groups(flags[first].loc[common], flags[second].loc[common])
    overlap_rows: list[dict[str, object]] = []
    for target in (None, *targets):
        for group, members in groups.groupby(groups):
            row: dict[str, object] = {
                "first": first,
                "second": second,
                "target": target if target else "none",
                "flagged_by": group,
                "share": len(members) / len(groups),
            }
            if target is None:
                row.update(
                    {
                        f"surrogates_{key}": value
                        for key, value in pair_agreement(
                            passage, signals.prediction, first, second, members.index
                        ).items()
                    }
                )
            else:
                for label, model in (("first", first), ("second", second)):
                    stats = pair_agreement(
                        passage, signals.prediction, model, target, members.index
                    )
                    row.update(
                        {f"{label}_{key}": value for key, value in stats.items()}
                    )
            overlap_rows.append(row)
    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(output_dir, "flag_overlap.csv"), index=False
    )
    both = (groups == "both").sum()
    either = (groups != "neither").sum()
    print(
        f"Wrote {output_dir}: cuts "
        + ", ".join(f"{s} {c} nats" for s, c in cuts.items())
        + f"; flag Jaccard {both / either:.3f} on {len(groups)} held-out prompts"
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
