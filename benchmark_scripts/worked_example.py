# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Worked example: when can Qwen2.5-14B explain GPT-4o on BoolQ?

Reanalysis of the published BoolQ sentence-level results for one open→closed
pair. No model or API calls; nothing is written under ``results/``.

``build`` writes, under ``--output-dir``:

- ``pair_segments.csv.gz``: one row per shared (prompt, sentence) with both
  models' Δℓ and Qwen's internal signals.
- ``prompts.csv``: one row per prompt with answers, correctness and
  per-prompt attribution agreement.
- ``strata.csv``: attribution agreement by answer agreement, gold label,
  correctness and confidence.
- ``gate.csv``: attribution agreement by the API-observable log-odds gap.
- ``pools/<theme>.csv``: candidate prompts for each case-study theme.
- ``gate_rule.csv``, ``gate_bins.csv``: the one-call trust gate, fit on even
  prompts and evaluated on odd prompts.
- ``pair_words.csv.gz``, ``word_summary.csv``: the sampled word level, if the
  consolidated word tables exist.

``inspect`` prints one prompt's sentence-by-sentence table from those files.

Conventions:

- Δℓ_i = ℓ(x) − ℓ(x without sentence i), so Δℓ_i > 0 means sentence i pushes
  toward True.
- Support_i = Δℓ_i · sign(ℓ): how much sentence i supports the model's own
  answer. The "decisive" sentence is the one with the largest support.
- A sentence "flips" a model if deleting it changes the sign of ℓ.
- Per-prompt agreement uses the sentences both models score with finite Δℓ.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from benchmark_scripts.reanalysis_rebuttal import Signals, load_signals, segment_parts

SOURCE: str = "qwen2.5-14b-instruct"
TARGET: str = "gpt-4o"
MIN_RANK_SEGMENTS: int = 3
POOL_SIZE: int = 40
SEED: int = 0


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def pair_segments(
    results_dir: str,
    source: str = SOURCE,
    target: str = TARGET,
    signals: Signals | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Return the shared segment table plus ℓ_S, ℓ_T and gold, indexed by prompt.

    Columns keep the ``_qwen`` (source) and ``_gpt`` (target) suffixes for any
    pair, so the per-prompt and screen functions apply unchanged.

    Args:
        results_dir: Result root (see :func:`load_signals`).
        source: Open surrogate model.
        target: Target model.
        signals: Already-loaded signals, to avoid reloading for several pairs.

    Returns:
        The shared segment table, ℓ_S, ℓ_T and gold.
    """
    if signals is None:
        signals = load_signals(results_dir)
    manifest: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "boolq", "sentence", "segments.tsv.gz"), sep="\t"
    )
    manifest["part"] = segment_parts(manifest)
    manifest = manifest.set_index(["prompt_idx", "seg_idx"])
    attr = signals.segment["F_attr"]
    frame = pd.DataFrame(
        {
            "d_qwen": attr[source],
            "d_gpt": attr[target],
            "align_qwen": signals.segment["F_align"][source],
            "attn_mean_qwen": signals.segment["F_attn_mean"][source],
            "attn_max_qwen": signals.segment["F_attn_max"][source],
            "attn_rollout_qwen": signals.segment["F_attn_rollout"][source],
            "dz_norm_qwen": signals.segment["F_mag"][source],
        }
    )
    frame.index = frame.index.set_names(["prompt_idx", "seg_idx"])
    frame = frame.join(manifest[["part", "message_role", "segment_text"]], how="left")
    # Overlap with the question, counted on the full prompt (the no-model
    # baseline shared with analysis 7).
    frame["question_overlap"] = signals.segments["question_overlap"]
    shared = np.isfinite(frame["d_qwen"]) & np.isfinite(frame["d_gpt"])
    frame = frame[shared].reset_index()
    return (
        frame,
        signals.prediction[source],
        signals.prediction[target],
        signals.gold,
    )


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_RANK_SEGMENTS:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_RANK_SEGMENTS or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _top(
    values: np.ndarray, seg: np.ndarray, rng: np.random.Generator
) -> tuple[int, bool]:
    """Return the seg_idx of the largest value (random tie-break) and a tie flag."""
    best = np.flatnonzero(values == values.max())
    return int(seg[rng.choice(best)]), len(best) > 1


def prompt_table(
    frame: pd.DataFrame, ell_s: pd.Series, ell_t: pd.Series, gold: pd.Series
) -> pd.DataFrame:
    """One row per prompt with answers and per-prompt attribution agreement."""
    rng = np.random.default_rng(SEED)
    # A separate stream, so the baseline's tie-breaks leave the others unchanged.
    overlap_rng = np.random.default_rng(SEED + 1)
    rows: list[dict[str, object]] = []
    for prompt, group in frame.groupby("prompt_idx", sort=True):
        ls = float(ell_s.get(prompt, np.nan))
        lt = float(ell_t.get(prompt, np.nan))
        row: dict[str, object] = {
            "prompt_idx": prompt,
            "gold": bool(gold[prompt]),
            "ell_qwen": ls,
            "ell_gpt": lt,
        }
        question = group.loc[group["part"].eq("question"), "segment_text"]
        row["question"] = " ".join(question).replace("Question:", "").strip()
        seg = group["seg_idx"].to_numpy()
        dq = group["d_qwen"].to_numpy()
        dg = group["d_gpt"].to_numpy()
        passage = group["part"].eq("passage").to_numpy()
        row["n_shared"] = len(group)
        row["n_passage"] = int(passage.sum())
        row["rho_all"] = _spearman(dq, dg)
        row["r_all"] = _pearson(dq, dg)
        row["rho_passage"] = _spearman(dq[passage], dg[passage])
        if passage.any() and np.isfinite(ls) and ls != 0:
            # Needs only Qwen: how strongly its best passage sentence supports
            # its own answer.
            row["qwen_top_support"] = float((dq[passage] * np.sign(ls)).max())
        row["r_passage"] = _pearson(dq[passage], dg[passage])
        if passage.sum() >= 2:
            p_seg = seg[passage]
            top_q, tie_q = _top(np.abs(dq[passage]), p_seg, rng)
            top_g, tie_g = _top(np.abs(dg[passage]), p_seg, rng)
            row.update(
                top_abs_passage_qwen=top_q,
                top_abs_passage_gpt=top_g,
                top_abs_passage_match=top_q == top_g,
                top_abs_passage_tie=tie_q or tie_g,
                chance_passage=1.0 / passage.sum(),
            )
            if "question_overlap" in group:
                overlap = group["question_overlap"].to_numpy()[passage]
                # A random pick for display; the match rate is its expectation:
                # k sentences tied for the most overlap share credit 1/k.
                top_o, _ = _top(overlap, p_seg, overlap_rng)
                tied = p_seg[overlap == overlap.max()]
                row["top_overlap_passage"] = top_o
                row["overlap_top_matches_gpt"] = float(top_g in tied) / len(tied)
            align = group["align_qwen"].to_numpy()[passage]
            attn = group["attn_mean_qwen"].to_numpy()[passage]
            if np.isfinite(align).all():
                top_a, _ = _top(np.abs(align), p_seg, rng)
                row["top_abs_align_qwen"] = top_a
                row["align_top_matches_gpt"] = top_a == top_g
                row["rho_align_qwen_vs_d_gpt_passage"] = _spearman(align, dg[passage])
            if np.isfinite(attn).all():
                top_at, _ = _top(attn, p_seg, rng)
                row["top_attn_qwen"] = top_at
                row["attn_top_matches_gpt"] = top_at == top_g
                row["rho_attn_qwen_vs_abs_d_gpt_passage"] = _spearman(
                    attn, np.abs(dg[passage])
                )
        for name, ell, d in (("qwen", ls, dq), ("gpt", lt, dg)):
            if np.isfinite(ell) and ell != 0:
                # Decisive and flip sentences are passage-only: deleting the
                # question trivially changes the answer.
                flipped = np.sign(ell - d) != np.sign(ell)
                row[f"question_flips_{name}"] = bool(
                    flipped[group["part"].eq("question").to_numpy()].any()
                )
                flips = seg[flipped & passage]
                row[f"flip_segs_{name}"] = " ".join(map(str, flips))
                row[f"n_flips_{name}"] = len(flips)
                if passage.any():
                    support = d[passage] * np.sign(ell)
                    row[f"decisive_{name}"], _ = _top(support, seg[passage], rng)
        rows.append(row)
    out = pd.DataFrame(rows)
    for name in ("qwen", "gpt"):
        for column in ("decisive", "flip_segs", "n_flips", "question_flips"):
            if f"{column}_{name}" not in out:
                out[f"{column}_{name}"] = np.nan
    out["pred_qwen"] = out["ell_qwen"] > 0
    out["pred_gpt"] = out["ell_gpt"] > 0
    valid = np.isfinite(out["ell_qwen"]) & np.isfinite(out["ell_gpt"])
    out["agree"] = (out["pred_qwen"] == out["pred_gpt"]).where(valid)
    out["correct_qwen"] = out["pred_qwen"] == out["gold"]
    out["correct_gpt"] = out["pred_gpt"] == out["gold"]
    out["case"] = np.select(
        [
            out["correct_qwen"] & out["correct_gpt"],
            ~out["correct_qwen"] & ~out["correct_gpt"],
            out["correct_qwen"] & ~out["correct_gpt"],
        ],
        ["both_right", "both_wrong", "only_qwen_right"],
        default="only_gpt_right",
    )
    out.loc[~valid, "case"] = "unscored"
    out["decisive_match"] = out["decisive_qwen"] == out["decisive_gpt"]
    out["gap"] = (out["ell_qwen"] - out["ell_gpt"]).abs()
    out["confidence"] = (out["ell_qwen"].abs() + out["ell_gpt"].abs()) / 2
    out["gpt_flip_found_by_qwen"] = [
        (
            bool(set(a.split()) & set(b.split()))
            if isinstance(a, str) and a and isinstance(b, str)
            else np.nan
        )
        for a, b in zip(out.get("flip_segs_gpt", ""), out.get("flip_segs_qwen", ""))
    ]
    return out


def _pooled_r2(frame: pd.DataFrame, prompts: pd.Index) -> float:
    sub = frame[frame["prompt_idx"].isin(prompts)]
    if len(sub) < 3:
        return float("nan")
    return float(np.corrcoef(sub["d_qwen"], sub["d_gpt"])[0, 1] ** 2)


def summarize(prompts: pd.DataFrame, frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Attribution agreement for each level of ``by``."""
    rows: list[dict[str, object]] = []
    groups = [("all", prompts)] + list(prompts.groupby(by, dropna=True))
    for level, group in groups:
        scored = group[group["top_abs_passage_match"].notna()]
        rows.append(
            {
                "stratifier": by,
                "level": level,
                "n_prompts": len(group),
                "pooled_r2_all": _pooled_r2(frame, pd.Index(group["prompt_idx"])),
                "pooled_r2_passage": _pooled_r2(
                    frame[frame["part"].eq("passage")], pd.Index(group["prompt_idx"])
                ),
                "median_rho_passage": group["rho_passage"].median(),
                "top1_passage_match": scored["top_abs_passage_match"]
                .astype(float)
                .mean(),
                "top1_passage_chance": scored["chance_passage"].mean(),
                "decisive_match": group["decisive_match"].astype(float).mean(),
                "align_top_matches_gpt": group["align_top_matches_gpt"]
                .astype(float)
                .mean(),
                "attn_top_matches_gpt": group["attn_top_matches_gpt"]
                .astype(float)
                .mean(),
                "share_gpt_has_passage_flip": (group["n_flips_gpt"] > 0).mean(),
                "gpt_flip_found_by_qwen": group["gpt_flip_found_by_qwen"]
                .astype(float)
                .mean(),
            }
        )
    return pd.DataFrame(rows)


def build_pools(prompts: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Candidate prompts per theme, with the ordering used to rank them."""
    p = prompts[prompts["case"].ne("unscored") & (prompts["n_passage"] >= 3)]
    pools: dict[str, pd.DataFrame] = {}
    wrong = p[p["case"].isin(["both_wrong", "only_qwen_right"])]
    pools["shared_failure"] = wrong.sort_values(
        ["case", "gpt_flip_found_by_qwen", "confidence"], ascending=[True, False, False]
    )
    gate_rows = []
    for (_, _), cell in p.groupby(
        [pd.qcut(p["gap"], 4, labels=False), p["top_abs_passage_match"]], dropna=True
    ):
        gate_rows.append(cell.sample(min(10, len(cell)), random_state=SEED))
    pools["trust_gate"] = pd.concat(gate_rows).sort_values("gap")
    both = p[p["case"].eq("both_right")]
    pools["different_reasons"] = both[~both["decisive_match"]].sort_values(
        ["rho_passage", "confidence"], ascending=[True, False]
    )
    inside = p[p["align_top_matches_gpt"].eq(True)]
    pools["inside_qwen"] = pd.concat(
        [
            inside.sort_values("rho_align_qwen_vs_d_gpt_passage", ascending=False).head(
                POOL_SIZE // 2
            ),
            p[p["attn_top_matches_gpt"].eq(False) & p["align_top_matches_gpt"].eq(True)]
            .sort_values("confidence", ascending=False)
            .head(POOL_SIZE // 2),
        ]
    ).drop_duplicates("prompt_idx")
    return {k: v.head(POOL_SIZE) for k, v in pools.items()}


def build(results_dir: str, output_dir: str) -> None:
    os.makedirs(os.path.join(output_dir, "pools"), exist_ok=True)
    frame, ell_s, ell_t, gold = pair_segments(results_dir)
    frame.to_csv(os.path.join(output_dir, "pair_segments.csv.gz"), index=False)
    prompts = prompt_table(frame, ell_s, ell_t, gold)
    prompts.to_csv(os.path.join(output_dir, "prompts.csv"), index=False)
    scored = prompts[prompts["case"].ne("unscored")].copy()
    scored["agree_x_gold"] = (
        scored["agree"].map({True: "agree", False: "disagree"})
        + "/gold_"
        + scored["gold"].map({True: "true", False: "false"})
    )
    scored["confidence_quintile"] = pd.qcut(scored["confidence"], 5, labels=False) + 1
    strata = pd.concat(
        [
            summarize(scored, frame, by)
            for by in ("agree_x_gold", "case", "confidence_quintile")
        ]
    )
    strata.to_csv(os.path.join(output_dir, "strata.csv"), index=False)
    agreeing = scored[scored["agree"].eq(True)].copy()
    agreeing["gap_quintile"] = pd.qcut(agreeing["gap"], 5, labels=False) + 1
    scored["gap_quintile_all"] = pd.qcut(scored["gap"], 5, labels=False) + 1
    gate = pd.concat(
        [
            summarize(scored, frame, "gap_quintile_all"),
            summarize(agreeing, frame, "gap_quintile").assign(subset="agree_only"),
        ]
    )
    gate.to_csv(os.path.join(output_dir, "gate.csv"), index=False)
    for theme, pool in build_pools(prompts).items():
        pool.to_csv(os.path.join(output_dir, "pools", f"{theme}.csv"), index=False)
    print(f"Wrote {output_dir}: {len(prompts)} prompts, {len(frame)} shared segments")
    retest = gpt_retest(output_dir, frame)
    rule, bins, threshold = gate_analysis(prompts, frame, retest)
    rule.to_csv(os.path.join(output_dir, "gate_rule.csv"), index=False)
    bins.to_csv(os.path.join(output_dir, "gate_bins.csv"), index=False)
    print(f"Trust gate: |ℓ_qwen − ℓ_gpt| < {threshold} nats (fit on even prompts)")
    words = word_table(results_dir, frame, prompts)
    if words is not None:
        words.to_csv(os.path.join(output_dir, "pair_words.csv.gz"), index=False)
        word_summary(words).to_csv(
            os.path.join(output_dir, "word_summary.csv"), index=False
        )
        print(f"Wrote word-level table: {len(words)} shared words")


# ---------------------------------------------------------------------------
# Word level (a fixed random sample of 10,000 word coordinates)
# ---------------------------------------------------------------------------


def word_table(
    results_dir: str, sentences: pd.DataFrame, prompts: pd.DataFrame
) -> pd.DataFrame | None:
    """
    One row per sampled word both models score, with its context.

    The word run ablates a fixed random sample of 10,000 words (about three
    per prompt), so it supports pooled and per-word analyses, not per-prompt
    rankings. Requires the consolidated ``boolq_word_*`` scratch tables
    (README step 3); returns None if they are absent.
    """
    from benchmark_scripts.f_table import (
        _per_prompt_logodds,
        _per_seg_ablation,
        _per_seg_align,
        _per_seg_segment_col,
    )
    from surrogate.utils import segment_text, WORD_RX

    odds_path = os.path.join(results_dir, "boolq_word_logodds.tsv")
    seg_path = os.path.join(results_dir, "boolq_word_segments.tsv")
    if not (os.path.exists(odds_path) and os.path.exists(seg_path)):
        print("Word-level scratch tables absent; skipping word table")
        return None
    odds = pd.read_csv(odds_path, sep="\t")
    seg_df = pd.read_csv(seg_path, sep="\t", low_memory=False)

    def indexed(signal: pd.Series | None, name: str) -> pd.Series:
        if signal is None:
            raise ValueError(f"word-level field missing: {name}")
        signal.index = signal.index.set_names(["prompt_idx", "seg_idx"])
        return signal.rename(name)

    frame = pd.concat(
        [
            indexed(_per_seg_ablation(odds, SOURCE), "d_qwen"),
            indexed(_per_seg_ablation(odds, TARGET), "d_gpt"),
            indexed(_per_seg_align(seg_df, SOURCE), "align_qwen"),
            indexed(
                _per_seg_segment_col(seg_df, SOURCE, "attention_mean"),
                "attn_mean_qwen",
            ),
            indexed(
                _per_seg_segment_col(seg_df, SOURCE, "delta_norm_postnorm"),
                "dz_norm_qwen",
            ),
        ],
        axis=1,
    )
    frame = frame[np.isfinite(frame["d_qwen"]) & np.isfinite(frame["d_gpt"])]
    word_ell = {
        name: _per_prompt_logodds(odds, model, "orig")
        for name, model in (("qwen", SOURCE), ("gpt", TARGET))
    }

    manifest = pd.read_csv(
        os.path.join(results_dir, "boolq", "word", "segments.tsv.gz"), sep="\t"
    ).set_index(["prompt_idx", "seg_idx"])
    frame = frame.join(
        manifest[["message_idx", "message_role", "message_seg_idx", "segment_text"]]
    ).reset_index()

    # Place each word in its message and in the sentence that contains it.
    sentence_manifest = pd.read_csv(
        os.path.join(results_dir, "boolq", "sentence", "segments.tsv.gz"), sep="\t"
    )
    sentence_manifest["part"] = segment_parts(sentence_manifest)
    messages = {
        key: group.sort_values("message_seg_idx")
        for key, group in sentence_manifest.groupby(["prompt_idx", "message_idx"])
    }
    sent_d = sentences.set_index(["prompt_idx", "seg_idx"])[["d_qwen", "d_gpt"]]
    split_cache: dict[tuple[int, int], tuple[str, list[str], np.ndarray]] = {}
    extra: list[dict[str, object]] = []
    for row in frame.itertuples():
        key = (int(row.prompt_idx), int(row.message_idx))
        if key not in split_cache:
            text = "".join(messages[key]["segment_text"])
            pieces = WORD_RX.split(text)
            delimiters, words = pieces[::2], pieces[1::2]
            if words != segment_text(text, level="word")[0]:
                raise ValueError(f"word split mismatch at {key}")
            lengths = np.array([len(w) for w in words])
            delim_lengths = np.array([len(d) for d in delimiters[:-1]])
            starts = np.cumsum(delim_lengths) + np.concatenate(
                [[0], np.cumsum(lengths)[:-1]]
            )
            split_cache[key] = (text, words, starts)
        text, words, starts = split_cache[key]
        i = int(row.message_seg_idx)
        if words[i] != row.segment_text:
            raise ValueError(f"word coordinate mismatch at {key}, word {i}")
        start, end = int(starts[i]), int(starts[i]) + len(words[i])
        lo = int(starts[max(0, i - 15)])
        hi = int(starts[i + 15]) if i + 15 < len(words) else len(text)
        context = (text[lo:start] + "[[" + words[i] + "]]" + text[end:hi]).replace(
            "\n", "⏎"
        )
        sents = messages[key]
        sent_starts = np.cumsum([0] + [len(s) for s in sents["segment_text"].iloc[:-1]])
        j = int(np.searchsorted(sent_starts, start, side="right") - 1)
        sentence = sents.iloc[j]
        s_key = (int(row.prompt_idx), int(sentence["seg_idx"]))
        extra.append(
            {
                "word": row.segment_text,
                "context": context.strip(),
                "sentence_seg_idx": int(sentence["seg_idx"]),
                "part": sentence["part"],
                "sentence_d_qwen": sent_d["d_qwen"].get(s_key, np.nan),
                "sentence_d_gpt": sent_d["d_gpt"].get(s_key, np.nan),
            }
        )
    frame = pd.concat([frame, pd.DataFrame(extra, index=frame.index)], axis=1)
    p = prompts.set_index("prompt_idx")
    for name in ("qwen", "gpt"):
        ell = word_ell[name]
        if ell is None:
            raise ValueError(f"word-level ℓ missing for {name}")
        frame[f"ell_{name}"] = frame["prompt_idx"].map(ell)
        sentence_ell = frame["prompt_idx"].map(p[f"ell_{name}"])
        # The unablated prompt is identical across pregroupers.
        frame[f"ell_{name}_matches_sentence_run"] = np.isclose(
            frame[f"ell_{name}"], sentence_ell, atol=1e-6
        ) | (frame[f"ell_{name}"] == sentence_ell)
        e = frame[f"ell_{name}"]
        frame[f"flips_{name}"] = np.isfinite(e) & (
            np.sign(e - frame[f"d_{name}"]) != np.sign(e)
        )
    frame["gold"] = frame["prompt_idx"].map(p["gold"])
    frame["case"] = frame["prompt_idx"].map(p["case"])
    frame["question"] = frame["prompt_idx"].map(p["question"])
    return frame


def word_summary(words: pd.DataFrame) -> pd.DataFrame:
    """Pooled word-level agreement by part and by answer agreement."""
    rows: list[dict[str, object]] = []
    groups = [("all", "all", words)]
    groups += [("part", k, g) for k, g in words.groupby("part")]
    groups += [("case", k, g) for k, g in words.groupby("case")]
    for stratifier, level, g in groups:
        r = np.corrcoef(g["d_qwen"], g["d_gpt"])[0, 1] if len(g) > 2 else np.nan
        flips_g = g["flips_gpt"]
        rows.append(
            {
                "stratifier": stratifier,
                "level": level,
                "n_words": len(g),
                "n_prompts": g["prompt_idx"].nunique(),
                "pearson_r2": r * r,
                "spearman": g["d_qwen"].rank().corr(g["d_gpt"].rank()),
                "sign_agreement_nonzero": (np.sign(g["d_qwen"]) == np.sign(g["d_gpt"]))[
                    (g["d_qwen"] != 0) & (g["d_gpt"] != 0)
                ].mean(),
                "n_gpt_flips": int(flips_g.sum()),
                "n_qwen_flips": int(g["flips_qwen"].sum()),
                "n_both_flip": int((flips_g & g["flips_qwen"]).sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trust gate: one GPT-4o call decides whether to trust Qwen's attributions
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: tuple[int, ...] = tuple(range(1, 26))
N_BOOTSTRAP: int = 2000
N_GATE_BINS: int = 10
SUPPORT_CUTS: tuple[float, ...] = tuple(np.arange(0.5, 40.5, 0.5))
REPEATS_FILE: str = "gpt4o_repeats.tsv.gz"


def _prompt_moments(segments: pd.DataFrame) -> pd.DataFrame:
    """Per-prompt sums for pooled passage r² under prompt resampling."""
    passage = segments[segments["part"].eq("passage")]
    x, y = passage["d_qwen"], passage["d_gpt"]
    return (
        pd.DataFrame(
            {
                "prompt_idx": passage["prompt_idx"],
                "n": 1.0,
                "x": x,
                "y": y,
                "xx": x * x,
                "yy": y * y,
                "xy": x * y,
            }
        )
        .groupby("prompt_idx")
        .sum()
    )


def _r2_from_moments(m: np.ndarray) -> np.ndarray:
    """Pooled Pearson r² from summed moments; rows are [n, x, y, xx, yy, xy]."""
    n, x, y, xx, yy, xy = (m[..., k] for k in range(6))
    cov = xy - x * y / n
    vx = xx - x * x / n
    vy = yy - y * y / n
    return cov * cov / (vx * vy)


def _within_r2(segments: pd.DataFrame, prompts: pd.Index, x: str, y: str) -> float:
    """Pooled r² of two passage columns after centring each within its prompt."""
    passage = segments[
        segments["part"].eq("passage") & segments["prompt_idx"].isin(prompts)
    ]
    sub = passage[["prompt_idx", x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    centred = sub[[x, y]] - sub.groupby("prompt_idx")[[x, y]].transform("mean")
    if len(centred) < 3 or centred[x].std() == 0:
        return float("nan")
    return float(np.corrcoef(centred[x], centred[y])[0, 1] ** 2)


def _r2_with_ci(
    moments: pd.DataFrame, prompts: pd.Index, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Pooled passage r² with a 95% prompt-cluster bootstrap interval."""
    m = moments.reindex(prompts).dropna().to_numpy()
    point = float(_r2_from_moments(m.sum(axis=0)))
    counts = rng.multinomial(len(m), np.full(len(m), 1 / len(m)), size=N_BOOTSTRAP)
    draws = _r2_from_moments(counts @ m)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def _chi2(accepted: pd.Series, rejected: pd.Series) -> float:
    """Pearson χ² for a 2×2 table of top-1 match by gate decision."""
    table = np.array(
        [
            [accepted.sum(), (~accepted).sum()],
            [rejected.sum(), (~rejected).sum()],
        ],
        dtype=float,
    )
    if (table.sum(axis=1) == 0).any() or (table.sum(axis=0) == 0).any():
        return 0.0
    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0) / table.sum()
    return float(((table - expected) ** 2 / expected).sum())


def _retest_summary(
    retest: tuple[pd.Series, pd.DataFrame] | None, prompts: pd.Index
) -> dict[str, float]:
    """GPT-4o's top-1 and pooled passage r² against its own repeats."""
    if retest is None:
        return {"gpt_retest_top1": np.nan, "gpt_retest_passage_r2": np.nan}
    shares, table = retest
    sub = table[table.index.get_level_values(0).isin(prompts)]
    r2s = []
    for column in sub.columns.drop("published"):
        pair = sub[["published", column]].replace([np.inf, -np.inf], np.nan).dropna()
        r2s.append(np.corrcoef(pair["published"], pair[column])[0, 1] ** 2)
    return {
        "gpt_retest_top1": float(shares.reindex(prompts).mean()),
        "gpt_retest_passage_r2": float(np.mean(r2s)),
    }


def gate_analysis(
    prompts: pd.DataFrame,
    segments: pd.DataFrame,
    retest: tuple[pd.Series, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Fit the two-stage screen on even prompts and evaluate it on odd prompts.

    Stage 1 (one GPT-4o call): accept when |ℓ_qwen − ℓ_gpt| < t.
    Stage 2 (Qwen only): among accepted prompts, also require Qwen's top
    passage support, max_i Δℓ_i·sign(ℓ_qwen), to be at least c.
    t, then c, each maximize the χ² of top-1 passage match between the two
    sides of the cut on even prompts. For comparison, a Qwen-only screen
    (support ≥ c' with no GPT-4o call) is fit the same way on all even
    prompts, and also at the two-stage screen's even-prompt coverage.

    Args:
        prompts: Output of :func:`prompt_table`.
        segments: The shared segment table.
        retest: Optional output of :func:`gpt_retest`: GPT-4o's agreement
            with its own repeat runs, a noise ceiling.

    Returns:
        The held-out evaluation rows, the per-bin rows for the figure, and t.
    """
    p = prompts[
        prompts["case"].ne("unscored") & prompts["top_abs_passage_match"].notna()
    ].copy()
    p["top1"] = p["top_abs_passage_match"].astype(bool)
    even = p[p["prompt_idx"] % 2 == 0]
    odd = p[p["prompt_idx"] % 2 == 1]
    threshold = max(
        GATE_THRESHOLDS,
        key=lambda t: _chi2(
            even.loc[even["gap"] < t, "top1"], even.loc[even["gap"] >= t, "top1"]
        ),
    )
    even_accepted = even[even["gap"] < threshold]
    support_cut = max(
        SUPPORT_CUTS,
        key=lambda c: _chi2(
            even_accepted.loc[even_accepted["qwen_top_support"] >= c, "top1"],
            even_accepted.loc[even_accepted["qwen_top_support"] < c, "top1"],
        ),
    )
    qwen_only_cut = max(
        SUPPORT_CUTS,
        key=lambda c: _chi2(
            even.loc[even["qwen_top_support"] >= c, "top1"],
            even.loc[even["qwen_top_support"] < c, "top1"],
        ),
    )
    two_stage_even = (
        (even["gap"] < threshold) & (even["qwen_top_support"] >= support_cut)
    ).mean()
    matched_cut = float(even["qwen_top_support"].quantile(1 - two_stage_even))
    moments = _prompt_moments(segments)
    abs_segments = segments.assign(
        abs_qwen=segments["d_qwen"].abs(), abs_gpt=segments["d_gpt"].abs()
    )
    rng = np.random.default_rng(SEED)
    answer_rng = np.random.default_rng(SEED + 2)
    rows: list[dict[str, object]] = []
    accepted = odd["gap"] < threshold
    supported = odd["qwen_top_support"] >= support_cut
    subsets = {
        "ungated": odd,
        "accepted": odd[accepted],
        "rejected": odd[~accepted],
        "accepted_supported": odd[accepted & supported],
        "accepted_unsupported": odd[accepted & ~supported],
        "not_accepted_supported": odd[~(accepted & supported)],
        "qwen_only_supported": odd[odd["qwen_top_support"] >= qwen_only_cut],
        "qwen_only_unsupported": odd[odd["qwen_top_support"] < qwen_only_cut],
        "qwen_only_matched_coverage": odd[odd["qwen_top_support"] >= matched_cut],
        "rejected_and_agree": odd[(odd["gap"] >= threshold) & odd["agree"].eq(True)],
        "disagree": odd[odd["agree"].eq(False)],
        # The screen passes mostly prompts Qwen answers True.
        "accepted_supported_pred_true": odd[accepted & supported & odd["pred_qwen"]],
        "accepted_supported_pred_false": odd[accepted & supported & ~odd["pred_qwen"]],
        "not_accepted_supported_pred_true": odd[
            ~(accepted & supported) & odd["pred_qwen"]
        ],
        "not_accepted_supported_pred_false": odd[
            ~(accepted & supported) & ~odd["pred_qwen"]
        ],
    }
    for name, sub in subsets.items():
        index = pd.Index(sub["prompt_idx"])
        # Answer-split rows draw from their own stream, so the rows and bins
        # above keep their bootstrap intervals.
        r2, lo, hi = _r2_with_ci(
            moments, index, answer_rng if "_pred_" in name else rng
        )
        rows.append(
            {
                "split": "odd",
                "subset": name,
                "threshold_nats": threshold,
                "support_cut_nats": support_cut,
                "qwen_only_cut_nats": qwen_only_cut,
                "qwen_only_matched_cut_nats": matched_cut,
                "n_prompts": len(sub),
                "coverage": len(sub) / len(odd),
                "passage_r2": r2,
                "passage_r2_lo": lo,
                "passage_r2_hi": hi,
                "top1_match": sub["top1"].mean(),
                "top1_chance": sub["chance_passage"].mean(),
                "top1_overlap_baseline": (
                    sub["overlap_top_matches_gpt"].astype(float).mean()
                    if "overlap_top_matches_gpt" in sub
                    else np.nan
                ),
                "abs_within_r2_qwen": _within_r2(abs_segments, index, "abs_qwen", "abs_gpt"),
                "abs_within_r2_overlap": (
                    _within_r2(abs_segments, index, "question_overlap", "abs_gpt")
                    if "question_overlap" in abs_segments
                    else np.nan
                ),
                "share_pred_true": sub["pred_qwen"].astype(float).mean(),
                "decisive_match": sub["decisive_match"].astype(float).mean(),
                "median_gpt_calls_full_attribution": float(
                    sub["n_shared"].median() + 1
                ),
                **_retest_summary(retest, pd.Index(sub["prompt_idx"])),
            }
        )
    bins = pd.qcut(odd["gap"], N_GATE_BINS, labels=False)
    bin_rows: list[dict[str, object]] = []
    for b, sub in odd.groupby(bins):
        r2, lo, hi = _r2_with_ci(moments, pd.Index(sub["prompt_idx"]), rng)
        bin_rows.append(
            {
                "bin": int(b) + 1,
                "gap_min": sub["gap"].min(),
                "gap_max": sub["gap"].max(),
                "gap_median": sub["gap"].median(),
                "n_prompts": len(sub),
                "share_agree": sub["agree"].astype(float).mean(),
                "passage_r2": r2,
                "passage_r2_lo": lo,
                "passage_r2_hi": hi,
                "top1_match": sub["top1"].mean(),
                "top1_chance": sub["chance_passage"].mean(),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bin_rows), threshold


def extract_repeats(reps_dir: str, output_dir: str) -> str:
    """
    Write GPT-4o's rows from repeat runs to a small table.

    ``reps_dir`` holds ``rep*/boolq_sentence_logodds.tsv`` from repeated
    hosted runs of the same coordinates (the noise-ceiling repeats).
    """
    frames: list[pd.DataFrame] = []
    for name in sorted(os.listdir(reps_dir)):
        path = os.path.join(reps_dir, name, "boolq_sentence_logodds.tsv")
        if not os.path.isfile(path):
            continue
        odds = pd.read_csv(path, sep="\t")
        odds = odds[odds["model"].eq(TARGET)]
        frames.append(
            odds[["prompt_idx", "seg_idx", "kind", "logodds"]].assign(rep=name)
        )
    if not frames:
        raise FileNotFoundError(f"no rep*/boolq_sentence_logodds.tsv under {reps_dir}")
    out = os.path.join(output_dir, REPEATS_FILE)
    pd.concat(frames).to_csv(out, sep="\t", index=False)
    return out


def gpt_retest(
    output_dir: str, segments: pd.DataFrame
) -> tuple[pd.Series, pd.DataFrame] | None:
    """
    GPT-4o's agreement with its own repeat runs on the pair's shared passage
    sentences.

    Returns:
        Per prompt, the share of repeats whose top-|Δℓ| passage sentence
        matches the published run's (prompts where any repeat is non-finite
        on a shared sentence drop those sentences), and the passage Δℓ table
        with one column per run. None if the repeats table is absent.
    """
    path = os.path.join(output_dir, REPEATS_FILE)
    if not os.path.isfile(path):
        return None
    reps = pd.read_csv(path, sep="\t")
    passage = segments[segments["part"].eq("passage")].set_index(
        ["prompt_idx", "seg_idx"]
    )["d_gpt"]
    columns: dict[str, pd.Series] = {"published": passage}
    for name, rep in reps.groupby("rep"):
        orig = rep[rep["kind"].eq("orig")].set_index("prompt_idx")["logodds"]
        abl = rep[rep["kind"].eq("ablated")].copy()
        abl["d"] = abl["prompt_idx"].map(orig) - abl["logodds"]
        abl["seg_idx"] = abl["seg_idx"].astype(int)
        columns[str(name)] = abl.set_index(["prompt_idx", "seg_idx"])["d"]
    table = pd.DataFrame(columns).reindex(passage.index)
    rng = np.random.default_rng(SEED)
    shares: dict[int, float] = {}
    for prompt, group in table.groupby(level=0):
        group = group.replace([np.inf, -np.inf], np.nan).dropna()
        if len(group) < 2:
            continue
        seg = group.index.get_level_values(1).to_numpy()
        top_pub, _ = _top(group["published"].abs().to_numpy(), seg, rng)
        hits = [
            _top(group[c].abs().to_numpy(), seg, rng)[0] == top_pub
            for c in group.columns
            if c != "published"
        ]
        shares[int(prompt)] = float(np.mean(hits))
    return pd.Series(shares), table


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------


def inspect_prompt(output_dir: str, prompt: int, width: int) -> str:
    prompts = pd.read_csv(os.path.join(output_dir, "prompts.csv")).set_index(
        "prompt_idx"
    )
    segs = pd.read_csv(os.path.join(output_dir, "pair_segments.csv.gz"))
    if prompt not in prompts.index:
        raise SystemExit(f"prompt {prompt} not in prompts.csv")
    row = prompts.loc[prompt]
    seg = segs[segs["prompt_idx"].eq(prompt)].copy()
    lines = [
        f"prompt {prompt} | gold={row['gold']} | case={row['case']}",
        f"Q: {row['question']}",
        f"ℓ_qwen={row['ell_qwen']:.3f} (pred {row['pred_qwen']}) | "
        f"ℓ_gpt={row['ell_gpt']:.3f} (pred {row['pred_gpt']}) | gap={row['gap']:.3f}",
        f"shared sentences={row['n_shared']} (passage {row['n_passage']}) | "
        f"rho_all={row['rho_all']:.3f} rho_passage={row['rho_passage']:.3f} "
        f"r_passage={row['r_passage']:.3f}",
        f"decisive: qwen={row['decisive_qwen']} gpt={row['decisive_gpt']} | "
        f"top|Δℓ| passage: qwen={row.get('top_abs_passage_qwen')} "
        f"gpt={row.get('top_abs_passage_gpt')} | "
        f"top|align| qwen={row.get('top_abs_align_qwen')} | "
        f"top attn qwen={row.get('top_attn_qwen')}",
        f"flip sentences: qwen=[{row.get('flip_segs_qwen', '')}] "
        f"gpt=[{row.get('flip_segs_gpt', '')}]",
        "",
        f"{'seg':>4} {'part':<8} {'Δℓ_qwen':>8} {'Δℓ_gpt':>8} {'rk_q':>4} {'rk_g':>4} "
        f"{'align_q':>8} {'attn_q':>7} {'|Δz|_q':>7}  text",
    ]
    seg["rk_q"] = seg["d_qwen"].abs().rank(ascending=False, method="min").astype(int)
    seg["rk_g"] = seg["d_gpt"].abs().rank(ascending=False, method="min").astype(int)
    for _, s in seg.iterrows():
        text = str(s["segment_text"]).replace("\n", "⏎")
        if width and len(text) > width:
            text = text[: width - 1] + "…"
        lines.append(
            f"{s['seg_idx']:>4} {s['part']:<8} {s['d_qwen']:>8.3f} {s['d_gpt']:>8.3f} "
            f"{s['rk_q']:>4} {s['rk_g']:>4} {s['align_qwen']:>8.3f} "
            f"{s['attn_mean_qwen']:>7.4f} {s['dz_norm_qwen']:>7.2f}  {text}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--results-dir", default="results")
    b.add_argument("--output-dir", default="worked_example")
    r = sub.add_parser("extract-repeats")
    r.add_argument("--reps-dir", required=True)
    r.add_argument("--output-dir", default="worked_example")
    i = sub.add_parser("inspect")
    i.add_argument("prompts", type=int, nargs="+")
    i.add_argument("--output-dir", default="worked_example")
    i.add_argument("--width", type=int, default=0, help="Truncate text (0 = full)")
    args = parser.parse_args()
    if args.command == "build":
        build(args.results_dir, args.output_dir)
    elif args.command == "extract-repeats":
        print(f"Wrote {extract_repeats(args.reps_dir, args.output_dir)}")
    else:
        print(
            "\n\n".join(
                inspect_prompt(args.output_dir, p, args.width) for p in args.prompts
            )
        )
    sys.stdout.flush()


if __name__ == "__main__":
    main()
