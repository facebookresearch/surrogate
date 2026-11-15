# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Plot the surrogate-only screen for the worked example (Qwen2.5-14B).

Reads ``prompts.csv``, ``pair_segments.csv.gz`` and ``gate_rule.csv`` written by
``benchmark_scripts.worked_example build`` and ``transfer/screen_cuts.csv`` and
``transfer/screen_transfer.csv`` written by ``benchmark_scripts.screen_transfer``,
and writes ``figures/fig_screen.{pdf,png}`` under the same directory.

Panels: (a) passage r² against GPT-4o by decile of the surrogate's own support;
(b) top-sentence match on passing and failing held-out prompts for each hosted
target; (c, d) one passing and one failing prompt.
"""

from __future__ import annotations

import argparse
import os
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from benchmark_scripts.plot_worked_example import (  # noqa: E402
    ATTR,
    CHANCE,
    GPT,
    INK,
    MUTED,
    QWEN,
    _short,
    _style,
)
from benchmark_scripts.screen_transfer import scored_prompts  # noqa: E402
from benchmark_scripts.worked_example import (  # noqa: E402
    SEED,
    _prompt_moments,
    _r2_with_ci,
)

SURROGATE: str = "qwen2.5-14b-instruct"
FIT_TARGET: str = "gpt-4o"
N_BINS: int = 10
# A passing prompt and a failing one. The failing prompt is the held-out prompt
# with the median surrogate support among failing prompts that both models
# answer correctly, with three to five passage sentences, differing top
# sentences, and neither top sentence first in the passage.
SHOWCASE: tuple[tuple[int, str], ...] = ((3228, "passes"), (525, "fails"))
TARGETS: tuple[tuple[str, str], ...] = (
    ("gpt-4o", "GPT-4o (cut fit here)"),
    ("gpt-4-1", "GPT-4.1"),
    ("gemini-2-5-flash-lite-vertex", "Gemini 2.5 Flash Lite"),
    ("llama3.1-70b-instruct", "Llama-3.1-70B"),
    ("llama3.3-70b-instruct", "Llama-3.3-70B"),
    ("llama4-maverick-17b-128e-instruct", "Llama-4 Maverick"),
)


def support_bins(
    prompts: pd.DataFrame, segments: pd.DataFrame, n_bins: int = N_BINS
) -> pd.DataFrame:
    """
    Passage r² on held-out prompts by quantile bin of surrogate support.

    Args:
        prompts: Output of ``worked_example.prompt_table``.
        segments: The shared segment table of the pair.
        n_bins: Number of quantile bins.

    Returns:
        One row per bin with its median support, passage r² and a 95%
        prompt-cluster bootstrap interval.
    """
    odd = scored_prompts(prompts)
    odd = odd[(odd["prompt_idx"] % 2 == 1) & odd["qwen_top_support"].notna()]
    bins = pd.qcut(odd["qwen_top_support"], n_bins, labels=False, duplicates="drop")
    moments = _prompt_moments(segments)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float]] = []
    for b, sub in odd.groupby(bins):
        r2, lo, hi = _r2_with_ci(moments, pd.Index(sub["prompt_idx"]), rng)
        rows.append(
            {
                "bin": int(b) + 1,
                "support_median": float(sub["qwen_top_support"].median()),
                "n_prompts": len(sub),
                "passage_r2": r2,
                "passage_r2_lo": lo,
                "passage_r2_hi": hi,
                "top1_match": float(sub["top1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _prompt_panel(
    ax: plt.Axes,
    letter: str,
    verdict: str,
    row: pd.Series,
    seg: pd.DataFrame,
    legend: bool,
) -> None:
    """Draw both models' Δℓ for the passage sentences and the question."""
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.5)
    seg = seg[seg["part"].isin(["passage", "question"])]
    n_passage = int(seg["part"].eq("passage").sum())
    # The question sits below the passage, separated by a gap and a rule.
    y = np.concatenate([np.arange(n_passage)[::-1] + 1.4, [0.0]])
    h = 0.36
    ax.barh(
        y + h / 2,
        seg["d_qwen"],
        height=h,
        color=QWEN,
        label="Qwen2.5-14B",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.barh(
        y - h / 2,
        seg["d_gpt"],
        height=h,
        color=GPT,
        label="GPT-4o",
        edgecolor="white",
        linewidth=0.5,
    )
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.axhline(0.7, color="#e6e5e1", linewidth=0.8)
    labels = [_short(t) for t in seg["segment_text"]]
    labels[-1] = "the question"
    ax.set_yticks(y, labels, fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Δℓ when the segment is deleted (nats)", fontsize=7, color=MUTED)
    question = textwrap.shorten(row["question"].rstrip("."), width=44, placeholder="…")
    answer = lambda value: "True" if value else "False"  # noqa: E731
    ax.set_title(
        f"{letter}  {verdict}: “{question}?”\n"
        f"support {row['qwen_top_support']:.1f} nats · Qwen {answer(row['pred_qwen'])}, "
        f"GPT-4o {answer(row['pred_gpt'])}, gold {row['gold']}",
        fontsize=7,
        loc="left",
        color=INK,
    )
    if pd.notna(row.get("top_overlap_passage")):
        hit = np.flatnonzero(seg["seg_idx"].to_numpy() == row["top_overlap_passage"])
        if len(hit) and row["top_overlap_passage"] != row["top_abs_passage_gpt"]:
            ax.annotate(
                (
                    "word-overlap pick →"
                    if seg["d_qwen"].iloc[hit[0]] > 0
                    else "← word-overlap pick"
                ),
                xy=(0.02 if seg["d_qwen"].iloc[hit[0]] > 0 else 0.98, y[hit[0]]),
                xycoords=("axes fraction", "data"),
                ha="left" if seg["d_qwen"].iloc[hit[0]] > 0 else "right",
                va="center",
                fontsize=6,
                color=MUTED,
            )
    if legend:
        ax.legend(fontsize=6.5, frameon=False, loc="center left")


def plot(output_dir: str) -> str:
    prompts = pd.read_csv(os.path.join(output_dir, "prompts.csv"))
    segments = pd.read_csv(os.path.join(output_dir, "pair_segments.csv.gz"))
    rule = pd.read_csv(os.path.join(output_dir, "gate_rule.csv")).set_index("subset")
    cuts = pd.read_csv(os.path.join(output_dir, "transfer", "screen_cuts.csv"))
    transfer = pd.read_csv(os.path.join(output_dir, "transfer", "screen_transfer.csv"))
    cut = float(cuts.set_index("surrogate").loc[SURROGATE, "support_cut_nats"])
    transfer = transfer[transfer["surrogate"].eq(SURROGATE)]

    fig = plt.figure(figsize=(7.0, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.3], hspace=0.62, wspace=1.05)

    # (a) agreement against the surrogate's own support
    bins = support_bins(prompts, segments)
    ax = fig.add_subplot(grid[0, 0])
    _style(ax)
    x = bins["bin"].to_numpy()
    ax.vlines(
        x, bins["passage_r2_lo"], bins["passage_r2_hi"], color=ATTR, linewidth=1.2
    )
    ax.plot(
        x,
        bins["passage_r2"],
        "o",
        color=ATTR,
        markersize=4.5,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x, [f"{v:.0f}" for v in bins["support_median"]])
    ax.set_xlabel(
        "Qwen's top passage support (nats), decile median", fontsize=7, color=MUTED
    )
    ax.set_ylabel("passage attribution r² with GPT-4o", fontsize=7, color=MUTED)
    ax.set_title(
        "a  Agreement rises with the surrogate's own support",
        fontsize=8,
        loc="left",
        color=INK,
    )
    cut_x = float(x[bins["support_median"] < cut].max()) + 0.5
    ax.axvline(cut_x, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    top = ax.get_ylim()[1]
    ax.text(cut_x - 0.15, top, "fail", ha="right", va="top", fontsize=7, color=INK)
    ax.text(
        cut_x + 0.15,
        top,
        f"pass (≥ {cut:g} nats)",
        ha="left",
        va="top",
        fontsize=7,
        color=INK,
    )

    # (b) the same prompts against every hosted target
    ax = fig.add_subplot(grid[0, 1])
    _style(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.5)
    ys = np.arange(len(TARGETS))[::-1]
    chance = []
    for y, (target, _) in zip(ys, TARGETS):
        sub = transfer[transfer["target"].eq(target)].set_index("subset")
        passed, failed = sub.loc["pass"], sub.loc["fail"]
        chance.append(passed["top1_chance"])
        ax.plot(
            [failed["top1_match"], passed["top1_match"]],
            [y, y],
            color=ATTR,
            linewidth=1.2,
            zorder=1,
        )
        ax.plot(
            failed["top1_match"],
            y,
            "o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=ATTR,
            markeredgewidth=1.2,
            zorder=3,
            label="fails screen" if y == ys[0] else None,
        )
        ax.plot(
            passed["top1_match"],
            y,
            "o",
            markersize=6,
            color=ATTR,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
            label="passes screen" if y == ys[0] else None,
        )
        ax.plot(
            passed["top1_overlap_baseline"],
            y,
            "|",
            color=INK,
            markersize=8,
            markeredgewidth=1.3,
            zorder=2,
            label="word overlap (passing)" if y == ys[0] else None,
        )
        if target == FIT_TARGET and "gpt_retest_top1" in rule:
            retest = rule.loc["qwen_only_supported", "gpt_retest_top1"]
            if pd.notna(retest):
                ax.plot(
                    retest,
                    y,
                    "D",
                    color=INK,
                    markersize=4.5,
                    markerfacecolor="white",
                    markeredgewidth=1.0,
                    zorder=2,
                    label="GPT-4o re-runs (passing)",
                )
    # Drawn over the data rows only, so it does not run through the legend.
    ax.plot(
        [float(np.mean(chance))] * 2,
        [-0.5, len(TARGETS) - 0.5],
        color=CHANCE,
        linewidth=1.4,
        linestyle=(0, (1, 1.2)),
        label="chance",
    )
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(-0.6, len(TARGETS) + 2.3)
    ax.set_yticks(ys, [label for _, label in TARGETS], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(
        "share of held-out prompts with the same top passage sentence",
        fontsize=7,
        color=MUTED,
    )
    ax.set_title(
        "b  The same prompts, six hosted targets", fontsize=8, loc="left", color=INK
    )
    ax.legend(
        fontsize=6,
        frameon=False,
        loc="upper left",
        ncol=2,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.1,
        bbox_to_anchor=(0.0, 1.0),
    )

    # (c, d) one passing and one failing prompt
    indexed = prompts.set_index("prompt_idx")
    for col, (prompt, verdict) in enumerate(SHOWCASE):
        ax = fig.add_subplot(grid[1, col])
        _prompt_panel(
            ax,
            "cd"[col],
            verdict,
            indexed.loc[prompt],
            segments[segments["prompt_idx"].eq(prompt)],
            legend=col == 1,
        )

    out_dir = os.path.join(output_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "fig_screen")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    bins.to_csv(os.path.join(output_dir, "transfer", "support_bins.csv"), index=False)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="worked_example")
    args = parser.parse_args()
    print(f"Wrote {plot(args.output_dir)}.{{pdf,png}}")


if __name__ == "__main__":
    main()
