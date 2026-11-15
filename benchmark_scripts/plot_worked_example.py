# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Plot the Qwen2.5-14B → GPT-4o trust-gate figure for the worked example.

Reads ``gate_bins.csv``, ``gate_rule.csv``, ``prompts.csv`` and
``pair_segments.csv.gz`` written by ``benchmark_scripts.worked_example build``
and writes ``figures/fig_trust_gate.{pdf,png}`` under the same directory.
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

# Paper palette: attribution agreement is teal, controls are grey.
ATTR: str = "#0091A1"
CHANCE: str = "#9A958F"
INK: str = "#0b0b0b"
MUTED: str = "#52514e"
# Validated two-series categorical pair for model identity.
QWEN: str = "#2a78d6"
GPT: str = "#eb6834"
SHOWCASE: tuple[tuple[int, str], ...] = (
    (3228, "passes both checks"),
    (582, "passes check 1, fails check 2"),
)
LABEL_CHARS: int = 34


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelsize=7, width=0.6, length=2.5)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.5)
    ax.set_axisbelow(True)


def _gate_line(ax: plt.Axes, x: float, label: bool) -> None:
    ax.axvline(x, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
    if label:
        ax.text(
            x - 0.15,
            ax.get_ylim()[1],
            "trust",
            ha="right",
            va="top",
            fontsize=7,
            color=INK,
        )
        ax.text(
            x + 0.15,
            ax.get_ylim()[1],
            "don't trust",
            ha="left",
            va="top",
            fontsize=7,
            color=INK,
        )


def _short(text: str) -> str:
    text = " ".join(str(text).replace("Context:", "").split())
    return textwrap.shorten(text, width=LABEL_CHARS, placeholder="…")


def plot(output_dir: str) -> str:
    bins = pd.read_csv(os.path.join(output_dir, "gate_bins.csv"))
    rule = pd.read_csv(os.path.join(output_dir, "gate_rule.csv")).set_index("subset")
    prompts = pd.read_csv(os.path.join(output_dir, "prompts.csv")).set_index(
        "prompt_idx"
    )
    segments = pd.read_csv(os.path.join(output_dir, "pair_segments.csv.gz"))
    threshold = float(rule.loc["accepted", "threshold_nats"])
    support_cut = float(rule.loc["accepted", "support_cut_nats"])

    fig = plt.figure(figsize=(7.0, 5.0))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.62, wspace=1.05)
    x = bins["bin"].to_numpy()
    labels = [f"{v:.0f}" for v in bins["gap_median"]]
    # The gate falls between the last accepted bin and the first rejected one.
    # Deciles are drawn on the accepted side when most of their prompts pass.
    gate_x = float(x[bins["gap_median"] < threshold].max()) + 0.5

    ax = fig.add_subplot(grid[0, 0])
    _style(ax)
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
    ax.set_ylim(0, 0.7)
    ax.set_xticks(x, labels)
    ax.set_xlabel(
        "|ℓ$_{Qwen}$ − ℓ$_{GPT-4o}$| (nats), decile median", fontsize=7, color=MUTED
    )
    ax.set_ylabel("passage attribution r²", fontsize=7, color=MUTED)
    ax.set_title(
        "a  Attribution agreement vs one-call gap", fontsize=8, loc="left", color=INK
    )
    _gate_line(ax, gate_x, label=True)

    ax = fig.add_subplot(grid[0, 1])
    _style(ax)
    groups = [
        ("qwen_only_matched_coverage", "Qwen\ncheck only"),
        ("accepted_supported", "both\nchecks"),
        ("accepted_unsupported", "check 1\nonly"),
        ("rejected", "fail\ncheck 1"),
    ]
    gx = np.arange(len(groups))
    sub = rule.loc[[g for g, _ in groups]]
    ax.bar(gx, sub["top1_match"], width=0.6, color=ATTR, label="Qwen vs GPT-4o")
    if sub["gpt_retest_top1"].notna().all():
        ax.plot(gx, sub["gpt_retest_top1"], "D", color=INK, markersize=4.5,
                markerfacecolor="white", markeredgewidth=1.0, label="GPT-4o vs its re-runs")
    ax.hlines(sub["top1_chance"], gx - 0.3, gx + 0.3, color=CHANCE, linewidth=1.4,
              linestyle=(0, (1, 1.2)), label="chance")
    if "top1_overlap_baseline" in sub and sub["top1_overlap_baseline"].notna().all():
        ax.hlines(sub["top1_overlap_baseline"], gx - 0.3, gx + 0.3, color=INK,
                  linewidth=1.2, label="word overlap (no model)")
    for k, (_, row) in enumerate(sub.iterrows()):
        ax.text(k, 0.02, f"{row['coverage']:.0%}\n{row['passage_r2']:.2f}", ha="center",
                va="bottom", fontsize=6.5, color="white")
    ax.set_xticks(gx, [label for _, label in groups], fontsize=6.5)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("top-sentence match", fontsize=7, color=MUTED)
    ax.set_title("b  Two-stage screen (held-out prompts)", fontsize=8, loc="left", color=INK)
    ax.legend(fontsize=6, frameon=False, loc="upper center", ncol=2, handlelength=1.2,
              columnspacing=0.8, borderaxespad=0.1, bbox_to_anchor=(0.5, 1.0))

    for col, (prompt, verdict) in enumerate(SHOWCASE):
        ax = fig.add_subplot(grid[1, col])
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color="#e6e5e1", linewidth=0.5)
        row = prompts.loc[prompt]
        seg = segments[
            segments["prompt_idx"].eq(prompt) & segments["part"].eq("passage")
        ]
        y = np.arange(len(seg))[::-1]
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
        ax.set_yticks(y, [_short(t) for t in seg["segment_text"]], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Δℓ when the sentence is deleted (nats)", fontsize=7, color=MUTED)
        question = textwrap.shorten(
            row["question"].rstrip("."), width=46, placeholder="…"
        )
        letter = "cd"[col]
        ax.set_title(
            f"{letter}  {verdict}: “{question}?”\n"
            f"gap {row['gap']:.1f} nats · Qwen {'True' if row['pred_qwen'] else 'False'}, "
            f"GPT-4o {'True' if row['pred_gpt'] else 'False'}, gold {row['gold']}",
            fontsize=7,
            loc="left",
            color=INK,
        )
        if "top_overlap_passage" in row and pd.notna(row["top_overlap_passage"]):
            hit = np.flatnonzero(seg["seg_idx"].to_numpy() == row["top_overlap_passage"])
            if len(hit) and row["top_overlap_passage"] != row["top_abs_passage_gpt"]:
                ax.annotate("← word-overlap pick", xy=(0.98, y[hit[0]]),
                            xycoords=("axes fraction", "data"), ha="right",
                            va="center", fontsize=6, color=MUTED)
        if col == 1:
            ax.legend(fontsize=6.5, frameon=False, loc="lower right")

    acc, rej = rule.loc["accepted"], rule.loc["rejected"]
    caption = (
        f"a–b: held-out (odd) BoolQ prompts; gate fit on even prompts: trust when the gap < {threshold:.0f} nats "
        f"({acc['coverage']:.0%} of prompts; passage r² {acc['passage_r2']:.2f} "
        f"[{acc['passage_r2_lo']:.2f}, {acc['passage_r2_hi']:.2f}] vs "
        f"{rej['passage_r2']:.2f} [{rej['passage_r2_lo']:.2f}, {rej['passage_r2_hi']:.2f}]). "
        f"Bars in a: 95% prompt-bootstrap intervals. b: check 2 needs only Qwen: its top passage sentence must support its "
        f"own answer by ≥ {support_cut:g} nats (fit on even prompts); \"Qwen check only\" uses no GPT-4o call, with its cut set to match the "
        "two-stage coverage. White text: coverage, then passage r². Black ticks: picking the passage sentence that shares the most "
        "content words with the question. Check 1 = gap < " f"{threshold:.0f} nats. "
        "c–d: passage sentences of two illustrative prompts. In c, Qwen and GPT-4o both hinge on a sentence that word overlap misses; "
        "d shows the models reading the same sentence in opposite directions, which is rare (about 3% of prompts)."
    )
    fig.text(0.01, -0.02, textwrap.fill(caption, 135), fontsize=6, color=MUTED, va="top")
    out_dir = os.path.join(output_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "fig_trust_gate")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", default="worked_example")
    args = parser.parse_args()
    print(f"Wrote {plot(args.output_dir)}.{{pdf,png}}")


if __name__ == "__main__":
    main()
