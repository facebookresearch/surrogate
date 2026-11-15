# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Rebuttal reanalyses of the published surrogate-fidelity results.

Every analysis reads existing result artifacts only: no model or API calls are
made, and nothing under ``results/`` is written. Signals come from the
F-table extractors, so log-odds, attribution, attention, perturbation
magnitude, and alignment mean exactly what they mean in ``f_table.tsv``.

Conventions follow the paper. F is Pearson r² over the coordinates that both
models observe with finite values, pooled across prompts. ``F_pred`` uses one
log-odds value per prompt; segment-level metrics use attribution
(``l(x) - l(x without segment i)``) and the internal signals. Cross-level
``X_to_attr`` correlates model S's signal X with model T's attribution;
``F_mag_to_attr`` takes the absolute value of the target. Pair-set summaries
report the median, minimum, and maximum over Open->Open, Open->Closed, and All
pairs.

Analyses (BoolQ, sentence level, unless noted):

1. Label conditioning of ``F_pred`` and ``F_attr``.
2. Segment-length control of the representation and cross-level metrics.
3. Unsigned attention-to-attribution transfer.
4. Within-prompt variants of the segment-level metrics.
5. Passage-only segment-level metrics.
6. Table 2 ``F_pred`` / ``F_attr`` medians by pair type, all benchmarks.
7. Decisive-segment agreement.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    write_derived_provenance,
)
from benchmark_scripts.f_table import (
    _analysis_rng,
    _contrast_column,
    _mask_unsupported_signals,
    _per_prompt_logodds,
    _per_seg_ablation,
    _per_seg_align,
    _per_seg_segment_col,
    _supporting_source_paths,
    _unsupported_model_components,
    OPEN_MODELS,
    PAPER_MODELS,
)

logger: logging.Logger = logging.getLogger(__name__)

BENCHMARK: str = "boolq"
PREGROUPER: str = "sentence"
CONTRAST: str = "canonical"
HOSTED_MODELS: tuple[str, ...] = tuple(m for m in PAPER_MODELS if m not in OPEN_MODELS)
FAMILY: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "qwen",
    "qwen2.5-3b-instruct": "qwen",
    "qwen2.5-7b-instruct": "qwen",
    "qwen2.5-14b-instruct": "qwen",
    "llama-3.1-8b-instruct": "llama",
    "llama3.1-70b-instruct": "llama",
    "llama3.3-70b-instruct": "llama",
    "llama4-maverick-17b-128e-instruct": "llama",
    "gpt-4o": "gpt",
    "gpt-4-1": "gpt",
    "gemini-2-5-flash-lite-vertex": "gemini",
}
ATTENTION_VARIANTS: tuple[str, ...] = ("mean", "max", "rollout")
ATTENTION_METRICS: tuple[str, ...] = tuple(f"F_attn_{v}" for v in ATTENTION_VARIANTS)
REPRESENTATION_METRICS: tuple[str, ...] = (*ATTENTION_METRICS, "F_mag", "F_align")
LENGTH_CONTROLLED_METRICS: tuple[str, ...] = (*ATTENTION_METRICS, "F_mag")
# Cross-level metric -> (source signal, whether the attribution keeps its sign).
TRANSFER_METRICS: dict[str, tuple[str, bool]] = {
    "F_mag_to_attr": ("F_mag", False),
    "F_align_to_attr": ("F_align", True),
    **{f"{metric}_to_attr": (metric, True) for metric in ATTENTION_METRICS},
}
PAIR_SETS: dict[str, frozenset[str]] = {
    "open_to_open": frozenset({"open_open"}),
    "open_to_closed": frozenset({"open_hosted"}),
    "all": frozenset({"open_open", "open_hosted", "hosted_hosted"}),
}
ROW_COLUMNS: list[str] = [
    "analysis",
    "metric",
    "variant",
    "statistic",
    "model_s",
    "model_t",
    "pair_population",
    "family_relation",
    "n_prompts",
    "n_observations",
    "value",
]
SUMMARY_KEYS: list[str] = ["analysis", "metric", "variant", "statistic"]
TIE_SEED: int = 0
# Analysis 7's no-model baseline picks the passage sentence sharing the most
# content words with the question.
STOPWORDS: frozenset[str] = frozenset(
    """a an the is are was were be been being do does did has have had can could
    will would shall should may might must of in on at to for from by with as and
    or but if then than that this these those it its into onto over under about
    who whom whose which what when where why how there their they them he she his
    her him i you your we our us not no any ever get got one also just very more
    most such own same so too s t context question""".split()
)
TOP_K: int = 3
N_CONFIDENCE_BINS: int = 5
ANALYSES: tuple[str, ...] = (
    "label_conditioning",
    "length_control",
    "unsigned_attention",
    "within_prompt",
    "passage_only",
    "table2_pair_type",
    "decisive_segments",
)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signals:
    """Every per-model signal used by the BoolQ analyses.

    Attributes:
        prediction: ``{model: log-odds}`` indexed by ``prompt_idx``.
        segment: ``{metric: {model: signal}}`` indexed by
            ``(prompt_idx, seg_idx)``; ``F_attr`` holds attribution for every
            model, the representation metrics hold open-model signals.
        segments: Coordinate metadata indexed by ``(prompt_idx, seg_idx)`` with
            ``part`` (system / passage / question), ``n_words`` and
            ``question_overlap`` (content words shared with the question).
        gold: Gold answer per prompt.
    """

    prediction: dict[str, pd.Series]
    segment: dict[str, dict[str, pd.Series]]
    segments: pd.DataFrame
    gold: pd.Series


def _int_index(signal: pd.Series) -> pd.Series:
    """Normalize prompt and segment index levels to integers."""
    if isinstance(signal.index, pd.MultiIndex):
        levels = [signal.index.get_level_values(i).astype(np.int64) for i in (0, 1)]
        index = pd.MultiIndex.from_arrays(levels, names=["prompt_idx", "seg_idx"])
    else:
        index = pd.Index(signal.index.astype(np.int64), name="prompt_idx")
    return pd.Series(signal.to_numpy(dtype=float), index=index)


def segment_parts(segments: pd.DataFrame) -> pd.Series:
    """
    Label each BoolQ coordinate as system, passage, or question.

    The BoolQ user message is ``Context:\\n<passage>.\\n\\nQuestion:\\n<q>.``.
    The question starts at the user segment beginning with ``Question:`` and
    includes every later user segment of that prompt.

    Args:
        segments: Rows of ``segments.tsv.gz``.

    Returns:
        Part label aligned with ``segments``.

    Raises:
        ValueError: If a prompt has no question segment.
    """
    user: pd.Series = segments["message_role"].eq("user")
    question_start: pd.Series = (
        segments[user & segments["segment_text"].str.startswith("Question:")]
        .groupby("prompt_idx")["seg_idx"]
        .min()
    )
    missing: set[int] = set(segments["prompt_idx"]) - set(question_start.index)
    if missing:
        raise ValueError(f"{len(missing)} prompts have no question segment")
    start: np.ndarray = segments["prompt_idx"].map(question_start).to_numpy()
    part: np.ndarray = np.where(
        ~user,
        "system",
        np.where(segments["seg_idx"].to_numpy() >= start, "question", "passage"),
    )
    return pd.Series(part, index=segments.index, name="part")


def content_words(text: object) -> set[str]:
    """Lower-cased content words of ``text``, with a crude plural strip."""
    words = re.findall(r"[a-z0-9]+", text.lower()) if isinstance(text, str) else []
    return {
        w[:-1] if len(w) > 3 and w.endswith("s") else w
        for w in words
        if w not in STOPWORDS
    }


def question_overlap(segments: pd.DataFrame) -> pd.Series:
    """
    Count each segment's content words shared with its prompt's question.

    Args:
        segments: Rows with ``prompt_idx``, ``part`` and ``segment_text``.

    Returns:
        Overlap counts aligned with ``segments``.
    """
    questions: pd.Series = (
        segments[segments["part"].eq("question")]
        .groupby("prompt_idx")["segment_text"]
        .agg(" ".join)
        .map(content_words)
    )
    return pd.Series(
        [
            len(content_words(text) & questions.get(prompt, set()))
            for prompt, text in zip(segments["prompt_idx"], segments["segment_text"])
        ],
        index=segments.index,
        dtype=float,
    )


def load_signals(results_dir: str) -> Signals:
    """
    Load BoolQ/sentence signals through the F-table extractors.

    Args:
        results_dir: Result root holding the consolidated
            ``boolq_sentence_{segments,logodds}.tsv`` and ``boolq/sentence``.

    Returns:
        The loaded signals.

    Raises:
        ValueError: If any model lacks a required signal.
    """
    prefix: str = os.path.join(results_dir, f"{BENCHMARK}_{PREGROUPER}")
    seg_df: pd.DataFrame = pd.read_csv(
        f"{prefix}_segments.tsv", sep="\t", low_memory=False
    )
    odds_df: pd.DataFrame = pd.read_csv(f"{prefix}_logodds.tsv", sep="\t")
    contrast_col: str = _contrast_column(CONTRAST)
    unsupported_prediction, unsupported_attribution = _unsupported_model_components(
        results_dir, BENCHMARK, PREGROUPER
    )

    def collect(
        models: Iterable[str], name: str, extract: Callable[[str], pd.Series | None]
    ) -> dict[str, pd.Series]:
        out: dict[str, pd.Series] = {}
        for model in models:
            signal = extract(model)
            if signal is None:
                raise ValueError(f"{model} lacks the field required for {name}")
            out[model] = _int_index(signal)
        return out

    prediction = _mask_unsupported_signals(
        collect(
            PAPER_MODELS,
            "F_pred",
            lambda m: _per_prompt_logodds(odds_df, m, "orig", contrast_col),
        ),
        unsupported_prediction,
    )
    segment: dict[str, dict[str, pd.Series]] = {
        "F_attr": _mask_unsupported_signals(
            collect(
                PAPER_MODELS,
                "F_attr",
                lambda m: _per_seg_ablation(odds_df, m, contrast_col),
            ),
            unsupported_attribution,
        ),
        "F_mag": collect(
            OPEN_MODELS,
            "F_mag",
            lambda m: _per_seg_segment_col(seg_df, m, "delta_norm_postnorm"),
        ),
        "F_align": collect(OPEN_MODELS, "F_align", lambda m: _per_seg_align(seg_df, m)),
    }
    for variant in ATTENTION_VARIANTS:
        column: str = f"attention_{variant}"
        segment[f"F_attn_{variant}"] = collect(
            OPEN_MODELS,
            f"F_attn_{variant}",
            lambda m, column=column: _per_seg_segment_col(seg_df, m, column),
        )

    manifest: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, BENCHMARK, PREGROUPER, "segments.tsv.gz"), sep="\t"
    )
    manifest["part"] = segment_parts(manifest)
    manifest["n_words"] = manifest["segment_text"].str.split().str.len().clip(lower=1)
    manifest["question_overlap"] = question_overlap(manifest)
    segments: pd.DataFrame = manifest.set_index(["prompt_idx", "seg_idx"])[
        ["answer", "message_role", "part", "n_words", "question_overlap"]
    ]
    gold: pd.Series = manifest.groupby("prompt_idx")["answer"].first().astype(bool)
    return Signals(prediction, segment, segments, gold)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def pair_population(model_s: str, model_t: str) -> str:
    kinds: list[str] = sorted(
        "open" if m in OPEN_MODELS else "hosted" for m in (model_s, model_t)
    )
    return {
        ("open", "open"): "open_open",
        ("hosted", "open"): "open_hosted",
        ("hosted", "hosted"): "hosted_hosted",
    }[(kinds[0], kinds[1])]


def family_relation(model_s: str, model_t: str) -> str:
    return "same_family" if FAMILY[model_s] == FAMILY[model_t] else "cross_family"


def symmetric_pairs(models: Iterable[str]) -> list[tuple[str, str]]:
    ordered: list[str] = [m for m in PAPER_MODELS if m in set(models)]
    return [(a, b) for i, a in enumerate(ordered) for b in ordered[i + 1 :]]


def transfer_pairs() -> list[tuple[str, str]]:
    return [(s, t) for s in OPEN_MODELS for t in PAPER_MODELS if s != t]


def joint(
    source: pd.Series,
    target: pd.Series,
    *,
    keys: pd.Index | None = None,
    prompts: pd.Index | None = None,
    abs_target: bool = False,
) -> pd.DataFrame:
    """
    Pair two signals on their shared finite coordinates.

    Args:
        source: Signal of model S.
        target: Signal of model T, on the same index type.
        keys: Optional coordinates to restrict to.
        prompts: Optional prompts to restrict to.
        abs_target: Take the absolute value of the target.

    Returns:
        Frame with ``x``, ``y``, and ``prompt_idx`` for each shared coordinate
        where both values are finite.
    """
    common: pd.Index = source.index.intersection(target.index)
    if keys is not None:
        common = common.intersection(keys)
    prompt: np.ndarray = (
        common.get_level_values(0).to_numpy()
        if isinstance(common, pd.MultiIndex)
        else common.to_numpy()
    )
    frame: pd.DataFrame = pd.DataFrame(
        {
            "x": source.loc[common].to_numpy(dtype=float),
            "y": target.loc[common].to_numpy(dtype=float),
            "prompt_idx": prompt,
        },
        # Unnamed index levels keep ``prompt_idx`` unambiguous as a column.
        index=common.set_names([None] * common.nlevels),
    )
    if prompts is not None:
        frame = frame[frame["prompt_idx"].isin(prompts)]
    frame = frame[np.isfinite(frame["x"]) & np.isfinite(frame["y"])]
    if abs_target:
        frame = frame.assign(y=frame["y"].abs())
    return frame


def pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r² of pooled rows; NaN with fewer than three rows or no variance."""
    if len(x) < 3:
        return float("nan")
    xc: np.ndarray = x - x.mean()
    yc: np.ndarray = y - y.mean()
    denominator: float = float(np.sqrt((xc * xc).sum() * (yc * yc).sum()))
    if denominator <= 0:
        return float("nan")
    return float((xc * yc).sum() / denominator) ** 2


def pooled_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation of pooled rows (average ranks for ties)."""
    if len(x) < 3:
        return float("nan")
    rx: np.ndarray = pd.Series(x).rank(method="average").to_numpy()
    ry: np.ndarray = pd.Series(y).rank(method="average").to_numpy()
    r2: float = pearson_r2(rx, ry)
    sign: float = float(np.sign(np.cov(rx, ry)[0, 1]))
    return sign * float(np.sqrt(r2)) if np.isfinite(r2) else float("nan")


def demean_within_prompt(frame: pd.DataFrame) -> pd.DataFrame:
    """Subtract each prompt's mean from ``x`` and ``y``."""
    grouped = frame.groupby("prompt_idx")
    return frame.assign(
        x=frame["x"] - grouped["x"].transform("mean"),
        y=frame["y"] - grouped["y"].transform("mean"),
    )


def mean_prompt_spearman(
    frame: pd.DataFrame, min_segments: int = 3
) -> tuple[float, int, int]:
    """
    Average per-prompt Spearman correlation.

    Prompts with fewer than ``min_segments`` shared coordinates, or with a
    constant signal, are skipped, as in the F-table prompt-equal sensitivity.

    Args:
        frame: Output of :func:`joint`.
        min_segments: Minimum shared coordinates per prompt.

    Returns:
        Mean correlation, number of prompts averaged, and their row count.
    """
    sizes: pd.Series = frame.groupby("prompt_idx")["x"].transform("size")
    kept: pd.DataFrame = frame[sizes >= min_segments]
    if kept.empty:
        return float("nan"), 0, 0
    grouped = kept.groupby("prompt_idx")
    rx: pd.Series = grouped["x"].rank(method="average")
    ry: pd.Series = grouped["y"].rank(method="average")
    ranks: pd.DataFrame = pd.DataFrame(
        {"prompt_idx": kept["prompt_idx"], "rx": rx, "ry": ry}
    )
    centered = ranks[["rx", "ry"]] - ranks.groupby("prompt_idx")[
        ["rx", "ry"]
    ].transform("mean")
    products: pd.DataFrame = pd.DataFrame(
        {
            "prompt_idx": kept["prompt_idx"],
            "xy": centered["rx"] * centered["ry"],
            "xx": centered["rx"] ** 2,
            "yy": centered["ry"] ** 2,
            "n": 1,
        }
    )
    sums: pd.DataFrame = products.groupby("prompt_idx")[["xy", "xx", "yy", "n"]].sum()
    valid: pd.Series = (sums["xx"] > 0) & (sums["yy"] > 0)
    rho: pd.Series = sums.loc[valid, "xy"] / np.sqrt(
        sums.loc[valid, "xx"] * sums.loc[valid, "yy"]
    )
    if rho.empty:
        return float("nan"), 0, 0
    return float(rho.mean()), int(len(rho)), int(sums.loc[valid, "n"].sum())


def residualize_on_length(
    signal: pd.Series, log_length: pd.Series
) -> tuple[pd.Series, dict[str, float]]:
    """
    Residualize a signal on log segment length by ordinary least squares.

    Args:
        signal: One model's segment-level signal.
        log_length: Log word count per coordinate.

    Returns:
        Residuals on the signal's index (NaN where the signal is non-finite)
        and the fit's ``intercept``, ``slope``, ``r2_length``, and
        ``n_observations``.
    """
    x: np.ndarray = log_length.reindex(signal.index).to_numpy(dtype=float)
    y: np.ndarray = signal.to_numpy(dtype=float)
    valid: np.ndarray = np.isfinite(x) & np.isfinite(y)
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    residual: np.ndarray = np.full(len(y), np.nan)
    residual[valid] = y[valid] - (intercept + slope * x[valid])
    fit: dict[str, float] = {
        "intercept": float(intercept),
        "slope": float(slope),
        "r2_length": pearson_r2(x[valid], y[valid]),
        "n_observations": int(valid.sum()),
    }
    return pd.Series(residual, index=signal.index), fit


def _row(
    analysis: str,
    metric: str,
    variant: str,
    statistic: str,
    model_s: str,
    model_t: str,
    frame: pd.DataFrame | None,
    value: float,
    *,
    n_prompts: int | None = None,
    n_observations: int | None = None,
) -> dict[str, str | float | int]:
    return {
        "analysis": analysis,
        "metric": metric,
        "variant": variant,
        "statistic": statistic,
        "model_s": model_s,
        "model_t": model_t,
        "pair_population": pair_population(model_s, model_t),
        "family_relation": family_relation(model_s, model_t),
        "n_prompts": (
            n_prompts
            if n_prompts is not None
            else int(frame["prompt_idx"].nunique()) if frame is not None else 0
        ),
        "n_observations": (
            n_observations
            if n_observations is not None
            else len(frame) if frame is not None else 0
        ),
        "value": value,
    }


def _r2_row(
    analysis: str,
    metric: str,
    variant: str,
    model_s: str,
    model_t: str,
    frame: pd.DataFrame,
) -> dict[str, str | float | int]:
    return _row(
        analysis,
        metric,
        variant,
        "pearson_r2",
        model_s,
        model_t,
        frame,
        pearson_r2(frame["x"].to_numpy(), frame["y"].to_numpy()),
    )


def metric_pairs(metric: str) -> list[tuple[str, str]]:
    """Pairs evaluated for a metric, in the F-table's orientation."""
    if metric in TRANSFER_METRICS:
        return transfer_pairs()
    if metric in {"F_pred", "F_attr"}:
        return symmetric_pairs(PAPER_MODELS)
    return symmetric_pairs(OPEN_MODELS)


def metric_frame(
    signals: Signals,
    metric: str,
    model_s: str,
    model_t: str,
    *,
    source: dict[str, dict[str, pd.Series]] | None = None,
    keys: pd.Index | None = None,
    prompts: pd.Index | None = None,
    abs_target: bool | None = None,
) -> pd.DataFrame:
    """
    Joint frame for one metric and pair.

    Args:
        signals: Loaded signals.
        metric: F-table metric name.
        model_s: Source model.
        model_t: Target model.
        source: Optional replacement source signals by metric (for example,
            length-residualized signals).
        keys: Optional coordinate restriction.
        prompts: Optional prompt restriction.
        abs_target: Override the metric's sign convention for the target.

    Returns:
        Output of :func:`joint`.
    """
    sources: dict[str, dict[str, pd.Series]] = source or signals.segment
    if metric == "F_pred":
        return joint(
            signals.prediction[model_s], signals.prediction[model_t], prompts=prompts
        )
    if metric in TRANSFER_METRICS:
        source_metric, signed = TRANSFER_METRICS[metric]
        return joint(
            sources[source_metric][model_s],
            signals.segment["F_attr"][model_t],
            keys=keys,
            prompts=prompts,
            abs_target=(not signed) if abs_target is None else abs_target,
        )
    by_model: dict[str, pd.Series] = (
        signals.segment[metric] if metric == "F_attr" else sources[metric]
    )
    return joint(by_model[model_s], by_model[model_t], keys=keys, prompts=prompts)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def correct_prompts(signals: Signals, model: str) -> pd.Index:
    """Prompts whose log-odds sign matches the gold label (ties are wrong)."""
    logodds: pd.Series = signals.prediction[model]
    gold: pd.Series = signals.gold.reindex(logodds.index)
    correct: pd.Series = (logodds.gt(0) & gold) | (logodds.lt(0) & ~gold)
    return logodds.index[correct.to_numpy()]


def label_conditioning(signals: Signals) -> pd.DataFrame:
    """Analysis 1: ``F_pred`` and ``F_attr`` within label and correctness subsets."""
    gold: pd.Series = signals.gold
    true_prompts: pd.Index = gold.index[gold.to_numpy()]
    false_prompts: pd.Index = gold.index[~gold.to_numpy()]
    correct: dict[str, pd.Index] = {
        m: correct_prompts(signals, m) for m in PAPER_MODELS
    }
    rows: list[dict[str, str | float | int]] = []
    for a, b in symmetric_pairs(PAPER_MODELS):
        both: pd.Index = correct[a].intersection(correct[b])
        subsets: dict[str, pd.Index | None] = {
            "unconditioned": None,
            "gold_true": true_prompts,
            "gold_false": false_prompts,
            "both_correct": both,
            "both_correct_gold_true": both.intersection(true_prompts),
            "both_correct_gold_false": both.intersection(false_prompts),
        }
        frames: dict[str, pd.DataFrame] = {
            metric: metric_frame(signals, metric, a, b)
            for metric in ("F_pred", "F_attr")
        }
        for subset, prompts in subsets.items():
            for metric, frame in frames.items():
                kept: pd.DataFrame = (
                    frame
                    if prompts is None
                    else frame[frame["prompt_idx"].isin(prompts)]
                )
                rows.append(_r2_row("label_conditioning", metric, subset, a, b, kept))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def model_accuracy(signals: Signals) -> pd.DataFrame:
    """Per-model accuracy over prompts with an observed log-odds sign."""
    records: list[dict[str, str | float | int]] = []
    for model in PAPER_MODELS:
        logodds: pd.Series = signals.prediction[model]
        observed: pd.Series = logodds.dropna()
        correct: pd.Index = correct_prompts(signals, model)
        gold: pd.Series = signals.gold.reindex(observed.index)
        records.append(
            {
                "model": model,
                "n_prompts_observed": len(observed),
                "accuracy": len(correct) / len(observed),
                "accuracy_gold_true": float(
                    observed.index[gold.to_numpy()].isin(correct).mean()
                ),
                "accuracy_gold_false": float(
                    observed.index[~gold.to_numpy()].isin(correct).mean()
                ),
                "n_infinite_logodds": int(np.isinf(observed).sum()),
            }
        )
    return pd.DataFrame(records)


def passage_keys(signals: Signals) -> pd.Index:
    """Coordinates of BoolQ passage segments (no system or question text)."""
    return signals.segments.index[signals.segments["part"].eq("passage").to_numpy()]


def length_residualized(
    signals: Signals, keys: pd.Index | None = None, scope: str = "all_segments"
) -> tuple[dict[str, dict[str, pd.Series]], pd.DataFrame]:
    """
    Residualize each open model's magnitude and attention on log word count.

    Args:
        signals: Loaded signals.
        keys: Optional coordinates to fit on; residuals exist only there.
        scope: Label recorded with each fit.

    Returns:
        Signals by metric with residualized magnitude and attention, and one
        fit record per (metric, model).
    """
    log_length: pd.Series = np.log(signals.segments["n_words"].astype(float))
    residualized: dict[str, dict[str, pd.Series]] = dict(signals.segment)
    fits: list[dict[str, str | float | int]] = []
    for metric in LENGTH_CONTROLLED_METRICS:
        residualized[metric] = {}
        for model in OPEN_MODELS:
            signal: pd.Series = signals.segment[metric][model]
            if keys is not None:
                signal = signal.loc[signal.index.intersection(keys)]
            residual, fit = residualize_on_length(signal, log_length)
            residualized[metric][model] = residual
            fits.append({"scope": scope, "metric": metric, "model": model, **fit})
    return residualized, pd.DataFrame(fits)


def length_control(
    signals: Signals, residualized: dict[str, dict[str, pd.Series]]
) -> pd.DataFrame:
    """Analysis 2: representation and cross-level metrics with length control."""
    metrics: list[str] = [
        *LENGTH_CONTROLLED_METRICS,
        "F_mag_to_attr",
        *(f"{metric}_to_attr" for metric in ATTENTION_METRICS),
    ]
    rows: list[dict[str, str | float | int]] = []
    for metric in metrics:
        for a, b in metric_pairs(metric):
            for variant, source in (
                ("raw", signals.segment),
                ("length_residualized", residualized),
            ):
                frame = metric_frame(signals, metric, a, b, source=source)
                rows.append(_r2_row("length_control", metric, variant, a, b, frame))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def unsigned_attention(
    signals: Signals,
    residualized: dict[str, dict[str, pd.Series]],
    passage_residualized: dict[str, dict[str, pd.Series]],
) -> pd.DataFrame:
    """
    Analysis 3: attention against unsigned attribution.

    Raw and length-controlled, over all segments and (as a diagnostic of
    segment-type structure) over passage segments only.
    """
    passage: pd.Index = passage_keys(signals)
    rows: list[dict[str, str | float | int]] = []
    for attention in ATTENTION_METRICS:
        metric: str = f"{attention}_to_attr"
        for a, b in transfer_pairs():
            for variant, source, absolute, keys in (
                ("signed_raw", signals.segment, False, None),
                ("unsigned_raw", signals.segment, True, None),
                ("unsigned_length_residualized", residualized, True, None),
                ("signed_passage_only", signals.segment, False, passage),
                ("unsigned_passage_only", signals.segment, True, passage),
                (
                    "unsigned_passage_only_length_residualized",
                    passage_residualized,
                    True,
                    passage,
                ),
            ):
                frame = metric_frame(
                    signals,
                    metric,
                    a,
                    b,
                    source=source,
                    abs_target=absolute,
                    keys=keys,
                )
                rows.append(_r2_row("unsigned_attention", metric, variant, a, b, frame))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


WITHIN_PROMPT_METRICS: tuple[str, ...] = (
    "F_attr",
    "F_mag",
    "F_align",
    *ATTENTION_METRICS,
    *TRANSFER_METRICS,
)


def within_prompt(signals: Signals) -> pd.DataFrame:
    """Analysis 4: prompt-demeaned pooled r² and mean per-prompt Spearman."""
    rows: list[dict[str, str | float | int]] = []
    for metric in WITHIN_PROMPT_METRICS:
        for a, b in metric_pairs(metric):
            frame = metric_frame(signals, metric, a, b)
            x, y = frame["x"].to_numpy(), frame["y"].to_numpy()
            rows.append(_r2_row("within_prompt", metric, "pooled", a, b, frame))
            rows.append(
                _row(
                    "within_prompt",
                    metric,
                    "pooled",
                    "spearman",
                    a,
                    b,
                    frame,
                    pooled_spearman(x, y),
                )
            )
            rows.append(
                _r2_row(
                    "within_prompt",
                    metric,
                    "prompt_demeaned",
                    a,
                    b,
                    demean_within_prompt(frame),
                )
            )
            rho, n_prompts, n_rows = mean_prompt_spearman(frame)
            rows.append(
                _row(
                    "within_prompt",
                    metric,
                    "mean_per_prompt",
                    "spearman",
                    a,
                    b,
                    frame,
                    rho,
                    n_prompts=n_prompts,
                    n_observations=n_rows,
                )
            )
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def passage_only(
    signals: Signals, passage_residualized: dict[str, dict[str, pd.Series]]
) -> pd.DataFrame:
    """
    Analysis 5: attribution and cross-level metrics on passage segments only.

    Cross-level metrics whose source is magnitude or attention are also
    reported with the source residualized on log length within passages.
    """
    passage: pd.Index = passage_keys(signals)
    rows: list[dict[str, str | float | int]] = []
    for metric in ("F_attr", *TRANSFER_METRICS):
        variants: list[tuple[str, pd.Index | None, dict[str, dict[str, pd.Series]]]] = [
            ("all_segments", None, signals.segment),
            ("passage_only", passage, signals.segment),
        ]
        if TRANSFER_METRICS.get(metric, ("",))[0] in LENGTH_CONTROLLED_METRICS:
            variants.append(
                ("passage_only_length_residualized", passage, passage_residualized)
            )
        for a, b in metric_pairs(metric):
            for variant, keys, source in variants:
                frame = metric_frame(signals, metric, a, b, keys=keys, source=source)
                rows.append(_r2_row("passage_only", metric, variant, a, b, frame))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def table2_pair_type(results_dir: str) -> pd.DataFrame:
    """Analysis 6: published Table 2 ``F_pred`` / ``F_attr`` values by pair type."""
    f_table: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "f_table.tsv"), sep="\t"
    )
    anli_rv: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "anli_rv.tsv"), sep="\t"
    )
    race_rv: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "race_rv.tsv"), sep="\t"
    )
    frames: list[pd.DataFrame] = []
    for label, benchmark, pregrouper in (
        ("BoolQ", "boolq", "sentence"),
        ("WinoGrande", "winogrande", "sentence"),
        ("BoolQ word", "boolq", "word"),
    ):
        rows = f_table[
            f_table["benchmark"].eq(benchmark)
            & f_table["pregrouper"].eq(pregrouper)
            & f_table["scope"].eq("all")
            & f_table["contrast"].eq("canonical")
            & f_table["statistic"].eq("pearson_r2")
            & f_table["availability_status"].eq("available")
            & f_table["api_infinity_policy"].eq("pairwise_complete")
            & f_table["metric"].isin(["F_pred", "F_attr"])
        ]
        frames.append(rows.assign(table2_column=label, statistic="pearson_r2"))
    for label, benchmark in (
        ("ANLI R1", "anli_r1"),
        ("ANLI R2", "anli_r2"),
        ("ANLI R3", "anli_r3"),
    ):
        rows = anli_rv[anli_rv["benchmark"].eq(benchmark) & anli_rv["scope"].eq("all")]
        frames.append(
            rows.assign(
                table2_column=label,
                statistic="centered_rv",
                metric=rows["metric"].str.removesuffix("_rv"),
            )
        )
    race = race_rv[
        race_rv["representation"].eq("all_pairs")
        & race_rv["metric"].isin(["F_pred_rv", "F_attr_rv"])
    ]
    frames.append(
        race.assign(
            table2_column="RACE RV",
            statistic="centered_rv",
            metric=race["metric"].str.removesuffix("_rv"),
        )
    )
    published: pd.DataFrame = pd.concat(frames, ignore_index=True)
    published = published[published["f_point"].notna()]
    return pd.DataFrame(
        {
            "analysis": "table2_pair_type",
            "table2_column": published["table2_column"],
            "benchmark": published["benchmark"],
            "pregrouper": published["pregrouper"],
            "metric": published["metric"],
            "statistic": published["statistic"],
            "model_s": published["model_s"],
            "model_t": published["model_t"],
            "pair_population": [
                pair_population(a, b)
                for a, b in zip(published["model_s"], published["model_t"])
            ],
            "family_relation": [
                family_relation(a, b)
                for a, b in zip(published["model_s"], published["model_t"])
            ],
            "n_prompts": published["n_prompts"],
            "n_observations": published["n_observations"],
            "value": published["f_point"],
        }
    )


TABLE2_PAIR_TYPES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "all": lambda f: pd.Series(True, index=f.index),
    "same_family": lambda f: f["family_relation"].eq("same_family"),
    "cross_family": lambda f: f["family_relation"].eq("cross_family"),
    "open_closed": lambda f: f["pair_population"].eq("open_hosted"),
    "open_open": lambda f: f["pair_population"].eq("open_open"),
    "closed_closed": lambda f: f["pair_population"].eq("hosted_hosted"),
    "same_family_open_closed": lambda f: f["family_relation"].eq("same_family")
    & f["pair_population"].eq("open_hosted"),
    "cross_family_open_closed": lambda f: f["family_relation"].eq("cross_family")
    & f["pair_population"].eq("open_hosted"),
}
TABLE2_COLUMNS: tuple[str, ...] = (
    "BoolQ",
    "ANLI R1",
    "ANLI R2",
    "ANLI R3",
    "WinoGrande",
    "BoolQ word",
    "RACE RV",
)


def summarize_table2(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str | float | int]] = []
    for (column, metric), group in rows.groupby(["table2_column", "metric"]):
        for pair_type, select in TABLE2_PAIR_TYPES.items():
            values: pd.Series = group.loc[select(group), "value"].dropna()
            records.append(
                {
                    "table2_column": column,
                    "metric": metric,
                    "statistic": group["statistic"].iloc[0],
                    "pair_type": pair_type,
                    "n_pairs": len(values),
                    "median": values.median() if len(values) else np.nan,
                    "min": values.min() if len(values) else np.nan,
                    "max": values.max() if len(values) else np.nan,
                }
            )
    out = pd.DataFrame(records)
    out["table2_column"] = pd.Categorical(out["table2_column"], TABLE2_COLUMNS)
    return out.sort_values(["metric", "table2_column", "pair_type"]).reset_index(
        drop=True
    )


def _top_k_sets(
    frame: pd.DataFrame, column: str, rng: np.random.Generator, k: int
) -> pd.Series:
    """Top-``k`` coordinates per prompt by ``column``, ties broken at random."""
    order: pd.DataFrame = frame.assign(
        _tie=rng.permutation(len(frame)), _seg=frame.index.get_level_values(1)
    ).sort_values(["prompt_idx", column, "_tie"], ascending=[True, False, True])
    top: pd.DataFrame = order.groupby("prompt_idx", sort=True).head(k)
    return top.groupby("prompt_idx", sort=True)["_seg"].agg(frozenset)


def decisive_agreement(
    frame: pd.DataFrame,
    confidence: pd.Series,
    rng: np.random.Generator,
    *,
    source_column: str = "ax",
    target_column: str = "ay",
    parts: pd.Series | None = None,
    baseline: pd.Series | None = None,
) -> list[dict[str, str | float | int]]:
    """
    Top-1 and top-3 decisive-segment agreement, overall and by confidence.

    Args:
        frame: Joint frame with ranking columns ``source_column`` and
            ``target_column`` (larger is more decisive).
        confidence: Pair confidence per prompt (mean absolute log-odds).
        rng: Tie-breaking generator.
        source_column: Ranking column of model S.
        target_column: Ranking column of model T.
        parts: Optional segment part (system / passage / question) indexed by
            ``(prompt_idx, seg_idx)``.
        baseline: Optional no-model score indexed by ``(prompt_idx, seg_idx)``;
            its top segment per prompt is compared with the target's. When k
            segments tie for its maximum, a match counts 1/k: the expected
            top-1 agreement under a random tie-break, with no randomness.

    Returns:
        One record per stratum (``all`` and confidence quintiles ``q1``-``q5``,
        q5 most confident) with top-1 agreement, its chance level (mean of
        ``1/N``), top-3 overlap fraction, its chance level (mean of ``3/N``),
        the share of prompts whose maximum is tied for either model, and, when
        ``parts`` is given, the share of prompts whose top segment is the
        question, and the system prompt, for each model. The target's question
        share is the top-1 agreement of a role-aware baseline that always
        picks the question segment. With ``baseline``, also the top-1
        agreement of the baseline's pick with the target.
    """
    sizes: pd.Series = frame.groupby("prompt_idx").size()
    prompts: pd.Index = sizes.index[sizes.to_numpy() >= 2]
    frame = frame[frame["prompt_idx"].isin(prompts)]
    top1_s = _top_k_sets(frame, source_column, rng, 1)
    top1_t = _top_k_sets(frame, target_column, rng, 1)
    top3_s = _top_k_sets(frame, source_column, rng, TOP_K)
    top3_t = _top_k_sets(frame, target_column, rng, TOP_K)
    grouped = frame.groupby("prompt_idx")

    def tied(column: str) -> pd.Series:
        maxima = grouped[column].transform("max")
        return frame[column].eq(maxima).groupby(frame["prompt_idx"]).sum() > 1

    per_prompt: pd.DataFrame = pd.DataFrame(
        {
            "n_segments": sizes.loc[prompts],
            "top1_match": [top1_s[p] == top1_t[p] for p in prompts],
            "top3_overlap": [len(top3_s[p] & top3_t[p]) / TOP_K for p in prompts],
            "tied": (tied(source_column) | tied(target_column)).loc[prompts],
            "confidence": confidence.reindex(prompts),
        },
        index=prompts,
    )
    for side, top1 in (("source", top1_s), ("target", top1_t)):
        for part in ("question", "system"):
            if parts is None:
                per_prompt[f"{side}_top_{part}"] = np.nan
                continue
            keys = pd.MultiIndex.from_arrays(
                [prompts, [next(iter(top1[p])) for p in prompts]]
            )
            per_prompt[f"{side}_top_{part}"] = parts.reindex(keys).eq(part).to_numpy()
    per_prompt["baseline_match"] = np.nan
    if baseline is not None:
        score = pd.Series(
            baseline.reindex(frame.index).to_numpy(), index=frame.index
        ).fillna(-np.inf)
        best = score.eq(score.groupby(frame["prompt_idx"].to_numpy()).transform("max"))
        ties = (
            frame.index[best.to_numpy()]
            .to_frame(index=False)
            .groupby(0)[1]
            .agg(frozenset)
        )
        per_prompt["baseline_match"] = [
            len(ties[p] & top1_t[p]) / len(ties[p]) for p in prompts
        ]
    rated: pd.DataFrame = per_prompt[per_prompt["confidence"].notna()]
    per_prompt["stratum"] = pd.Series(dtype=object)
    if len(rated) >= N_CONFIDENCE_BINS:
        bins = pd.qcut(
            rated["confidence"].rank(method="first"),
            N_CONFIDENCE_BINS,
            labels=[f"q{i}" for i in range(1, N_CONFIDENCE_BINS + 1)],
        )
        per_prompt.loc[rated.index, "stratum"] = bins.astype(str)
    records: list[dict[str, str | float | int]] = []
    strata: list[tuple[str, pd.DataFrame]] = [("all", per_prompt)] + [
        (f"q{i}", per_prompt[per_prompt["stratum"].eq(f"q{i}")])
        for i in range(1, N_CONFIDENCE_BINS + 1)
    ]
    for stratum, group in strata:
        top3: pd.DataFrame = group[group["n_segments"] > TOP_K]
        records.append(
            {
                "stratum": stratum,
                "n_prompts": len(group),
                "confidence_min": group["confidence"].min(),
                "confidence_max": group["confidence"].max(),
                "top1_agreement": group["top1_match"].mean(),
                "top1_chance": (1.0 / group["n_segments"]).mean(),
                "n_prompts_top3": len(top3),
                "top3_overlap": top3["top3_overlap"].mean(),
                "top3_chance": (TOP_K / top3["n_segments"]).mean(),
                "share_prompts_tied_max": group["tied"].mean(),
                "share_source_top_question": group["source_top_question"].mean(),
                "share_target_top_question": group["target_top_question"].mean(),
                "share_source_top_system": group["source_top_system"].mean(),
                "share_target_top_system": group["target_top_system"].mean(),
                "top1_overlap_baseline": group["baseline_match"].astype(float).mean(),
            }
        )
    return records


def decisive_segments(signals: Signals) -> pd.DataFrame:
    """Analysis 7: decisive-segment agreement by |attribution| and by attention."""
    rows: list[dict[str, str | float | int]] = []
    attribution: dict[str, pd.Series] = signals.segment["F_attr"]

    def confidence(a: str, b: str) -> pd.Series:
        return (
            signals.prediction[a].abs() + signals.prediction[b].abs()
        ).dropna() / 2.0

    parts: pd.Series = signals.segments["part"]
    overlap: pd.Series = signals.segments["question_overlap"]
    passage: pd.Index = passage_keys(signals)

    def emit(
        comparison: str,
        scope: str,
        a: str,
        b: str,
        frame: pd.DataFrame,
        conf: pd.Series,
    ) -> None:
        rng = _analysis_rng(TIE_SEED, "decisive_segments", comparison, scope, a, b)
        # Overlap with the question is only a baseline for passage sentences:
        # over all segments the question trivially overlaps itself, and the
        # question-top share already plays that role.
        passage_scope = scope == "passage_only"
        for record in decisive_agreement(
            frame,
            conf,
            rng,
            parts=parts,
            baseline=overlap if passage_scope else None,
        ):
            rows.append(
                {
                    "analysis": "decisive_segments",
                    "comparison": comparison,
                    "scope": scope,
                    "model_s": a,
                    "model_t": b,
                    "pair_population": pair_population(a, b),
                    "family_relation": family_relation(a, b),
                    **record,
                }
            )

    for scope, keys in (("all_segments", None), ("passage_only", passage)):
        for a, b in symmetric_pairs(PAPER_MODELS):
            frame = joint(attribution[a], attribution[b], keys=keys)
            frame = frame.assign(ax=frame["x"].abs(), ay=frame["y"].abs())
            emit("abs_attr_vs_abs_attr", scope, a, b, frame, confidence(a, b))
        for attention in ATTENTION_METRICS:
            for a, b in transfer_pairs():
                frame = joint(signals.segment[attention][a], attribution[b], keys=keys)
                frame = frame.assign(ax=frame["x"], ay=frame["y"].abs())
                emit(
                    f"{attention}_vs_abs_attr", scope, a, b, frame, confidence(a, b)
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def summarize(rows: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Median, minimum, and maximum of ``value`` over each pair set."""
    keys = keys or SUMMARY_KEYS
    records: list[dict[str, str | float | int]] = []
    for key_values, group in rows.groupby(keys, sort=False):
        for pair_set, populations in PAIR_SETS.items():
            values: pd.Series = group.loc[
                group["pair_population"].isin(populations), "value"
            ].dropna()
            if values.empty:
                continue
            records.append(
                {
                    **dict(zip(keys, key_values)),
                    "pair_set": pair_set,
                    "n_pairs": len(values),
                    "median": float(values.median()),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "median_n_prompts": float(
                        group.loc[values.index, "n_prompts"].median()
                    ),
                    "median_n_observations": float(
                        group.loc[values.index, "n_observations"].median()
                    ),
                }
            )
    return pd.DataFrame(records)


def published_medians(results_dir: str) -> pd.DataFrame:
    """Published BoolQ/sentence Pearson r² pair-set medians, for verification."""
    f_table: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "f_table.tsv"), sep="\t"
    )
    rows = f_table[
        f_table["benchmark"].eq(BENCHMARK)
        & f_table["pregrouper"].eq(PREGROUPER)
        & f_table["scope"].eq("all")
        & f_table["contrast"].eq(CONTRAST)
        & f_table["statistic"].eq("pearson_r2")
        & f_table["availability_status"].eq("available")
        & f_table["api_infinity_policy"].eq("pairwise_complete")
        & f_table["aggregation"].eq("row_pooled")
    ]
    return rows[["metric", "model_s", "model_t", "f_point"]]


def verify_against_published(rows: pd.DataFrame, published: pd.DataFrame) -> float:
    """
    Check recomputed baselines against ``f_table.tsv``.

    Args:
        rows: Analysis rows containing baseline Pearson r² values.
        published: Output of :func:`published_medians`.

    Returns:
        Largest absolute difference over matched pairs.

    Raises:
        ValueError: If a baseline differs from the published value by more
            than 1e-9 or no baseline could be matched.
    """
    merged: pd.DataFrame = rows.merge(
        published, on=["metric", "model_s", "model_t"], how="inner"
    )
    if merged.empty:
        raise ValueError("No baseline rows matched f_table.tsv")
    difference: float = float((merged["value"] - merged["f_point"]).abs().max())
    if not difference <= 1e-9:
        raise ValueError(f"Baseline differs from f_table.tsv by {difference:.3g}")
    return difference


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

PAIR_SET_LABELS: dict[str, str] = {
    "open_to_open": "Open→Open",
    "open_to_closed": "Open→Closed",
    "all": "All",
}


def _fmt(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def _change_table(
    summary: pd.DataFrame,
    baseline: str,
    variants: list[str],
    *,
    metrics: list[str] | None = None,
    statistic: str = "pearson_r2",
    baseline_statistic: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Markdown rows of metric | pair set | baseline | variant | value | change."""
    data: pd.DataFrame = summary
    baseline_rows = data[
        data["variant"].eq(baseline)
        & data["statistic"].eq(baseline_statistic or statistic)
    ]
    lines: list[str] = [
        "| Metric | Pair set | Baseline median | Variant | Variant median "
        "[min, max] | Change |",
        "|---|---|---:|---|---:|---:|",
    ]
    records: list[dict[str, str | float]] = []
    ordered_metrics: list[str] = metrics or list(dict.fromkeys(data["metric"]))
    for metric in ordered_metrics:
        for pair_set in PAIR_SETS:
            base = baseline_rows[
                baseline_rows["metric"].eq(metric)
                & baseline_rows["pair_set"].eq(pair_set)
            ]
            if base.empty:
                continue
            base_value: float = float(base["median"].iloc[0])
            for variant in variants:
                hit = data[
                    data["metric"].eq(metric)
                    & data["pair_set"].eq(pair_set)
                    & data["variant"].eq(variant)
                    & data["statistic"].eq(statistic)
                ]
                if hit.empty:
                    continue
                record = hit.iloc[0]
                change: float = float(record["median"]) - base_value
                lines.append(
                    f"| `{metric}` | {PAIR_SET_LABELS[pair_set]} | {_fmt(base_value)} "
                    f"| {variant} | {_fmt(record['median'])} "
                    f"[{_fmt(record['min'])}, {_fmt(record['max'])}] "
                    f"| {change:+.3f} |"
                )
                records.append(
                    {
                        "metric": metric,
                        "pair_set": pair_set,
                        "variant": variant,
                        "baseline": base_value,
                        "value": float(record["median"]),
                        "change": change,
                    }
                )
    return lines, pd.DataFrame(records)


def _median(
    summary: pd.DataFrame,
    metric: str,
    variant: str,
    pair_set: str,
    statistic: str = "pearson_r2",
) -> float:
    hit = summary[
        summary["metric"].eq(metric)
        & summary["variant"].eq(variant)
        & summary["pair_set"].eq(pair_set)
        & summary["statistic"].eq(statistic)
    ]
    return float(hit["median"].iloc[0]) if len(hit) else float("nan")


def write_report(
    path: str,
    summaries: dict[str, pd.DataFrame],
    extras: dict[str, pd.DataFrame],
) -> None:
    """Write the summary markdown: notes on claims, then one table per analysis."""
    s1: pd.DataFrame = summaries["label_conditioning"]
    s2: pd.DataFrame = summaries["length_control"]
    s3: pd.DataFrame = summaries["unsigned_attention"]
    s4: pd.DataFrame = summaries["within_prompt"]
    s5: pd.DataFrame = summaries["passage_only"]
    s6: pd.DataFrame = summaries["table2_pair_type"]
    s7: pd.DataFrame = summaries["decisive_segments"]
    fits: pd.DataFrame = extras["length_fits"]
    detail: pd.DataFrame = extras["decisive_rows"]
    accuracy: pd.DataFrame = extras["model_accuracy"]

    def med(summary: pd.DataFrame, metric: str, variant: str, pair_set: str = "all",
            statistic: str = "pearson_r2") -> float:
        return _median(summary, metric, variant, pair_set, statistic)

    def table2(metric: str, pair_type: str, column: str) -> float:
        hit = s6[
            s6["metric"].eq(metric)
            & s6["pair_type"].eq(pair_type)
            & s6["table2_column"].astype(str).eq(column)
        ]
        return float(hit["median"].iloc[0]) if len(hit) else float("nan")

    def decisive(comparison: str, scope: str, stratum: str, column: str,
                 pair_set: str = "all") -> float:
        hit = s7[
            s7["comparison"].eq(comparison)
            & s7["scope"].eq(scope)
            & s7["stratum"].eq(stratum)
            & s7["pair_set"].eq(pair_set)
        ]
        return float(hit[column].iloc[0]) if len(hit) else float("nan")

    def highlight(comparison: str, scope: str, stratum: str, column: str) -> float:
        hit = detail[
            detail["model_s"].eq("qwen2.5-14b-instruct")
            & detail["model_t"].eq("gpt-4o")
            & detail["comparison"].eq(comparison)
            & detail["scope"].eq(scope)
            & detail["stratum"].eq(stratum)
        ]
        return float(hit[column].iloc[0]) if len(hit) else float("nan")

    subsets: list[str] = [
        "gold_true",
        "gold_false",
        "both_correct",
        "both_correct_gold_true",
        "both_correct_gold_false",
    ]
    flags: list[str] = []
    for subset in ["unconditioned", *subsets]:
        for pair_set in PAIR_SETS:
            pred = med(s1, "F_pred", subset, pair_set)
            attr = med(s1, "F_attr", subset, pair_set)
            if np.isfinite(pred) and np.isfinite(attr) and pred <= attr:
                flags.append(
                    f"{subset} / {PAIR_SET_LABELS[pair_set]} "
                    f"({pred:.3f} ≤ {attr:.3f})"
                )
    all_fits = fits[fits["scope"].eq("all_segments")]
    passage_fits = fits[fits["scope"].eq("passage_only")]
    _, changes2 = _change_table(s2, "raw", ["length_residualized"])
    _, changes4 = _change_table(
        s4, "pooled", ["prompt_demeaned"], metrics=list(WITHIN_PROMPT_METRICS)
    )
    _, changes4s = _change_table(
        s4, "pooled", ["mean_per_prompt"], metrics=list(WITHIN_PROMPT_METRICS),
        statistic="spearman",
    )
    big4 = pd.concat([changes4, changes4s])
    big4 = big4[big4["change"].abs() >= 0.05]
    attn_max_unsigned: float = max(
        med(s3, f"{m}_to_attr", "unsigned_raw", ps)
        for m in ATTENTION_METRICS for ps in ("open_to_open", "open_to_closed")
    )
    def passage_range(variant: str) -> str:
        values = [
            med(s3, f"{m}_to_attr", variant, "all") for m in ("F_attn_mean", "F_attn_max")
        ]
        return f"{min(values):.2f}–{max(values):.2f}"

    _, changes5 = _change_table(s5, "passage_only", ["passage_only_length_residualized"])
    open_closed_below_all: dict[str, bool] = {
        metric: all(
            table2(metric, "open_closed", column) < table2(metric, "all", column)
            for column in TABLE2_COLUMNS
        )
        for metric in ("F_pred", "F_attr")
    }
    same_family_higher_attr: list[str] = [
        column
        for column in TABLE2_COLUMNS
        if table2("F_attr", "same_family", column)
        <= table2("F_attr", "cross_family", column)
    ]

    notes: list[str] = [
        "## Notes on the paper's claims",
        "",
        "1. **Label conditioning (changes the F_pred > F_attr reading).** Within one "
        "gold label the ordering mostly holds (All: gold-True F_pred "
        f"{med(s1, 'F_pred', 'gold_true'):.3f} vs F_attr {med(s1, 'F_attr', 'gold_true'):.3f}; "
        f"gold-False {med(s1, 'F_pred', 'gold_false'):.3f} vs "
        f"{med(s1, 'F_attr', 'gold_false'):.3f}), but among prompts both models answer "
        "correctly, within one label, median F_pred falls to "
        f"{med(s1, 'F_pred', 'both_correct_gold_true'):.3f} / "
        f"{med(s1, 'F_pred', 'both_correct_gold_false'):.3f} while F_attr is "
        f"{med(s1, 'F_attr', 'both_correct_gold_true'):.3f} / "
        f"{med(s1, 'F_attr', 'both_correct_gold_false'):.3f}. Conditioning removes the "
        "between-class variance (range restriction), and F_pred loses most of its "
        "r² while F_attr is unchanged or higher in the same subsets: the headline "
        "F_pred mainly reflects agreement on the answer, not on confidence within "
        "a class.",
        "2. **Length control (no change).** Pooled over all segments, log word count "
        f"explains at most {all_fits['r2_length'].max():.2f} of any open-model signal's "
        "variance, and residualizing moves no pair-set median by more than "
        f"{changes2['change'].abs().max():.3f}. Within passages length explains "
        f"{passage_fits['r2_length'].min():.2f}–{passage_fits['r2_length'].max():.2f}; "
        "controlling it there moves passage-only medians by at most "
        f"{changes5['change'].abs().max():.3f} (analysis 5).",
        "3. **Unsigned attribution (qualifies the attention claim).** Attention is "
        "nearly uninformative about the sign of attribution (signed medians ≤ "
        f"{max(med(s3, f'{m}_to_attr', 'signed_raw', ps) for m in ATTENTION_METRICS for ps in PAIR_SETS):.3f}) "
        "but predicts its magnitude: unsigned medians reach "
        f"{attn_max_unsigned:.3f} (mean attention, Open→Closed) and survive length "
        "control "
        f"({med(s3, 'F_attn_mean_to_attr', 'unsigned_length_residualized', 'open_to_closed'):.3f}). "
        "Part of this is segment role (the question segment has the highest mean "
        "attention and the largest |Δℓ|), but not all: on passage segments alone, "
        f"mean and max attention still explain {passage_range('unsigned_passage_only')} "
        f"of |Δℓ| ({passage_range('unsigned_passage_only_length_residualized')} after "
        "within-passage length control); rollout explains "
        f"{med(s3, 'F_attn_rollout_to_attr', 'unsigned_passage_only', 'all'):.3f}.",
        "4. **Within-prompt (no change for r²).** Removing per-prompt means changes "
        "every pooled-r² median by less than 0.05. "
        + (
            "Per-prompt Spearman moves "
            + "; ".join(
                f"`{r.metric}` {PAIR_SET_LABELS[r.pair_set]} {r.baseline:.3f}→{r.value:.3f}"
                for r in big4.itertuples(index=False)
            )
            + "."
            if len(big4)
            else "Per-prompt Spearman medians also move by less than 0.05."
        ),
        "5. **Passage-only (changes the magnitude claim).** On passage segments "
        f"F_mag→|attr| rises from {med(s5, 'F_mag_to_attr', 'all_segments'):.3f} to "
        f"{med(s5, 'F_mag_to_attr', 'passage_only'):.3f} (All; "
        f"{med(s5, 'F_mag_to_attr', 'passage_only_length_residualized'):.3f} with "
        "within-passage length control). The published value is depressed by "
        "system segments, which have large ‖Δz‖ but small |Δℓ|. F_align→attr "
        f"falls from {med(s5, 'F_align_to_attr', 'all_segments'):.3f} to "
        f"{med(s5, 'F_align_to_attr', 'passage_only'):.3f}; F_attr changes from "
        f"{med(s5, 'F_attr', 'all_segments'):.3f} to {med(s5, 'F_attr', 'passage_only'):.3f}.",
        "6. **Table 2 by pair type (no change).** BoolQ F_pred is similar within "
        f"and across families ({table2('F_pred', 'same_family', 'BoolQ'):.3f} vs "
        f"{table2('F_pred', 'cross_family', 'BoolQ'):.3f}). F_attr is higher within "
        f"family on BoolQ ({table2('F_attr', 'same_family', 'BoolQ'):.3f} vs "
        f"{table2('F_attr', 'cross_family', 'BoolQ'):.3f})"
        + (
            " and every other benchmark except "
            + ", ".join(c for c in same_family_higher_attr if c != "BoolQ")
            if same_family_higher_attr
            else " and on every other benchmark"
        )
        + ". Open↔closed medians are "
        + (
            "below the all-pairs median on every benchmark for both metrics"
            if all(open_closed_below_all.values())
            else "not uniformly below the all-pairs median"
        )
        + ". Only three same-family open↔closed pairs exist (Llama-3.1-8B with "
        "each hosted Llama).",
        "7. **Decisive segments (structural, qualify before citing).** Top-1 "
        f"agreement is {decisive('abs_attr_vs_abs_attr', 'all_segments', 'all', 'top1_agreement'):.3f} "
        f"vs 1/N chance {decisive('abs_attr_vs_abs_attr', 'all_segments', 'all', 'top1_chance'):.3f} "
        f"(All), rising from {decisive('abs_attr_vs_abs_attr', 'all_segments', 'q1', 'top1_agreement'):.3f} "
        f"(q1) to {decisive('abs_attr_vs_abs_attr', 'all_segments', 'q5', 'top1_agreement'):.3f} "
        "(q5). But the question segment is the target's top segment in "
        f"{decisive('abs_attr_vs_abs_attr', 'all_segments', 'all', 'share_target_top_question'):.0%} "
        "of prompts (rising with confidence), so 'always pick the question' "
        "matches about as often and accounts for the confidence trend. On passage "
        "segments agreement is "
        f"{decisive('abs_attr_vs_abs_attr', 'passage_only', 'all', 'top1_agreement'):.3f} vs "
        f"chance {decisive('abs_attr_vs_abs_attr', 'passage_only', 'all', 'top1_chance'):.3f}, "
        "nearly flat across quintiles "
        f"({decisive('abs_attr_vs_abs_attr', 'passage_only', 'q1', 'top1_agreement'):.3f} → "
        f"{decisive('abs_attr_vs_abs_attr', 'passage_only', 'q5', 'top1_agreement'):.3f}). "
        "Qwen-14B → GPT-4o: "
        f"{highlight('abs_attr_vs_abs_attr', 'all_segments', 'all', 'top1_agreement'):.3f} "
        f"(passage-only {highlight('abs_attr_vs_abs_attr', 'passage_only', 'all', 'top1_agreement'):.3f}, "
        f"chance {highlight('abs_attr_vs_abs_attr', 'passage_only', 'all', 'top1_chance'):.3f}). "
        "Qwen-14B's top mean-attention segment matches GPT-4o's top |Δℓ| segment in "
        f"{highlight('F_attn_mean_vs_abs_attr', 'all_segments', 'all', 'top1_agreement'):.3f} "
        "of prompts, essentially the share in which GPT-4o's top is the question "
        f"({highlight('F_attn_mean_vs_abs_attr', 'all_segments', 'all', 'share_target_top_question'):.3f}), "
        "because Qwen-14B's mean attention almost always peaks on the question; "
        "on passages it is "
        f"{highlight('F_attn_mean_vs_abs_attr', 'passage_only', 'all', 'top1_agreement'):.3f} vs "
        f"chance {highlight('F_attn_mean_vs_abs_attr', 'passage_only', 'all', 'top1_chance'):.3f}.",
        "",
    ]

    lines: list[str] = [
        "# Rebuttal reanalyses: summary",
        "",
        "Reanalysis of existing BoolQ/sentence results (analysis 6 uses every",
        "Table 2 benchmark). No model or API calls; regenerate with",
        "`python -m benchmark_scripts.reanalysis_rebuttal`. Values are pair-set",
        "medians of Pearson r² unless the statistic says otherwise; per-pair rows",
        "are in the CSVs. Pair sets follow the paper: `F_pred` / `F_attr` use the",
        "55 unordered pairs, representation metrics the 10 open pairs, and",
        "cross-level metrics the 50 ordered open-source pairs (All = Open→All).",
        "Every baseline equals the published `f_table.tsv` value for the same",
        "pair (checked to 1e-9).",
        "",
        *notes,
    ]

    table, _ = _change_table(s1, "unconditioned", subsets)
    lines += [
        "## 1. Label conditioning",
        "",
        "Subsets restrict prompts; correctness is sign(ℓ) = gold for both models.",
        "",
        *table,
        "",
        "Median prompts per subset (All pairs): "
        + ", ".join(
            f"{subset} {s1[s1['variant'].eq(subset) & s1['metric'].eq('F_pred') & s1['pair_set'].eq('all')]['median_n_prompts'].iloc[0]:.0f}"
            for subset in ["unconditioned", *subsets]
        )
        + ".",
        "",
        "Model accuracy (sign of ℓ vs. gold): "
        + ", ".join(
            f"{r.model} {r.accuracy:.3f}" for r in accuracy.itertuples(index=False)
        )
        + ".",
        "",
        "**Flag (median F_pred ≤ F_attr):** " + ("; ".join(flags) if flags else "none."),
        "",
    ]

    table, _ = _change_table(s2, "raw", ["length_residualized"])
    lines += [
        "## 2. Segment-length control",
        "",
        "Each open model's ‖Δz‖ and attention scores are residualized on",
        "log(whitespace word count) by OLS over that model's segments; the",
        "attribution target is not residualized.",
        "",
        *table,
        "",
        "Variance explained by log length (r², range over open models), all",
        "segments / passage segments: "
        + ", ".join(
            f"`{metric}` {group['r2_length'].min():.2f}–{group['r2_length'].max():.2f} / "
            f"{passage_fits[passage_fits['metric'].eq(metric)]['r2_length'].min():.2f}–"
            f"{passage_fits[passage_fits['metric'].eq(metric)]['r2_length'].max():.2f}"
            for metric, group in all_fits.groupby("metric", sort=False)
        )
        + ".",
        "",
    ]

    table, _ = _change_table(
        s3,
        "signed_raw",
        [
            "unsigned_raw",
            "unsigned_length_residualized",
            "signed_passage_only",
            "unsigned_passage_only",
            "unsigned_passage_only_length_residualized",
        ],
    )
    lines += [
        "## 3. Unsigned attribution",
        "",
        "F_attn→|attr| for each attention variant. Passage-only rows (diagnostic)",
        "drop system and question segments; their length control is fitted",
        "within passages.",
        "",
        *table,
        "",
    ]

    table_a, _ = _change_table(
        s4, "pooled", ["prompt_demeaned"], metrics=list(WITHIN_PROMPT_METRICS)
    )
    table_b, _ = _change_table(
        s4,
        "pooled",
        ["mean_per_prompt"],
        metrics=list(WITHIN_PROMPT_METRICS),
        statistic="spearman",
    )
    lines += [
        "## 4. Within-prompt",
        "",
        "(a) Pearson r² after subtracting per-prompt means (baseline: pooled r²).",
        "",
        *table_a,
        "",
        "(b) Mean per-prompt Spearman ρ over prompts with ≥ 3 shared segments",
        "(baseline: pooled Spearman ρ).",
        "",
        *table_b,
        "",
    ]

    table, _ = _change_table(
        s5, "all_segments", ["passage_only", "passage_only_length_residualized"]
    )
    lines += [
        "## 5. Passage-only",
        "",
        "System-prompt and question segments are dropped. For magnitude and",
        "attention sources, `passage_only_length_residualized` also residualizes",
        "the source on log length within passages.",
        "",
        *table,
        "",
    ]

    pair_types: list[str] = [
        "all",
        "same_family",
        "cross_family",
        "open_closed",
        "same_family_open_closed",
        "cross_family_open_closed",
    ]
    lines += [
        "## 6. Table 2 by pair type",
        "",
        "Published per-pair values (BoolQ / WinoGrande / BoolQ word: Pearson r²;",
        "ANLI / RACE: centered RV, as in Table 2), median (n pairs). Families:",
        "Qwen (4 open), Llama (Llama-3.1-8B open; Llama-3.1-70B, Llama-3.3-70B,",
        "Llama-4 Maverick hosted), GPT (GPT-4o, GPT-4.1), Gemini. Same/cross",
        "family span all 55 pairs; open↔closed pairs are further split by family.",
        "",
        "| Metric | Pair type | " + " | ".join(TABLE2_COLUMNS) + " |",
        "|---|---|" + "---:|" * len(TABLE2_COLUMNS),
    ]
    for metric in ("F_pred", "F_attr"):
        for pair_type in pair_types:
            cells: list[str] = []
            for column in TABLE2_COLUMNS:
                hit = s6[
                    s6["metric"].eq(metric)
                    & s6["pair_type"].eq(pair_type)
                    & s6["table2_column"].astype(str).eq(column)
                ]
                cells.append(
                    f"{_fmt(float(hit['median'].iloc[0]))} ({int(hit['n_pairs'].iloc[0])})"
                    if len(hit) and hit["n_pairs"].iloc[0]
                    else "—"
                )
            lines.append(f"| `{metric}` | {pair_type} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## 7. Decisive-segment agreement",
        "",
        "Per prompt, each model's top segment by |Δℓ| (or by attention for the",
        "source, in attention rows) over the pair's shared finite segments",
        "(prompts with ≥ 2; top-3 overlap uses prompts with > 3); ties are broken",
        "at random with a fixed seed. Chance is the mean of 1/N (top-1) and 3/N",
        "(top-3 overlap fraction). 'Q-top' is the share of prompts whose target",
        "top segment is the question, i.e. the top-1 agreement of always picking",
        "the question. 'S-top' is the same share for the system prompt.",
        "'Overlap' (passage rows only) is the top-1 agreement of a no-model",
        "baseline that picks the passage sentence sharing the most content words",
        "with the question (stopwords removed, crude plural strip; k tied",
        "sentences share credit 1/k). Confidence quintiles use the mean |ℓ| of the two models",
        "(q5 most confident).",
        "",
        "| Scope | Pair set | Stratum | Top-1 | Chance | Q-top | S-top | Overlap | Top-3 | Chance |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("all_segments", "passage_only"):
        for pair_set in PAIR_SETS:
            for stratum in ["all", "q1", "q3", "q5"]:
                hit = s7[
                    s7["comparison"].eq("abs_attr_vs_abs_attr")
                    & s7["scope"].eq(scope)
                    & s7["pair_set"].eq(pair_set)
                    & s7["stratum"].eq(stratum)
                ]
                if hit.empty:
                    continue
                r = hit.iloc[0]
                lines.append(
                    f"| {scope} | {PAIR_SET_LABELS[pair_set]} | {stratum} | "
                    f"{r['top1_agreement']:.3f} | {r['top1_chance']:.3f} | "
                    f"{r['share_target_top_question']:.3f} | "
                    f"{r['share_target_top_system']:.3f} | "
                    f"{_fmt3(r['top1_overlap_baseline'])} | "
                    f"{r['top3_overlap']:.3f} | {r['top3_chance']:.3f} |"
                )
    lines += [
        "",
        "Qwen2.5-14B → GPT-4o. Rows `F_attn_*` rank Qwen-14B's segments by",
        "attention and GPT-4o's by |Δℓ|.",
        "",
        "| Comparison | Scope | Stratum | Top-1 | Chance | Q-top | S-top | Overlap | Top-3 | Chance | n |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    highlight_rows = detail[
        detail["model_s"].eq("qwen2.5-14b-instruct") & detail["model_t"].eq("gpt-4o")
    ]
    for r in highlight_rows.itertuples(index=False):
        if r.comparison != "abs_attr_vs_abs_attr" and r.stratum not in {"all", "q1", "q5"}:
            continue
        lines.append(
            f"| {r.comparison} | {r.scope} | {r.stratum} | {r.top1_agreement:.3f} | "
            f"{r.top1_chance:.3f} | {r.share_target_top_question:.3f} | "
            f"{r.share_target_top_system:.3f} | {_fmt3(r.top1_overlap_baseline)} | "
            f"{r.top3_overlap:.3f} | {r.top3_chance:.3f} | {r.n_prompts} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as sink:
        sink.write("\n".join(lines) + "\n")


def _fmt3(value: float) -> str:
    """Three decimals, or an em dash for a missing value."""
    return "—" if pd.isna(value) else f"{value:.3f}"


def summarize_decisive(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, str | float | int]] = []
    metrics: list[str] = [
        "top1_agreement",
        "top1_chance",
        "top3_overlap",
        "top3_chance",
        "share_prompts_tied_max",
        "share_source_top_question",
        "share_target_top_question",
        "share_source_top_system",
        "share_target_top_system",
        "top1_overlap_baseline",
    ]
    for (comparison, scope, stratum), group in rows.groupby(
        ["comparison", "scope", "stratum"], sort=False
    ):
        for pair_set, populations in PAIR_SETS.items():
            subset = group[group["pair_population"].isin(populations)]
            if subset.empty:
                continue
            records.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "stratum": stratum,
                    "pair_set": pair_set,
                    "n_pairs": len(subset),
                    "median_n_prompts": float(subset["n_prompts"].median()),
                    **{metric: float(subset[metric].median()) for metric in metrics},
                    "median_top1_lift": float(
                        (subset["top1_agreement"] - subset["top1_chance"]).median()
                    ),
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _write_csv(
    frame: pd.DataFrame,
    path: str,
    *,
    inputs: dict[str, str],
    parameters: dict[str, object],
    root_dir: str,
) -> None:
    frame.to_csv(path, index=False, float_format="%.10g")
    write_derived_provenance(
        path,
        generator_name="benchmark_scripts.reanalysis_rebuttal",
        generator_path=os.path.abspath(__file__),
        input_paths=inputs,
        parameters=parameters,
        root_dir=root_dir,
        supporting_source_paths=_supporting_source_paths(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-dir", default="results")
    # Kept outside results/, whose release inventory is validated exactly.
    parser.add_argument("--output-dir", default="reanalysis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    os.makedirs(args.output_dir, exist_ok=True)
    root_dir: str = os.path.dirname(os.path.abspath(args.results_dir))
    prefix: str = os.path.join(args.results_dir, f"{BENCHMARK}_{PREGROUPER}")
    boolq_inputs: dict[str, str] = collect_result_inputs(
        args.results_dir,
        BENCHMARK,
        PREGROUPER,
        {
            "segments": f"{prefix}_segments.tsv",
            "logodds": f"{prefix}_logodds.tsv",
            "f_table": os.path.join(args.results_dir, "f_table.tsv"),
        },
        PAPER_MODELS,
    )
    table2_inputs: dict[str, str] = {
        name: os.path.join(args.results_dir, f"{name}.tsv")
        for name in ("f_table", "anli_rv", "race_rv")
    }
    parameters: dict[str, object] = {
        "benchmark": BENCHMARK,
        "pregrouper": PREGROUPER,
        "contrast": CONTRAST,
        "api_infinity_policy": "pairwise_complete",
        "statistic": "pearson_r2",
        "length_measure": "log whitespace word count, minimum 1",
        "within_prompt_min_segments": 3,
        "decisive_top_k": TOP_K,
        "decisive_tie_seed": TIE_SEED,
        "confidence_bins": N_CONFIDENCE_BINS,
        "families": FAMILY,
        "cohort": list(PAPER_MODELS),
    }

    def out(name: str) -> str:
        return os.path.join(args.output_dir, name)

    logger.info("Loading BoolQ/sentence signals")
    signals: Signals = load_signals(args.results_dir)
    published: pd.DataFrame = published_medians(args.results_dir)
    residualized, fits = length_residualized(signals)
    passage_residualized, passage_fits = length_residualized(
        signals, passage_keys(signals), scope="passage_only"
    )
    fits = pd.concat([fits, passage_fits], ignore_index=True)
    summaries: dict[str, pd.DataFrame] = {}
    extras: dict[str, pd.DataFrame] = {"length_fits": fits}

    analyses: list[tuple[str, Callable[[], pd.DataFrame], str]] = [
        ("label_conditioning", lambda: label_conditioning(signals), "unconditioned"),
        ("length_control", lambda: length_control(signals, residualized), "raw"),
        (
            "unsigned_attention",
            lambda: unsigned_attention(signals, residualized, passage_residualized),
            "signed_raw",
        ),
        ("within_prompt", lambda: within_prompt(signals), "pooled"),
        (
            "passage_only",
            lambda: passage_only(signals, passage_residualized),
            "all_segments",
        ),
    ]
    for index, (name, run, baseline) in enumerate(analyses, start=1):
        logger.info(f"Analysis {index}: {name}")
        rows: pd.DataFrame = run()
        base = rows[rows["variant"].eq(baseline) & rows["statistic"].eq("pearson_r2")]
        difference = verify_against_published(base, published)
        logger.info(f"  baseline matches f_table.tsv (max |diff| {difference:.2g})")
        summary = summarize(rows)
        summaries[name] = summary
        _write_csv(
            rows,
            out(f"{index}_{name}.csv"),
            inputs=boolq_inputs,
            parameters=parameters,
            root_dir=root_dir,
        )
        _write_csv(
            summary,
            out(f"{index}_{name}_summary.csv"),
            inputs=boolq_inputs,
            parameters=parameters,
            root_dir=root_dir,
        )
    accuracy: pd.DataFrame = model_accuracy(signals)
    extras["model_accuracy"] = accuracy
    _write_csv(
        accuracy,
        out("1_label_conditioning_model_accuracy.csv"),
        inputs=boolq_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )
    _write_csv(
        fits,
        out("2_length_control_fits.csv"),
        inputs=boolq_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )

    logger.info("Analysis 6: table2_pair_type")
    table2: pd.DataFrame = table2_pair_type(args.results_dir)
    table2_summary: pd.DataFrame = summarize_table2(table2)
    summaries["table2_pair_type"] = table2_summary
    _write_csv(
        table2,
        out("6_table2_pair_type.csv"),
        inputs=table2_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )
    _write_csv(
        table2_summary,
        out("6_table2_pair_type_summary.csv"),
        inputs=table2_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )

    logger.info("Analysis 7: decisive_segments")
    decisive: pd.DataFrame = decisive_segments(signals)
    decisive_summary: pd.DataFrame = summarize_decisive(decisive)
    summaries["decisive_segments"] = decisive_summary
    extras["decisive_rows"] = decisive
    _write_csv(
        decisive,
        out("7_decisive_segments.csv"),
        inputs=boolq_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )
    _write_csv(
        decisive_summary,
        out("7_decisive_segments_summary.csv"),
        inputs=boolq_inputs,
        parameters=parameters,
        root_dir=root_dir,
    )

    write_report(out("summary.md"), summaries, extras)
    logger.info(f"Wrote {args.output_dir}/summary.md")


if __name__ == "__main__":
    main()
