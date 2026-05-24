# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Generate the F-table TSV from consolidated benchmark results.

Reads two consolidated TSVs per benchmark:
  ``{results_dir}/{benchmark}_{pregrouper}_segments.tsv``
      one row per (model, prompt, seg) — attention + rep metrics
  ``{results_dir}/{benchmark}_{pregrouper}_logodds.tsv``
      one row per (model, prompt, seg, kind) — label_lp_* and logodds_* cols

Writes one long-format TSV with one row per
``(benchmark, model_s, model_t, metric, statistic)`` tuple, including a
bootstrap confidence interval ``[f_lo, f_hi]`` alongside the point
estimate.

Output schema (tab-separated)::

    benchmark  model_s  model_t  metric  statistic  f_point  f_lo  f_hi

``metric`` is one of:
  - Symmetric (correlation of model_s vs model_t signal):
      F_pred, F_attr, F_attn_rollout, F_attn_mean, F_attn_max, F_mag, F_align
  - Asymmetric (model_s signal predicts model_t ablation, concat-all
    correlation across all (prompt, seg) pairs — same bootstrap path as
    symmetric metrics, just with two different signal types and ordered
    pairs):
      F_align_to_attr, F_mag_to_attr, F_attn_rollout_to_attr,
      F_attn_mean_to_attr, F_attn_max_to_attr

``statistic`` is ``spearman`` or ``pearson_r2``. Eligible models are
auto-discovered from the segment / logodds TSVs by required columns.
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BENCHMARKS: list[str] = [
    "boolq",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "winogrande",
    "lambada",
]


# ---------------------------------------------------------------------------
# Signal extractors: each returns a per-model series indexed by (prompt_idx,
# seg_idx) for segment-level signals or by prompt_idx for prompt-level.
# ---------------------------------------------------------------------------


def _per_prompt_logodds(
    logodds_df: pd.DataFrame, model: str, kind: str
) -> pd.Series | None:
    """Original log-odds for the binary case: pick the first ``logodds_*`` col."""
    cols: list[str] = [c for c in logodds_df.columns if c.startswith("logodds_")]
    if not cols:
        return None
    sub: pd.DataFrame = logodds_df[
        (logodds_df["model"] == model) & (logodds_df["kind"] == kind)
    ]
    if sub.empty:
        return None
    return sub.set_index("prompt_idx")[cols[0]]


def _per_seg_ablation(
    logodds_df: pd.DataFrame, model: str
) -> pd.Series | None:
    """Ablation response = orig_logodds - ablated_logodds, per (prompt, seg)."""
    cols: list[str] = [c for c in logodds_df.columns if c.startswith("logodds_")]
    if not cols:
        return None
    col: str = cols[0]
    sub: pd.DataFrame = logodds_df[logodds_df["model"] == model]
    orig: pd.DataFrame = sub[sub["kind"] == "orig"][["prompt_idx", col]].rename(
        columns={col: "orig"}
    )
    abl: pd.DataFrame = sub[sub["kind"] == "ablated"][
        ["prompt_idx", "seg_idx", col]
    ].rename(columns={col: "ablated"})
    if orig.empty or abl.empty:
        return None
    merged: pd.DataFrame = abl.merge(orig, on="prompt_idx", how="inner")
    merged["ablation"] = merged["orig"] - merged["ablated"]
    return merged.set_index(["prompt_idx", "seg_idx"])["ablation"]


def _per_seg_segment_col(
    seg_df: pd.DataFrame, model: str, col: str
) -> pd.Series | None:
    if col not in seg_df.columns:
        return None
    sub: pd.DataFrame = seg_df[seg_df["model"] == model]
    if sub.empty or sub[col].isna().all():
        return None
    return sub.set_index(["prompt_idx", "seg_idx"])[col]


def _per_seg_align(seg_df: pd.DataFrame, model: str) -> pd.Series | None:
    needed: list[str] = ["w_dot_delta_z_postnorm", "w_norm", "delta_norm_postnorm"]
    for c in needed:
        if c not in seg_df.columns:
            return None
    sub: pd.DataFrame = seg_df[seg_df["model"] == model]
    if sub.empty or sub[needed].isna().all().any():
        return None
    s: pd.Series = sub["w_dot_delta_z_postnorm"] / (sub["w_norm"] * sub["delta_norm_postnorm"])
    return s.set_axis(pd.MultiIndex.from_arrays([sub["prompt_idx"], sub["seg_idx"]]))


def _eligible_logodds_models(logodds_df: pd.DataFrame, kind: str) -> list[str]:
    cols: list[str] = [c for c in logodds_df.columns if c.startswith("logodds_")]
    if not cols:
        return []
    sub: pd.DataFrame = logodds_df[logodds_df["kind"] == kind]
    return sorted(sub["model"].dropna().unique().tolist())


def _eligible_seg_models(seg_df: pd.DataFrame, required_cols: list[str]) -> list[str]:
    for c in required_cols:
        if c not in seg_df.columns:
            return []
    out: list[str] = []
    for m in sorted(seg_df["model"].dropna().unique()):
        sub: pd.DataFrame = seg_df[seg_df["model"] == m]
        if not sub[required_cols].isna().all().any():
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _pearson_batch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    denom = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out = np.full(num.shape, np.nan)
    nz = denom > 0
    out[nz] = num[nz] / denom[nz]
    return out


def _bootstrap_corrs(
    x: np.ndarray,
    y: np.ndarray,
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    if len(x) < 3:
        return {}
    rx, ry = rankdata(x), rankdata(y)
    idx = rng.integers(0, len(x), (n_resamples, len(x)))
    spear = _pearson_batch(rx[idx], ry[idx])
    pear_r2 = _pearson_batch(x[idx], y[idx]) ** 2
    point_spear = float(_pearson_batch(rx[None, :], ry[None, :])[0])
    point_pear_r2 = float(_pearson_batch(x[None, :], y[None, :])[0] ** 2)
    alpha = (1.0 - conf) / 2.0 * 100.0
    out: dict[str, tuple[float, float, float]] = {}
    for name, point, samples in [
        ("spearman", point_spear, spear),
        ("pearson_r2", point_pear_r2, pear_r2),
    ]:
        finite = samples[np.isfinite(samples)]
        if not np.isfinite(point) or len(finite) == 0:
            continue
        lo, hi = np.percentile(finite, [alpha, 100.0 - alpha])
        out[name] = (point, float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def _pair_to_arrays(sa: pd.Series, sb: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join two series on their (multi-)index, drop ±inf/NaN, return arrays."""
    common = sa.index.intersection(sb.index)
    if len(common) == 0:
        return np.array([]), np.array([])
    a, b = sa.loc[common], sb.loc[common]
    paired = (
        pd.DataFrame({"x": a.values, "y": b.values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    return paired["x"].values, paired["y"].values


def _emit_pair_corrs(
    benchmark: str,
    metric: str,
    sig_by_model: dict[str, pd.Series],
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    models: list[str] = sorted(sig_by_model.keys())
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            xs, ys = _pair_to_arrays(sig_by_model[a], sig_by_model[b])
            for stat, (point, lo, hi) in _bootstrap_corrs(
                xs, ys, n_resamples, conf, rng
            ).items():
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_s": a,
                        "model_t": b,
                        "metric": metric,
                        "statistic": stat,
                        "f_point": point,
                        "f_lo": lo,
                        "f_hi": hi,
                    }
                )
    return rows


def _emit_transfer(
    benchmark: str,
    metric: str,
    src_by_model: dict[str, pd.Series],
    tgt_by_model: dict[str, pd.Series],
    signed: bool,
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
) -> list[dict[str, str | float]]:
    """Concat-all transfer correlation: like ``_emit_pair_corrs`` but the two
    series are different signal types (source signal vs target ablation), so
    pairs are ordered (a != b, both directions emitted) instead of unordered."""
    rows: list[dict[str, str | float]] = []
    for a, sa in src_by_model.items():
        for b, sb in tgt_by_model.items():
            if a == b:
                continue
            common = sa.index.intersection(sb.index)
            if len(common) == 0:
                continue
            sv = sa.loc[common].values
            abl = sb.loc[common].values
            abl = abl if signed else np.abs(abl)
            paired = (
                pd.DataFrame({"x": sv, "y": abl})
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            xs, ys = paired["x"].values, paired["y"].values
            for stat, (point, lo, hi) in _bootstrap_corrs(
                xs, ys, n_resamples, conf, rng
            ).items():
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_s": a,
                        "model_t": b,
                        "metric": metric,
                        "statistic": stat,
                        "f_point": point,
                        "f_lo": lo,
                        "f_hi": hi,
                    }
                )
    return rows


# ---------------------------------------------------------------------------
# Per-benchmark driver
# ---------------------------------------------------------------------------


def _process_benchmark(
    benchmark: str,
    pregrouper: str,
    results_dir: str,
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
) -> list[dict[str, str | float]]:
    seg_path: str = os.path.join(
        results_dir, f"{benchmark}_{pregrouper}_segments.tsv"
    )
    odds_path: str = os.path.join(
        results_dir, f"{benchmark}_{pregrouper}_logodds.tsv"
    )
    if not os.path.exists(seg_path):
        logger.warning(f"Missing {seg_path}, skipping")
        return []
    seg_df: pd.DataFrame = pd.read_csv(seg_path, sep="\t")
    odds_df: pd.DataFrame = (
        pd.read_csv(odds_path, sep="\t") if os.path.exists(odds_path) else pd.DataFrame()
    )

    out: list[dict[str, str | float]] = []

    # F_pred — per-prompt orig logodds, model_s vs model_t.
    if not odds_df.empty:
        models_pred: list[str] = _eligible_logodds_models(odds_df, "orig")
        sig_by_model: dict[str, pd.Series] = {
            m: s for m in models_pred
            if (s := _per_prompt_logodds(odds_df, m, "orig")) is not None
        }
        out.extend(_emit_pair_corrs(
            benchmark, "F_pred", sig_by_model, n_resamples, conf, rng
        ))

    # F_attr — per-(prompt, seg) ablation response.
    abl_by_model: dict[str, pd.Series] = {}
    if not odds_df.empty:
        for m in odds_df["model"].dropna().unique():
            s = _per_seg_ablation(odds_df, m)
            if s is not None:
                abl_by_model[m] = s
        out.extend(_emit_pair_corrs(
            benchmark, "F_attr", abl_by_model, n_resamples, conf, rng
        ))

    # F_attn_{rollout,mean,max} — per-(prompt, seg) attention scores.
    for variant in ("rollout", "mean", "max"):
        col: str = f"attention_{variant}"
        models_attn: list[str] = _eligible_seg_models(seg_df, [col])
        sig_by_model = {
            m: s for m in models_attn
            if (s := _per_seg_segment_col(seg_df, m, col)) is not None
        }
        out.extend(_emit_pair_corrs(
            benchmark, f"F_attn_{variant}", sig_by_model, n_resamples, conf, rng
        ))

    # F_mag — per-(prompt, seg) post-norm delta_norm.
    models_mag: list[str] = _eligible_seg_models(seg_df, ["delta_norm_postnorm"])
    sig_by_model = {
        m: s for m in models_mag
        if (s := _per_seg_segment_col(seg_df, m, "delta_norm_postnorm")) is not None
    }
    out.extend(_emit_pair_corrs(
        benchmark, "F_mag", sig_by_model, n_resamples, conf, rng
    ))

    # F_align — cos(Δz, w) signed via w_dot_delta_z_postnorm / (w_norm × delta_norm_postnorm).
    models_align: list[str] = _eligible_seg_models(
        seg_df, ["w_dot_delta_z_postnorm", "w_norm", "delta_norm_postnorm"]
    )
    align_by_model: dict[str, pd.Series] = {
        m: s for m in models_align
        if (s := _per_seg_align(seg_df, m)) is not None
    }
    out.extend(_emit_pair_corrs(
        benchmark, "F_align", align_by_model, n_resamples, conf, rng
    ))

    # Transfer metrics: source signal × target ablation, concat-all (same
    # bootstrap path as symmetric pair-corrs).
    if abl_by_model:
        # F_align_to_attr (signed)
        out.extend(_emit_transfer(
            benchmark, "F_align_to_attr", align_by_model, abl_by_model,
            signed=True, n_resamples=n_resamples, conf=conf, rng=rng,
        ))
        # F_mag_to_attr (unsigned target)
        mag_by_model: dict[str, pd.Series] = {
            m: s for m in models_mag
            if (s := _per_seg_segment_col(seg_df, m, "delta_norm_postnorm")) is not None
        }
        out.extend(_emit_transfer(
            benchmark, "F_mag_to_attr", mag_by_model, abl_by_model,
            signed=False, n_resamples=n_resamples, conf=conf, rng=rng,
        ))
        for variant in ("rollout", "mean", "max"):
            col = f"attention_{variant}"
            attn_by_model: dict[str, pd.Series] = {
                m: s for m in _eligible_seg_models(seg_df, [col])
                if (s := _per_seg_segment_col(seg_df, m, col)) is not None
            }
            out.extend(_emit_transfer(
                benchmark, f"F_attn_{variant}_to_attr", attn_by_model, abl_by_model,
                signed=True, n_resamples=n_resamples, conf=conf, rng=rng,
            ))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate long-format F-table TSV with bootstrap CIs."
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Directory holding consolidated TSVs and where the F-table is written.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output TSV path (default: {results-dir}/f_table.tsv).",
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=DEFAULT_BENCHMARKS,
        help="Benchmarks to include.",
    )
    parser.add_argument(
        "--pregrouper", default="sentence", choices=["word", "sentence"],
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=1000,
        help="Number of bootstrap resamples per (pair, metric, statistic).",
    )
    parser.add_argument(
        "--confidence-level", type=float, default=0.95,
        help="Confidence level for f_lo/f_hi (e.g., 0.95 -> 2.5%/97.5% percentiles).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducible bootstraps."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)
    out_path: str = args.output or os.path.join(args.results_dir, "f_table.tsv")

    all_rows: list[dict[str, str | float]] = []
    for bench in tqdm(args.benchmarks, desc="benchmarks", unit="bench"):
        all_rows.extend(_process_benchmark(
            bench, args.pregrouper, args.results_dir,
            args.bootstrap_resamples, args.confidence_level, rng,
        ))

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
