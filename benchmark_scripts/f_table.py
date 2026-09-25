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

Writes one long-format TSV with one row per benchmark configuration, message
scope, label contrast, model pair, metric, and statistic. Segment-level
confidence intervals use a prompt-cluster bootstrap.

Output schema (tab-separated)::

    cohort  benchmark  pregrouper  scope  requested_scope  resolved_scope
    contrast  requested_contrast
    resolved_source_contrast  resolved_target_contrast  readout_contrast
    model_s  model_t  metric ...

``metric`` is one of:
  - Symmetric (correlation of model_s vs model_t signal):
      F_pred, F_attr, F_attn_rollout, F_attn_mean, F_attn_max, F_mag, F_align
  - Asymmetric (model_s signal predicts model_t ablation, averaged over
    pooled segment rows by default and evaluated on ordered model pairs):
      F_align_to_attr, F_mag_to_attr, F_attn_rollout_to_attr,
      F_attn_mean_to_attr, F_attn_max_to_attr

``statistic`` is ``spearman``, ``pearson_r``, or ``pearson_r2``. The default
analysis uses full-dialog segment coordinates, the entailment-minus-
contradiction ANLI contrast, pairwise-complete observations, and the fixed
11-model release cohort. ``--all-models`` opts into every discovered model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tqdm.auto import tqdm

from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    derived_supporting_source_paths,
    write_derived_provenance,
)

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_CONFIGS: list[tuple[str, str]] = [
    ("boolq", "sentence"),
    ("anli_r1", "sentence"),
    ("anli_r2", "sentence"),
    ("anli_r3", "sentence"),
    ("winogrande", "sentence"),
    ("boolq", "word"),
    ("lambada", "word"),
]

OPEN_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
)
API_MODELS: tuple[str, ...] = (
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)
# The API-served Llama-3.1-8B duplicates the locally evaluated model and is
# retained only as a diagnostic. The paper's headline cohort excludes it.
PAPER_MODELS: tuple[str, ...] = OPEN_MODELS + API_MODELS[1:]


def _unsupported_model_components(
    results_dir: str, benchmark: str, pregrouper: str
) -> tuple[set[str], set[str]]:
    """Return hosted models unsupported for prediction and attribution.

    The fixed canary controls whether a full LAMBADA rerun is attempted. The
    derived-table decision is based on the coverage of the final public
    artifact itself, so a usable prediction component is never discarded only
    because attribution coverage was inadequate (or vice versa).
    """
    config_dir: str = os.path.join(results_dir, benchmark, pregrouper)
    prediction: set[str] = set()
    attribution: set[str] = set()
    for model in API_MODELS:
        run_path: str = os.path.join(config_dir, f"{model}_run.json")
        if not os.path.isfile(run_path):
            continue
        with open(run_path, encoding="utf-8") as source:
            provenance: Any = json.load(source)
        if (
            not isinstance(provenance, dict)
            or provenance.get("availability_status") != "unsupported_after_canary"
        ):
            continue
        segment_base: str = os.path.join(config_dir, f"{model}_segment.tsv")
        segment_path: str | None = next(
            (
                candidate
                for candidate in (segment_base + ".gz", segment_base)
                if os.path.isfile(candidate)
            ),
            None,
        )
        if segment_path is None:
            raise FileNotFoundError(f"Missing unsupported output for {model}")
        component_frame: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
        required_columns: set[str] = {
            "prompt_idx",
            "original_result_available",
            "segment_result_available",
        }
        if not required_columns.issubset(component_frame.columns):
            raise ValueError(f"Invalid unsupported output schema in {segment_path}")
        original_by_prompt: pd.Series = component_frame.groupby("prompt_idx")[
            "original_result_available"
        ].any()
        original_coverage: float = float(original_by_prompt.eq(True).mean())
        paired_coverage: float = float(
            (
                component_frame["original_result_available"].eq(True)
                & component_frame["segment_result_available"].eq(True)
            ).mean()
        )
        if original_coverage < 0.8:
            prediction.add(model)
        if paired_coverage < 0.8:
            attribution.add(model)
    return prediction, attribution


def _mask_unsupported_signals(
    signals: dict[str, pd.Series], unsupported: set[str]
) -> dict[str, pd.Series]:
    """Preserve the semantic pair grid but suppress rejected partial samples."""
    return {
        model: (
            pd.Series(np.nan, index=signal.index, dtype=float)
            if model in unsupported
            else signal
        )
        for model, signal in signals.items()
    }


def _clamp_api_infinities(signal: pd.Series, model: str) -> pd.Series:
    """Apply the finite-extreme sensitivity policy described in the paper."""
    if model not in API_MODELS:
        return signal
    finite: pd.Series = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return signal
    low: float = float(finite.min() - abs(finite.min()) * 0.01 - 0.01)
    high: float = float(finite.max() + abs(finite.max()) * 0.01 + 0.01)
    return signal.replace({np.inf: high, -np.inf: low})


# ---------------------------------------------------------------------------
# Signal extractors: each returns a per-model series indexed by (prompt_idx,
# seg_idx) for segment-level signals or by prompt_idx for prompt-level.
# ---------------------------------------------------------------------------


def _per_prompt_logodds(
    logodds_df: pd.DataFrame,
    model: str,
    kind: str,
    contrast_col: str = "logodds",
    finite_extreme_api: bool = False,
) -> pd.Series | None:
    """Return the benchmark's explicitly defined canonical log-odds."""
    if contrast_col not in logodds_df.columns:
        return None
    sub: pd.DataFrame = logodds_df[
        (logodds_df["model"] == model) & (logodds_df["kind"] == kind)
    ]
    if sub.empty:
        return None
    signal: pd.Series = sub.set_index("prompt_idx")[contrast_col]
    return _clamp_api_infinities(signal, model) if finite_extreme_api else signal


def _per_seg_ablation(
    logodds_df: pd.DataFrame,
    model: str,
    contrast_col: str = "logodds",
    finite_extreme_api: bool = False,
) -> pd.Series | None:
    """Ablation response = orig_logodds - ablated_logodds, per (prompt, seg)."""
    if contrast_col not in logodds_df.columns:
        return None
    col: str = contrast_col
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
    signal: pd.Series = merged.set_index(["prompt_idx", "seg_idx"])["ablation"]
    return _clamp_api_infinities(signal, model) if finite_extreme_api else signal


def _per_prompt_completion(seg_df: pd.DataFrame, model: str) -> pd.Series | None:
    """Return one original completion log-probability per prompt."""
    col: str = "orig_completion_logprob"
    if col not in seg_df.columns:
        return None
    sub: pd.DataFrame = seg_df[seg_df["model"] == model]
    if sub.empty:
        return None
    return sub.groupby("prompt_idx")[col].first()


def _per_seg_completion_ablation(seg_df: pd.DataFrame, model: str) -> pd.Series | None:
    """Return original-minus-ablated completion score per segment."""
    needed: list[str] = ["orig_completion_logprob", "ablated_completion_logprob"]
    if any(col not in seg_df.columns for col in needed):
        return None
    sub: pd.DataFrame = seg_df[seg_df["model"] == model]
    if sub.empty:
        return None
    signal: pd.Series = (
        sub["orig_completion_logprob"] - sub["ablated_completion_logprob"]
    )
    return signal.set_axis(
        pd.MultiIndex.from_arrays([sub["prompt_idx"], sub["seg_idx"]])
    )


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
    s: pd.Series = sub["w_dot_delta_z_postnorm"] / (
        sub["w_norm"] * sub["delta_norm_postnorm"]
    )
    return s.set_axis(pd.MultiIndex.from_arrays([sub["prompt_idx"], sub["seg_idx"]]))


def _eligible_logodds_models(
    logodds_df: pd.DataFrame,
    kind: str,
    contrast_col: str = "logodds",
) -> list[str]:
    if contrast_col not in logodds_df.columns:
        return []
    sub: pd.DataFrame = logodds_df[logodds_df["kind"] == kind]
    return sorted(sub["model"].dropna().unique().tolist())


def _in_cohort(models: list[str], cohort: tuple[str, ...] | None) -> list[str]:
    if cohort is None:
        return models
    available: set[str] = set(models)
    return [model for model in cohort if model in available]


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


def _contrast_column(contrast: str) -> str:
    """Map a CLI contrast name to the corresponding log-odds column."""
    return "logodds" if contrast == "canonical" else f"logodds_{contrast}"


def _readout_matches_contrast(benchmark: str, contrast: str) -> bool:
    """Whether stored w projections correspond to the requested contrast."""
    return contrast == "canonical" or (
        benchmark.startswith("anli_") and contrast == "entailment_neutral"
    )


def _uses_layer_readout(benchmark: str, contrast: str) -> bool:
    """Whether the requested alignment is supplied by a layer artifact."""
    return benchmark.startswith("anli_") and contrast == "entailment_contradiction"


def _resolved_contrast(benchmark: str, contrast: str) -> str:
    """Resolve a requested contrast to an explicit ordered scientific quantity."""
    if contrast != "canonical":
        explicit: dict[str, str] = {
            "entailment_neutral": "entailment_minus_neutral",
            "entailment_contradiction": "entailment_minus_contradiction",
        }
        if contrast not in explicit:
            raise ValueError(f"Unknown explicit contrast {contrast!r}")
        return explicit[contrast]
    if benchmark == "boolq":
        return "true_minus_false"
    if benchmark.startswith("anli_"):
        return "entailment_minus_neutral"
    if benchmark == "winogrande":
        return "option_1_minus_option_2"
    if benchmark == "race":
        return "correct_answer_minus_logsumexp_other_answers"
    if benchmark == "lambada":
        return "target_completion_logprob"
    raise ValueError(f"No canonical contrast is defined for {benchmark!r}")


def _resolved_scope(metric: str, scope: str) -> str:
    """Make clear that scope filters coordinates, not the full model input."""
    if metric == "F_pred":
        return "prompt_level_full_dialog"
    return f"{scope}_segment_coordinates_from_full_dialog"


def _analysis_jobs(
    benchmark_configs: list[tuple[str, str]],
    scopes: list[str],
    contrasts: list[str],
    anli_contrast: str,
) -> list[tuple[str, str, str, str]]:
    """Resolve requested analyses without overriding explicit ANLI contrasts.

    ``--anli-contrast`` defines only what ``canonical`` means for ANLI. An
    explicitly requested contrast remains explicit, which permits E-N and E-C
    sensitivity analyses in one invocation. Duplicate resolved jobs are
    removed while preserving request order.
    """
    jobs: list[tuple[str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for benchmark, pregrouper in benchmark_configs:
        for scope in scopes:
            for contrast in contrasts:
                resolved_contrast: str = (
                    anli_contrast
                    if benchmark.startswith("anli_") and contrast == "canonical"
                    else contrast
                )
                job: tuple[str, str, str, str] = (
                    benchmark,
                    pregrouper,
                    scope,
                    resolved_contrast,
                )
                if job not in seen:
                    seen.add(job)
                    jobs.append(job)
    return jobs


def _contrast_metadata(
    benchmark: str, contrast: str, metric: str
) -> tuple[str, str, str]:
    """Return resolved source, target, and readout contrasts for one metric."""
    requested: str = _resolved_contrast(benchmark, contrast)
    readout: str = (
        requested
        if _uses_layer_readout(benchmark, contrast)
        else _resolved_contrast(benchmark, "canonical")
    )
    if metric in {"F_pred", "F_attr"}:
        return requested, requested, "not_applicable"
    if metric == "F_align":
        return readout, readout, readout
    if metric == "F_align_to_attr":
        return readout, requested, readout
    if metric.endswith("_to_attr"):
        return "not_applicable", requested, "not_applicable"
    return "not_applicable", "not_applicable", "not_applicable"


def _analysis_rng(seed: int, *identity_parts: str) -> np.random.Generator:
    """Return a stable RNG derived from an explicit scientific cell identity."""
    identity: bytes = "\0".join((str(seed), *identity_parts)).encode("utf-8")
    derived_seed: int = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def _log_final_layer_crosscheck(
    benchmark: str,
    model: str,
    signal_name: str,
    layer_signal: pd.Series,
    ordinary_signal: pd.Series | None,
) -> None:
    """Log a diagnostic comparison without imposing an acceptance threshold."""
    if ordinary_signal is None:
        return
    joined: pd.DataFrame = pd.concat(
        [layer_signal.rename("layer"), ordinary_signal.rename("ordinary")],
        axis=1,
        join="inner",
    )
    finite: np.ndarray = np.isfinite(joined.to_numpy(dtype=float)).all(axis=1)
    complete: pd.DataFrame = joined.loc[finite]
    if len(complete) < 2:
        logger.warning(
            "%s/%s final-layer %s cross-check has only %d complete cells",
            benchmark,
            model,
            signal_name,
            len(complete),
        )
        return
    error: pd.Series = (complete["layer"] - complete["ordinary"]).abs()
    correlation: float = float(
        np.corrcoef(
            complete["layer"].to_numpy(dtype=float),
            complete["ordinary"].to_numpy(dtype=float),
        )[0, 1]
    )
    logger.info(
        "%s/%s final-layer %s diagnostic: n=%d pearson_r=%.9g "
        "median_abs_delta=%.9g max_abs_delta=%.9g",
        benchmark,
        model,
        signal_name,
        len(complete),
        correlation,
        float(error.median()),
        float(error.max()),
    )


def _derived_input_paths(
    results_dir: str,
    benchmark_configs: list[tuple[str, str]],
    cohort: tuple[str, ...] | None,
    layer_alignment_configs: set[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """Return immediate tables and committed inputs for an F-table invocation."""
    inputs: dict[str, str] = {}
    layer_configs: set[tuple[str, str]] = layer_alignment_configs or set()
    for benchmark, pregrouper in benchmark_configs:
        segment_path: str = os.path.join(
            results_dir, f"{benchmark}_{pregrouper}_segments.tsv"
        )
        logodds_path: str = os.path.join(
            results_dir, f"{benchmark}_{pregrouper}_logodds.tsv"
        )
        config_inputs: dict[str, str] = collect_result_inputs(
            results_dir,
            benchmark,
            pregrouper,
            {"segments": segment_path, "logodds": logodds_path},
            cohort,
        )
        config_inputs = {
            identifier: path
            for identifier, path in config_inputs.items()
            if not os.path.basename(path).endswith("_layers_run.json")
        }
        if (benchmark, pregrouper) in layer_configs:
            from benchmark_scripts.layerwise_fidelity import (
                _layer_sidecar_path,
                _model_path,
            )

            layer_models: tuple[str, ...] = tuple(
                model for model in OPEN_MODELS if cohort is None or model in cohort
            )
            prefix: str = f"{benchmark}/{pregrouper}/raw"
            for model in layer_models:
                layer_path: str = _model_path(results_dir, benchmark, pregrouper, model)
                sidecar_path: str = _layer_sidecar_path(layer_path)
                config_inputs[f"{prefix}/{os.path.basename(layer_path)}"] = layer_path
                config_inputs[f"{prefix}/{os.path.basename(sidecar_path)}"] = (
                    sidecar_path
                )
        inputs.update(config_inputs)
    return inputs


def _supporting_source_paths() -> dict[str, str]:
    """Return public sources that materially define the F-table."""
    paths: dict[str, str] = derived_supporting_source_paths()
    relative_path: str = "benchmark_scripts/layerwise_fidelity.py"
    paths[relative_path] = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), relative_path
    )
    return paths


def _scope_keys(
    benchmark: str,
    pregrouper: str,
    results_dir: str,
    seg_df: pd.DataFrame,
    scope: str,
) -> pd.MultiIndex | None:
    """Load the canonical segment keys for a requested message-role scope."""
    if scope == "all":
        return None
    manifest_base: str = os.path.join(
        results_dir, benchmark, pregrouper, "segments.tsv"
    )
    manifest_path: str | None = next(
        (
            path
            for path in (manifest_base + ".gz", manifest_base)
            if os.path.exists(path)
        ),
        None,
    )
    manifest: pd.DataFrame = (
        pd.read_csv(manifest_path, sep="\t") if manifest_path is not None else seg_df
    )
    required: set[str] = {"prompt_idx", "seg_idx", "message_role"}
    missing: set[str] = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Cannot select scope {scope!r} for {benchmark}/{pregrouper}; "
            f"segment metadata is missing {sorted(missing)}"
        )
    scoped: pd.DataFrame = manifest[manifest["message_role"] == scope]
    return pd.MultiIndex.from_frame(scoped[["prompt_idx", "seg_idx"]].drop_duplicates())


def _filter_segment_frame(
    frame: pd.DataFrame,
    allowed_keys: pd.MultiIndex | None,
) -> pd.DataFrame:
    """Restrict segment-indexed rows while retaining every prompt-level row."""
    if allowed_keys is None or frame.empty:
        return frame
    segment_rows: pd.Series = frame["seg_idx"].notna()
    keys: pd.MultiIndex = pd.MultiIndex.from_frame(
        frame.loc[segment_rows, ["prompt_idx", "seg_idx"]]
    )
    keep: pd.Series = ~segment_rows
    keep.loc[segment_rows] = keys.isin(allowed_keys)
    return frame.loc[keep].copy()


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


def _weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Return Pearson correlation after repeating rows by integer weights."""
    total: float = float(weights.sum())
    if total <= 0:
        return float("nan")
    mean_x: float = float(np.dot(weights, x) / total)
    mean_y: float = float(np.dot(weights, y) / total)
    centered_x: np.ndarray = x - mean_x
    centered_y: np.ndarray = y - mean_y
    numerator: float = float(np.dot(weights, centered_x * centered_y))
    denominator: float = float(
        np.sqrt(
            np.dot(weights, centered_x * centered_x)
            * np.dot(weights, centered_y * centered_y)
        )
    )
    return numerator / denominator if denominator > 0 else float("nan")


def _weighted_ranks(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return exact midranks for a sample represented by row multiplicities."""
    order: np.ndarray = np.argsort(values, kind="stable")
    sorted_values: np.ndarray = values[order]
    group_start: np.ndarray = np.concatenate(
        ([True], sorted_values[1:] != sorted_values[:-1])
    )
    group_ids: np.ndarray = np.cumsum(group_start) - 1
    return _weighted_ranks_prepared(weights, order, group_ids)


def _weighted_ranks_prepared(
    weights: np.ndarray,
    order: np.ndarray,
    group_ids: np.ndarray,
) -> np.ndarray:
    """Compute weighted midranks using a precomputed value ordering."""
    sorted_weights: np.ndarray = weights[order]
    group_weights: np.ndarray = np.bincount(
        group_ids,
        weights=sorted_weights,
    )
    cumulative_before: np.ndarray = np.concatenate(
        ([0.0], np.cumsum(group_weights)[:-1])
    )
    group_ranks: np.ndarray = cumulative_before + (group_weights + 1.0) / 2.0
    ranks: np.ndarray = np.empty(len(weights), dtype=float)
    ranks[order] = group_ranks[group_ids]
    return ranks


def _bootstrap_corrs(
    x: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
) -> dict[str, tuple[float, float, float]]:
    """Correlate paired values with an exact prompt-cluster bootstrap."""
    if len(x) < 3:
        missing: tuple[float, float, float] = (
            float("nan"),
            float("nan"),
            float("nan"),
        )
        return {
            "spearman": missing,
            "pearson_r": missing,
            "pearson_r2": missing,
        }
    rx, ry = rankdata(x), rankdata(y)
    unique_clusters, cluster_inverse = np.unique(clusters, return_inverse=True)

    def _cluster_sufficient_statistics(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        columns: list[np.ndarray] = []
        for values in (np.ones(len(a)), a, b, a * a, b * b, a * b):
            columns.append(
                np.bincount(
                    cluster_inverse,
                    weights=values,
                    minlength=len(unique_clusters),
                )
            )
        return np.stack(columns, axis=1)

    def _corr_from_stats(stats: np.ndarray) -> np.ndarray:
        n, sum_x, sum_y, sum_x2, sum_y2, sum_xy = stats.T
        numerator: np.ndarray = sum_xy - sum_x * sum_y / n
        denominator: np.ndarray = np.sqrt(
            (sum_x2 - sum_x * sum_x / n) * (sum_y2 - sum_y * sum_y / n)
        )
        result: np.ndarray = np.full(len(stats), np.nan)
        nonzero: np.ndarray = denominator > 0
        result[nonzero] = numerator[nonzero] / denominator[nonzero]
        return result

    sampled_clusters: np.ndarray = rng.integers(
        0,
        len(unique_clusters),
        (n_resamples, len(unique_clusters)),
    )
    raw_stats: np.ndarray = _cluster_sufficient_statistics(x, y)
    pearson: np.ndarray = _corr_from_stats(raw_stats[sampled_clusters].sum(axis=1))
    x_order: np.ndarray = np.argsort(x, kind="stable")
    y_order: np.ndarray = np.argsort(y, kind="stable")
    x_sorted: np.ndarray = x[x_order]
    y_sorted: np.ndarray = y[y_order]
    x_groups: np.ndarray = (
        np.cumsum(np.concatenate(([True], x_sorted[1:] != x_sorted[:-1]))) - 1
    )
    y_groups: np.ndarray = (
        np.cumsum(np.concatenate(([True], y_sorted[1:] != y_sorted[:-1]))) - 1
    )
    spear_values: list[float] = []
    for sampled in sampled_clusters:
        cluster_weights: np.ndarray = np.bincount(
            sampled,
            minlength=len(unique_clusters),
        )
        row_weights: np.ndarray = cluster_weights[cluster_inverse]
        weights: np.ndarray = row_weights.astype(float)
        bootstrap_rx: np.ndarray = _weighted_ranks_prepared(weights, x_order, x_groups)
        bootstrap_ry: np.ndarray = _weighted_ranks_prepared(weights, y_order, y_groups)
        spear_values.append(_weighted_correlation(bootstrap_rx, bootstrap_ry, weights))
    spear: np.ndarray = np.asarray(spear_values)
    pear_r2: np.ndarray = pearson**2
    point_spear = float(_pearson_batch(rx[None, :], ry[None, :])[0])
    point_pear = float(_pearson_batch(x[None, :], y[None, :])[0])
    point_pear_r2 = point_pear**2
    alpha = (1.0 - conf) / 2.0 * 100.0
    out: dict[str, tuple[float, float, float]] = {}
    for name, point, samples in [
        ("spearman", point_spear, spear),
        ("pearson_r", point_pear, pearson),
        ("pearson_r2", point_pear_r2, pear_r2),
    ]:
        finite = samples[np.isfinite(samples)]
        if not np.isfinite(point) or len(finite) == 0:
            out[name] = (point, float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(finite, [alpha, 100.0 - alpha])
            out[name] = (point, float(lo), float(hi))
    return out


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------


def _pair_to_arrays(
    sa: pd.Series,
    sb: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Inner-join two series and return valid values and expected pair counts."""
    common = sa.index.intersection(sb.index)
    if len(common) == 0:
        return np.array([]), np.array([]), np.array([]), 0, 0
    a, b = sa.loc[common], sb.loc[common]
    paired = (
        pd.DataFrame({"x": a.values, "y": b.values})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    clusters: np.ndarray = np.asarray(
        [index[0] if isinstance(index, tuple) else index for index in common]
    )
    valid: np.ndarray = np.isfinite(a.values) & np.isfinite(b.values)
    return (
        paired["x"].values,
        paired["y"].values,
        clusters[valid],
        len(common),
        len(np.unique(clusters)),
    )


def _coverage_fraction(observed: int, expected: int) -> float:
    """Return a coverage fraction, or NaN when no observations are expected."""
    return observed / expected if expected else float("nan")


def _emit_pair_corrs(
    benchmark: str,
    metric: str,
    sig_by_model: dict[str, pd.Series],
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
    *,
    bootstrap_seed: int | None = None,
    rng_context: tuple[str, ...] = (),
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    # Dictionaries are built in explicit cohort order. Preserve it so output
    # pair orientation is deterministic and matches the documented cohort.
    models: list[str] = list(sig_by_model)
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            (
                xs,
                ys,
                clusters,
                expected_observations,
                expected_prompts,
            ) = _pair_to_arrays(sig_by_model[a], sig_by_model[b])
            n_observations: int = len(xs)
            n_prompts: int = len(np.unique(clusters))
            pair_rng: np.random.Generator = (
                _analysis_rng(bootstrap_seed, *rng_context, metric, a, b)
                if bootstrap_seed is not None
                else rng
            )
            for stat, (point, lo, hi) in _bootstrap_corrs(
                xs, ys, clusters, n_resamples, conf, pair_rng
            ).items():
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_s": a,
                        "model_t": b,
                        "metric": metric,
                        "statistic": stat,
                        "n_observations": n_observations,
                        "expected_observations": expected_observations,
                        "observation_coverage": _coverage_fraction(
                            n_observations, expected_observations
                        ),
                        "n_prompts": n_prompts,
                        "expected_prompts": expected_prompts,
                        "prompt_coverage": _coverage_fraction(
                            n_prompts, expected_prompts
                        ),
                        "f_point": point,
                        "f_lo": lo,
                        "f_hi": hi,
                    }
                )
    return rows


def _emit_unavailable_pairs(
    benchmark: str,
    metric: str,
    source_models: list[str],
    target_models: list[str],
    *,
    directed: bool,
    reason: str,
    aggregation: str = "row_pooled",
) -> list[dict[str, str | float]]:
    """Materialize an expected pair grid whose estimand is unavailable."""
    pairs: list[tuple[str, str]] = (
        [
            (source, target)
            for source in source_models
            for target in target_models
            if source != target
        ]
        if directed
        else [
            (source, target)
            for index, source in enumerate(source_models)
            for target in source_models[index + 1 :]
        ]
    )
    return [
        {
            "benchmark": benchmark,
            "model_s": source,
            "model_t": target,
            "metric": metric,
            "statistic": statistic,
            "aggregation": aggregation,
            "n_observations": 0,
            "expected_observations": 0,
            "observation_coverage": float("nan"),
            "n_prompts": 0,
            "expected_prompts": 0,
            "prompt_coverage": float("nan"),
            "f_point": float("nan"),
            "f_lo": float("nan"),
            "f_hi": float("nan"),
            "availability_status": "unavailable",
            "unavailable_reason": reason,
        }
        for source, target in pairs
        for statistic in ("spearman", "pearson_r", "pearson_r2")
    ]


def _emit_transfer(
    benchmark: str,
    metric: str,
    src_by_model: dict[str, pd.Series],
    tgt_by_model: dict[str, pd.Series],
    signed: bool,
    n_resamples: int,
    conf: float,
    rng: np.random.Generator,
    aggregation: str = "row_pooled",
    *,
    bootstrap_seed: int | None = None,
    rng_context: tuple[str, ...] = (),
) -> list[dict[str, str | float]]:
    """Compute row-pooled paper transfer or prompt-equal sensitivity."""
    if aggregation not in {"row_pooled", "prompt_equal_mean_r2"}:
        raise ValueError(f"Unknown transfer aggregation {aggregation!r}")
    rows: list[dict[str, str | float]] = []
    for a, sa in src_by_model.items():
        for b, sb in tgt_by_model.items():
            if a == b:
                continue
            common = sa.index.intersection(sb.index)
            if len(common) == 0:
                continue
            expected_observations: int = len(common)
            expected_prompt_values: np.ndarray = np.asarray(
                [index[0] if isinstance(index, tuple) else index for index in common]
            )
            expected_prompts: int = len(np.unique(expected_prompt_values))
            paired = (
                pd.DataFrame(
                    {
                        "x": sa.loc[common].values,
                        "y": sb.loc[common].values,
                        "prompt_idx": [
                            index[0] if isinstance(index, tuple) else index
                            for index in common
                        ],
                    }
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if not signed:
                paired["y"] = paired["y"].abs()

            if aggregation == "row_pooled":
                clusters: np.ndarray = paired["prompt_idx"].to_numpy()
                pair_rng: np.random.Generator = (
                    _analysis_rng(bootstrap_seed, *rng_context, metric, a, b)
                    if bootstrap_seed is not None
                    else rng
                )
                for stat, (point, lo, hi) in _bootstrap_corrs(
                    paired["x"].to_numpy(),
                    paired["y"].to_numpy(),
                    clusters,
                    n_resamples,
                    conf,
                    pair_rng,
                ).items():
                    n_observations = len(paired)
                    n_prompts = len(np.unique(clusters))
                    rows.append(
                        {
                            "benchmark": benchmark,
                            "model_s": a,
                            "model_t": b,
                            "metric": metric,
                            "statistic": stat,
                            "aggregation": aggregation,
                            "n_observations": n_observations,
                            "expected_observations": expected_observations,
                            "observation_coverage": _coverage_fraction(
                                n_observations, expected_observations
                            ),
                            "n_prompts": n_prompts,
                            "expected_prompts": expected_prompts,
                            "prompt_coverage": _coverage_fraction(
                                n_prompts, expected_prompts
                            ),
                            "f_point": point,
                            "f_lo": lo,
                            "f_hi": hi,
                        }
                    )
                continue

            per_prompt: dict[str, list[float]] = {
                "spearman": [],
                "pearson_r": [],
                "pearson_r2": [],
            }
            per_prompt_observations: dict[str, int] = {
                "spearman": 0,
                "pearson_r": 0,
                "pearson_r2": 0,
            }
            for _, prompt_rows in paired.groupby("prompt_idx", sort=False):
                x: np.ndarray = prompt_rows["x"].to_numpy()
                y: np.ndarray = prompt_rows["y"].to_numpy()
                if len(x) < 3 or x.std() == 0 or y.std() == 0:
                    continue
                pearson: float = float(np.corrcoef(x, y)[0, 1])
                spearman: float = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
                if np.isfinite(spearman):
                    per_prompt["spearman"].append(spearman)
                    per_prompt_observations["spearman"] += len(prompt_rows)
                if np.isfinite(pearson):
                    per_prompt["pearson_r"].append(pearson)
                    per_prompt_observations["pearson_r"] += len(prompt_rows)
                    per_prompt["pearson_r2"].append(pearson**2)
                    per_prompt_observations["pearson_r2"] += len(prompt_rows)

            for stat, values_list in per_prompt.items():
                values: np.ndarray = np.asarray(values_list)
                if len(values):
                    point: float = float(values.mean())
                    statistic_rng: np.random.Generator = (
                        _analysis_rng(
                            bootstrap_seed,
                            *rng_context,
                            metric,
                            a,
                            b,
                            stat,
                        )
                        if bootstrap_seed is not None
                        else rng
                    )
                    indices: np.ndarray = statistic_rng.integers(
                        0, len(values), (n_resamples, len(values))
                    )
                    samples: np.ndarray = values[indices].mean(axis=1)
                    alpha: float = (1.0 - conf) / 2.0 * 100.0
                    lo, hi = np.percentile(samples, [alpha, 100.0 - alpha])
                else:
                    point = float("nan")
                    lo, hi = float("nan"), float("nan")
                rows.append(
                    {
                        "benchmark": benchmark,
                        "model_s": a,
                        "model_t": b,
                        "metric": metric,
                        "statistic": stat,
                        "aggregation": aggregation,
                        "n_observations": per_prompt_observations[stat],
                        "expected_observations": expected_observations,
                        "observation_coverage": _coverage_fraction(
                            per_prompt_observations[stat], expected_observations
                        ),
                        "n_prompts": len(values),
                        "expected_prompts": expected_prompts,
                        "prompt_coverage": _coverage_fraction(
                            len(values), expected_prompts
                        ),
                        "f_point": point,
                        "f_lo": float(lo),
                        "f_hi": float(hi),
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
    cohort: tuple[str, ...] | None = PAPER_MODELS,
    scope: str = "all",
    contrast: str = "canonical",
    api_infinity_policy: str = "pairwise_complete",
    transfer_aggregation: str = "row_pooled",
    bootstrap_seed: int | None = None,
    requested_metrics: set[str] | None = None,
) -> list[dict[str, str | float]]:
    seg_path: str = os.path.join(results_dir, f"{benchmark}_{pregrouper}_segments.tsv")
    odds_path: str = os.path.join(results_dir, f"{benchmark}_{pregrouper}_logodds.tsv")
    if not os.path.exists(seg_path):
        logger.warning(f"Missing {seg_path}, skipping")
        return []
    seg_df: pd.DataFrame = pd.read_csv(seg_path, sep="\t")
    odds_df: pd.DataFrame = (
        pd.read_csv(odds_path, sep="\t")
        if os.path.exists(odds_path)
        else pd.DataFrame()
    )
    if cohort is not None:
        available_models: set[str] = set(seg_df["model"].dropna().astype(str))
        missing_models: set[str] = set(cohort) - available_models
        if missing_models:
            raise ValueError(
                f"{benchmark}/{pregrouper} omits required models "
                f"{sorted(missing_models)}"
            )
    unfiltered_seg_df: pd.DataFrame = seg_df
    allowed_keys: pd.MultiIndex | None = _scope_keys(
        benchmark, pregrouper, results_dir, seg_df, scope
    )
    seg_df = _filter_segment_frame(seg_df, allowed_keys)
    if not odds_df.empty:
        odds_df = _filter_segment_frame(odds_df, allowed_keys)
    contrast_col: str = _contrast_column(contrast)
    finite_extreme_api: bool = api_infinity_policy == "finite_extreme"
    if not odds_df.empty and contrast_col not in odds_df.columns:
        raise ValueError(
            f"Contrast {contrast!r} requires column {contrast_col!r} in " f"{odds_path}"
        )
    unsupported_prediction, unsupported_attribution = _unsupported_model_components(
        results_dir, benchmark, pregrouper
    )

    def rng_context(metric: str) -> tuple[str, ...]:
        source_contrast, target_contrast, readout_contrast = _contrast_metadata(
            benchmark, contrast, metric
        )
        return (
            benchmark,
            pregrouper,
            _resolved_scope(metric, scope),
            source_contrast,
            target_contrast,
            readout_contrast,
            transfer_aggregation if metric.endswith("_to_attr") else "row_pooled",
        )

    out: list[dict[str, str | float]] = []

    def finish_rows() -> list[dict[str, str | float]]:
        for row in out:
            metric: str = str(row["metric"])
            source_contrast, target_contrast, readout_contrast = _contrast_metadata(
                benchmark, contrast, metric
            )
            row.setdefault("aggregation", "row_pooled")
            row["pregrouper"] = pregrouper
            row["scope"] = scope
            row["requested_scope"] = scope
            row["resolved_scope"] = _resolved_scope(metric, scope)
            row["contrast"] = contrast
            row["requested_contrast"] = contrast
            row["resolved_source_contrast"] = source_contrast
            row["resolved_target_contrast"] = target_contrast
            row["readout_contrast"] = readout_contrast
            row["api_infinity_policy"] = api_infinity_policy
            point: float = float(row["f_point"])
            row.setdefault(
                "availability_status",
                "available" if np.isfinite(point) else "unavailable",
            )
            row.setdefault(
                "unavailable_reason",
                "" if np.isfinite(point) else "insufficient_joint_observations",
            )
        return out

    # F_pred — per-prompt original classification or completion score.
    if not odds_df.empty:
        models_pred: list[str] = _in_cohort(
            _eligible_logodds_models(odds_df, "orig", contrast_col), cohort
        )
        sig_by_model: dict[str, pd.Series] = {
            m: s
            for m in models_pred
            if (
                s := _per_prompt_logodds(
                    odds_df,
                    m,
                    "orig",
                    contrast_col,
                    finite_extreme_api,
                )
            )
            is not None
        }
        sig_by_model = _mask_unsupported_signals(sig_by_model, unsupported_prediction)
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_pred",
                sig_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_pred"),
            )
        )
    else:
        models_pred = _in_cohort(
            sorted(unfiltered_seg_df["model"].dropna().unique().tolist()), cohort
        )
        sig_by_model = {
            m: s
            for m in models_pred
            if (s := _per_prompt_completion(unfiltered_seg_df, m)) is not None
        }
        sig_by_model = _mask_unsupported_signals(sig_by_model, unsupported_prediction)
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_pred",
                sig_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_pred"),
            )
        )

    prediction_by_model: dict[str, pd.Series] = dict(sig_by_model)

    # F_attr — per-(prompt, seg) ablation response.
    abl_by_model: dict[str, pd.Series] = {}
    if not odds_df.empty:
        ablation_models: list[str] = _in_cohort(
            sorted(odds_df["model"].dropna().unique().tolist()), cohort
        )
        for m in ablation_models:
            s = _per_seg_ablation(
                odds_df,
                m,
                contrast_col,
                finite_extreme_api,
            )
            if s is not None:
                abl_by_model[m] = s
        abl_by_model = _mask_unsupported_signals(abl_by_model, unsupported_attribution)
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_attr",
                abl_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_attr"),
            )
        )
    else:
        completion_models: list[str] = _in_cohort(
            sorted(seg_df["model"].dropna().unique().tolist()),
            cohort,
        )
        abl_by_model = {
            m: s
            for m in completion_models
            if (s := _per_seg_completion_ablation(seg_df, m)) is not None
        }
        abl_by_model = _mask_unsupported_signals(abl_by_model, unsupported_attribution)
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_attr",
                abl_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_attr"),
            )
        )

    if requested_metrics is not None and requested_metrics <= {"F_pred", "F_attr"}:
        return finish_rows()

    # F_attn_{rollout,mean,max} — per-(prompt, seg) attention scores.
    for variant in ("rollout", "mean", "max"):
        col: str = f"attention_{variant}"
        models_attn: list[str] = _in_cohort(_eligible_seg_models(seg_df, [col]), cohort)
        sig_by_model = {
            m: s
            for m in models_attn
            if (s := _per_seg_segment_col(seg_df, m, col)) is not None
        }
        out.extend(
            _emit_pair_corrs(
                benchmark,
                f"F_attn_{variant}",
                sig_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context(f"F_attn_{variant}"),
            )
        )

    # F_mag — per-(prompt, seg) post-norm delta_norm.
    models_mag: list[str] = _in_cohort(
        _eligible_seg_models(seg_df, ["delta_norm_postnorm"]), cohort
    )
    sig_by_model = {
        m: s
        for m in models_mag
        if (s := _per_seg_segment_col(seg_df, m, "delta_norm_postnorm")) is not None
    }
    out.extend(
        _emit_pair_corrs(
            benchmark,
            "F_mag",
            sig_by_model,
            n_resamples,
            conf,
            rng,
            bootstrap_seed=bootstrap_seed,
            rng_context=rng_context("F_mag"),
        )
    )

    # F_align uses the linear-readout direction materialized during the ordinary
    # run when it matches the requested contrast. ANLI E-C instead uses the
    # requested sum-unembedding direction at the sidecar-bound final block.
    readout_matches_contrast: bool = _readout_matches_contrast(benchmark, contrast)
    uses_layer_readout: bool = _uses_layer_readout(benchmark, contrast)
    needs_alignment: bool = requested_metrics is None or bool(
        requested_metrics & {"F_align", "F_align_to_attr"}
    )
    models_align: list[str] = []
    align_by_model: dict[str, pd.Series] = {}
    if uses_layer_readout and needs_alignment:
        from benchmark_scripts.layerwise_fidelity import load_final_layer_readout

        models_align = [
            model for model in OPEN_MODELS if cohort is None or model in cohort
        ]
        for model in models_align:
            final_readout = load_final_layer_readout(
                results_dir,
                benchmark,
                pregrouper,
                model,
                scope,
                contrast,
            )
            align_by_model[model] = final_readout.alignment
            _log_final_layer_crosscheck(
                benchmark,
                model,
                "F_pred",
                final_readout.prediction,
                prediction_by_model.get(model),
            )
            _log_final_layer_crosscheck(
                benchmark,
                model,
                "F_attr",
                final_readout.attribution,
                abl_by_model.get(model),
            )
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_align",
                align_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_align"),
            )
        )
    elif readout_matches_contrast:
        models_align = _in_cohort(
            _eligible_seg_models(
                seg_df,
                ["w_dot_delta_z_postnorm", "w_norm", "delta_norm_postnorm"],
            ),
            cohort,
        )
        align_by_model = {
            m: s for m in models_align if (s := _per_seg_align(seg_df, m)) is not None
        }
        out.extend(
            _emit_pair_corrs(
                benchmark,
                "F_align",
                align_by_model,
                n_resamples,
                conf,
                rng,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_align"),
            )
        )
    elif needs_alignment:
        models_align = _in_cohort(
            _eligible_seg_models(
                seg_df,
                ["w_dot_delta_z_postnorm", "w_norm", "delta_norm_postnorm"],
            ),
            cohort,
        )
        out.extend(
            _emit_unavailable_pairs(
                benchmark,
                "F_align",
                models_align,
                models_align,
                directed=False,
                reason="readout_contrast_mismatch",
            )
        )

    # Transfer metrics use pooled segment rows by default. Prompt-equal
    # mean-r² remains available as an explicitly selected sensitivity.
    if abl_by_model:
        # F_align_to_attr (signed)
        if align_by_model:
            out.extend(
                _emit_transfer(
                    benchmark,
                    "F_align_to_attr",
                    align_by_model,
                    abl_by_model,
                    signed=True,
                    n_resamples=n_resamples,
                    conf=conf,
                    rng=rng,
                    aggregation=transfer_aggregation,
                    bootstrap_seed=bootstrap_seed,
                    rng_context=rng_context("F_align_to_attr"),
                )
            )
        elif needs_alignment and not (readout_matches_contrast or uses_layer_readout):
            out.extend(
                _emit_unavailable_pairs(
                    benchmark,
                    "F_align_to_attr",
                    models_align,
                    list(abl_by_model),
                    directed=True,
                    reason="readout_contrast_mismatch",
                    aggregation=transfer_aggregation,
                )
            )
        # F_mag_to_attr (unsigned target)
        mag_by_model: dict[str, pd.Series] = {
            m: s
            for m in models_mag
            if (s := _per_seg_segment_col(seg_df, m, "delta_norm_postnorm")) is not None
        }
        out.extend(
            _emit_transfer(
                benchmark,
                "F_mag_to_attr",
                mag_by_model,
                abl_by_model,
                signed=False,
                n_resamples=n_resamples,
                conf=conf,
                rng=rng,
                aggregation=transfer_aggregation,
                bootstrap_seed=bootstrap_seed,
                rng_context=rng_context("F_mag_to_attr"),
            )
        )
        for variant in ("rollout", "mean", "max"):
            col = f"attention_{variant}"
            attn_by_model: dict[str, pd.Series] = {
                m: s
                for m in _in_cohort(_eligible_seg_models(seg_df, [col]), cohort)
                if (s := _per_seg_segment_col(seg_df, m, col)) is not None
            }
            out.extend(
                _emit_transfer(
                    benchmark,
                    f"F_attn_{variant}_to_attr",
                    attn_by_model,
                    abl_by_model,
                    signed=True,
                    n_resamples=n_resamples,
                    conf=conf,
                    rng=rng,
                    aggregation=transfer_aggregation,
                    bootstrap_seed=bootstrap_seed,
                    rng_context=rng_context(f"F_attn_{variant}_to_attr"),
                )
            )

    return finish_rows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate long-format F-table TSV with bootstrap CIs."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory holding consolidated TSVs and where the F-table is written.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output TSV path (default: {results-dir}/f_table.tsv).",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Benchmarks to include with --pregrouper. By default, process the "
            "paper configurations, including BoolQ sentence+word and LAMBADA word."
        ),
    )
    parser.add_argument(
        "--pregrouper",
        default="sentence",
        choices=["word", "sentence"],
    )
    parser.add_argument(
        "--scopes",
        nargs="+",
        default=["all"],
        choices=["all", "system", "user"],
        help="Message-role strata to compute from the same raw artifact.",
    )
    parser.add_argument(
        "--contrasts",
        nargs="+",
        default=["canonical"],
        help=(
            "Canonical or label-pair suffixes such as "
            "entailment_contradiction and entailment_neutral."
        ),
    )
    parser.add_argument(
        "--anli-contrast",
        choices=["entailment_neutral", "entailment_contradiction"],
        default="entailment_contradiction",
        help=(
            "Override the canonical contrast for ANLI configurations while "
            "leaving each non-ANLI benchmark on its canonical contrast."
        ),
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=1000,
        help="Number of bootstrap resamples per (pair, metric, statistic).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for f_lo/f_hi (e.g., 0.95 -> 2.5%%/97.5%% percentiles).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for reproducible bootstraps."
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Deprecated alias for --cohort all.",
    )
    parser.add_argument(
        "--cohort",
        choices=["paper", "open", "all"],
        default="paper",
        help=(
            "Model cohort. The paper cohort has five open and six hosted "
            "models and excludes the duplicate API-served Llama-8B."
        ),
    )
    parser.add_argument(
        "--api-infinity-policy",
        choices=["pairwise_complete", "finite_extreme"],
        default="pairwise_complete",
        help=(
            "How to handle hosted top-k infinities. 'pairwise_complete' uses "
            "jointly finite rows; 'finite_extreme' is a separately labeled "
            "sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--transfer-aggregation",
        choices=["row_pooled", "prompt_equal_mean_r2"],
        default="row_pooled",
        help=(
            "Aggregation for cross-level transfer. row_pooled is canonical; "
            "prompt_equal_mean_r2 is an optional sensitivity analysis."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optionally retain only these metric names in the output.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out_path: str = args.output or os.path.join(args.results_dir, "f_table.tsv")

    all_rows: list[dict[str, str | float]] = []
    cohort_name: str = "all" if args.all_models else args.cohort
    cohorts: dict[str, tuple[str, ...] | None] = {
        "paper": PAPER_MODELS,
        "open": OPEN_MODELS,
        "all": None,
    }
    cohort: tuple[str, ...] | None = cohorts[cohort_name]
    benchmark_configs: list[tuple[str, str]] = (
        [(benchmark, args.pregrouper) for benchmark in args.benchmarks]
        if args.benchmarks is not None
        else DEFAULT_BENCHMARK_CONFIGS
    )
    jobs: list[tuple[str, str, str, str]] = _analysis_jobs(
        benchmark_configs,
        list(args.scopes),
        list(args.contrasts),
        args.anli_contrast,
    )
    for bench, pregrouper, scope, contrast in tqdm(
        jobs, desc="benchmark analyses", unit="analysis"
    ):
        all_rows.extend(
            _process_benchmark(
                bench,
                pregrouper,
                args.results_dir,
                args.bootstrap_resamples,
                args.confidence_level,
                _analysis_rng(args.seed, bench, pregrouper),
                cohort,
                scope,
                contrast,
                args.api_infinity_policy,
                args.transfer_aggregation,
                args.seed,
                set(args.metrics) if args.metrics is not None else None,
            )
        )

    out_df = pd.DataFrame(all_rows)
    if args.metrics is not None and not out_df.empty:
        requested_metrics: set[str] = set(args.metrics)
        unavailable_metrics: set[str] = requested_metrics - set(
            out_df["metric"].astype(str)
        )
        if unavailable_metrics:
            raise ValueError(
                f"Requested metrics were not produced: {sorted(unavailable_metrics)}"
            )
        out_df = out_df[out_df["metric"].isin(requested_metrics)].copy()
    out_df["cohort"] = cohort_name
    if not out_df.empty:
        source_open: pd.Series = out_df["model_s"].isin(OPEN_MODELS)
        target_open: pd.Series = out_df["model_t"].isin(OPEN_MODELS)
        out_df["pair_population"] = np.select(
            [source_open & target_open, source_open | target_open],
            ["open_open", "open_hosted"],
            default="hosted_hosted",
        )
    else:
        out_df["pair_population"] = pd.Series(dtype=str)
    output_cols: list[str] = [
        "cohort",
        "pair_population",
        "benchmark",
        "pregrouper",
        "scope",
        "requested_scope",
        "resolved_scope",
        "contrast",
        "requested_contrast",
        "resolved_source_contrast",
        "resolved_target_contrast",
        "readout_contrast",
        "availability_status",
        "unavailable_reason",
        "api_infinity_policy",
        "aggregation",
        "model_s",
        "model_t",
        "metric",
        "statistic",
        "n_observations",
        "expected_observations",
        "observation_coverage",
        "n_prompts",
        "expected_prompts",
        "prompt_coverage",
        "f_point",
        "f_lo",
        "f_hi",
    ]
    if not out_df.empty:
        out_df = out_df[output_cols]
    out_df.to_csv(out_path, sep="\t", index=False)
    output_models: list[str] = sorted(
        set(out_df.get("model_s", pd.Series(dtype=str)).dropna().astype(str))
        | set(out_df.get("model_t", pd.Series(dtype=str)).dropna().astype(str))
    )
    alignment_requested: bool = args.metrics is None or bool(
        set(args.metrics) & {"F_align", "F_align_to_attr"}
    )
    layer_alignment_configs: set[tuple[str, str]] = {
        (benchmark, pregrouper)
        for benchmark, pregrouper, _scope, contrast in jobs
        if alignment_requested and _uses_layer_readout(benchmark, contrast)
    }
    write_derived_provenance(
        out_path,
        generator_name="benchmark_scripts.f_table",
        generator_path=__file__,
        input_paths=_derived_input_paths(
            args.results_dir,
            benchmark_configs,
            cohort,
            layer_alignment_configs,
        ),
        parameters={
            "benchmark_configs": [
                {"benchmark": benchmark, "pregrouper": pregrouper}
                for benchmark, pregrouper in benchmark_configs
            ],
            "scopes": list(args.scopes),
            "contrasts": list(args.contrasts),
            "anli_contrast": args.anli_contrast,
            "resolved_jobs": [
                {
                    "benchmark": benchmark,
                    "pregrouper": pregrouper,
                    "scope": scope,
                    "contrast": contrast,
                }
                for benchmark, pregrouper, scope, contrast in jobs
            ],
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "bootstrap_rng": "sha256_cell_key_v1",
            "cohort": cohort_name,
            "requested_models": list(cohort) if cohort is not None else None,
            "output_models": output_models,
            "api_infinity_policy": args.api_infinity_policy,
            "transfer_aggregation": args.transfer_aggregation,
            "missingness_policy": (
                "pair_specific_complete_case"
                if args.api_infinity_policy == "pairwise_complete"
                else "model_specific_finite_extreme_replacement"
            ),
            "metrics": sorted(args.metrics) if args.metrics is not None else None,
        },
        root_dir=args.results_dir,
        supporting_source_paths=_supporting_source_paths(),
    )
    logger.info(f"Wrote {len(out_df)} rows to {out_path}")


if __name__ == "__main__":
    main()
