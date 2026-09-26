# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot multiclass fidelity as missing label log-probabilities are floored."""

from __future__ import annotations

import argparse
import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_scripts.anli_rv import OPEN_MODELS, PAPER_MODELS
from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    derived_supporting_source_paths,
    write_derived_provenance,
)
from benchmark_scripts.rv import (
    FINITE_EXTREME_ABSOLUTE_MARGIN,
    FINITE_EXTREME_RELATIVE_MARGIN,
    centered_rv,
    finite_extreme_offset,
    replace_censored_label_logprobs,
)

OFFSETS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
LABELS: dict[str, tuple[str, ...]] = {
    "anli_r1": ("entailment", "neutral", "contradiction"),
    "anli_r2": ("entailment", "neutral", "contradiction"),
    "anli_r3": ("entailment", "neutral", "contradiction"),
    "race": ("a", "b", "c", "d"),
}
COLORS: dict[str, str] = {
    "F_pred": "#584486",
    "F_attr": "#0091A1",
    "F_mag": "#8E5313",
    "F_align": "#E9A86E",
    "F_attn_mean": "#8594A6",
}


def _floor_missing_labels(
    frame: pd.DataFrame, label_columns: list[str], offset: float
) -> pd.DataFrame:
    """Apply a fixed floor to partial censoring while preserving empty rows."""
    output: pd.DataFrame = frame.copy()
    for indices in output.groupby("model", sort=False).groups.values():
        output.loc[indices, label_columns] = replace_censored_label_logprobs(
            output.loc[indices, label_columns].to_numpy(dtype=float),
            offset=offset,
        )
    return output


def _pair_vectors(frame: pd.DataFrame, labels: tuple[str, ...]) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.column_stack(
            [
                frame[f"label_lp_{left}"].to_numpy(dtype=float)
                - frame[f"label_lp_{right}"].to_numpy(dtype=float)
                for left, right in itertools.combinations(labels, 2)
            ]
        )


def _signals(
    frame: pd.DataFrame,
    model: str,
    labels: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: pd.DataFrame = frame[frame["model"] == model]
    original: pd.DataFrame = rows[rows["kind"] == "orig"]
    ablated: pd.DataFrame = rows[rows["kind"] == "ablated"]
    label_columns: list[str] = [f"label_lp_{label}" for label in labels]
    prediction: pd.DataFrame = pd.DataFrame(
        _pair_vectors(original, labels), index=original["prompt_idx"]
    )
    merged: pd.DataFrame = ablated.merge(
        original[["prompt_idx", *label_columns]],
        on="prompt_idx",
        suffixes=("_ablated", "_original"),
        validate="many_to_one",
    )
    original_labels: pd.DataFrame = merged[
        [f"{column}_original" for column in label_columns]
    ].rename(columns={f"{column}_original": column for column in label_columns})
    ablated_labels: pd.DataFrame = merged[
        [f"{column}_ablated" for column in label_columns]
    ].rename(columns={f"{column}_ablated": column for column in label_columns})
    with np.errstate(invalid="ignore"):
        attribution_values: np.ndarray = _pair_vectors(
            original_labels, labels
        ) - _pair_vectors(ablated_labels, labels)
    index: pd.MultiIndex = pd.MultiIndex.from_frame(merged[["prompt_idx", "seg_idx"]])
    attribution: pd.DataFrame = pd.DataFrame(attribution_values, index=index)
    return prediction, attribution


def _pair_rv(first: pd.DataFrame, second: pd.DataFrame) -> float:
    common: pd.Index = first.index.intersection(second.index)
    return centered_rv(
        first.loc[common].to_numpy(dtype=float),
        second.loc[common].to_numpy(dtype=float),
    )


def compute_sensitivity(results_dir: str) -> pd.DataFrame:
    """Return per-model-pair fidelity for each finite floor offset."""
    records: list[dict[str, str | float]] = []
    for benchmark, labels in LABELS.items():
        columns: list[str] = [
            "model",
            "prompt_idx",
            "seg_idx",
            "kind",
            *[f"label_lp_{label}" for label in labels],
        ]
        frame: pd.DataFrame = pd.read_csv(
            os.path.join(results_dir, f"{benchmark}_sentence_logodds.tsv"),
            sep="\t",
            usecols=columns,
        )
        label_columns: list[str] = [f"label_lp_{label}" for label in labels]
        policy_offsets: dict[str, float] = {}
        for model, indices in frame.groupby("model", sort=False).groups.items():
            values: np.ndarray = frame.loc[indices, label_columns].to_numpy(dtype=float)
            policy_offset: float | None = finite_extreme_offset(values)
            if policy_offset is None:
                raise ValueError(f"{benchmark}/{model} has no finite label scores")
            policy_offsets[str(model)] = policy_offset
        for offset in OFFSETS:
            floored: pd.DataFrame = _floor_missing_labels(frame, label_columns, offset)
            by_model: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {
                model: _signals(floored, model, labels) for model in PAPER_MODELS
            }
            for model_s, model_t in itertools.combinations(PAPER_MODELS, 2):
                for metric, position in (("F_pred", 0), ("F_attr", 1)):
                    records.append(
                        {
                            "benchmark": benchmark,
                            "scope": "all",
                            "floor_offset": offset,
                            "model_s": model_s,
                            "model_t": model_t,
                            "current_policy_offset_s": policy_offsets[model_s],
                            "current_policy_offset_t": policy_offsets[model_t],
                            "metric": metric,
                            "f_point": _pair_rv(
                                by_model[model_s][position],
                                by_model[model_t][position],
                            ),
                        }
                    )
    return pd.DataFrame(records)


def reference_values(results_dir: str) -> dict[str, dict[str, float]]:
    """Load canonical multiclass and representation reference medians."""
    f_table: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "f_table.tsv"), sep="\t"
    )
    race_rv: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "race_rv.tsv"), sep="\t"
    )
    anli_rv: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "anli_rv.tsv"), sep="\t"
    )
    references: dict[str, dict[str, float]] = {}
    anli_scalar: pd.DataFrame = f_table[
        f_table["benchmark"].isin(LABELS.keys() - {"race"})
        & f_table["scope"].eq("all")
        & f_table["statistic"].eq("pearson_r2")
    ]
    anli_canonical: pd.DataFrame = anli_rv[anli_rv["scope"].eq("all")]
    references["ANLI"] = {
        "F_pred_complete": float(
            anli_canonical.loc[
                anli_canonical["metric"].eq("F_pred_rv"), "f_point"
            ].median()
        ),
        "F_attr_complete": float(
            anli_canonical.loc[
                anli_canonical["metric"].eq("F_attr_rv"), "f_point"
            ].median()
        ),
        **{
            metric: float(
                anli_scalar.loc[anli_scalar["metric"].eq(metric), "f_point"].median()
            )
            for metric in ("F_attn_mean", "F_mag", "F_align")
        },
    }
    race_scalar: pd.DataFrame = race_rv[
        race_rv["scope"].eq("all") & race_rv["representation"].eq("canonical_scalar")
    ]
    race_canonical: pd.DataFrame = race_rv[
        race_rv["scope"].eq("all") & race_rv["representation"].eq("all_pairs")
    ]
    references["RACE"] = {
        "F_pred_complete": float(
            race_canonical.loc[
                race_canonical["metric"].eq("F_pred_rv"), "f_point"
            ].median()
        ),
        "F_attr_complete": float(
            race_canonical.loc[
                race_canonical["metric"].eq("F_attr_rv"), "f_point"
            ].median()
        ),
        **{
            output_name: float(
                race_scalar.loc[
                    race_scalar["metric"].eq(input_name), "f_point"
                ].median()
            )
            for output_name, input_name in (
                ("F_attn_mean", "F_attn_mean_rv"),
                ("F_mag", "F_mag_rv"),
                ("F_align", "F_align_rv"),
            )
        },
    }
    return references


def plot_sensitivity(
    sensitivity: pd.DataFrame, references: dict[str, dict[str, float]], output: str
) -> None:
    """Write the two-panel ICLR-width sensitivity figure."""
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": False,
            "lines.linewidth": 1.25,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.25), sharey=True)
    panels: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("ANLI", ("anli_r1", "anli_r2", "anli_r3")),
        ("RACE", ("race",)),
    )
    labels: dict[str, str] = {
        "F_pred": r"$F_{\mathrm{pred}}$",
        "F_attr": r"$F_{\mathrm{attr}}$",
        "F_attn_mean": r"$F_{\mathrm{attn}}$",
        "F_mag": r"$F_{\mathrm{mag}}$",
        "F_align": r"$F_{\mathrm{align}}$",
    }
    for axis, (title, benchmarks) in zip(axes, panels):
        panel: pd.DataFrame = sensitivity[sensitivity["benchmark"].isin(benchmarks)]
        x_values: np.ndarray = np.log2(1.0 + np.asarray(OFFSETS))
        policy_rows: pd.DataFrame = pd.concat(
            [
                panel[["benchmark", "model_s", "current_policy_offset_s"]].rename(
                    columns={"model_s": "model", "current_policy_offset_s": "offset"}
                ),
                panel[["benchmark", "model_t", "current_policy_offset_t"]].rename(
                    columns={"model_t": "model", "current_policy_offset_t": "offset"}
                ),
            ],
            ignore_index=True,
        ).drop_duplicates(["benchmark", "model"])
        hosted_policy_offsets: np.ndarray = policy_rows.loc[
            ~policy_rows["model"].isin(OPEN_MODELS), "offset"
        ].to_numpy(dtype=float)
        policy_low, policy_median, policy_high = np.quantile(
            hosted_policy_offsets, [0.0, 0.5, 1.0]
        )
        policy_x_low: float = float(np.log2(1.0 + policy_low))
        policy_x_median: float = float(np.log2(1.0 + policy_median))
        policy_x_high: float = float(np.log2(1.0 + policy_high))
        axis.axvspan(
            policy_x_low,
            policy_x_high,
            color="#9A958F",
            alpha=0.18,
            linewidth=0,
            zorder=0,
        )
        axis.axvline(
            policy_x_median,
            color="#77726D",
            linestyle=(0, (1.5, 1.5)),
            linewidth=0.8,
            zorder=1,
        )
        for metric in ("F_pred", "F_attr"):
            grouped = panel[panel["metric"].eq(metric)].groupby("floor_offset")[
                "f_point"
            ]
            median: pd.Series = grouped.median()
            axis.plot(x_values, median.values, color=COLORS[metric])
            axis.fill_between(
                x_values,
                grouped.quantile(0.25).values,
                grouped.quantile(0.75).values,
                color=COLORS[metric],
                alpha=0.12,
                linewidth=0,
            )
            axis.hlines(
                references[title][f"{metric}_complete"],
                x_values[0],
                x_values[-1],
                color=COLORS[metric],
                linestyle="--",
                linewidth=0.8,
            )
            axis.annotate(
                labels[metric],
                (x_values[-1], float(median.iloc[-1])),
                xytext=(4, 5 if metric == "F_attr" else 0),
                textcoords="offset points",
                color=COLORS[metric],
                va="center",
            )
        for metric, linestyle in (
            ("F_attn_mean", ":"),
            ("F_mag", "-."),
            ("F_align", (0, (3, 1, 1, 1))),
        ):
            value: float = references[title][metric]
            axis.hlines(
                value,
                x_values[0],
                x_values[-1],
                color=COLORS[metric],
                linestyle=linestyle,
                linewidth=1.0,
            )
            axis.annotate(
                labels[metric],
                (x_values[-1], value),
                xytext=(4, -6 if metric == "F_align" else 0),
                textcoords="offset points",
                color=COLORS[metric],
                va="center",
            )
        axis.set_title(title)
        axis.set_xticks(x_values, [str(int(offset)) for offset in OFFSETS])
        axis.set_xlim(-0.2, x_values[-1] + 0.9)
        axis.set_ylim(-0.05, 1.0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"Median fidelity ($r^2$ or RV)")
    fig.supxlabel("Floor offset below observed minimum (nats)", fontsize=8, y=0.01)
    fig.tight_layout(w_pad=1.4, rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(
        output,
        metadata={
            "Creator": "benchmark_scripts.plot_missingness_sensitivity",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(os.path.splitext(output)[0] + ".png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--output-table", default="results/multiclass_floor_sensitivity.tsv"
    )
    parser.add_argument(
        "--output-figure",
        default="figures/0625_cameraready/appendix_multiclass_floor_sensitivity.pdf",
    )
    args: argparse.Namespace = parser.parse_args()
    sensitivity: pd.DataFrame = compute_sensitivity(args.results_dir)
    sensitivity.to_csv(args.output_table, sep="\t", index=False)
    input_paths: dict[str, str] = {}
    for benchmark in LABELS:
        input_paths.update(
            collect_result_inputs(
                args.results_dir,
                benchmark,
                "sentence",
                {
                    "logodds": os.path.join(
                        args.results_dir, f"{benchmark}_sentence_logodds.tsv"
                    )
                },
                PAPER_MODELS,
            )
        )
    for filename in ("f_table.tsv", "anli_rv.tsv", "race_rv.tsv"):
        input_paths[f"canonical/{filename}"] = os.path.join(args.results_dir, filename)
    supporting_sources: dict[str, str] = derived_supporting_source_paths()
    supporting_sources["benchmark_scripts/anli_rv.py"] = os.path.join(
        os.path.dirname(__file__), "anli_rv.py"
    )
    supporting_sources["benchmark_scripts/rv.py"] = os.path.join(
        os.path.dirname(__file__), "rv.py"
    )
    write_derived_provenance(
        args.output_table,
        generator_name="benchmark_scripts.plot_missingness_sensitivity",
        generator_path=__file__,
        input_paths=input_paths,
        parameters={
            "benchmarks": list(LABELS),
            "scope": "all",
            "floor_offsets_nats": list(OFFSETS),
            "floor_reference": "model_benchmark_minimum_finite_label_logprob",
            "missing_value": "partial_negative_infinity_only",
            "all_labels_missing": "nan",
            "finite_extreme_relative_margin": FINITE_EXTREME_RELATIVE_MARGIN,
            "finite_extreme_absolute_margin": FINITE_EXTREME_ABSOLUTE_MARGIN,
            "cohort": "paper",
            "summary": "median_and_interquartile_model_pair_range",
        },
        root_dir=args.results_dir,
        supporting_source_paths=supporting_sources,
    )
    plot_sensitivity(
        sensitivity,
        reference_values(args.results_dir),
        args.output_figure,
    )


if __name__ == "__main__":
    main()
