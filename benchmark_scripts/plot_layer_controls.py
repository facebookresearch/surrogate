# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Plot the compact BoolQ layer-control summary for an ICLR wrapfigure.

The observed-target ribbons are prompt-cluster bootstrap confidence intervals.
The control ribbon instead shows empirical variation over sampled compatible
readouts. The plot deliberately contains only the two headline fidelities and
their closest readout-compatible null.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_TARGET_POINT_SUFFIX: str = "mean_pair_pearson_r2"
_TARGET_LOWER_SUFFIX: str = "bootstrap_lower"
_TARGET_UPPER_SUFFIX: str = "bootstrap_upper"
_CONTROL_MEDIAN_SUFFIX: str = "median"
_CONTROL_LOWER_SUFFIX: str = "q025"
_CONTROL_UPPER_SUFFIX: str = "q975"

_PREDICTION: str = "grouped_logsumexp_prediction"
_ATTRIBUTION: str = "grouped_logsumexp_attribution"
_SINGLE_TOKEN_ATTRIBUTION: str = "single_token_true_false_attribution_diagnostic"
_GROUPED_CONTROL: str = "grouped_9v8_pseudo_label"
_ISOTROPIC_CONTROL: str = "independent_isotropic"
_OBSERVATION_CONTROL: str = "observation_pair_permutation"
_GAP: str = "prediction_minus_attribution_gap"

_COLOR_PRED: str = "#584486"
_COLOR_ATTR: str = "#0091A1"
_COLOR_CONTROL: str = "#9A958F"
_TARGET_LINE_WIDTH: float = 1.25
_CONTROL_LINE_WIDTH: float = 1.0
_ANNOTATION_LINE_WIDTH: float = 0.6


def _column(prefix: str, suffix: str) -> str:
    """Join an analyzer curve name and statistic suffix."""
    return f"{prefix}_{suffix}"


_SERIES_COLUMNS: tuple[str, ...] = (
    _column(_PREDICTION, _TARGET_POINT_SUFFIX),
    _column(_PREDICTION, _TARGET_LOWER_SUFFIX),
    _column(_PREDICTION, _TARGET_UPPER_SUFFIX),
    _column(_ATTRIBUTION, _TARGET_POINT_SUFFIX),
    _column(_ATTRIBUTION, _TARGET_LOWER_SUFFIX),
    _column(_ATTRIBUTION, _TARGET_UPPER_SUFFIX),
    _column(_SINGLE_TOKEN_ATTRIBUTION, _TARGET_POINT_SUFFIX),
    _column(_GAP, _TARGET_POINT_SUFFIX),
    _column(_GAP, _TARGET_LOWER_SUFFIX),
    _column(_GAP, _TARGET_UPPER_SUFFIX),
    _column(_GROUPED_CONTROL, _CONTROL_MEDIAN_SUFFIX),
    _column(_GROUPED_CONTROL, _CONTROL_LOWER_SUFFIX),
    _column(_GROUPED_CONTROL, _CONTROL_UPPER_SUFFIX),
    _column(_ISOTROPIC_CONTROL, _CONTROL_MEDIAN_SUFFIX),
    _column(_OBSERVATION_CONTROL, _CONTROL_MEDIAN_SUFFIX),
)


@dataclass(frozen=True)
class LayerControlPlotData:
    """Validated values and metadata used by the two-panel plot."""

    relative_depth: np.ndarray
    series: dict[str, np.ndarray]
    benchmark: str
    pregrouper: str
    scope: str
    embedding_slot_included: bool

    @property
    def depth_count(self) -> int:
        """Return the number of aligned depth positions."""
        return len(self.relative_depth)


def _single_text(frame: pd.DataFrame, column: str) -> str:
    """Return the single textual value shared by all rows."""
    if column not in frame.columns:
        raise ValueError(f"summary TSV is missing required column {column!r}")
    values: list[str] = sorted(set(frame[column].astype(str)))
    if len(values) != 1 or not values[0]:
        raise ValueError(f"summary TSV must contain one nonempty {column!r} value")
    return values[0]


def _single_bool(frame: pd.DataFrame, column: str) -> bool:
    """Parse a single strict Boolean metadata value."""
    text: str = _single_text(frame, column).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"summary TSV {column!r} must be true or false")


def _numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return one required finite numeric column as float64."""
    if column not in frame.columns:
        raise ValueError(f"summary TSV is missing required column {column!r}")
    values: np.ndarray = pd.to_numeric(frame[column], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError(f"summary TSV column {column!r} must be finite")
    return values


def _validate_interval(
    series: dict[str, np.ndarray], lower: str, median: str, upper: str
) -> None:
    """Validate pointwise ordering of an empirical control interval."""
    if np.any(series[lower] > series[median]) or np.any(series[median] > series[upper]):
        raise ValueError(f"summary TSV has an invalid interval around {median!r}")


def load_plot_data(summary_path: str) -> LayerControlPlotData:
    """Load and strictly validate an analyzer summary TSV.

    Args:
        summary_path: Path to the compact TSV written by
            ``analyze_layer_controls.py``.

    Returns:
        The ordered relative-depth values, plotted series, and caption metadata.

    Raises:
        ValueError: If the input is incomplete, inconsistent, or not a BoolQ
            sentence-level relative-depth summary.
    """
    frame: pd.DataFrame = pd.read_csv(summary_path, sep="\t")
    required_metadata: tuple[str, ...] = (
        "summary_kind",
        "benchmark",
        "pregrouper",
        "scope",
        "relative_depth_index",
        "relative_depth",
        "depth_grid_size",
        "depth_alignment",
        "embedding_slot_included",
    )
    missing_metadata: list[str] = [
        column for column in required_metadata if column not in frame.columns
    ]
    if missing_metadata:
        raise ValueError(
            "summary TSV is missing required columns: " + ", ".join(missing_metadata)
        )
    depth_rows: pd.DataFrame = frame.loc[
        frame["summary_kind"].astype(str).eq("relative_depth")
    ].copy()
    if len(depth_rows) < 2:
        raise ValueError("summary TSV must contain at least two relative-depth rows")

    depth_indices_raw: np.ndarray = _numeric_series(depth_rows, "relative_depth_index")
    if not np.equal(depth_indices_raw, np.floor(depth_indices_raw)).all():
        raise ValueError("relative_depth_index values must be integers")
    depth_rows["relative_depth_index"] = depth_indices_raw.astype(np.int64)
    depth_rows = depth_rows.sort_values("relative_depth_index", kind="stable")
    expected_indices: np.ndarray = np.arange(len(depth_rows), dtype=np.int64)
    actual_indices: np.ndarray = depth_rows["relative_depth_index"].to_numpy(
        dtype=np.int64
    )
    if not np.array_equal(actual_indices, expected_indices):
        raise ValueError("relative_depth_index must be complete and zero-based")

    relative_depth: np.ndarray = _numeric_series(depth_rows, "relative_depth")
    if (
        not np.isclose(relative_depth[0], 0.0)
        or not np.isclose(relative_depth[-1], 1.0)
        or np.any(np.diff(relative_depth) <= 0.0)
    ):
        raise ValueError("relative_depth must increase strictly from zero to one")

    depth_grid_values: np.ndarray = _numeric_series(depth_rows, "depth_grid_size")
    if not np.equal(depth_grid_values, float(len(depth_rows))).all():
        raise ValueError("depth_grid_size must equal the relative-depth row count")

    benchmark: str = _single_text(depth_rows, "benchmark")
    pregrouper: str = _single_text(depth_rows, "pregrouper")
    scope: str = _single_text(depth_rows, "scope")
    if benchmark != "boolq" or pregrouper != "sentence":
        raise ValueError("plotting requires the BoolQ sentence-level summary")
    if scope not in {"all", "system", "user"}:
        raise ValueError("summary TSV has an unsupported segment scope")

    embedding_slot_included: bool = _single_bool(depth_rows, "embedding_slot_included")
    alignment: str = _single_text(depth_rows, "depth_alignment")
    expected_alignment: str = (
        "linear_interpolation_all_slots"
        if embedding_slot_included
        else "linear_interpolation_block_outputs_only"
    )
    if alignment != expected_alignment:
        raise ValueError("depth alignment and embedding policy disagree")

    series: dict[str, np.ndarray] = {
        column: _numeric_series(depth_rows, column) for column in _SERIES_COLUMNS
    }
    for column, values in series.items():
        lower_bound: float = -1.0 if column.startswith(_GAP) else 0.0
        if np.any(values < lower_bound) or np.any(values > 1.0):
            valid_range: str = "[-1, 1]" if column.startswith(_GAP) else "[0, 1]"
            raise ValueError(f"summary TSV column {column!r} is outside {valid_range}")

    expected_gap: np.ndarray = (
        series[_column(_PREDICTION, _TARGET_POINT_SUFFIX)]
        - series[_column(_ATTRIBUTION, _TARGET_POINT_SUFFIX)]
    )
    if not np.allclose(
        series[_column(_GAP, _TARGET_POINT_SUFFIX)],
        expected_gap,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("summary TSV gap does not equal F_pred minus F_attr")

    _validate_interval(
        series,
        _column(_GROUPED_CONTROL, _CONTROL_LOWER_SUFFIX),
        _column(_GROUPED_CONTROL, _CONTROL_MEDIAN_SUFFIX),
        _column(_GROUPED_CONTROL, _CONTROL_UPPER_SUFFIX),
    )
    for target in (_PREDICTION, _ATTRIBUTION, _GAP):
        lower: str = _column(target, _TARGET_LOWER_SUFFIX)
        upper: str = _column(target, _TARGET_UPPER_SUFFIX)
        if np.any(series[lower] > series[upper]):
            raise ValueError(
                f"summary TSV has an invalid bootstrap interval for {target}"
            )

    return LayerControlPlotData(
        relative_depth=relative_depth,
        series=series,
        benchmark=benchmark,
        pregrouper=pregrouper,
        scope=scope,
        embedding_slot_included=embedding_slot_included,
    )


def _load_pyplot() -> Any:
    """Load matplotlib lazily so non-plot analysis has no plotting dependency."""
    try:
        import matplotlib  # pyre-ignore[21]

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot  # pyre-ignore[21]
    except ImportError as error:
        raise RuntimeError(
            "plot_layer_controls requires matplotlib; install it in the plotting "
            "environment before running this command"
        ) from error
    return pyplot


def _plot_target(
    axis: Any,
    data: LayerControlPlotData,
    target: str,
    *,
    color: str,
    line_label: str,
    ribbon_label: str,
    linestyle: str = "-",
) -> None:
    """Plot an observed target and its prompt-bootstrap interval."""
    x: np.ndarray = data.relative_depth
    axis.fill_between(
        x,
        data.series[_column(target, _TARGET_LOWER_SUFFIX)],
        data.series[_column(target, _TARGET_UPPER_SUFFIX)],
        color=color,
        alpha=0.16,
        linewidth=0.0,
        label=ribbon_label,
        zorder=1,
    )
    axis.plot(
        x,
        data.series[_column(target, _TARGET_POINT_SUFFIX)],
        color=color,
        linestyle=linestyle,
        linewidth=_TARGET_LINE_WIDTH,
        label=line_label,
        zorder=3,
    )


def _plot_control_band(
    axis: Any,
    data: LayerControlPlotData,
    family: str,
    *,
    color: str,
    line_label: str,
    ribbon_label: str,
) -> None:
    """Plot a control median and empirical direction interval."""
    x: np.ndarray = data.relative_depth
    axis.fill_between(
        x,
        data.series[_column(family, _CONTROL_LOWER_SUFFIX)],
        data.series[_column(family, _CONTROL_UPPER_SUFFIX)],
        color=color,
        alpha=0.13,
        linewidth=0.0,
        label=ribbon_label,
        zorder=1,
    )
    axis.plot(
        x,
        data.series[_column(family, _CONTROL_MEDIAN_SUFFIX)],
        color=color,
        linestyle="--",
        linewidth=_CONTROL_LINE_WIDTH,
        label=line_label,
        zorder=2,
    )


def _save_figure(figure: Any, output_path: str, output_format: str) -> None:
    """Atomically save a figure with stable, format-specific metadata."""
    directory: str = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=directory, prefix=".layer-controls-", suffix=f".{output_format}"
    )
    os.close(descriptor)
    metadata: dict[str, Any]
    if output_format == "pdf":
        metadata = {
            "Title": "BoolQ layer-control fidelity",
            "Subject": "Surrogate-fidelity layer controls",
            "Creator": "benchmark_scripts.plot_layer_controls",
            "Producer": "matplotlib",
            "CreationDate": None,
            "ModDate": None,
        }
    else:
        metadata = {
            "Title": "BoolQ layer-control fidelity",
            "Description": "Surrogate-fidelity layer controls",
            "Software": "benchmark_scripts.plot_layer_controls",
        }
    try:
        figure.savefig(
            temporary_path,
            format=output_format,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.08,
            metadata=metadata,
        )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def plot_summary(summary_path: str, output_path: str) -> None:
    """Render a deterministic half-width layer-control figure.

    Args:
        summary_path: Analyzer-produced compact summary TSV.
        output_path: Destination ending in ``.pdf`` or ``.png``.

    Raises:
        ValueError: If the output extension or summary data is invalid.
        RuntimeError: If matplotlib is unavailable.
    """
    output_format: str = os.path.splitext(output_path)[1].lower().lstrip(".")
    if output_format not in {"pdf", "png"}:
        raise ValueError("output path must end in .pdf or .png")
    data: LayerControlPlotData = load_plot_data(summary_path)
    pyplot: Any = _load_pyplot()

    style: dict[str, Any] = {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    with pyplot.rc_context():
        # Do not inherit workstation- or notebook-specific Matplotlib styles.
        pyplot.rcdefaults()
        pyplot.rcParams.update(style)
        # ICLR uses a 5.5-inch single-column text block. The paper currently
        # places this plot in a 0.5\textwidth wrapfigure.
        figure, axis = pyplot.subplots(figsize=(2.75, 1.35))

        _plot_target(
            axis,
            data,
            _PREDICTION,
            color=_COLOR_PRED,
            line_label=r"$F_{\mathrm{pred}}$",
            ribbon_label=r"$F_{\mathrm{pred}}$: 95% prompt-bootstrap CI",
        )
        _plot_target(
            axis,
            data,
            _ATTRIBUTION,
            color=_COLOR_ATTR,
            line_label=r"$F_{\mathrm{attr}}$",
            ribbon_label=r"$F_{\mathrm{attr}}$: 95% prompt-bootstrap CI",
        )
        _plot_control_band(
            axis,
            data,
            _GROUPED_CONTROL,
            color=_COLOR_CONTROL,
            line_label="Control",
            ribbon_label="Control: empirical 2.5–97.5% readout interval",
        )

        endpoints: tuple[tuple[str, str, float], ...] = (
            (
                r"$F_{\mathrm{pred}}$",
                _COLOR_PRED,
                float(data.series[_column(_PREDICTION, _TARGET_POINT_SUFFIX)][-1]),
            ),
            (
                r"$F_{\mathrm{attr}}$",
                _COLOR_ATTR,
                float(data.series[_column(_ATTRIBUTION, _TARGET_POINT_SUFFIX)][-1]),
            ),
            (
                "Control",
                _COLOR_CONTROL,
                float(
                    data.series[
                        _column(_GROUPED_CONTROL, _CONTROL_MEDIAN_SUFFIX)
                    ][-1]
                ),
            ),
        )
        for label, color, endpoint in endpoints:
            axis.annotate(
                label,
                xy=(1.0, endpoint),
                xytext=(1.025, endpoint),
                color=color,
                fontsize=7.2,
                va="center",
                ha="left",
                annotation_clip=False,
                arrowprops={
                    "arrowstyle": "-",
                    "color": color,
                    "linewidth": _ANNOTATION_LINE_WIDTH,
                },
            )

        prediction_upper: float = float(
            max(
                data.series[_column(_PREDICTION, _TARGET_POINT_SUFFIX)].max(),
                data.series[_column(_PREDICTION, _TARGET_UPPER_SUFFIX)].max(),
            )
        )
        y_max: float = min(1.0, float(np.ceil((prediction_upper + 0.03) * 10.0) / 10.0))
        axis.set_xlim(0.0, 1.18)
        axis.set_ylim(0.0, y_max)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xlabel("Relative decoder depth")
        axis.set_ylabel(r"Mean pairwise $r^2$")
        figure.subplots_adjust(left=0.20, right=0.96, top=0.98, bottom=0.28)
        try:
            _save_figure(figure, output_path, output_format)
        finally:
            pyplot.close(figure)


def main() -> None:
    """Parse CLI arguments and render the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_tsv", help="Analyzer-produced compact summary TSV")
    parser.add_argument("output", help="Output .pdf or .png path")
    args: argparse.Namespace = parser.parse_args()
    plot_summary(args.summary_tsv, args.output)


if __name__ == "__main__":
    main()
