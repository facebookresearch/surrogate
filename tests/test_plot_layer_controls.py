# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the public BoolQ layer-control plotting module."""

from __future__ import annotations

import contextlib
import os
import tempfile
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmark_scripts import plot_layer_controls as plotter


def _summary_frame(depth_count: int = 5) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for depth_index, depth in enumerate(np.linspace(0.0, 1.0, depth_count)):
        row: dict[str, Any] = {
            "summary_kind": "relative_depth",
            "benchmark": "boolq",
            "pregrouper": "sentence",
            "scope": "user",
            "relative_depth_index": depth_index,
            "relative_depth": depth,
            "depth_grid_size": depth_count,
            "depth_alignment": "linear_interpolation_block_outputs_only",
            "embedding_slot_included": False,
        }
        for column in plotter._SERIES_COLUMNS:
            if column.endswith("bootstrap_lower") or column.endswith("q025"):
                value: float = 0.20 + depth * 0.10
            elif column.endswith("bootstrap_upper") or column.endswith("q975"):
                value = 0.40 + depth * 0.10
            else:
                value = 0.30 + depth * 0.10
            row[column] = value
        row["grouped_logsumexp_prediction_mean_pair_pearson_r2"] = 0.60 + depth * 0.10
        row["grouped_logsumexp_attribution_mean_pair_pearson_r2"] = 0.30 + depth * 0.10
        row["prediction_minus_attribution_gap_mean_pair_pearson_r2"] = 0.30
        rows.append(row)
    depth_mean: dict[str, Any] = dict(rows[-1])
    depth_mean["summary_kind"] = "equal_weight_depth_mean"
    depth_mean["relative_depth_index"] = ""
    depth_mean["relative_depth"] = ""
    rows.append(depth_mean)
    return pd.DataFrame(rows)


class _FakeAxis:
    def __init__(self) -> None:
        self.plots: list[dict[str, Any]] = []
        self.bands: list[dict[str, Any]] = []
        self.annotations: list[tuple[str, dict[str, Any]]] = []
        self.ylim: tuple[float, float] | None = None

    def plot(self, *_args: Any, **kwargs: Any) -> list[object]:
        self.plots.append(kwargs)
        return [object()]

    def fill_between(self, *_args: Any, **kwargs: Any) -> object:
        self.bands.append(kwargs)
        return object()

    def annotate(self, text: str, **kwargs: Any) -> object:
        self.annotations.append((text, kwargs))
        return object()

    def set_xlim(self, *_args: Any) -> None:
        pass

    def set_ylim(self, lower: float, upper: float) -> None:
        self.ylim = (lower, upper)

    def set_xlabel(self, *_args: Any) -> None:
        pass

    def set_ylabel(self, *_args: Any) -> None:
        pass

    def set_xticks(self, *_args: Any) -> None:
        pass


class _FakeFigure:
    def __init__(self) -> None:
        self.saved_metadata: dict[str, Any] = {}

    def subplots_adjust(self, **_kwargs: Any) -> None:
        pass

    def savefig(self, path: str, **kwargs: Any) -> None:
        self.saved_metadata = kwargs["metadata"]
        with open(path, "wb") as output:
            output.write(b"synthetic figure")


class _FakePyplot:
    def __init__(self) -> None:
        self.figure: _FakeFigure = _FakeFigure()
        self.axis: _FakeAxis = _FakeAxis()
        self.rcParams: dict[str, Any] = {}
        self.defaults_reset: bool = False
        self.subplots_kwargs: dict[str, Any] = {}
        self.closed: bool = False

    def rc_context(
        self, _style: dict[str, Any] | None = None
    ) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def rcdefaults(self) -> None:
        self.defaults_reset = True

    def subplots(
        self, *_args: Any, **kwargs: Any
    ) -> tuple[_FakeFigure, _FakeAxis]:
        self.subplots_kwargs = kwargs
        return self.figure, self.axis

    def close(self, _figure: _FakeFigure) -> None:
        self.closed = True


class PlotLayerControlsTest(TestCase):
    def _write_summary(self, directory: str, frame: pd.DataFrame) -> str:
        path: str = os.path.join(directory, "summary.tsv")
        frame.to_csv(path, sep="\t", index=False)
        return path

    def test_loads_relative_depth_rows_and_infers_grid_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = self._write_summary(directory, _summary_frame(7))
            data: plotter.LayerControlPlotData = plotter.load_plot_data(path)

        self.assertEqual(data.depth_count, 7)
        self.assertEqual(data.scope, "user")
        self.assertFalse(data.embedding_slot_included)
        np.testing.assert_allclose(data.relative_depth, np.linspace(0.0, 1.0, 7))

    def test_rejects_missing_series_and_invalid_control_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing: pd.DataFrame = _summary_frame()
            missing = missing.drop(columns=[plotter._SERIES_COLUMNS[0]])
            missing_path: str = self._write_summary(directory, missing)
            with self.assertRaisesRegex(ValueError, "missing required column"):
                plotter.load_plot_data(missing_path)

            invalid: pd.DataFrame = _summary_frame()
            invalid.loc[
                invalid["summary_kind"] == "relative_depth",
                "grouped_9v8_pseudo_label_q025",
            ] = 0.8
            invalid_path: str = os.path.join(directory, "invalid.tsv")
            invalid.to_csv(invalid_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "invalid interval"):
                plotter.load_plot_data(invalid_path)

    def test_rejects_inconsistent_depth_or_embedding_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad_depth: pd.DataFrame = _summary_frame()
            bad_depth.loc[1, "relative_depth_index"] = 3
            depth_path: str = self._write_summary(directory, bad_depth)
            with self.assertRaisesRegex(ValueError, "complete and zero-based"):
                plotter.load_plot_data(depth_path)

            bad_alignment: pd.DataFrame = _summary_frame()
            bad_alignment["embedding_slot_included"] = True
            alignment_path: str = os.path.join(directory, "alignment.tsv")
            bad_alignment.to_csv(alignment_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "embedding policy disagree"):
                plotter.load_plot_data(alignment_path)

    def test_rejects_gap_inconsistent_with_target_curves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid: pd.DataFrame = _summary_frame()
            invalid.loc[
                invalid["summary_kind"] == "relative_depth",
                "prediction_minus_attribution_gap_mean_pair_pearson_r2",
            ] += 0.01
            path: str = self._write_summary(directory, invalid)
            with self.assertRaisesRegex(ValueError, "gap does not equal"):
                plotter.load_plot_data(path)

    def test_plot_labels_uncertainty_types_and_writes_atomically(self) -> None:
        fake_pyplot: _FakePyplot = _FakePyplot()
        with tempfile.TemporaryDirectory() as directory:
            summary_path: str = self._write_summary(directory, _summary_frame(6))
            output_path: str = os.path.join(directory, "figure.png")
            with patch.object(plotter, "_load_pyplot", return_value=fake_pyplot):
                plotter.plot_summary(summary_path, output_path)
            with open(output_path, "rb") as output:
                self.assertEqual(output.read(), b"synthetic figure")

        line_labels: set[str] = {call["label"] for call in fake_pyplot.axis.plots}
        band_labels: set[str] = {
            call["label"] for call in fake_pyplot.axis.bands
        }
        self.assertEqual(
            line_labels,
            {
                r"$F_{\mathrm{pred}}$",
                r"$F_{\mathrm{attr}}$",
                "Control",
            },
        )
        line_widths: dict[str, float] = {
            call["label"]: call["linewidth"] for call in fake_pyplot.axis.plots
        }
        self.assertEqual(line_widths[r"$F_{\mathrm{pred}}$"], 1.25)
        self.assertEqual(line_widths[r"$F_{\mathrm{attr}}$"], 1.25)
        self.assertEqual(line_widths["Control"], 1.0)
        self.assertEqual(
            band_labels,
            {
                r"$F_{\mathrm{pred}}$: 95% prompt-bootstrap CI",
                r"$F_{\mathrm{attr}}$: 95% prompt-bootstrap CI",
                "Control: empirical 2.5–97.5% readout interval",
            },
        )
        self.assertEqual(
            {text for text, _kwargs in fake_pyplot.axis.annotations},
            {r"$F_{\mathrm{pred}}$", r"$F_{\mathrm{attr}}$", "Control"},
        )
        self.assertEqual(fake_pyplot.subplots_kwargs["figsize"], (2.75, 1.35))
        self.assertEqual(fake_pyplot.axis.ylim, (0.0, 0.8))
        self.assertTrue(fake_pyplot.defaults_reset)
        self.assertEqual(fake_pyplot.rcParams["font.family"], "serif")
        self.assertEqual(
            fake_pyplot.figure.saved_metadata["Software"],
            "benchmark_scripts.plot_layer_controls",
        )
        self.assertTrue(fake_pyplot.closed)

    def test_rejects_unsupported_output_extension_before_loading_matplotlib(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary_path: str = self._write_summary(directory, _summary_frame())
            with patch.object(plotter, "_load_pyplot") as load_pyplot:
                with self.assertRaisesRegex(ValueError, "must end in .pdf or .png"):
                    plotter.plot_summary(summary_path, os.path.join(directory, "x.svg"))
                load_pyplot.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
