# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
from benchmark_scripts.noise_ceiling import (
    autocorrelation_rows,
    PUBLISHED_RUN,
    SIGNAL_METRICS,
    spread_table,
    TRANSFER_METRICS,
)

OPEN: str = "qwen2.5-0.5b-instruct"
HOSTED: str = "gpt-4o"


def _segment_index(n_prompts: int, n_segments: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [range(n_prompts), range(n_segments)], names=["prompt_idx", "seg_idx"]
    )


def _signals(
    base: dict[str, np.ndarray], noise: float, rng: np.random.Generator
) -> dict[str, dict[str, pd.Series]]:
    seg_index: pd.MultiIndex = _segment_index(20, 4)
    prompt_index: pd.Index = pd.Index(range(20), name="prompt_idx")
    signals: dict[str, dict[str, pd.Series]] = {}
    for metric in SIGNAL_METRICS:
        index = prompt_index if metric == "F_pred" else seg_index
        models = [OPEN, HOSTED] if metric in {"F_pred", "F_attr"} else [OPEN]
        signals[metric] = {
            model: pd.Series(
                base[f"{metric}/{model}"] + noise * rng.normal(size=len(index)),
                index=index,
            )
            for model in models
        }
    return signals


class TestNoiseCeiling(TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.base: dict[str, np.ndarray] = {
            f"{metric}/{model}": rng.normal(size=20 if metric == "F_pred" else 80)
            for metric in SIGNAL_METRICS
            for model in (OPEN, HOSTED)
        }

    def test_identical_runs_have_unit_autocorrelation_and_full_grid(self) -> None:
        rng = np.random.default_rng(1)
        run = _signals(self.base, 0.0, rng)
        rows = pd.DataFrame(
            autocorrelation_rows(
                {PUBLISHED_RUN: run, "rep1": run, "rep2": run},
                "boolq",
                "sentence",
                n_resamples=20,
                conf=0.95,
                seed=0,
            )
        )
        symmetric = rows[rows["metric"].isin(SIGNAL_METRICS)]
        np.testing.assert_allclose(
            symmetric[symmetric["statistic"] != "pearson_r2"]["f_point"], 1.0
        )
        self.assertTrue((rows["run_s"] != rows["run_t"]).all())
        # Three runs: three unordered pairs per signal, six ordered pairs per transfer.
        counts = (
            rows[rows["statistic"] == "spearman"].groupby(["metric", "model"]).size()
        )
        self.assertEqual(counts[("F_pred", HOSTED)], 3)
        self.assertEqual(counts[("F_attn_mean", OPEN)], 3)
        for metric in TRANSFER_METRICS:
            self.assertEqual(counts[(metric, OPEN)], 6)
        self.assertNotIn((("F_mag_to_attr", HOSTED)), counts.index)
        self.assertEqual(set(rows["comparison"]), {"rep_vs_published", "rep_vs_rep"})
        rep_pairs = rows[rows["comparison"] == "rep_vs_rep"]
        self.assertTrue(
            set(rep_pairs["run_s"]) | set(rep_pairs["run_t"]) <= {"rep1", "rep2"}
        )
        self.assertEqual(set(rows["pair_population"]), {"open_open", "hosted_hosted"})
        self.assertTrue((rows["availability_status"] == "available").all())

    def test_run_noise_lowers_autocorrelation(self) -> None:
        rng = np.random.default_rng(2)
        rows = pd.DataFrame(
            autocorrelation_rows(
                {
                    "rep1": _signals(self.base, 0.5, rng),
                    "rep2": _signals(self.base, 0.5, rng),
                },
                "boolq",
                "sentence",
                n_resamples=20,
                conf=0.95,
                seed=0,
            )
        )
        attr = rows[(rows["metric"] == "F_attr") & (rows["statistic"] == "pearson_r")]
        self.assertEqual(len(attr), 2)
        self.assertTrue(((attr["f_point"] > 0.5) & (attr["f_point"] < 0.99)).all())
        self.assertTrue((attr["f_lo"] <= attr["f_point"]).all())

    def test_spread_summarizes_repetitions_against_published(self) -> None:
        keys = {
            "pair_population": "open_open",
            "model_s": OPEN,
            "model_t": "qwen2.5-3b-instruct",
            "metric": "F_pred",
            "statistic": "spearman",
            "aggregation": "row_pooled",
        }
        published = pd.DataFrame([{**keys, "f_point": 0.50}])
        reps = {
            f"rep{i}": pd.DataFrame([{**keys, "f_point": value, "n_observations": n}])
            for i, (value, n) in enumerate([(0.40, 10), (0.60, 9), (0.50, 10)], 1)
        }
        row = spread_table(published, reps).iloc[0]
        self.assertAlmostEqual(row["rep_mean"], 0.50)
        self.assertAlmostEqual(row["rep_sd"], 0.10)
        self.assertAlmostEqual(row["rep_min"], 0.40)
        self.assertAlmostEqual(row["rep_max"], 0.60)
        self.assertAlmostEqual(row["published_minus_rep_mean"], 0.0)
        self.assertEqual(row["rep_min_n_observations"], 9)
        self.assertEqual(row["f_point_rep2"], 0.60)
