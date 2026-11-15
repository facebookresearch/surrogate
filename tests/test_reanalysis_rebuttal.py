# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
from benchmark_scripts.f_table import _emit_transfer
from benchmark_scripts.reanalysis_rebuttal import (
    decisive_agreement,
    demean_within_prompt,
    joint,
    mean_prompt_spearman,
    pearson_r2,
    residualize_on_length,
    segment_parts,
    verify_against_published,
)


def _segment_index(n_prompts: int, n_segments: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [range(n_prompts), range(n_segments)], names=["prompt_idx", "seg_idx"]
    )


class TestReanalysisRebuttal(TestCase):
    def test_segment_parts_split_system_passage_and_question(self) -> None:
        segments = pd.DataFrame(
            {
                "prompt_idx": [0, 0, 0, 0, 0],
                "seg_idx": [0, 1, 2, 3, 4],
                "message_role": ["system", "user", "user", "user", "user"],
                "segment_text": [
                    "Answer true or false.",
                    "Context:\nA passage mentioning Question: in a title.",
                    "More passage.",
                    "Question:\ndoes it split",
                    " across two segments.",
                ],
            }
        )
        self.assertEqual(
            segment_parts(segments).tolist(),
            ["system", "passage", "passage", "question", "question"],
        )

    def test_joint_drops_non_finite_and_takes_absolute_target(self) -> None:
        index = _segment_index(2, 3)
        source = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0], index=index)
        target = pd.Series([-1.0, np.inf, 3.0, -4.0, 5.0, -6.0], index=index)
        frame = joint(source, target, abs_target=True)
        self.assertEqual(len(frame), 4)
        self.assertTrue((frame["y"] >= 0).all())
        self.assertEqual(frame["prompt_idx"].tolist(), [0, 1, 1, 1])

    def test_within_prompt_estimators_match_f_table_prompt_equal(self) -> None:
        rng = np.random.default_rng(0)
        index = _segment_index(40, 6)
        offsets = np.repeat(rng.normal(scale=5.0, size=40), 6)
        source = pd.Series(offsets + rng.normal(size=240), index=index)
        target = pd.Series(offsets + rng.normal(size=240), index=index)
        frame = joint(source, target)
        # Shared prompt offsets inflate the pooled r²; demeaning removes them.
        self.assertGreater(pearson_r2(frame["x"].values, frame["y"].values), 0.5)
        demeaned = demean_within_prompt(frame)
        self.assertLess(pearson_r2(demeaned["x"].values, demeaned["y"].values), 0.1)
        reference = pd.DataFrame(
            _emit_transfer(
                "boolq",
                "F_mag_to_attr",
                {"s": source},
                {"t": target},
                True,
                1,
                0.95,
                rng,
                aggregation="prompt_equal_mean_r2",
            )
        )
        expected = reference[reference["statistic"].eq("spearman")].iloc[0]
        rho, n_prompts, n_rows = mean_prompt_spearman(frame)
        self.assertAlmostEqual(rho, expected["f_point"])
        self.assertEqual(n_prompts, expected["n_prompts"])
        self.assertEqual(n_rows, expected["n_observations"])

    def test_residualize_on_length_removes_linear_length_dependence(self) -> None:
        rng = np.random.default_rng(1)
        index = _segment_index(50, 4)
        log_length = pd.Series(rng.uniform(0, 3, size=200), index=index)
        signal = pd.Series(2.0 * log_length.values + rng.normal(size=200), index=index)
        signal.iloc[0] = np.nan
        residual, fit = residualize_on_length(signal, log_length)
        self.assertAlmostEqual(fit["slope"], 2.0, delta=0.3)
        self.assertEqual(fit["n_observations"], 199)
        valid = residual.notna()
        self.assertFalse(valid.iloc[0])
        self.assertLess(
            pearson_r2(log_length[valid].values, residual[valid].values), 1e-20
        )

    def test_decisive_agreement_counts_matches_chance_and_question_share(self) -> None:
        index = _segment_index(10, 5)
        values = np.tile([1.0, 2.0, 3.0, 4.0, 5.0], 10)
        frame = pd.DataFrame(
            {"ax": values, "ay": values, "prompt_idx": index.get_level_values(0)},
            index=index.set_names([None, None]),
        )
        parts = pd.Series(
            np.tile(["system", "passage", "passage", "passage", "question"], 10),
            index=index,
        )
        confidence = pd.Series(np.arange(10, dtype=float))
        records = decisive_agreement(
            frame, confidence, np.random.default_rng(0), parts=parts
        )
        overall = records[0]
        self.assertEqual(overall["stratum"], "all")
        self.assertEqual(overall["top1_agreement"], 1.0)
        self.assertAlmostEqual(overall["top1_chance"], 0.2)
        self.assertEqual(overall["top3_overlap"], 1.0)
        self.assertAlmostEqual(overall["top3_chance"], 0.6)
        self.assertEqual(overall["share_target_top_question"], 1.0)
        self.assertEqual(overall["share_target_top_system"], 0.0)
        self.assertTrue(np.isnan(overall["top1_overlap_baseline"]))
        self.assertEqual([r["n_prompts"] for r in records[1:]], [2] * 5)
        # A baseline peaking on the system segment never matches the target,
        # and a model peaking there is counted in the system share.
        system_first = pd.Series(np.tile([9.0, 1.0, 1.0, 1.0, 1.0], 10), index=index)
        records = decisive_agreement(
            frame.assign(ay=system_first.to_numpy()),
            confidence,
            np.random.default_rng(0),
            parts=parts,
            baseline=pd.Series(values, index=index),
        )
        self.assertEqual(records[0]["share_target_top_system"], 1.0)
        self.assertEqual(records[0]["top1_overlap_baseline"], 0.0)
        # Ties share credit: a flat baseline over 5 segments scores 1/5.
        records = decisive_agreement(
            frame,
            confidence,
            np.random.default_rng(0),
            parts=parts,
            baseline=pd.Series(1.0, index=index),
        )
        self.assertAlmostEqual(records[0]["top1_overlap_baseline"], 0.2)

    def test_verify_against_published_rejects_mismatch(self) -> None:
        rows = pd.DataFrame(
            {"metric": ["F_pred"], "model_s": ["a"], "model_t": ["b"], "value": [0.5]}
        )
        published = rows.rename(columns={"value": "f_point"})
        self.assertEqual(verify_against_published(rows, published), 0.0)
        with self.assertRaises(ValueError):
            verify_against_published(rows, published.assign(f_point=0.6))
