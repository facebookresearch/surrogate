# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from benchmark_scripts.race_rv import (
    _apply_missingness_policy,
    _aligned,
    _attribution_vectors,
    _canonical_attribution,
    _centered_rv,
    _pair_vectors,
    compute_race_rv,
)


class TestRaceRV(TestCase):
    def test_missingness_policy_rebuilds_correct_vs_rest_logodds(self) -> None:
        frame: pd.DataFrame = pd.DataFrame(
            [
                {
                    "model": "m",
                    "answer": "A",
                    "label_lp_a": -1.0,
                    "label_lp_b": -2.0,
                    "label_lp_c": -np.inf,
                    "label_lp_d": -4.0,
                    "logodds": np.inf,
                },
                {
                    "model": "m",
                    "answer": "B",
                    "label_lp_a": -np.inf,
                    "label_lp_b": -np.inf,
                    "label_lp_c": -np.inf,
                    "label_lp_d": -np.inf,
                    "logodds": np.nan,
                },
            ]
        )

        result: pd.DataFrame = _apply_missingness_policy(frame)
        expected: float = -1.0 - float(
            np.logaddexp.reduce(np.asarray([-2.0, -4.05, -4.0]))
        )

        self.assertAlmostEqual(result.loc[0, "label_lp_c"], -4.05)
        self.assertAlmostEqual(result.loc[0, "logodds"], expected)
        self.assertTrue(
            result.loc[1, ["label_lp_a", "label_lp_b", "label_lp_c", "label_lp_d"]]
            .isna()
            .all()
        )
        self.assertTrue(np.isnan(result.loc[1, "logodds"]))

    def test_identical_matrices_have_unit_rv(self) -> None:
        values = np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 3.0]])
        self.assertAlmostEqual(_centered_rv(values, values), 1.0)

    def test_all_pairs_vector_order(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "label_lp_a": 4.0,
                    "label_lp_b": 3.0,
                    "label_lp_c": 1.0,
                    "label_lp_d": 0.0,
                }
            ]
        )
        self.assertEqual(
            _pair_vectors(frame, "all_pairs").tolist()[0],
            [1.0, 3.0, 4.0, 2.0, 3.0, 1.0],
        )

    def test_attribution_is_original_minus_ablated_pairwise_margin(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "model": "m",
                    "prompt_idx": 0,
                    "seg_idx": None,
                    "kind": "orig",
                    "label_lp_a": 4.0,
                    "label_lp_b": 2.0,
                    "label_lp_c": 1.0,
                    "label_lp_d": 0.0,
                },
                {
                    "model": "m",
                    "prompt_idx": 0,
                    "seg_idx": 3,
                    "kind": "ablated",
                    "label_lp_a": 3.0,
                    "label_lp_b": 2.0,
                    "label_lp_c": 1.0,
                    "label_lp_d": 0.0,
                },
            ]
        )

        result = _attribution_vectors(frame, "m", "anchor_a")

        self.assertEqual(result.loc[(0, 3)].tolist(), [1.0, 1.0, 1.0])

        canonical: pd.DataFrame = _canonical_attribution(
            pd.DataFrame(
                [
                    {**frame.iloc[0].to_dict(), "logodds": 2.5},
                    {**frame.iloc[1].to_dict(), "logodds": 1.0},
                ]
            ),
            "m",
        )
        self.assertEqual(canonical.loc[(0, 3), "value"], 1.5)

    def test_alignment_uses_pair_specific_complete_cases(self) -> None:
        index = pd.Index([0, 1, 2])
        first = pd.DataFrame([[1.0], [np.inf], [3.0]], index=index)
        second = pd.DataFrame([[1.0], [2.0], [np.nan]], index=index)

        x, y, clusters = _aligned(first, second)

        self.assertEqual(x.tolist(), [[1.0]])
        self.assertEqual(y.tolist(), [[1.0]])
        self.assertEqual(clusters.tolist(), [0])

    def test_prediction_bootstrap_is_scope_and_order_invariant(self) -> None:
        rows: list[dict[str, object]] = []
        manifest_rows: list[dict[str, object]] = []
        segment_rows: list[dict[str, object]] = []
        for prompt_idx in range(8):
            for seg_idx, role in enumerate(("system", "user")):
                manifest_rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "seg_idx": seg_idx,
                        "message_role": role,
                    }
                )
            for model_index, model in enumerate(("m1", "m2")):
                base: float = float(prompt_idx + model_index)
                rows.append(
                    {
                        "model": model,
                        "prompt_idx": prompt_idx,
                        "seg_idx": np.nan,
                        "kind": "orig",
                        "answer": "A",
                        "label_lp_a": base + 0.1 * prompt_idx,
                        "label_lp_b": -base,
                        "label_lp_c": 0.5 * base,
                        "label_lp_d": -0.25 * base,
                    }
                )
                for seg_idx in range(2):
                    segment_rows.append(
                        {
                            "model": model,
                            "prompt_idx": prompt_idx,
                            "seg_idx": seg_idx,
                            "attention_mean": base + 0.2 * seg_idx,
                            "attention_max": base + 0.3 * seg_idx,
                            "attention_rollout": base - 0.1 * seg_idx,
                            "w_norm": 2.0,
                            "delta_norm_postnorm": base + seg_idx + 1.0,
                            "w_dot_delta_z_postnorm": base - seg_idx + 0.5,
                        }
                    )
                    rows.append(
                        {
                            "model": model,
                            "prompt_idx": prompt_idx,
                            "seg_idx": seg_idx,
                            "kind": "ablated",
                            "answer": "A",
                            "label_lp_a": base - 0.2 * (seg_idx + 1),
                            "label_lp_b": -base + 0.1 * prompt_idx,
                            "label_lp_c": 0.5 * base - 0.05 * seg_idx,
                            "label_lp_d": -0.25 * base + 0.03 * prompt_idx,
                            "logodds": base - 0.4 * seg_idx,
                        }
                    )

        for row in rows:
            if row["kind"] == "orig":
                row["logodds"] = float(row["label_lp_a"]) - float(row["label_lp_b"])

        with tempfile.TemporaryDirectory() as directory:
            logodds_path: str = os.path.join(directory, "race.tsv")
            manifest_path: str = os.path.join(directory, "segments.tsv")
            segments_path: str = os.path.join(directory, "segment_values.tsv")
            pd.DataFrame(rows).to_csv(logodds_path, sep="\t", index=False)
            pd.DataFrame(manifest_rows).to_csv(manifest_path, sep="\t", index=False)
            pd.DataFrame(segment_rows).to_csv(segments_path, sep="\t", index=False)
            forward: pd.DataFrame = compute_race_rv(
                logodds_path,
                manifest_path,
                segments_path,
                ["all", "system", "user"],
                n_resamples=40,
                confidence=0.9,
                seed=42,
                cohort=("m1", "m2"),
            )
            reverse: pd.DataFrame = compute_race_rv(
                logodds_path,
                manifest_path,
                segments_path,
                ["user", "all", "system"],
                n_resamples=40,
                confidence=0.9,
                seed=42,
                cohort=("m1", "m2"),
            )

        prediction: pd.DataFrame = forward[forward["metric"] == "F_pred_rv"]
        self.assertEqual(set(forward["aggregation"]), {"row_pooled"})
        self.assertEqual(
            set(forward["metric"]),
            {
                "F_pred_rv",
                "F_attr_rv",
                "F_attn_mean_rv",
                "F_attn_max_rv",
                "F_attn_rollout_rv",
                "F_mag_rv",
                "F_align_rv",
                "F_attn_mean_to_attr_rv",
                "F_attn_max_to_attr_rv",
                "F_attn_rollout_to_attr_rv",
                "F_mag_to_attr_rv",
                "F_align_to_attr_rv",
            },
        )
        for _, representation_rows in prediction.groupby("representation"):
            self.assertEqual(representation_rows["f_point"].nunique(), 1)
            self.assertEqual(representation_rows["f_lo"].nunique(), 1)
            self.assertEqual(representation_rows["f_hi"].nunique(), 1)
        sort_columns: list[str] = [
            "representation",
            "scope",
            "metric",
            "model_s",
            "model_t",
        ]
        pd.testing.assert_frame_equal(
            forward.sort_values(sort_columns).reset_index(drop=True),
            reverse.sort_values(sort_columns).reset_index(drop=True),
        )
