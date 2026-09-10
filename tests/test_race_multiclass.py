# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import math
import os
import tempfile
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from benchmark_scripts.race_multiclass import (
    MAGNITUDE_METRICS,
    VECTOR_METRICS,
    _aligned_moments,
    _attribution_features,
    _bootstrap,
    _evaluate,
    _helmert_basis,
    _prediction_features,
    compute_race_multiclass,
)


class TestRaceMulticlass(TestCase):
    def test_shared_clr_coordinates_and_geometric_magnitudes(self) -> None:
        basis: np.ndarray = _helmert_basis(4)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.ones(4) @ basis, np.zeros(3), atol=1e-12)

        baseline: np.ndarray = np.asarray(
            [[math.log(0.4), math.log(0.3), math.log(0.2), math.log(0.1)]]
        )
        vectors: np.ndarray = _prediction_features(baseline)
        shifted: np.ndarray = _prediction_features(baseline + 17.0)
        np.testing.assert_allclose(vectors, shifted, atol=1e-12)

        attribution, magnitudes = _attribution_features(baseline, baseline)
        np.testing.assert_allclose(attribution, np.zeros((1, 3)), atol=1e-12)
        np.testing.assert_allclose(magnitudes, np.zeros((1, 3)), atol=1e-7)

    def test_vector_metrics_preserve_sign_but_cka_does_not(self) -> None:
        keys: list[int] = [0, 1, 2]
        x: np.ndarray = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
        left: pd.DataFrame = pd.DataFrame(
            x, index=keys, columns=["vector_0", "vector_1"]
        )
        right: pd.DataFrame = pd.DataFrame(-x, index=keys, columns=left.columns)
        moments = _aligned_moments(left, right, set(keys), ())
        metrics: dict[str, float] = _evaluate(moments, np.ones(3), "row_pooled")
        self.assertAlmostEqual(metrics["signed_frobenius_r"], -1.0)
        self.assertAlmostEqual(metrics["direction_cosine"], -1.0)
        self.assertAlmostEqual(metrics["linear_cka"], 1.0)

    def test_prompt_equal_direction_gives_each_prompt_equal_weight(self) -> None:
        keys: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0)]
        left: pd.DataFrame = pd.DataFrame(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            index=pd.MultiIndex.from_tuples(keys, names=["prompt_idx", "seg_idx"]),
            columns=["vector_0", "vector_1"],
        )
        right: pd.DataFrame = pd.DataFrame(
            [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]],
            index=left.index,
            columns=left.columns,
        )
        moments = _aligned_moments(left, right, set(keys), ())
        multiplicities: np.ndarray = np.ones(2)
        row_metrics: dict[str, float] = _evaluate(moments, multiplicities, "row_pooled")
        prompt_metrics: dict[str, float] = _evaluate(
            moments, multiplicities, "prompt_equal"
        )
        self.assertAlmostEqual(row_metrics["direction_cosine"], 1.0 / 3.0)
        self.assertAlmostEqual(prompt_metrics["direction_cosine"], 0.0)

    def test_bootstrap_is_deterministic_and_retains_valid_draws(self) -> None:
        keys: list[int] = list(range(30))
        x: np.ndarray = np.asarray(
            [[math.sin(index), math.cos(index)] for index in keys]
        )
        y: np.ndarray = np.asarray(
            [[row[0] + 0.1 * row[1], 0.2 * row[0] + row[1]] for row in x]
        )
        left: pd.DataFrame = pd.DataFrame(
            x, index=keys, columns=["vector_0", "vector_1"]
        )
        right: pd.DataFrame = pd.DataFrame(y, index=keys, columns=left.columns)
        moments = _aligned_moments(left, right, set(keys), ())
        first = _bootstrap(moments, "row_pooled", 25, 17)
        second = _bootstrap(moments, "row_pooled", 25, 17)
        self.assertEqual(first, second)
        self.assertTrue(all(result[3] == 25 for result in first.values()))

    def test_strict_and_global_complete_case_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            logodds_path: str = os.path.join(directory, "race_sentence_logodds.tsv")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": prompt_idx,
                        "seg_idx": seg_idx,
                        "message_role": role,
                        "answer": "A",
                        "n_segments": 2,
                    }
                    for prompt_idx in range(4)
                    for seg_idx, role in ((0, "system"), (1, "user"))
                ]
            )
            manifest.to_csv(manifest_path, sep="\t", index=False, compression="gzip")
            rows: list[dict[str, str | int | float]] = []
            for model_index, model in enumerate(("model-a", "model-b", "model-c")):
                for prompt_idx in range(4):
                    baseline: list[float] = [
                        -0.2 * (prompt_idx + 1) * (label_index + 1)
                        + 0.03 * model_index * label_index
                        for label_index in range(4)
                    ]
                    if model == "model-c" and prompt_idx == 3:
                        baseline[0] = -float("inf")
                    rows.append(
                        self._logodds_row(model, prompt_idx, None, "orig", baseline)
                    )
                    for seg_idx in range(2):
                        ablated: list[float] = [
                            value
                            + 0.02
                            * (seg_idx + 1)
                            * (label_index + 1)
                            * (model_index + 1)
                            for label_index, value in enumerate(baseline)
                        ]
                        rows.append(
                            self._logodds_row(
                                model, prompt_idx, seg_idx, "ablated", ablated
                            )
                        )
            pd.DataFrame(rows).to_csv(logodds_path, sep="\t", index=False)

            strict: pd.DataFrame = compute_race_multiclass(
                logodds_path,
                manifest_path,
                ("model-a", "model-b"),
                "strict_complete",
                n_bootstrap=0,
            )
            self.assertEqual(len(strict), 39)
            prediction: pd.DataFrame = strict[strict["scope"] == "prediction"]
            self.assertEqual(set(prediction["metric"]), set(VECTOR_METRICS))
            all_attribution: pd.DataFrame = strict[
                (strict["scope"] == "all") & (strict["aggregation"] == "row_pooled")
            ]
            self.assertEqual(
                set(all_attribution["metric"]),
                set((*VECTOR_METRICS, *MAGNITUDE_METRICS)),
            )
            self.assertTrue((all_attribution["n_observations"] == 8).all())

            global_result: pd.DataFrame = compute_race_multiclass(
                logodds_path,
                manifest_path,
                ("model-a", "model-b", "model-c"),
                "global_complete_case_mnar",
                n_bootstrap=0,
            )
            self.assertEqual(len(global_result), 117)
            global_prediction: pd.DataFrame = global_result[
                global_result["scope"] == "prediction"
            ]
            self.assertTrue((global_prediction["n_observations"] == 3).all())
            global_all: pd.DataFrame = global_result[global_result["scope"] == "all"]
            self.assertTrue((global_all["n_observations"] == 6).all())
            self.assertTrue(
                (
                    global_result["missingness_policy"] == "global_complete_case_mnar"
                ).all()
            )

            with self.assertRaisesRegex(ValueError, "finite prediction rows"):
                compute_race_multiclass(
                    logodds_path,
                    manifest_path,
                    ("model-a", "model-c"),
                    "strict_complete",
                    n_bootstrap=0,
                )

    @staticmethod
    def _logodds_row(
        model: str,
        prompt_idx: int,
        seg_idx: int | None,
        kind: str,
        values: list[float],
    ) -> dict[str, str | int | float]:
        """Create one synthetic consolidated log-odds row."""
        row: dict[str, str | int | float] = {
            "model": model,
            "prompt_idx": prompt_idx,
            "seg_idx": float("nan") if seg_idx is None else seg_idx,
            "kind": kind,
            "answer": "A",
        }
        for label, value in zip(("a", "b", "c", "d"), values):
            row[f"label_lp_{label}"] = value
        return row


if __name__ == "__main__":
    unittest.main()
