# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

import numpy as np
import pandas as pd

from benchmark_scripts.anli_rv import _pair_vectors
from benchmark_scripts.rv import centered_rv, replace_censored_label_logprobs


class TestAnliRV(TestCase):
    def test_finite_extreme_preserves_all_label_missing_rows(self) -> None:
        values: np.ndarray = np.asarray(
            [
                [-1.0, -np.inf, -3.0],
                [-np.inf, -np.inf, -np.inf],
                [np.nan, np.nan, np.nan],
                [-2.0, -4.0, -np.inf],
            ]
        )

        result: np.ndarray = replace_censored_label_logprobs(values)

        self.assertAlmostEqual(result[0, 1], -4.05)
        self.assertAlmostEqual(result[3, 2], -4.05)
        self.assertTrue(np.isnan(result[1]).all())
        self.assertTrue(np.isnan(result[2]).all())
        self.assertTrue(np.isneginf(values[0, 1]))

    def test_pair_vector_order(self) -> None:
        frame: pd.DataFrame = pd.DataFrame(
            [
                {
                    "label_lp_entailment": 4.0,
                    "label_lp_neutral": 2.0,
                    "label_lp_contradiction": 1.0,
                }
            ]
        )
        self.assertEqual(_pair_vectors(frame).tolist()[0], [2.0, 3.0, 1.0])

    def test_pairwise_margins_equal_centered_class_score_rv(self) -> None:
        first: np.ndarray = np.asarray(
            [[4.0, 2.0, 1.0], [1.0, 3.0, 2.0], [2.0, 0.0, 4.0], [5.0, 1.0, 3.0]]
        )
        second: np.ndarray = np.asarray(
            [[3.0, 2.0, 0.0], [2.0, 4.0, 1.0], [1.0, 0.0, 5.0], [4.0, 2.0, 3.0]]
        )
        pair_map: np.ndarray = np.asarray(
            [[1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]
        ).T
        centered_first: np.ndarray = first - first.mean(axis=1, keepdims=True)
        centered_second: np.ndarray = second - second.mean(axis=1, keepdims=True)
        self.assertAlmostEqual(
            centered_rv(first @ pair_map, second @ pair_map),
            centered_rv(centered_first, centered_second),
        )
