# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
from benchmark_scripts.screen_transfer import (
    fit_support_cut,
    flag_groups,
    pair_agreement,
    scored_prompts,
    screen_rows,
    surrogate_support,
)


def _passage() -> pd.DataFrame:
    # Four prompts with three passage sentences each. Models a and b share a
    # top sentence in prompts 0-2 and differ in prompt 3.
    index = pd.MultiIndex.from_product(
        [range(4), range(3)], names=["prompt_idx", "seg_idx"]
    )
    return pd.DataFrame(
        {
            "a": [5.0, 1.0, 0.0, -6.0, 1.0, 0.5, 0.2, 4.0, -1.0, 3.0, 0.1, 0.2],
            "b": [4.0, 0.5, 0.1, -3.0, 0.2, 0.1, 0.0, 2.0, -0.5, 0.1, 0.2, -2.5],
        },
        index=index,
    )


def _prediction() -> dict[str, pd.Series]:
    return {
        "a": pd.Series({0: 2.0, 1: -3.0, 2: 1.0, 3: 4.0}),
        "b": pd.Series({0: 1.0, 1: -1.0, 2: 2.0, 3: 1.0}),
    }


class TestScreenTransfer(TestCase):
    def test_fit_support_cut_separates_matches_from_misses(self) -> None:
        prompts = pd.DataFrame(
            {
                "qwen_top_support": [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0, 13.0],
                "top1": [False, False, False, True, True, True, True, True],
            }
        )
        # Every cut in (3, 4] puts the three misses below and all matches above;
        # the first such grid point wins.
        self.assertEqual(fit_support_cut(prompts), 3.5)

    def test_scored_prompts_drops_unscored_and_single_sentence_prompts(self) -> None:
        prompts = pd.DataFrame(
            {
                "prompt_idx": [0, 1, 2],
                "case": ["both_right", "unscored", "both_right"],
                "top_abs_passage_match": [True, True, np.nan],
            }
        )
        out = scored_prompts(prompts)
        self.assertEqual(out["prompt_idx"].tolist(), [0])
        self.assertEqual(out["top1"].tolist(), [True])

    def test_surrogate_support_is_signed_by_the_models_own_answer(self) -> None:
        support = surrogate_support(_passage(), _prediction(), "a")
        # Prompt 1 answers False (ℓ < 0), so the −6 sentence supports it by +6.
        self.assertEqual(support.tolist(), [5.0, 6.0, 4.0, 3.0])

    def test_pair_agreement_top1_and_chance(self) -> None:
        out = pair_agreement(_passage(), _prediction(), "a", "b", pd.Index(range(4)))
        self.assertEqual(out["n_prompts"], 4)
        self.assertAlmostEqual(out["top1_match"], 0.75)
        self.assertAlmostEqual(out["top1_chance"], 1 / 3)
        passage = _passage()
        expected = np.corrcoef(passage["a"], passage["b"])[0, 1] ** 2
        self.assertAlmostEqual(out["passage_r2"], expected)

    def test_pair_agreement_skips_prompts_without_finite_logodds(self) -> None:
        prediction = _prediction()
        prediction["b"] = prediction["b"].copy()
        prediction["b"].loc[3] = -np.inf
        out = pair_agreement(_passage(), prediction, "a", "b", pd.Index(range(4)))
        self.assertEqual(out["n_prompts"], 3)
        self.assertAlmostEqual(out["top1_match"], 1.0)

    def test_flag_groups(self) -> None:
        first = pd.Series([True, True, False, False])
        second = pd.Series([True, False, True, False])
        self.assertEqual(
            flag_groups(first, second).tolist(),
            ["both", "first_only", "second_only", "neither"],
        )

    def test_screen_rows_use_odd_prompts_and_fail_missing_support(self) -> None:
        n = 12
        prompts = pd.DataFrame(
            {
                "prompt_idx": range(n),
                "case": ["both_right"] * n,
                "top_abs_passage_match": [True] * n,
                "chance_passage": [0.5] * n,
                "overlap_top_matches_gpt": [1.0] * n,
                "pred_qwen": [True] * n,
                # Odd prompts: 1, 3, 5 pass; 7, 9 fail; 11 has no support.
                "qwen_top_support": [
                    0,
                    20,
                    0,
                    20,
                    0,
                    20,
                    0,
                    1,
                    0,
                    1,
                    0,
                    np.nan,
                ],
            }
        )
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {
                "prompt_idx": np.repeat(np.arange(n), 3),
                "part": "passage",
                "d_qwen": rng.normal(size=3 * n),
                "d_gpt": rng.normal(size=3 * n),
                "question_overlap": rng.integers(0, 3, size=3 * n).astype(float),
            }
        )
        rows = {r["subset"]: r for r in screen_rows(frame, prompts, cut=10.0)}
        self.assertEqual(rows["no_screen"]["n_prompts"], 6)
        self.assertEqual(rows["pass"]["n_prompts"], 3)
        self.assertEqual(rows["fail"]["n_prompts"], 3)
        self.assertAlmostEqual(rows["pass"]["coverage"], 0.5)
        self.assertEqual(rows["pass_answer_false"]["n_prompts"], 0)
