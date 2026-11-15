# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

from unittest import TestCase

import numpy as np
import pandas as pd
from benchmark_scripts.reanalysis_rebuttal import question_overlap
from benchmark_scripts.worked_example import prompt_table


def _frame() -> pd.DataFrame:
    # One prompt: system, three passage sentences, question.
    return pd.DataFrame(
        {
            "prompt_idx": [0] * 5,
            "seg_idx": [0, 1, 2, 3, 4],
            "part": ["system", "passage", "passage", "passage", "question"],
            "segment_text": ["sys", "a", "b", "c", "Question:\nis it"],
            # ℓ_qwen = 2: sentence 2 flips Qwen (2 - 3 < 0); the question too.
            "d_qwen": [0.1, 1.0, 3.0, -0.5, 5.0],
            # ℓ_gpt = -4: sentence 1 is GPT-4o's decisive sentence and flips it.
            "d_gpt": [0.2, -5.0, -1.0, 0.5, -2.0],
            "align_qwen": [0.0, 0.1, 0.9, -0.2, 0.3],
            "attn_mean_qwen": [0.1, 0.2, 0.1, 0.3, 0.9],
        }
    )


class TestWorkedExample(TestCase):
    def test_prompt_table_answers_and_passage_only_decisive_and_flips(self) -> None:
        out = prompt_table(
            _frame(),
            pd.Series({0: 2.0}),
            pd.Series({0: -4.0}),
            pd.Series({0: True}),
        ).iloc[0]
        self.assertTrue(out["pred_qwen"])
        self.assertFalse(out["pred_gpt"])
        self.assertEqual(out["agree"], False)
        self.assertEqual(out["case"], "only_qwen_right")
        self.assertEqual(out["n_passage"], 3)
        # Decisive = largest Δℓ·sign(ℓ) among passage sentences.
        self.assertEqual(out["decisive_qwen"], 2)
        self.assertEqual(out["decisive_gpt"], 1)
        self.assertFalse(out["decisive_match"])
        # Flips are passage-only; deleting the question is recorded separately.
        self.assertEqual(out["flip_segs_qwen"], "2")
        self.assertTrue(out["question_flips_qwen"])
        self.assertEqual(out["flip_segs_gpt"], "1")
        self.assertFalse(out["gpt_flip_found_by_qwen"])
        self.assertEqual(out["top_abs_passage_qwen"], 2)
        self.assertEqual(out["top_abs_passage_gpt"], 1)
        self.assertEqual(out["top_abs_align_qwen"], 2)
        self.assertEqual(out["top_attn_qwen"], 3)
        self.assertAlmostEqual(out["chance_passage"], 1 / 3)
        self.assertAlmostEqual(out["gap"], 6.0)

    def test_infinite_target_logodds_is_unscored(self) -> None:
        out = prompt_table(
            _frame(),
            pd.Series({0: 2.0}),
            pd.Series({0: -np.inf}),
            pd.Series({0: False}),
        ).iloc[0]
        self.assertEqual(out["case"], "unscored")
        self.assertTrue(np.isnan(out["agree"]))

    def test_question_overlap_baseline_picks_the_most_overlapping_sentence(self) -> None:
        frame = _frame()
        frame["segment_text"] = [
            "sys",
            "Context:\nCats are mammals.",
            "Cats like milk a lot.",
            "Dogs bark.",
            "Question:\ndo cats like milk",
        ]
        frame["question_overlap"] = question_overlap(frame)
        # Plurals are stripped and "Context"/"Question" headers don't count.
        self.assertEqual(frame["question_overlap"].tolist()[1:4], [1.0, 3.0, 0.0])
        out = prompt_table(
            frame, pd.Series({0: 2.0}), pd.Series({0: -4.0}), pd.Series({0: True})
        ).iloc[0]
        self.assertEqual(out["top_overlap_passage"], 2)
        # GPT-4o's top passage sentence is 1, so the baseline misses.
        self.assertFalse(out["overlap_top_matches_gpt"])
