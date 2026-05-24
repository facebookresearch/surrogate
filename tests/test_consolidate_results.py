# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from unittest import TestCase

import pandas as pd
from benchmark_scripts.consolidate_results import consolidate


class TestConsolidate(TestCase):
    def test_concat_two_models_segment_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            in_dir: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(in_dir)

            pd.DataFrame(
                [{"prompt_idx": 0, "seg_idx": 0, "n_segments": 1, "answer": 1,
                  "attention_mean": 0.1, "w_norm": 2.0}]
            ).to_csv(os.path.join(in_dir, "qa_segment.tsv"), sep="\t", index=False)
            pd.DataFrame(
                [{"prompt_idx": 0, "seg_idx": 0, "n_segments": 1, "answer": 1,
                  "attention_mean": 0.2, "w_norm": 3.0}]
            ).to_csv(os.path.join(in_dir, "qb_segment.tsv"), sep="\t", index=False)

            pd.DataFrame(
                [{"prompt_idx": 0, "seg_idx": 0, "kind": "ablated",
                  "label": "true", "token": "true", "logprob": -0.2}]
            ).to_csv(os.path.join(in_dir, "qa_tokens.tsv"), sep="\t", index=False)

            consolidate("boolq", "sentence", results_dir=results_dir)

            seg_df = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"), sep="\t"
            )
            self.assertEqual(set(seg_df["model"].unique()), {"qa", "qb"})
            self.assertEqual(len(seg_df), 2)

            tok_df = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"), sep="\t"
            )
            self.assertEqual(set(tok_df["model"].unique()), {"qa"})
            self.assertEqual(len(tok_df), 1)

    def test_no_tokens_emits_only_segments_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            in_dir: str = os.path.join(results_dir, "lambada", "sentence")
            os.makedirs(in_dir)

            pd.DataFrame(
                [{"prompt_idx": 0, "seg_idx": 0, "n_segments": 1,
                  "orig_completion_logprob": -1.5}]
            ).to_csv(os.path.join(in_dir, "qa_segment.tsv"), sep="\t", index=False)

            consolidate("lambada", "sentence", results_dir=results_dir)

            self.assertTrue(
                os.path.exists(
                    os.path.join(results_dir, "lambada_sentence_segments.tsv")
                )
            )
            self.assertFalse(
                os.path.exists(
                    os.path.join(results_dir, "lambada_sentence_tokens.tsv")
                )
            )
