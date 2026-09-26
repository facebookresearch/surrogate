# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import math
import os
import tempfile
from unittest import TestCase

import pandas as pd

from benchmark_scripts.compute_logodds import compute


class TestComputeLogodds(TestCase):
    def test_anli_canonical_contrast_uses_configured_label_order(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            rows = [
                {
                    "model": "model",
                    "prompt_idx": 0,
                    "seg_idx": 0,
                    "kind": "ablated",
                    "label": label,
                    "token": token,
                    "logprob": logprob,
                }
                for label, token, logprob in [
                    ("entailment", "ent", -1.0),
                    ("neutral", "neutral", -2.0),
                    ("contradiction", "contr", -10.0),
                ]
            ]
            pd.DataFrame(rows).to_csv(
                os.path.join(results_dir, "anli_r1_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )

            compute("anli_r1", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "anli_r1_sentence_logodds.tsv"),
                sep="\t",
            )
            self.assertEqual(result.loc[0, "logodds"], 1.0)
            self.assertEqual(result.loc[0, "logodds_contradiction_entailment"], -9.0)

    def test_missing_api_label_has_negative_infinite_logprob(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "label": "true",
                        "token": "true",
                        "logprob": -1.0,
                    }
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "original_request_status": "ok",
                        "segment_request_status": "ok",
                    }
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"),
                sep="\t",
                index=False,
            )

            compute("boolq", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_logodds.tsv"),
                sep="\t",
            )
            self.assertEqual(result.loc[0, "label_lp_true"], -1.0)
            self.assertTrue(math.isinf(result.loc[0, "label_lp_false"]))
            self.assertGreater(result.loc[0, "logodds"], 0.0)

    def test_race_uses_correct_label_against_all_incorrect_labels(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            rows = [
                {
                    "model": "model",
                    "prompt_idx": 7,
                    "seg_idx": 0,
                    "kind": "ablated",
                    "answer": "B",
                    "label": label,
                    "token": label,
                    "logprob": logprob,
                }
                for label, logprob in [
                    ("A", -3.0),
                    ("B", -1.0),
                    ("C", -4.0),
                    ("D", -5.0),
                ]
            ]
            pd.DataFrame(rows).to_csv(
                os.path.join(results_dir, "race_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )

            compute("race", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "race_sentence_logodds.tsv"),
                sep="\t",
            )
            expected = -1.0 - math.log(math.exp(-3.0) + math.exp(-4.0) + math.exp(-5.0))
            self.assertAlmostEqual(result.loc[0, "logodds"], expected)

    def test_label_aggregate_rows_resolve_from_label_not_token_alias(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            pd.DataFrame(
                [
                    {
                        "model": "model",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "label": label,
                        "token": label,
                        "logprob": value,
                        "logprob_granularity": "label_aggregate",
                    }
                    for label, value in [
                        ("entailment", -1.0),
                        ("neutral", -2.0),
                        ("contradiction", -3.0),
                    ]
                ]
            ).to_csv(
                os.path.join(results_dir, "anli_r1_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )

            compute("anli_r1", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "anli_r1_sentence_logodds.tsv"),
                sep="\t",
            )
            self.assertEqual(result.loc[0, "label_lp_entailment"], -1.0)
            self.assertEqual(result.loc[0, "logodds_entailment_contradiction"], 2.0)

    def test_all_missing_aggregate_labels_preserve_observation_row(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            pd.DataFrame(
                [
                    {
                        "model": "hosted",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "label": label,
                        "token": label,
                        "logprob": -float("inf"),
                        "logprob_granularity": "label_aggregate",
                    }
                    for label in ("true", "false")
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )

            compute("boolq", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_logodds.tsv"),
                sep="\t",
            )
            self.assertEqual(len(result), 1)
            self.assertTrue(math.isnan(result.loc[0, "logodds"]))

    def test_failed_request_masks_materialized_label_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            pd.DataFrame(
                [
                    {
                        "model": "hosted",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "label": label,
                        "token": label,
                        "logprob": -float("inf"),
                        "logprob_granularity": "label_aggregate",
                    }
                    for label in ("true", "false")
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "model": "hosted",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "original_request_status": "ok",
                        "segment_request_status": "transient_exhausted",
                    }
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"),
                sep="\t",
                index=False,
            )

            compute("boolq", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_logodds.tsv"),
                sep="\t",
            )
            value_columns = [
                column
                for column in result.columns
                if column == "logodds"
                or column.startswith("logodds_")
                or column.startswith("label_lp_")
            ]
            self.assertTrue(result.loc[0, value_columns].isna().all())

    def test_failed_original_masks_only_original_derived_values(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            pd.DataFrame(
                [
                    {
                        "model": "hosted",
                        "prompt_idx": 0,
                        "seg_idx": seg_idx,
                        "kind": kind,
                        "label": label,
                        "token": label,
                        "logprob": logprob,
                        "logprob_granularity": "label_aggregate",
                    }
                    for seg_idx, kind in [(float("nan"), "orig"), (0, "ablated")]
                    for label, logprob in [("true", -1.0), ("false", -2.0)]
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                [
                    {
                        "model": "hosted",
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "original_request_status": "transient_exhausted",
                        "segment_request_status": "ok",
                    }
                ]
            ).to_csv(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"),
                sep="\t",
                index=False,
            )

            compute("boolq", "sentence", results_dir=results_dir)

            result = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_logodds.tsv"),
                sep="\t",
            )
            value_columns = [
                column
                for column in result.columns
                if column == "logodds"
                or column.startswith("logodds_")
                or column.startswith("label_lp_")
            ]
            original = result[result["kind"] == "orig"].iloc[0]
            ablated = result[result["kind"] == "ablated"].iloc[0]
            self.assertTrue(original[value_columns].isna().all())
            self.assertTrue(ablated[value_columns].notna().all())
            self.assertEqual(ablated["logodds"], 1.0)
