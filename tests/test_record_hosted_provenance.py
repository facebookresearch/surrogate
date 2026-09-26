# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile
from unittest import TestCase

import pandas as pd

from benchmark_scripts.hosted_completion import (
    LAMBADA_GOLD_CANARY,
    LAMBADA_GOLD_CANARY_ANSWERS,
)
from benchmark_scripts.record_hosted_provenance import (
    _copy_and_summarize_canary,
    record,
)


class TestRecordHostedProvenance(TestCase):
    def test_frozen_canary_identity_matches_independent_audit(self) -> None:
        self.assertEqual(
            [
                (prompt_idx, LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx], n_segments)
                for prompt_idx, n_segments in LAMBADA_GOLD_CANARY
            ],
            [
                (442, "wendy", 83),
                (459, "crib", 117),
                (485, "bare", 86),
                (1037, "owl", 73),
                (2229, "bree", 73),
                (2258, "then", 84),
                (3368, "winnie", 81),
                (3592, "jared", 75),
                (3982, "tiger", 72),
                (4420, "angie", 79),
            ],
        )

    def test_copies_and_summarizes_fixed_completion_canary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path: str = os.path.join(directory, "source.json")
            output_dir: str = os.path.join(directory, "output")
            os.makedirs(output_dir)
            with open(source_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                            "n_segments": n_segments,
                            "orig_logprob": -1.0 if index < 5 else None,
                            "ablated_logprob": [-2.0] * n_segments,
                            "private_runner_trace": "must not be published",
                        }
                        for index, (prompt_idx, n_segments) in enumerate(
                            LAMBADA_GOLD_CANARY
                        )
                    ],
                    output,
                )

            summary = _copy_and_summarize_canary(
                source_path,
                output_dir,
                "model",
                42,
            )

            self.assertEqual(summary["sample_size"], 10)
            self.assertEqual(summary["original_coverage"], 0.5)
            self.assertEqual(summary["ablated_coverage"], 1.0)
            self.assertEqual(
                summary["paired_attribution_coverage"],
                sum(count for _, count in LAMBADA_GOLD_CANARY[:5])
                / sum(count for _, count in LAMBADA_GOLD_CANARY),
            )
            self.assertEqual(
                summary["expected_segments"],
                sum(count for _, count in LAMBADA_GOLD_CANARY),
            )
            self.assertTrue(
                os.path.isfile(os.path.join(output_dir, "model_canary.json"))
            )
            with open(
                os.path.join(output_dir, "model_canary.json"), encoding="utf-8"
            ) as source:
                public_payload = json.load(source)
            self.assertNotIn("private_runner_trace", public_payload[0])
            self.assertEqual(
                set(public_payload[0]),
                {
                    "prompt_idx",
                    "answer",
                    "n_segments",
                    "orig_logprob",
                    "ablated_logprob",
                },
            )

    def test_canary_rejects_non_deterministic_prompt_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path: str = os.path.join(directory, "source.json")
            selected = list(LAMBADA_GOLD_CANARY)
            selected[0] = (441, selected[0][1])
            with open(source_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": LAMBADA_GOLD_CANARY_ANSWERS.get(
                                prompt_idx, "invalid"
                            ),
                            "n_segments": n_segments,
                            "orig_logprob": None,
                            "ablated_logprob": [],
                        }
                        for prompt_idx, n_segments in selected
                    ],
                    output,
                )

            with self.assertRaisesRegex(ValueError, "audited frozen LAMBADA sample"):
                _copy_and_summarize_canary(
                    source_path,
                    directory,
                    "model",
                    42,
                )

    def test_canary_rejects_short_nonempty_ablation_list(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path: str = os.path.join(directory, "source.json")
            with open(source_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                            "n_segments": n_segments,
                            "orig_logprob": -1.0,
                            "ablated_logprob": (
                                [-2.0] * (n_segments - 1) if index == 0 else []
                            ),
                        }
                        for index, (prompt_idx, n_segments) in enumerate(
                            LAMBADA_GOLD_CANARY
                        )
                    ],
                    output,
                )

            with self.assertRaisesRegex(ValueError, "exactly 83 ablated scores"):
                _copy_and_summarize_canary(
                    source_path,
                    directory,
                    "model",
                    42,
                )

    def test_canary_rejects_wrong_segment_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path: str = os.path.join(directory, "source.json")
            with open(source_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                            "n_segments": n_segments + (1 if index == 0 else 0),
                            "orig_logprob": None,
                            "ablated_logprob": [None] * n_segments,
                        }
                        for index, (prompt_idx, n_segments) in enumerate(
                            LAMBADA_GOLD_CANARY
                        )
                    ],
                    output,
                )

            with self.assertRaisesRegex(ValueError, "n_segments=83"):
                _copy_and_summarize_canary(
                    source_path,
                    directory,
                    "model",
                    42,
                )

    def test_missing_canary_scores_count_against_canonical_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path: str = os.path.join(directory, "source.json")
            with open(source_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                            "n_segments": n_segments,
                            "orig_logprob": -1.0,
                            "ablated_logprob": (
                                [-2.0] * n_segments
                                if index < 5
                                else [None] * n_segments
                            ),
                        }
                        for index, (prompt_idx, n_segments) in enumerate(
                            LAMBADA_GOLD_CANARY
                        )
                    ],
                    output,
                )

            summary = _copy_and_summarize_canary(
                source_path,
                directory,
                "model",
                42,
            )

            expected_segments = sum(count for _, count in LAMBADA_GOLD_CANARY)
            successful_segments = sum(count for _, count in LAMBADA_GOLD_CANARY[:5])
            self.assertEqual(summary["expected_segments"], expected_segments)
            self.assertEqual(
                summary["ablated_coverage"],
                successful_segments / expected_segments,
            )
            self.assertEqual(
                summary["paired_attribution_coverage"],
                successful_segments / expected_segments,
            )

    def test_records_source_hashes_and_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_segment: str = os.path.join(directory, "source_segment.tsv.gz")
            source_tokens: str = os.path.join(directory, "source_tokens.tsv.gz")
            output_segment: str = os.path.join(directory, "model_segment.tsv.gz")
            manifest: str = os.path.join(directory, "segments.tsv.gz")
            pd.DataFrame(
                [
                    {"segment_result_available": True},
                    {"segment_result_available": False},
                ]
            ).to_csv(output_segment, sep="\t", index=False)
            pd.DataFrame([{"value": 1}]).to_csv(source_segment, sep="\t", index=False)
            pd.DataFrame([{"value": 2}]).to_csv(source_tokens, sep="\t", index=False)
            pd.DataFrame([{"prompt_idx": 0, "seg_idx": 0}]).to_csv(
                manifest, sep="\t", index=False
            )

            record(
                directory,
                "model",
                "boolq",
                "sentence",
                "abc123",
                source_segment,
                source_tokens,
                "Compared every key and segment count to the frozen manifest.",
                "2026-09-08T00:00:00Z",
                "hosted-fixture",
                {"top_logprobs": 20},
            )

            with open(
                os.path.join(directory, "model_run.json"), encoding="utf-8"
            ) as source:
                metadata = json.load(source)
            self.assertEqual(metadata["source_revision"], "abc123")
            self.assertEqual(metadata["segment_result_coverage"], 0.5)
            self.assertIn("artifact_sha256", metadata)
            self.assertIn("manifest_sha256", metadata)
            self.assertIn("transformation_source_sha256", metadata)
            self.assertEqual(set(metadata["source_sha256"]), {"segment", "tokens"})
            self.assertEqual(metadata["producer"]["served_model"], "hosted-fixture")
