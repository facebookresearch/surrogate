# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import shutil
import tempfile
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from benchmark_scripts.hosted_audit_receipt import (
    canonical_projection_digest,
    classification_payload_projection,
)
from benchmark_scripts.hosted_completion_audit_receipt import (
    FULL_POPULATION,
    RAW_SOURCE_FORMATS,
    completion_payload_projection,
)
from benchmark_scripts.import_hosted_results import (
    _label_rows,
    _sha256,
    import_results,
    prepare_completion_audit_receipt,
    prepare_producer_audit_receipt,
)
from benchmark_scripts.provenance_sources import (
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
)

PRODUCER_METADATA: dict[str, object] = {
    "producer_revision": "fixture-revision",
    "generated_at": "2026-09-08T00:00:00Z",
    "served_model": "hosted-fixture",
    "request_parameters": {"top_logprobs": 20},
}


class TestImportHostedResults(TestCase):
    def test_receipt_preparation_binds_raw_manifest_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir: str = os.path.join(directory, "boolq", "sentence")
            os.makedirs(output_dir)
            manifest_path: str = os.path.join(output_dir, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            receipt_path: str = os.path.join(directory, "receipt.json")
            pd.DataFrame(
                {
                    "prompt_idx": [0],
                    "answer": [True],
                    "seg_idx": [0],
                    "message_idx": [1],
                    "message_role": ["user"],
                    "message_seg_idx": [0],
                    "segment_text": ["word"],
                    "n_segments": [1],
                }
            ).to_csv(manifest_path, sep="\t", index=False)
            payload: list[dict[str, Any]] = [
                {
                    "prompt_idx": 0,
                    "answer": True,
                    "n_segments": 1,
                    "orig_label_logprobs": {"true": -1.0, "false": -2.0},
                    "original_request_status": "ok",
                    "ablated_label_logprobs": [{"true": -1.5, "false": -2.5}],
                    "ablated_request_statuses": ["ok"],
                }
            ]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            with open(receipt_path, "w", encoding="utf-8") as output:
                output.write("{}")
            metadata: dict[str, Any] = {
                "producer_revision": "1" * 64,
                "request_parameters": {"protocol": "fixture"},
            }
            receipt: dict[str, Any] = {
                "request_parameters": metadata["request_parameters"],
                "configurations": [
                    {
                        "benchmark": "boolq",
                        "pregrouper": "sentence",
                        "manifest_sha256": _sha256(manifest_path),
                    }
                ],
                "entries": [
                    {
                        "benchmark": "boolq",
                        "pregrouper": "sentence",
                        "model": "gpt-4-1",
                        "raw_artifact_sha256": _sha256(input_path),
                        "raw_artifact_size_bytes": os.path.getsize(input_path),
                        "producer_revision_sha256": "1" * 64,
                        "availability_status": "complete",
                        "projection_sha256": canonical_projection_digest(
                            classification_payload_projection(
                                payload, ("true", "false")
                            )
                        ),
                    }
                ],
            }
            with patch(
                "benchmark_scripts.import_hosted_results.load_receipt",
                return_value=receipt,
            ) as load_receipt_mock:
                reference = prepare_producer_audit_receipt(
                    receipt_path,
                    input_path,
                    manifest_path,
                    output_dir,
                    "boolq",
                    "sentence",
                    "gpt-4-1",
                    metadata,
                    "complete",
                )
                load_receipt_mock.assert_called_once_with(
                    receipt_path,
                    expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
                )
                self.assertEqual(reference["entry_id"], "boolq/sentence/gpt-4-1")
                self.assertEqual(
                    reference["sha256"],
                    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
                )
                load_receipt_mock.reset_mock()
                payload[0]["orig_label_logprobs"] = {
                    "true": -9.0,
                    "false": -2.0,
                }
                with open(input_path, "w", encoding="utf-8") as output:
                    json.dump(payload, output)
                with self.assertRaisesRegex(ValueError, "raw artifact identity"):
                    prepare_producer_audit_receipt(
                        receipt_path,
                        input_path,
                        manifest_path,
                        output_dir,
                        "boolq",
                        "sentence",
                        "gpt-4-1",
                        metadata,
                        "complete",
                    )
                load_receipt_mock.assert_called_once_with(
                    receipt_path,
                    expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
                )

    def test_completion_receipt_binds_raw_manifest_and_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir: str = os.path.join(directory, "lambada", "word")
            os.makedirs(output_dir)
            manifest_path: str = os.path.join(output_dir, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            receipt_path: str = os.path.join(directory, "receipt.json")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": "target",
                        "seg_idx": 0,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 0,
                        "segment_text": "word",
                        "n_segments": 1,
                    }
                ]
            )
            manifest.to_csv(manifest_path, sep="\t", index=False)
            payload: list[dict[str, Any]] = [
                {
                    "prompt_idx": 0,
                    "answer": "target",
                    "ablation_idx": 0,
                    "n_segments": 1,
                    "orig_logprob": -1.0,
                    "orig_status": "ok",
                    "ablated_logprob": -2.0,
                    "ablated_status": "ok",
                }
            ]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            with open(receipt_path, "w", encoding="utf-8") as output:
                output.write("{}")
            revision: str = "1" * 64
            metadata: dict[str, Any] = {
                "producer_revision": revision,
                "request_parameters": {"protocol": "fixture"},
            }
            receipt: dict[str, Any] = {
                "manifest_sha256": _sha256(manifest_path),
                "request_parameters": metadata["request_parameters"],
                "entries": [
                    {
                        "model": "llama3.1-8b-instruct",
                        "source_population": FULL_POPULATION,
                        "raw_source_format": RAW_SOURCE_FORMATS[FULL_POPULATION],
                        "raw_artifact_sha256": _sha256(input_path),
                        "raw_artifact_size_bytes": os.path.getsize(input_path),
                        "producer_revision_sha256": revision,
                        "availability_status": "complete",
                        "raw_projection_sha256": canonical_projection_digest(
                            completion_payload_projection(payload, manifest)
                        ),
                    }
                ],
            }
            with patch(
                "benchmark_scripts.import_hosted_results.load_completion_receipt",
                return_value=receipt,
            ) as load_receipt_mock:
                reference: dict[str, str] = prepare_completion_audit_receipt(
                    receipt_path,
                    input_path,
                    manifest_path,
                    output_dir,
                    "lambada",
                    "word",
                    "llama3.1-8b-instruct",
                    metadata,
                    "complete",
                )
            load_receipt_mock.assert_called_once_with(
                receipt_path,
                expected_sha256=GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
            )
            self.assertEqual(reference["entry_id"], "lambada/word/llama3.1-8b-instruct")
            self.assertEqual(
                reference["sha256"],
                GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
            )

    def test_rejects_partial_label_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing expected labels.*B"):
            _label_rows(
                {"A": -1.0},
                ("A", "B"),
                prompt_idx=0,
                seg_idx=None,
                kind="orig",
                answer="A",
            )

    def test_rejects_prompt_level_failure_expansion_without_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": True,
                        "seg_idx": index,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": index,
                        "segment_text": f"segment {index}",
                        "n_segments": 2,
                    }
                    for index in range(2)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 0,
                            "answer": True,
                            "n_segments": 0,
                            "orig_label_logprobs": {
                                "true": -1.0,
                                "false": -2.0,
                            },
                            "original_request_status": "ok",
                            "ablated_label_logprobs": None,
                            "segment_request_status": "content_filter",
                        }
                    ],
                    output,
                )

            with self.assertRaisesRegex(ValueError, "JSON n_segments=0 disagrees"):
                import_results(
                    input_path,
                    manifest_path,
                    directory,
                    "hosted",
                    "boolq",
                    "sentence",
                    "Matched against the canonical full-dialog manifest.",
                    PRODUCER_METADATA,
                )

    def test_marks_returned_nonfinite_completion_score_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": prompt_idx,
                        "answer": "target",
                        "seg_idx": 7,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 3,
                        "segment_text": "word",
                        "n_segments": 10,
                    }
                    for prompt_idx in (4, 5)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": "target",
                            "ablation_idx": 7,
                            "n_segments": 10,
                            "orig_logprob": float("nan"),
                            "orig_status": "ok",
                            "ablated_logprob": float("nan"),
                            "ablated_status": "ok",
                        },
                        {
                            "prompt_idx": 5,
                            "answer": "target",
                            "ablation_idx": 7,
                            "n_segments": 10,
                            "orig_logprob": float("inf"),
                            "orig_status": "ok",
                            "ablated_logprob": -float("inf"),
                            "ablated_status": "ok",
                        },
                    ],
                    output,
                )

            import_results(input_path, manifest_path, directory, "hosted")

            segments: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertFalse(segments["original_result_available"].any())
            self.assertFalse(segments["segment_result_available"].any())
            self.assertEqual(
                set(segments["original_result_status"]), {"nonfinite_score"}
            )
            self.assertEqual(
                set(segments["segment_result_status"]), {"nonfinite_score"}
            )

    def test_imports_label_aggregates_with_segment_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": "B",
                        "seg_idx": index,
                        "message_idx": index,
                        "message_role": "system" if index == 0 else "user",
                        "message_seg_idx": 0,
                        "segment_text": f"segment {index}",
                        "n_segments": 2,
                    }
                    for index in range(2)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            payload = [
                {
                    "prompt_idx": 4,
                    "answer": "B",
                    "n_segments": 2,
                    "orig_label_logprobs": {"A": -2.0, "B": -1.0},
                    "ablated_label_logprobs": [
                        {"A": -3.0, "B": -1.0},
                        {"A": None, "B": -2.0},
                    ],
                }
            ]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)

            import_results(input_path, manifest_path, directory, "hosted")

            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            tokens = pd.read_csv(
                os.path.join(directory, "hosted_tokens.tsv.gz"), sep="\t"
            )
            self.assertEqual(segments["message_role"].tolist(), ["system", "user"])
            self.assertEqual(len(tokens), 6)
            self.assertEqual(set(tokens["logprob_granularity"]), {"label_aggregate"})
            missing = tokens[
                (tokens["kind"] == "ablated")
                & (tokens["seg_idx"] == 1)
                & (tokens["label"] == "A")
            ]
            self.assertTrue((missing["logprob"] == -float("inf")).all())

    def test_preserves_explicit_classification_content_filter_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": True,
                        "seg_idx": 0,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 0,
                        "segment_text": "word",
                        "n_segments": 1,
                    }
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            payload: list[dict[str, object]] = [
                {
                    "prompt_idx": 0,
                    "answer": True,
                    "n_segments": 1,
                    "orig_label_logprobs": {
                        "true": -1.0,
                        "false": -2.0,
                    },
                    "original_request_status": "ok",
                    "ablated_label_logprobs": [None],
                    "ablated_request_statuses": ["content_filter"],
                }
            ]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)

            import_results(input_path, manifest_path, directory, "hosted")

            segments: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(segments.loc[0, "original_request_status"], "ok")
            self.assertEqual(
                segments.loc[0, "segment_request_status"], "content_filter"
            )
            self.assertFalse(bool(segments.loc[0, "segment_result_available"]))

            payload[0]["ablated_label_logprobs"] = [{"true": -1.0, "false": -2.0}]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            with self.assertRaisesRegex(ValueError, "disagrees with its payload"):
                import_results(input_path, manifest_path, directory, "hosted")

            payload[0]["ablated_label_logprobs"] = [None]
            payload[0]["orig_label_logprobs"] = None
            payload[0]["original_request_status"] = "content_filter"
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            import_results(input_path, manifest_path, directory, "hosted")
            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(
                segments.loc[0, "original_request_status"], "content_filter"
            )
            self.assertFalse(bool(segments.loc[0, "original_result_available"]))

            payload[0]["orig_label_logprobs"] = {"true": -1.0, "false": -2.0}
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            with self.assertRaisesRegex(ValueError, "disagrees with its payload"):
                import_results(input_path, manifest_path, directory, "hosted")

            payload[0]["original_request_status"] = "ok"
            payload[0]["ablated_request_statuses"] = []
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            with self.assertRaisesRegex(ValueError, "statuses disagree"):
                import_results(input_path, manifest_path, directory, "hosted")

    def test_preserves_literal_na_and_marks_all_missing_labels_unavailable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": False,
                        "seg_idx": 0,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 0,
                        "segment_text": "NA",
                        "n_segments": 1,
                    }
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 0,
                            "answer": False,
                            "n_segments": 1,
                            "orig_label_logprobs": {},
                            "ablated_label_logprobs": [{"true": None, "false": None}],
                        }
                    ],
                    output,
                )
            import_results(
                input_path,
                manifest_path,
                directory,
                "hosted",
                "boolq",
                "sentence",
                "Matched against the canonical full-dialog manifest.",
                PRODUCER_METADATA,
            )

            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"),
                sep="\t",
                keep_default_na=False,
            )
            tokens = pd.read_csv(
                os.path.join(directory, "hosted_tokens.tsv.gz"), sep="\t"
            )
            self.assertEqual(segments.loc[0, "segment_text"], "NA")
            self.assertFalse(bool(segments.loc[0, "original_result_available"]))
            self.assertFalse(bool(segments.loc[0, "segment_result_available"]))
            self.assertEqual(segments.loc[0, "original_request_status"], "ok")
            self.assertEqual(segments.loc[0, "segment_request_status"], "ok")
            self.assertEqual(len(tokens), 4)
            self.assertTrue((tokens["logprob"] == -float("inf")).all())

    def test_rejects_answer_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": "A",
                        "seg_idx": 0,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 0,
                        "segment_text": "text",
                        "n_segments": 1,
                    }
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 0,
                            "answer": "B",
                            "n_segments": 1,
                            "orig_label_logprobs": {"A": -1.0},
                            "ablated_label_logprobs": [{"A": -1.0}],
                        }
                    ],
                    output,
                )

            with self.assertRaisesRegex(ValueError, "disagrees with manifest"):
                import_results(input_path, manifest_path, directory, "hosted")

    def test_imports_subsampled_completion_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": "target",
                        "seg_idx": 7,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": 3,
                        "segment_text": "word",
                        "n_segments": 10,
                    }
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": "target",
                            "ablation_idx": 7,
                            "n_segments": 10,
                            "orig_logprob": -1.0,
                            "orig_status": "ok",
                            "ablated_logprob": -2.0,
                            "ablated_status": "ok",
                        }
                    ],
                    output,
                )

            import_results(input_path, manifest_path, directory, "hosted")

            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(segments.loc[0, "seg_idx"], 7)
            self.assertEqual(segments.loc[0, "orig_completion_logprob"], -1.0)
            self.assertEqual(segments.loc[0, "original_result_status"], "ok")
            self.assertEqual(segments.loc[0, "segment_result_status"], "ok")
            with open(
                os.path.join(directory, "hosted_run.json"), encoding="utf-8"
            ) as source:
                metadata = json.load(source)
            self.assertIn("transformation_source_sha256", metadata)

    def test_unsupported_canary_materializes_all_missing_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "canary.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": "target",
                        "seg_idx": seg_idx,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": seg_idx,
                        "segment_text": f"word {seg_idx}",
                        "n_segments": 2,
                    }
                    for seg_idx in range(2)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": "target",
                            "n_segments": 2,
                            "orig_logprob": -1.0,
                            "ablated_logprob": [-2.0, None],
                        }
                    ],
                    output,
                )
            public_canary_path: str = os.path.join(directory, "hosted_canary.json")
            shutil.copyfile(input_path, public_canary_path)
            canary_metadata: dict[str, Any] = {
                "sample_size": 10,
                "artifact_path": os.path.basename(public_canary_path),
                "artifact_sha256": _sha256(public_canary_path),
            }

            import_results(
                input_path,
                manifest_path,
                directory,
                "hosted",
                "lambada",
                "word",
                "Canary coordinates match the frozen manifest.",
                PRODUCER_METADATA,
                "unsupported_after_canary",
                canary_metadata,
            )

            segments: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(len(segments), 2)
            self.assertTrue(segments["orig_completion_logprob"].isna().all())
            self.assertTrue(segments["ablated_completion_logprob"].isna().all())
            self.assertFalse(segments["original_result_available"].any())
            self.assertFalse(segments["segment_result_available"].any())
            self.assertEqual(
                set(segments["original_result_status"]),
                {"unsupported_after_canary"},
            )
            self.assertEqual(
                set(segments["segment_result_status"]),
                {"unsupported_after_canary"},
            )
            with open(
                os.path.join(directory, "hosted_run.json"), encoding="utf-8"
            ) as source:
                metadata: dict[str, object] = json.load(source)
            self.assertEqual(
                metadata["source_format"],
                "unsupported_completion_placeholder_after_canary",
            )

            with self.assertRaisesRegex(
                ValueError, "input must be the audited canary artifact"
            ):
                with open(input_path, "w", encoding="utf-8") as output:
                    json.dump(
                        [
                            {
                                "prompt_idx": 4,
                                "answer": "different target",
                                "n_segments": 2,
                                "orig_logprob": -1.0,
                                "ablated_logprob": [-2.0, None],
                            }
                        ],
                        output,
                    )
                import_results(
                    input_path,
                    manifest_path,
                    directory,
                    "hosted",
                    "lambada",
                    "word",
                    "Canary coordinates match the frozen manifest.",
                    PRODUCER_METADATA,
                    "unsupported_after_canary",
                    canary_metadata,
                )

    def test_imports_prompt_level_completion_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": prompt_idx,
                        "answer": f"target-{prompt_idx}",
                        "seg_idx": seg_idx,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": seg_idx,
                        "segment_text": f"word {seg_idx}",
                        "n_segments": 3,
                    }
                    for prompt_idx in (4, 9)
                    for seg_idx in range(3)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": prompt_idx,
                            "answer": f"target-{prompt_idx}",
                            "n_segments": 3,
                            "orig_logprob": -1.0,
                            "ablated_logprob": [-2.0, None, -4.0],
                        }
                        for prompt_idx in (4, 9)
                    ]
                    + [
                        {
                            "prompt_idx": 99,
                            "answer": "source-only-extra",
                            "n_segments": 1,
                            "orig_logprob": -1.0,
                            "ablated_logprob": [-2.0],
                        }
                    ],
                    output,
                )

            import_results(input_path, manifest_path, directory, "hosted")

            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(len(segments), 6)
            self.assertEqual(segments["seg_idx"].tolist(), [0, 1, 2, 0, 1, 2])
            self.assertEqual(segments["original_result_available"].tolist(), [True] * 6)
            self.assertEqual(
                segments["segment_result_available"].tolist(),
                [True, False, True, True, False, True],
            )

    def test_prompt_level_completion_preserves_failure_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": "target",
                        "seg_idx": seg_idx,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": seg_idx,
                        "segment_text": f"word {seg_idx}",
                        "n_segments": 3,
                    }
                    for seg_idx in range(3)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": "target",
                            "n_segments": 3,
                            "orig_logprob": None,
                            "orig_status": "application_error",
                            "ablated_logprob": [None, -2.0, -4.0],
                            "ablated_statuses": [
                                "target_parse_unavailable",
                                "ok",
                                "ok",
                            ],
                        }
                    ],
                    output,
                )

            import_results(input_path, manifest_path, directory, "hosted")

            segments: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(
                set(segments["original_result_status"]), {"application_error"}
            )
            self.assertEqual(
                segments["segment_result_status"].tolist(),
                ["target_parse_unavailable", "ok", "ok"],
            )

    def test_prompt_level_completion_uses_explicit_sampled_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": "target",
                        "seg_idx": seg_idx,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": seg_idx,
                        "segment_text": f"word {seg_idx}",
                        "n_segments": 10,
                    }
                    for seg_idx in (2, 7)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": "target",
                            "n_segments": 10,
                            "ablation_indices": [2, 7],
                            "orig_logprob": -1.0,
                            "ablated_logprob": [-2.0, -3.0],
                        }
                    ],
                    output,
                )

            import_results(input_path, manifest_path, directory, "hosted")

            segments = pd.read_csv(
                os.path.join(directory, "hosted_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(segments["seg_idx"].tolist(), [2, 7])
            self.assertEqual(
                segments["ablated_completion_logprob"].tolist(), [-2.0, -3.0]
            )

    def test_imports_sampled_label_rows_at_manifest_indices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            input_path: str = os.path.join(directory, "input.json")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 4,
                        "answer": False,
                        "seg_idx": seg_idx,
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": seg_idx,
                        "segment_text": f"word {seg_idx}",
                        "n_segments": 10,
                    }
                    for seg_idx in (2, 7)
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 4,
                            "answer": False,
                            "n_segments": 10,
                            "ablation_indices": [2, 7],
                            "orig_label_logprobs": {
                                "true": -2.0,
                                "false": -1.0,
                            },
                            "ablated_label_logprobs": [
                                {"true": -3.0, "false": -1.0},
                                {"true": -4.0, "false": -1.0},
                            ],
                        }
                    ],
                    output,
                )

            import_results(
                input_path,
                manifest_path,
                directory,
                "hosted",
                "boolq",
                "word",
                "Matched to the canonical sampled manifest.",
                PRODUCER_METADATA,
            )

            tokens = pd.read_csv(
                os.path.join(directory, "hosted_tokens.tsv.gz"), sep="\t"
            )
            ablated_indices = set(
                tokens.loc[tokens["kind"] == "ablated", "seg_idx"].astype(int)
            )
            self.assertEqual(ablated_indices, {2, 7})
