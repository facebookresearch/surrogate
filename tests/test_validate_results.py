# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import hashlib
import json
import os
import tempfile
from typing import Any, cast
from unittest import TestCase
from unittest.mock import AsyncMock, patch

import pandas as pd

from benchmark_scripts.derived_provenance import (
    collect_result_inputs,
    write_derived_provenance,
)
from benchmark_scripts.f_table import OPEN_MODELS
from benchmark_scripts.hosted_completion import (
    LAMBADA_GOLD_CANARY,
    LAMBADA_GOLD_CANARY_ANSWERS,
    summarize_completion_canary,
)
from benchmark_scripts.hosted_audit_receipt import (
    CLASSIFICATION_CONFIGURATIONS,
    CONFIGURATION_IDENTITY_SHA256,
)
from benchmark_scripts.hosted_completion_audit_receipt import (
    CANARY_POPULATION,
    canonical_projection_digest as canonical_completion_projection_digest,
    completion_payload_projection,
    completion_table_projection,
    projection_summary as completion_projection_summary,
)
from benchmark_scripts.import_hosted_results import import_results
from benchmark_scripts.provenance_sources import (
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_IDENTITY_ATTESTATION,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    HOSTED_RECORD_SOURCE_FILES,
    canonical_file_hash_manifest_sha256,
)
from benchmark_scripts.validate_results import (
    DERIVED_SUPPORTING_SOURCE_FILES,
    HOSTED_CLASSIFICATION_REQUEST_PARAMETERS,
    OPEN_SEGMENT_COLUMNS,
    _allowed_release_files,
    _require_pair_grid,
    _read_tsv,
    _sha256,
    _token_coverage,
    _validate_artifact_manifest,
    _validate_derived_sidecar,
    _validate_f_table_parameters,
    _validate_gold_terminal_failure_counts,
    _validate_gold_hosted_metadata,
    _validate_gold_run_metadata_schema,
    _validate_hosted_audit_binding,
    _validate_hosted_completion_audit_binding,
    _validate_hosted_dialog_identities,
    _validate_manifest,
    _validate_open_model_identity,
    _validate_open_model_identity_consistency,
    _validate_open_segment_metrics,
    _validate_release_inventory,
    _validate_segment_file,
    _write_artifact_manifest,
    validate_configuration,
)


class TestValidateResults(TestCase):
    def test_read_tsv_preserves_decimal_float_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "scores.tsv")
            with open(path, "w", encoding="utf-8") as output:
                output.write("score\n-10.702245578169823\n")
            frame: pd.DataFrame = _read_tsv(path)
            self.assertEqual(
                float(frame.loc[0, "score"]).hex(), "-0x1.5678cbb800000p+3"
            )

    def test_all_hosted_receipt_consumers_require_the_gold_digest(self) -> None:
        configurations: list[dict[str, str]] = [
            {
                "benchmark": benchmark,
                "pregrouper": pregrouper,
                "prompt_identity_sha256": CONFIGURATION_IDENTITY_SHA256[
                    (benchmark, pregrouper)
                ][0],
                "ablated_dialog_identity_sha256": CONFIGURATION_IDENTITY_SHA256[
                    (benchmark, pregrouper)
                ][1],
            }
            for benchmark, pregrouper in CLASSIFICATION_CONFIGURATIONS
        ]
        receipt: dict[str, Any] = {
            "configurations": configurations,
            "entries": [],
        }
        identities: list[tuple[str, str]] = [
            CONFIGURATION_IDENTITY_SHA256[configuration]
            for configuration in CLASSIFICATION_CONFIGURATIONS
        ]
        with (
            patch(
                "benchmark_scripts.validate_results.load_receipt",
                return_value=receipt,
            ) as load_receipt_mock,
            patch(
                "benchmark_scripts.validate_results.compute_dialog_identity",
                new=AsyncMock(side_effect=identities),
            ),
        ):
            _validate_hosted_dialog_identities("/results", "/datasets")
            load_receipt_mock.assert_called_once_with(
                "/results/hosted_classification_audit_receipt.json",
                expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
            )

        completion_provenance: dict[str, Any] = {
            "producer_audit_receipt": {
                "path": "hosted_completion_audit_receipt.json",
                "sha256": GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
                "entry_id": "lambada/word/gpt-4-1",
            }
        }
        completion_receipt: dict[str, Any] = {
            "manifest_sha256": "manifest",
            "entries": [],
        }
        with (
            patch(
                "benchmark_scripts.validate_results.load_completion_receipt",
                return_value=completion_receipt,
            ) as completion_load_mock,
            patch(
                "benchmark_scripts.validate_results._sha256",
                return_value="manifest",
            ),
        ):
            with self.assertRaises(StopIteration):
                _validate_hosted_completion_audit_binding(
                    "/results",
                    "gpt-4-1",
                    "/manifest.tsv.gz",
                    "/segment.tsv.gz",
                    pd.DataFrame(),
                    completion_provenance,
                )
            completion_load_mock.assert_called_once_with(
                "/results/hosted_completion_audit_receipt.json",
                expected_sha256=GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
            )

        provenance: dict[str, Any] = {
            "producer_audit_receipt": {
                "path": "hosted_classification_audit_receipt.json",
                "sha256": GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
                "entry_id": "boolq/sentence/gpt-4-1",
            }
        }
        receipt["configurations"] = []
        with patch(
            "benchmark_scripts.validate_results.load_receipt",
            return_value=receipt,
        ) as load_receipt_mock:
            with self.assertRaises(StopIteration):
                _validate_hosted_audit_binding(
                    "/results",
                    "boolq",
                    "sentence",
                    "gpt-4-1",
                    "/manifest.tsv.gz",
                    pd.DataFrame(),
                    "/tokens.tsv.gz",
                    provenance,
                )
            load_receipt_mock.assert_called_once_with(
                "/results/hosted_classification_audit_receipt.json",
                expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
            )

    def test_gold_hosted_metadata_is_sanitized_and_structured(self) -> None:
        provenance: dict[str, Any] = {
            "identity_attestation": GOLD_HOSTED_IDENTITY_ATTESTATION
        }
        producer: dict[str, Any] = {
            "producer_revision": "1" * 64,
            "generated_at": "2026-09-09T12:34:56+00:00",
            "served_model": "gpt-4-1",
        }
        _validate_gold_hosted_metadata(provenance, producer, "gpt-4-1", "run.json")
        mutations: tuple[tuple[str, str, object], ...] = (
            ("provenance", "identity_attestation", "/private/producer/path"),
            ("producer", "producer_revision", "FIXTURE"),
            ("producer", "generated_at", "yesterday"),
            ("producer", "served_model", "private-routing-alias"),
        )
        for owner, field, value in mutations:
            with self.subTest(field=field):
                altered_provenance: dict[str, Any] = dict(provenance)
                altered_producer: dict[str, Any] = dict(producer)
                target: dict[str, Any] = (
                    altered_provenance if owner == "provenance" else altered_producer
                )
                target[field] = value
                with self.assertRaises(ValueError):
                    _validate_gold_hosted_metadata(
                        altered_provenance,
                        altered_producer,
                        "gpt-4-1",
                        "run.json",
                    )

    def test_completion_receipt_rejects_score_changed_canary(self) -> None:
        fixed_manifest: pd.DataFrame = pd.DataFrame(
            [
                {
                    "prompt_idx": prompt_idx,
                    "seg_idx": seg_idx,
                    "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                    "n_segments": n_segments,
                }
                for prompt_idx, n_segments in LAMBADA_GOLD_CANARY
                for seg_idx in range(n_segments)
            ]
        )
        canary_payload: list[dict[str, Any]] = [
            {
                "prompt_idx": prompt_idx,
                "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                "n_segments": n_segments,
                "orig_logprob": -1.0 if index == 0 else None,
                "ablated_logprob": [
                    -2.0 if index == 0 and seg_idx == 0 else None
                    for seg_idx in range(n_segments)
                ],
            }
            for index, (prompt_idx, n_segments) in enumerate(LAMBADA_GOLD_CANARY)
        ]
        prompt_indices: list[int] = [
            prompt_idx for prompt_idx, _ in LAMBADA_GOLD_CANARY
        ]
        raw_projection: list[list[Any]] = completion_payload_projection(
            canary_payload,
            fixed_manifest,
            expected_prompt_indices=prompt_indices,
        )
        raw_summary = completion_projection_summary(raw_projection)
        public_frame: pd.DataFrame = pd.DataFrame(
            [
                {
                    "prompt_idx": 0,
                    "seg_idx": 0,
                    "orig_completion_logprob": None,
                    "ablated_completion_logprob": None,
                    "original_result_status": "unsupported_after_canary",
                    "segment_result_status": "unsupported_after_canary",
                }
            ]
        )
        public_projection: list[list[Any]] = completion_table_projection(
            public_frame.to_dict("records")
        )
        public_summary = completion_projection_summary(public_projection)
        with tempfile.TemporaryDirectory() as directory:
            config_dir: str = os.path.join(directory, "lambada", "word")
            os.makedirs(config_dir)
            manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
            segment_path: str = os.path.join(config_dir, "gpt-4-1_segment.tsv.gz")
            canary_path: str = os.path.join(config_dir, "gpt-4-1_canary.json")
            fixed_manifest.iloc[:1].to_csv(
                manifest_path, sep="\t", index=False, compression="gzip"
            )
            public_frame.to_csv(segment_path, sep="\t", index=False, compression="gzip")
            canary_payload[0]["orig_logprob"] = -3.0
            with open(canary_path, "w", encoding="utf-8") as output:
                json.dump(canary_payload, output)
            revision: str = "1" * 64
            entry: dict[str, Any] = {
                "model": "gpt-4-1",
                "source_population": CANARY_POPULATION,
                "raw_source_format": "hosted_completion_canary_logprob_json",
                "public_source_format": (
                    "unsupported_completion_placeholder_after_canary"
                ),
                "raw_artifact_sha256": "2" * 64,
                "raw_artifact_size_bytes": 123,
                "producer_revision_sha256": revision,
                "availability_status": "unsupported_after_canary",
                "public_artifact_sha256": _sha256(segment_path),
                "public_artifact_size_bytes": os.path.getsize(segment_path),
                "raw_projection_sha256": canonical_completion_projection_digest(
                    raw_projection
                ),
                "public_projection_sha256": canonical_completion_projection_digest(
                    public_projection
                ),
                "raw_prompt_count": raw_summary["prompt_count"],
                "raw_segment_count": raw_summary["segment_count"],
                "raw_original_available_count": raw_summary["original_available_count"],
                "raw_ablated_available_count": raw_summary["ablated_available_count"],
                "raw_paired_attribution_available_count": raw_summary[
                    "paired_attribution_available_count"
                ],
                "public_prompt_count": public_summary["prompt_count"],
                "public_segment_count": public_summary["segment_count"],
                "public_original_available_count": public_summary[
                    "original_available_count"
                ],
                "public_ablated_available_count": public_summary[
                    "ablated_available_count"
                ],
                "public_paired_attribution_available_count": public_summary[
                    "paired_attribution_available_count"
                ],
            }
            receipt: dict[str, Any] = {
                "manifest_sha256": _sha256(manifest_path),
                "request_parameters": {},
                "entries": [entry],
            }
            provenance: dict[str, Any] = {
                "producer_audit_receipt": {
                    "path": "hosted_completion_audit_receipt.json",
                    "sha256": GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
                    "entry_id": "lambada/word/gpt-4-1",
                },
                "source_sha256": entry["raw_artifact_sha256"],
                "source_size_bytes": entry["raw_artifact_size_bytes"],
                "producer": {
                    "producer_revision": revision,
                    "request_parameters": {},
                },
                "availability_status": "unsupported_after_canary",
                "source_format": entry["public_source_format"],
                "canary": {
                    "artifact_path": os.path.basename(canary_path),
                    "artifact_sha256": _sha256(canary_path),
                },
            }
            with patch(
                "benchmark_scripts.validate_results.load_completion_receipt",
                return_value=receipt,
            ):
                with self.assertRaisesRegex(ValueError, "canary projection disagrees"):
                    _validate_hosted_completion_audit_binding(
                        directory,
                        "gpt-4-1",
                        manifest_path,
                        segment_path,
                        public_frame,
                        provenance,
                    )

    def test_gold_run_metadata_schema_is_closed(self) -> None:
        model: str = "qwen2.5-0.5b-instruct"
        provenance: dict[str, Any] = {
            "artifact_sha256": {"segment": "0" * 64, "tokens": "1" * 64},
            "benchmark": "boolq",
            "dataset": {
                "hf_name": "boolq",
                "hf_path": "aps/super_glue",
                "hf_split": "validation",
                "normalized_frame_sha256": "2" * 64,
                "rows": 3270,
                "snapshot_filename": "google_boolq_validation.tsv",
                "snapshot_sha256": (
                    "80040aa10f18e5b01082386dae3bdde48931a0311e807f6cb10f7173995f346a"
                ),
            },
            "execution_model_source": "Qwen/Qwen2.5-0.5B-Instruct",
            "execution_source_hash_timing": "run_completion",
            "manifest_sha256": "3" * 64,
            "model": model,
            "model_artifact_sha256": {"model.safetensors": "4" * 64},
            "model_artifact_manifest_sha256": "7" * 64,
            "model_artifact_hash_timing": "post_run_seal",
            "model_identity_files_sha256": {"config.json": "5" * 64},
            "model_revision": "8" * 40,
            "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
            "parameters": {
                "batch_size": 8,
                "max_forward_passes": None,
                "max_samples": None,
                "phase_attention_implementation": {
                    "ablation": "sdpa",
                    "attention": "eager",
                },
                "phases": ["ablation", "attention"],
                "seed": 42,
            },
            "pregrouper": "sentence",
            "provenance_seal_sha256": "6" * 64,
            "release_source_sha256": {},
            "release_source_corrections": {},
            "schema_version": 3,
            "segmentation_scope": "full_dialog_in_message_order",
            "software": {
                "cuda_runtime": "13.0",
                "numpy": "2.2.1",
                "pandas": "2.2.3",
                "torch": "2.15.0a0+fb",
                "transformers": "4.0.0",
            },
            "source_sha256": {},
        }
        _validate_gold_run_metadata_schema(provenance, model, "boolq", "", "run.json")
        provenance["internal_path"] = "/private/service"
        with self.assertRaisesRegex(ValueError, "metadata fields disagree"):
            _validate_gold_run_metadata_schema(
                provenance, model, "boolq", "", "run.json"
            )
        del provenance["internal_path"]
        provenance["model_source"] = "private-routing-alias"
        with self.assertRaisesRegex(ValueError, "model source disagrees"):
            _validate_gold_run_metadata_schema(
                provenance, model, "boolq", "", "run.json"
            )

        hosted: dict[str, Any] = {
            "artifact_sha256": {},
            "availability_status": "complete",
            "benchmark": "boolq",
            "canary": None,
            "identity_attestation": "public",
            "logprob_granularity": "label",
            "manifest_sha256": "1" * 64,
            "model": "gpt-4o",
            "pregrouper": "sentence",
            "producer": {},
            "producer_audit_receipt": {},
            "prompts": 3270,
            "schema_version": 1,
            "segmentation_scope": "full_dialog_in_message_order",
            "segments": 27516,
            "source_format": "hosted_label_logprob_json",
            "source_sha256": "2" * 64,
            "source_size_bytes": 1,
            "transformation_source_sha256": {},
        }
        _validate_gold_run_metadata_schema(
            hosted,
            "gpt-4o",
            "boolq",
            "hosted_label_logprob_json",
            "hosted_run.json",
        )
        hosted["benchmark"] = "lambada"
        hosted["pregrouper"] = "word"
        hosted["segments"] = 10_000
        hosted["prompts"] = 4_387
        with self.assertRaisesRegex(ValueError, "source format is invalid"):
            _validate_gold_run_metadata_schema(
                hosted,
                "gpt-4o",
                "lambada",
                "hosted_label_logprob_json",
                "hosted_run.json",
            )
        hosted["benchmark"] = "boolq"
        hosted["pregrouper"] = "sentence"
        hosted["segments"] = 27_516
        hosted["prompts"] = 3_270
        hosted["canary"] = {"private_endpoint": "secret"}
        with self.assertRaisesRegex(ValueError, "canary metadata must be null"):
            _validate_gold_run_metadata_schema(
                hosted,
                "gpt-4o",
                "boolq",
                "hosted_label_logprob_json",
                "hosted_run.json",
            )

    def test_open_model_identity_is_pinned_and_consistent_across_cells(self) -> None:
        model: str = "qwen2.5-0.5b-instruct"
        artifact_hashes: dict[str, str] = {
            "config.json": "1" * 64,
            "model.safetensors": "2" * 64,
        }
        aggregate: str = canonical_file_hash_manifest_sha256(artifact_hashes)
        provenance: dict[str, Any] = {
            "execution_model_source": GOLD_OPEN_MODEL_REPOSITORIES[model],
            "model_source": GOLD_OPEN_MODEL_REPOSITORIES[model],
            "model_revision": GOLD_OPEN_MODEL_REVISIONS[model],
            "model_artifact_manifest_sha256": aggregate,
            "model_artifact_sha256": artifact_hashes,
            "model_identity_files_sha256": {"config.json": "1" * 64},
        }
        with patch.dict(GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256, {model: aggregate}):
            _validate_open_model_identity(provenance, model, "run.json")
            with tempfile.TemporaryDirectory() as results_dir:
                for benchmark, pregrouper in (
                    ("boolq", "sentence"),
                    ("lambada", "word"),
                ):
                    directory: str = os.path.join(results_dir, benchmark, pregrouper)
                    os.makedirs(directory)
                    with open(
                        os.path.join(directory, f"{model}_run.json"),
                        "w",
                        encoding="utf-8",
                    ) as output:
                        json.dump(provenance, output)
                _validate_open_model_identity_consistency(results_dir)
                second_path: str = os.path.join(
                    results_dir, "lambada", "word", f"{model}_run.json"
                )
                altered: dict[str, Any] = dict(provenance)
                altered["model_revision"] = "0" * 40
                with open(second_path, "w", encoding="utf-8") as output:
                    json.dump(altered, output)
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    _validate_open_model_identity_consistency(results_dir)

    def test_every_open_segment_metric_must_be_finite(self) -> None:
        complete: pd.DataFrame = pd.DataFrame(
            {column: [0.0] for column in OPEN_SEGMENT_COLUMNS}
        )
        _validate_open_segment_metrics(complete, "open-model")
        for column in OPEN_SEGMENT_COLUMNS:
            with self.subTest(column=column):
                corrupted: pd.DataFrame = complete.copy()
                corrupted.loc[0, column] = float("nan")
                with self.assertRaisesRegex(ValueError, repr(column)):
                    _validate_open_segment_metrics(corrupted, "open-model")

    def test_gold_terminal_failure_claim_is_fixed(self) -> None:
        _validate_gold_terminal_failure_counts("race", "sentence", "gpt-4o", 0, 110)
        with self.assertRaisesRegex(ValueError, r"expected \(0, 110\)"):
            _validate_gold_terminal_failure_counts("race", "sentence", "gpt-4o", 0, 109)
        _validate_gold_terminal_failure_counts("boolq", "sentence", "gpt-4o", 7, 9)
        _validate_gold_terminal_failure_counts("race", "sentence", "gpt-4-1", 0, 890)

    def test_gold_hosted_audit_requires_complete_diagnostic_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            directory: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(directory)
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
            ).to_csv(os.path.join(directory, "segments.tsv.gz"), sep="\t", index=False)
            with self.assertRaisesRegex(
                ValueError, "missing models.*llama3.1-8b-instruct"
            ):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=(),
                    require_gold_manifest=False,
                    require_hosted_audit=True,
                )

    def test_complete_hosted_classification_rejects_failed_calls(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            directory: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(directory)
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
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
            input_path: str = os.path.join(results_dir, "hosted.json")
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 0,
                            "answer": True,
                            "n_segments": 1,
                            "orig_label_logprobs": {"true": -1.0, "false": -2.0},
                            "ablated_label_logprobs": [None],
                        }
                    ],
                    output,
                )
            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "boolq",
                "sentence",
                "Every coordinate was attempted against the manifest.",
                {
                    "producer_revision": "fixture-revision",
                    "generated_at": "2026-09-08T00:00:00Z",
                    "served_model": "gpt-4-1",
                    "request_parameters": dict(
                        HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                    ),
                },
            )
            with self.assertRaisesRegex(ValueError, "failed calls"):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
                        {
                            "prompt_idx": 0,
                            "answer": True,
                            "n_segments": 1,
                            "orig_label_logprobs": {"true": -1.0, "false": -2.0},
                            "original_request_status": "ok",
                            "ablated_label_logprobs": [None],
                            "ablated_request_statuses": ["transient_exhausted"],
                        }
                    ],
                    output,
                )
            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "boolq",
                "sentence",
                "Every coordinate was attempted against the manifest.",
                {
                    "producer_revision": "fixture-revision",
                    "generated_at": "2026-09-08T00:00:00Z",
                    "served_model": "gpt-4-1",
                    "request_parameters": dict(
                        HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                    ),
                },
                availability_status="complete_with_terminal_failures",
            )
            summaries = validate_configuration(
                results_dir,
                "boolq",
                "sentence",
                required_models=("gpt-4-1",),
                require_gold_manifest=False,
            )
            self.assertEqual(summaries[0].original_terminal_failure_count, 0)
            self.assertEqual(summaries[0].ablated_terminal_failure_count, 1)
            terminal_segments: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "gpt-4-1_segment.tsv.gz"), sep="\t"
            )
            self.assertFalse(bool(terminal_segments.loc[0, "segment_result_available"]))
            terminal_tokens: pd.DataFrame = pd.read_csv(
                os.path.join(directory, "gpt-4-1_tokens.tsv.gz"), sep="\t"
            )
            self.assertTrue(
                (
                    terminal_tokens.loc[
                        terminal_tokens["kind"].eq("ablated"), "logprob"
                    ]
                    == -float("inf")
                ).all()
            )
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(
                    [
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
                    ],
                    output,
                )
            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "boolq",
                "sentence",
                "Every coordinate was attempted against the manifest.",
                {
                    "producer_revision": "fixture-revision",
                    "generated_at": "2026-09-08T00:00:00Z",
                    "served_model": "gpt-4-1",
                    "request_parameters": dict(
                        HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                    ),
                },
            )
            summaries = validate_configuration(
                results_dir,
                "boolq",
                "sentence",
                required_models=("gpt-4-1",),
                require_gold_manifest=False,
            )
            self.assertEqual(summaries[0].original_content_filter_count, 0)
            self.assertEqual(summaries[0].ablated_content_filter_count, 1)
            run_path: str = os.path.join(directory, "gpt-4-1_run.json")
            segment_path: str = os.path.join(directory, "gpt-4-1_segment.tsv.gz")
            token_path: str = os.path.join(directory, "gpt-4-1_tokens.tsv.gz")
            with open(run_path, encoding="utf-8") as source:
                run: dict[str, object] = json.load(source)
            run_producer: dict[str, object] = cast(dict[str, object], run["producer"])
            request_parameters: dict[str, object] = cast(
                dict[str, object], run_producer["request_parameters"]
            )
            request_parameters["top_logprobs"] = 20
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "classification protocol"):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            request_parameters["top_logprobs"] = 19
            artifact_hashes: dict[str, object] = cast(
                dict[str, object], run["artifact_sha256"]
            )
            segment: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
            segment.loc[0, "segment_request_status"] = "transient_exhausted"
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "failed calls"):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )

            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "boolq",
                "sentence",
                "Every coordinate was attempted against the manifest.",
                {
                    "producer_revision": "fixture-revision",
                    "generated_at": "2026-09-08T00:00:00Z",
                    "served_model": "gpt-4-1",
                    "request_parameters": dict(
                        HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                    ),
                },
            )
            with open(run_path, encoding="utf-8") as source:
                run = json.load(source)
            artifact_hashes = cast(dict[str, object], run["artifact_sha256"])
            segment = pd.read_csv(segment_path, sep="\t")
            tokens: pd.DataFrame = pd.read_csv(
                token_path,
                sep="\t",
                dtype={"label": str, "token": str, "kind": str},
            )
            segment.loc[0, "segment_result_available"] = True
            tokens.loc[tokens["kind"] == "ablated", "logprob"] = [-1.0, -2.0]
            segment.to_csv(segment_path, sep="\t", index=False)
            tokens.to_csv(token_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            artifact_hashes["tokens"] = _sha256(token_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "failed-call status"):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )

            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "boolq",
                "sentence",
                "Every coordinate was attempted against the manifest.",
                {
                    "producer_revision": "fixture-revision",
                    "generated_at": "2026-09-08T00:00:00Z",
                    "served_model": "gpt-4-1",
                    "request_parameters": dict(
                        HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                    ),
                },
            )
            with open(run_path, encoding="utf-8") as source:
                run = json.load(source)
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            run["source_format"] = "precomputed_portable_tsv"
            run["source_revision"] = "fixture-revision"
            run["transformation_source_sha256"] = {
                relative_path: _sha256(os.path.join(repository_root, relative_path))
                for relative_path in HOSTED_RECORD_SOURCE_FILES
            }
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "raw-JSON request status"):
                validate_configuration(
                    results_dir,
                    "boolq",
                    "sentence",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )

    def test_complete_hosted_completion_protocol_is_strictly_validated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            directory: str = os.path.join(results_dir, "lambada", "word")
            os.makedirs(directory)
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            manifest: pd.DataFrame = pd.DataFrame(
                {
                    "prompt_idx": range(10_000),
                    "answer": ["target"] * 10_000,
                    "seg_idx": [0] * 10_000,
                    "message_idx": [1] * 10_000,
                    "message_role": ["user"] * 10_000,
                    "message_seg_idx": [0] * 10_000,
                    "segment_text": ["word"] * 10_000,
                    "n_segments": [1] * 10_000,
                }
            )
            manifest.to_csv(manifest_path, sep="\t", index=False)
            input_path: str = os.path.join(results_dir, "hosted.json")
            payload: list[dict[str, object]] = [
                {
                    "prompt_idx": prompt_idx,
                    "answer": "target",
                    "ablation_idx": 0,
                    "n_segments": 1,
                    "orig_logprob": -1.0,
                    "ablated_logprob": -2.0,
                }
                for prompt_idx in range(10_000)
            ]
            with open(input_path, "w", encoding="utf-8") as output:
                json.dump(payload, output)
            producer: dict[str, object] = {
                "producer_revision": "fixture-revision",
                "generated_at": "2026-09-08T00:00:00Z",
                "served_model": "gpt-4-1",
                "request_parameters": {
                    "max_tokens": 0,
                    "echo": True,
                    "scoring": "teacher_forced_echo_target_logprob_sum",
                    "top_logprobs": 20,
                    "max_transient_attempts": 5,
                },
            }
            import_results(
                input_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "lambada",
                "word",
                "Every scored coordinate was verified against the manifest.",
                producer,
            )
            validate_configuration(
                results_dir,
                "lambada",
                "word",
                required_models=("gpt-4-1",),
                require_gold_manifest=False,
            )
            run_path: str = os.path.join(directory, "gpt-4-1_run.json")
            with open(run_path, encoding="utf-8") as source:
                run: dict[str, object] = json.load(source)
            segment_path: str = os.path.join(directory, "gpt-4-1_segment.tsv.gz")
            segment: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
            artifact_hashes: dict[str, object] = cast(
                dict[str, object], run["artifact_sha256"]
            )
            run["availability_status"] = "complete_with_terminal_failures"
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "only for hosted classification"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            run["availability_status"] = "complete"
            segment.loc[0, "ablated_completion_logprob"] = float("nan")
            segment.loc[0, "segment_result_available"] = False
            segment.loc[0, "segment_result_status"] = "nonfinite_score"
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            validate_configuration(
                results_dir,
                "lambada",
                "word",
                required_models=("gpt-4-1",),
                require_gold_manifest=False,
            )
            segment.loc[0, "ablated_completion_logprob"] = -2.0
            segment.loc[0, "segment_result_available"] = True
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "statuses disagree"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            segment.loc[0, "segment_result_status"] = "ok"
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            run_producer: dict[str, object] = cast(dict[str, object], run["producer"])
            request_parameters: dict[str, object] = cast(
                dict[str, object], run_producer["request_parameters"]
            )
            request_parameters["echo"] = False
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "completion protocol"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            request_parameters["echo"] = True
            segment.loc[0, "ablated_completion_logprob"] = float("nan")
            segment.loc[0, "segment_result_available"] = False
            segment.loc[0, "segment_result_status"] = "transport_error"
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes["segment"] = _sha256(segment_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "failed-call status"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )

    def test_unsupported_completion_placeholder_is_strictly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            directory: str = os.path.join(results_dir, "lambada", "word")
            os.makedirs(directory)
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            pd.DataFrame(
                {
                    "prompt_idx": range(10_000),
                    "answer": ["target"] * 10_000,
                    "seg_idx": [0] * 10_000,
                    "message_idx": [1] * 10_000,
                    "message_role": ["user"] * 10_000,
                    "message_seg_idx": [0] * 10_000,
                    "segment_text": ["word"] * 10_000,
                    "n_segments": [1] * 10_000,
                }
            ).to_csv(manifest_path, sep="\t", index=False)
            canary_path: str = os.path.join(directory, "gpt-4-1_canary.json")
            canary_payload: list[dict[str, Any]] = [
                {
                    "prompt_idx": prompt_idx,
                    "answer": LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx],
                    "n_segments": n_segments,
                    "orig_logprob": None,
                    "ablated_logprob": [None] * n_segments,
                }
                for prompt_idx, n_segments in LAMBADA_GOLD_CANARY
            ]
            with open(canary_path, "w", encoding="utf-8") as output:
                json.dump(canary_payload, output)
            canary_metadata: dict[str, object] = {
                "artifact_path": "gpt-4-1_canary.json",
                "artifact_sha256": _sha256(canary_path),
                **summarize_completion_canary(canary_payload, 42),
            }
            producer: dict[str, object] = {
                "producer_revision": "fixture-revision",
                "generated_at": "2026-09-08T00:00:00Z",
                "served_model": "gpt-4-1",
                "request_parameters": {
                    "max_tokens": 0,
                    "echo": True,
                    "scoring": "teacher_forced_echo_target_logprob_sum",
                    "top_logprobs": 20,
                    "max_transient_attempts": 5,
                },
            }
            import_results(
                canary_path,
                manifest_path,
                directory,
                "gpt-4-1",
                "lambada",
                "word",
                "The fixed canary and canonical manifest were verified.",
                producer,
                "unsupported_after_canary",
                canary_metadata,
            )

            validate_configuration(
                results_dir,
                "lambada",
                "word",
                required_models=("gpt-4-1",),
                require_gold_manifest=False,
            )
            run_path: str = os.path.join(directory, "gpt-4-1_run.json")
            with open(run_path, encoding="utf-8") as source:
                run: dict[str, object] = json.load(source)
            run_producer: dict[str, object] = cast(dict[str, object], run["producer"])
            request_parameters: dict[str, object] = cast(
                dict[str, object], run_producer["request_parameters"]
            )
            request_parameters["top_logprobs"] = 19
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "canary protocol"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            request_parameters["top_logprobs"] = 20

            run_canary: dict[str, object] = cast(dict[str, object], run["canary"])
            run_canary["private_path"] = "/private/canary"
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "canary evidence"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            del run_canary["private_path"]

            run_producer["private_path"] = "/private/producer"
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "producer metadata"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            del run_producer["private_path"]

            original_transformation_hashes: object = run["transformation_source_sha256"]
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            run["source_format"] = "precomputed_portable_tsv"
            run["transformation_source_sha256"] = {
                relative_path: _sha256(os.path.join(repository_root, relative_path))
                for relative_path in HOSTED_RECORD_SOURCE_FILES
            }
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "raw-JSON request status evidence"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            run["source_format"] = "unsupported_completion_placeholder_after_canary"
            run["transformation_source_sha256"] = original_transformation_hashes

            canary_payload[0]["private_path"] = "/private/canary-row"
            with open(canary_path, "w", encoding="utf-8") as output:
                json.dump(canary_payload, output)
            run_canary["artifact_sha256"] = _sha256(canary_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "Canary artifact contents"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )
            del canary_payload[0]["private_path"]
            with open(canary_path, "w", encoding="utf-8") as output:
                json.dump(canary_payload, output)
            run_canary["artifact_sha256"] = _sha256(canary_path)

            segment_path: str = os.path.join(directory, "gpt-4-1_segment.tsv.gz")
            segment: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
            segment.loc[0, "orig_completion_logprob"] = -1.0
            segment.loc[0, "original_result_available"] = True
            segment.loc[0, "original_result_status"] = "ok"
            segment.to_csv(segment_path, sep="\t", index=False)
            artifact_hashes: dict[str, object] = cast(
                dict[str, object], run["artifact_sha256"]
            )
            artifact_hashes["segment"] = _sha256(segment_path)
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(run, output)
            with self.assertRaisesRegex(ValueError, "all-missing placeholder"):
                validate_configuration(
                    results_dir,
                    "lambada",
                    "word",
                    required_models=("gpt-4-1",),
                    require_gold_manifest=False,
                )

    def test_existing_artifact_manifest_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            artifact_path: str = os.path.join(results_dir, "artifact.tsv.gz")
            with open(artifact_path, "wb") as output:
                output.write(b"original")
            _write_artifact_manifest(results_dir)

            _validate_artifact_manifest(results_dir)
            manifest_path: str = os.path.join(results_dir, "artifact_manifest.json")
            with open(manifest_path, encoding="utf-8") as source:
                manifest: dict[str, Any] = json.load(source)
            manifest["internal_path"] = "/private/artifact"
            with open(manifest_path, "w", encoding="utf-8") as output:
                json.dump(manifest, output)
            with self.assertRaisesRegex(ValueError, "manifest schema"):
                _validate_artifact_manifest(results_dir)
            _write_artifact_manifest(results_dir)

            with open(artifact_path, "wb") as output:
                output.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "inventory or hashes"):
                _validate_artifact_manifest(results_dir)

    def test_release_inventory_rejects_unexpected_files_and_links(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            with open(
                os.path.join(results_dir, "README.md"), "w", encoding="utf-8"
            ) as output:
                output.write("public documentation\n")
            _validate_release_inventory(results_dir, "paper", require_complete=False)

            unexpected_path: str = os.path.join(results_dir, "provider_raw.json")
            with open(unexpected_path, "w", encoding="utf-8") as output:
                output.write("{}\n")
            with self.assertRaisesRegex(ValueError, "provider_raw.json"):
                _validate_release_inventory(
                    results_dir, "paper", require_complete=False
                )
            os.unlink(unexpected_path)

            link_path: str = os.path.join(results_dir, "coverage.tsv")
            os.symlink("README.md", link_path)
            with self.assertRaisesRegex(ValueError, "coverage.tsv"):
                _validate_release_inventory(
                    results_dir, "paper", require_complete=False
                )
            os.unlink(link_path)

            os.makedirs(os.path.join(results_dir, "private_empty_directory"))
            with self.assertRaisesRegex(ValueError, "private_empty_directory"):
                _validate_release_inventory(
                    results_dir, "paper", require_complete=False
                )

        with tempfile.TemporaryDirectory() as parent:
            real_root: str = os.path.join(parent, "real")
            linked_root: str = os.path.join(parent, "linked")
            os.makedirs(real_root)
            os.symlink(real_root, linked_root)
            with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                _validate_release_inventory(
                    linked_root, "paper", require_complete=False
                )

    def test_release_inventory_allows_only_named_root_scratch_tables(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            with open(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"),
                "w",
                encoding="utf-8",
            ) as output:
                output.write("intermediate\n")
            _validate_release_inventory(results_dir, "paper", require_complete=False)

            with open(
                os.path.join(results_dir, "private_segments.tsv"),
                "w",
                encoding="utf-8",
            ) as output:
                output.write("not an approved intermediate\n")
            with self.assertRaisesRegex(ValueError, "private_segments.tsv"):
                _validate_release_inventory(
                    results_dir, "paper", require_complete=False
                )

    def test_release_inventory_is_cohort_aware_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            open_table: str = os.path.join(results_dir, "f_table_open.tsv")
            with open(open_table, "w", encoding="utf-8") as output:
                output.write("open only\n")
            _validate_release_inventory(results_dir, "open", require_complete=False)
            with self.assertRaisesRegex(ValueError, "f_table_open.tsv"):
                _validate_release_inventory(
                    results_dir, "paper", require_complete=False
                )
            with self.assertRaisesRegex(ValueError, "Missing files"):
                _validate_release_inventory(results_dir, "open")

    def test_release_inventory_accepts_complete_open_workflow(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            expected: set[str] = _allowed_release_files("open") - {
                "README.md",
                "artifact_manifest.json",
            }
            for relative in expected:
                path: str = os.path.join(results_dir, relative)
                os.makedirs(os.path.dirname(path) or results_dir, exist_ok=True)
                with open(path, "wb") as output:
                    output.write(b"placeholder")
            _validate_release_inventory(results_dir, "open")

            with open(
                os.path.join(results_dir, "artifact_manifest.json"), "wb"
            ) as output:
                output.write(b"{}")
            _validate_release_inventory(results_dir, "open", require_manifest=True)

    def test_release_inventory_accepts_complete_gold_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            expected: set[str] = _allowed_release_files("paper") - {
                "artifact_manifest.json"
            }
            for relative in expected:
                path: str = os.path.join(results_dir, relative)
                os.makedirs(os.path.dirname(path) or results_dir, exist_ok=True)
                with open(path, "wb") as output:
                    output.write(b"placeholder")
            _validate_release_inventory(results_dir, "paper")

            with open(
                os.path.join(results_dir, "artifact_manifest.json"), "wb"
            ) as output:
                output.write(b"{}")
            _validate_release_inventory(
                results_dir,
                "paper",
                require_manifest=True,
            )

    def test_derived_sidecar_binds_shipped_raw_inputs_and_output(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(config_dir)
            for name in ("segments.tsv.gz", "model_segment.tsv.gz", "model_run.json"):
                with open(os.path.join(config_dir, name), "wb") as output:
                    output.write(name.encode("utf-8"))
            output_path: str = os.path.join(results_dir, "derived.tsv")
            with open(output_path, "w", encoding="utf-8") as output:
                output.write("value\n1\n")
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            input_paths = collect_result_inputs(results_dir, "boolq", "sentence", {})
            parameters: dict[str, object] = {"seed": 42}
            write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.f_table",
                generator_path=os.path.join(
                    repository_root, "benchmark_scripts/f_table.py"
                ),
                input_paths=input_paths,
                parameters=parameters,
                root_dir=results_dir,
                supporting_source_paths={
                    relative_path: os.path.join(repository_root, relative_path)
                    for relative_path in DERIVED_SUPPORTING_SOURCE_FILES
                },
            )

            self.assertEqual(
                _validate_derived_sidecar(
                    results_dir,
                    output_path,
                    "benchmark_scripts.f_table",
                    [("boolq", "sentence")],
                    ("model", "model-a"),
                ),
                parameters,
            )
            sidecar_path: str = f"{output_path}.provenance.json"
            with open(sidecar_path, encoding="utf-8") as source:
                sidecar: dict[str, Any] = json.load(source)
            sidecar["internal_path"] = "/private/analysis"
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(sidecar, output)
            with self.assertRaisesRegex(ValueError, "provenance schema"):
                _validate_derived_sidecar(
                    results_dir,
                    output_path,
                    "benchmark_scripts.f_table",
                    [("boolq", "sentence")],
                    ("model", "model-a"),
                )
            write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.f_table",
                generator_path=os.path.join(
                    repository_root, "benchmark_scripts/f_table.py"
                ),
                input_paths=input_paths,
                parameters=parameters,
                root_dir=results_dir,
                supporting_source_paths={
                    relative_path: os.path.join(repository_root, relative_path)
                    for relative_path in DERIVED_SUPPORTING_SOURCE_FILES
                },
            )
            with open(sidecar_path, encoding="utf-8") as source:
                sidecar = json.load(source)
            software: dict[str, str] = cast(dict[str, str], sidecar["software"])
            software["internal_path"] = "/private/runtime"
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(sidecar, output)
            with self.assertRaisesRegex(ValueError, "software metadata"):
                _validate_derived_sidecar(
                    results_dir,
                    output_path,
                    "benchmark_scripts.f_table",
                    [("boolq", "sentence")],
                    ("model", "model-a"),
                )
            write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.f_table",
                generator_path=os.path.join(
                    repository_root, "benchmark_scripts/f_table.py"
                ),
                input_paths=input_paths,
                parameters=parameters,
                root_dir=results_dir,
                supporting_source_paths={
                    relative_path: os.path.join(repository_root, relative_path)
                    for relative_path in DERIVED_SUPPORTING_SOURCE_FILES
                },
            )
            with open(output_path, "a", encoding="utf-8") as output:
                output.write("2\n")
            with self.assertRaisesRegex(ValueError, "output seal"):
                _validate_derived_sidecar(
                    results_dir,
                    output_path,
                    "benchmark_scripts.f_table",
                    [("boolq", "sentence")],
                    ("model", "model-a"),
                )

    def test_f_table_parameters_reject_undeclared_fields(self) -> None:
        models: tuple[str, ...] = ("model-a", "model-b")
        parameters: dict[str, Any] = {
            "benchmark_configs": [{"benchmark": "boolq", "pregrouper": "sentence"}],
            "scopes": ["all"],
            "contrasts": ["canonical"],
            "bootstrap_resamples": 1000,
            "confidence_level": 0.95,
            "seed": 42,
            "bootstrap_rng": "sha256_cell_key_v1",
            "anli_contrast": None,
            "cohort": "paper",
            "requested_models": list(models),
            "output_models": sorted(models),
            "api_infinity_policy": "pairwise_complete",
            "transfer_aggregation": "row_pooled",
            "missingness_policy": "pair_specific_complete_case",
            "metrics": None,
        }
        expected_configs: set[tuple[str, str, str, str]] = {
            ("boolq", "sentence", "all", "canonical")
        }
        _validate_f_table_parameters(
            parameters,
            expected_configs,
            models,
            "pairwise_complete",
            "row_pooled",
            False,
            "f_table.tsv",
        )
        parameters["internal_path"] = "/private/analysis"
        with self.assertRaisesRegex(ValueError, "derivation parameters"):
            _validate_f_table_parameters(
                parameters,
                expected_configs,
                models,
                "pairwise_complete",
                "row_pooled",
                False,
                "f_table.tsv",
            )

    def test_derived_grid_requires_nan_for_unsupported_model(self) -> None:
        rows: list[dict[str, object]] = []
        for metric in ("F_pred", "F_attr"):
            for statistic in ("spearman", "pearson_r", "pearson_r2"):
                supported_component: bool = metric == "F_pred"
                rows.append(
                    {
                        "cohort": "paper",
                        "pair_population": "hosted_hosted",
                        "benchmark": "lambada",
                        "pregrouper": "word",
                        "scope": "all",
                        "requested_scope": "all",
                        "resolved_scope": (
                            "prompt_level_full_dialog"
                            if metric == "F_pred"
                            else "all_segment_coordinates_from_full_dialog"
                        ),
                        "contrast": "canonical",
                        "requested_contrast": "canonical",
                        "resolved_source_contrast": "target_completion_logprob",
                        "resolved_target_contrast": "target_completion_logprob",
                        "readout_contrast": "not_applicable",
                        "availability_status": (
                            "available" if supported_component else "unavailable"
                        ),
                        "unavailable_reason": (
                            ""
                            if supported_component
                            else "insufficient_joint_observations"
                        ),
                        "api_infinity_policy": "pairwise_complete",
                        "aggregation": "row_pooled",
                        "model_s": "open",
                        "model_t": "hosted",
                        "metric": metric,
                        "statistic": statistic,
                        "n_observations": 3 if supported_component else 0,
                        "expected_observations": 3 if supported_component else 0,
                        "observation_coverage": (
                            1.0 if supported_component else float("nan")
                        ),
                        "n_prompts": 3 if supported_component else 0,
                        "expected_prompts": 3 if supported_component else 0,
                        "prompt_coverage": (
                            1.0 if supported_component else float("nan")
                        ),
                        "f_point": 0.5 if supported_component else float("nan"),
                        "f_lo": 0.4 if supported_component else float("nan"),
                        "f_hi": 0.6 if supported_component else float("nan"),
                    }
                )
        frame = pd.DataFrame(rows)
        unsupported = {("lambada", "word", "hosted")}

        _require_pair_grid(
            frame,
            {("lambada", "word", "all", "canonical")},
            ("open", "hosted"),
            "results.tsv",
            "pairwise_complete",
            "row_pooled",
            set(),
            unsupported,
            scalar_only=True,
        )
        frame.loc[frame["metric"] == "F_attr", "f_point"] = 0.5
        with self.assertRaisesRegex(ValueError, "unsupported"):
            _require_pair_grid(
                frame,
                {("lambada", "word", "all", "canonical")},
                ("open", "hosted"),
                "results.tsv",
                "pairwise_complete",
                "row_pooled",
                set(),
                unsupported,
                scalar_only=True,
            )

    def test_rejects_system_only_segment_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            segment_path: str = os.path.join(directory, "model_segment.tsv.gz")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": False,
                        "seg_idx": index,
                        "message_idx": 0 if index < 3 else 1,
                        "message_role": "system" if index < 3 else "user",
                        "message_seg_idx": index if index < 3 else index - 3,
                        "segment_text": f"segment {index}",
                        "n_segments": 5,
                    }
                    for index in range(5)
                ]
            )
            manifest.to_csv(manifest_path, sep="\t", index=False)
            partial: pd.DataFrame = manifest.iloc[:3].copy()
            partial["segment_result_available"] = True
            partial.to_csv(segment_path, sep="\t", index=False)
            loaded, keys = _validate_manifest(manifest_path)

            with self.assertRaisesRegex(ValueError, "2 missing"):
                _validate_segment_file(segment_path, loaded, keys)

    def test_rejects_inconsistent_original_availability_within_prompt(self) -> None:
        manifest: pd.DataFrame = pd.DataFrame(
            [
                {
                    "prompt_idx": 0,
                    "answer": False,
                    "seg_idx": index,
                    "message_idx": 1,
                    "message_role": "user",
                    "message_seg_idx": index,
                    "segment_text": f"segment {index}",
                    "n_segments": 2,
                }
                for index in range(2)
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "model_segment.tsv.gz")
            frame: pd.DataFrame = manifest.copy()
            frame["segment_result_available"] = True
            frame["original_result_available"] = [False, True]
            frame.to_csv(path, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "inconsistent"):
                _validate_segment_file(path, manifest, {(0, 0), (0, 1)})

    def test_reports_sparse_label_coverage_without_rejecting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path: str = os.path.join(directory, "model_tokens.tsv.gz")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": None,
                        "kind": "orig",
                        "answer": False,
                        "label": "true",
                        "token": "true",
                        "logprob": -1.0,
                    },
                    {
                        "prompt_idx": 0,
                        "seg_idx": None,
                        "kind": "orig",
                        "answer": False,
                        "label": "false",
                        "token": "false",
                        "logprob": -float("inf"),
                    },
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "answer": False,
                        "label": "true",
                        "token": "true",
                        "logprob": -2.0,
                    },
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "answer": False,
                        "label": "false",
                        "token": "false",
                        "logprob": -float("inf"),
                    },
                ]
            ).to_csv(token_path, sep="\t", index=False)

            original, ablated, labels = _token_coverage(
                token_path,
                {(0, 0), (0, 1)},
                {0},
                {"true", "false"},
            )

            self.assertEqual(original, 1.0)
            self.assertEqual(ablated, 0.5)
            self.assertAlmostEqual(labels, 2 / 6)

    def test_rejects_token_answers_that_disagree_with_segment_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            token_path: str = os.path.join(directory, "model_tokens.tsv.gz")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": None,
                        "kind": "orig",
                        "answer": "B",
                        "label": "a",
                        "token": "a",
                        "logprob": -1.0,
                    },
                    {
                        "prompt_idx": 0,
                        "seg_idx": None,
                        "kind": "orig",
                        "answer": "B",
                        "label": "b",
                        "token": "b",
                        "logprob": -2.0,
                    },
                ]
            ).to_csv(token_path, sep="\t", index=False)
            segment_frame: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "answer": "A",
                        "segment_result_available": False,
                    }
                ]
            )

            with self.assertRaisesRegex(ValueError, "answers"):
                _token_coverage(
                    token_path,
                    {(0, 0)},
                    {0},
                    {"a", "b"},
                    segment_frame,
                )
