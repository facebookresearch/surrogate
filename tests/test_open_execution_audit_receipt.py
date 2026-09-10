# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the corrected open-model execution audit receipt."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any

import pandas as pd

from benchmark_scripts import open_execution_audit_receipt
from benchmark_scripts.f_table import OPEN_MODELS
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
)


class OpenExecutionAuditReceiptTest(unittest.TestCase):
    """Require the exact 5x8 grid and bind every numerical verdict to bytes."""

    def _execution_run(self, row: dict[str, Any]) -> dict[str, Any]:
        model: str = str(row["model"])
        return {
            "benchmark": row["benchmark"],
            "dataset": {"rows": 1},
            "model": model,
            "model_identity_files_sha256": {"config.json": "1" * 64},
            "model_source": GOLD_OPEN_EXECUTION_MODEL_SOURCES[model],
            "parameters": {
                "phase_attention_implementation": {
                    "ablation": "sdpa",
                    "attention": "eager",
                },
                "phases": ["ablation", "attention"],
            },
            "pregrouper": row["pregrouper"],
            "schema_version": 2,
            "segmentation_scope": "full_dialog_in_message_order",
            "software": {"torch": "test"},
            "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
        }

    def _write_run(
        self,
        results_dir: str,
        row: dict[str, Any],
        execution_run: dict[str, Any],
        sealed: bool,
    ) -> str:
        path: str = open_execution_audit_receipt._run_path(results_dir, row)
        record: dict[str, Any] = dict(execution_run)
        if sealed:
            record.update(
                {
                    "execution_model_source": execution_run["model_source"],
                    "model_source": GOLD_OPEN_MODEL_REPOSITORIES[str(row["model"])],
                    "schema_version": 3,
                    "software": {
                        **execution_run["software"],
                        "transformers": "test",
                    },
                }
            )
        with open(path, "w", encoding="utf-8") as output:
            json.dump(record, output, indent=2, sort_keys=True)
            output.write("\n")
        return open_execution_audit_receipt._sha256(path)

    def _special_row(self, receipt: dict[str, Any]) -> dict[str, Any]:
        return next(
            row
            for row in receipt["numerical_cells"]
            if (
                row["model"],
                row["benchmark"],
                row["pregrouper"],
            )
            == open_execution_audit_receipt.Q05_BOOLQ_WORD_CELL
        )

    def _fixture(self, results_dir: str) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for benchmark, pregrouper in open_execution_audit_receipt.CONFIGURATIONS:
            directory: str = os.path.join(results_dir, benchmark, pregrouper)
            os.makedirs(directory)
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": 0,
                        "seg_idx": 0,
                        "message_idx": 0,
                        "message_role": "user",
                        "message_seg_idx": 0,
                        "segment_text": "segment",
                        "n_segments": 1,
                    }
                ]
            ).to_csv(manifest_path, sep="\t", index=False)
            manifest_hash: str = open_execution_audit_receipt._sha256(manifest_path)
            for model in OPEN_MODELS:
                segment_path: str = os.path.join(directory, f"{model}_segment.tsv.gz")
                is_special: bool = (
                    model == "qwen2.5-0.5b-instruct"
                    and benchmark == "boolq"
                    and pregrouper == "word"
                )
                segment_columns: tuple[str, ...] = (
                    *open_execution_audit_receipt.ABLATION_SEGMENT_COLUMNS[:8],
                    *open_execution_audit_receipt.ATTENTION_SEGMENT_COLUMNS,
                    *open_execution_audit_receipt.ABLATION_SEGMENT_COLUMNS[8:],
                    *(
                        open_execution_audit_receipt.COMPLETION_SEGMENT_COLUMNS
                        if benchmark == "lambada"
                        else ()
                    ),
                )
                segment_frame: pd.DataFrame = pd.DataFrame(
                    [
                        {
                            column: (
                                "user"
                                if column == "message_role"
                                else (
                                    "segment"
                                    if column == "segment_text"
                                    else 0 if column != "n_segments" else 1
                                )
                            )
                            for column in segment_columns
                        }
                    ]
                )
                segment_frame.to_csv(segment_path, sep="\t", index=False)
                execution_segment_sha256: str = open_execution_audit_receipt._sha256(
                    segment_path
                )
                segment_frame["segment_result_available"] = True
                segment_frame["original_result_available"] = True
                segment_frame.to_csv(segment_path, sep="\t", index=False)
                row: dict[str, Any] = {
                    "archive_comparison": "same_coordinates",
                    "benchmark": benchmark,
                    "manifest_sha256": manifest_hash,
                    "model": model,
                    "pregrouper": pregrouper,
                    "prompts": 1,
                    "execution_segment_sha256": execution_segment_sha256,
                    "release_segment_projection_sha256": (
                        open_execution_audit_receipt._release_equivalent_segment_projection_sha256(
                            segment_path,
                            benchmark,
                        )
                    ),
                    "release_segment_sha256": (
                        open_execution_audit_receipt._sha256(segment_path)
                    ),
                    "segments": 1,
                    "execution_numerical_audit_status": "passed",
                    "historical_replication_status": (
                        "documented_historical_difference"
                        if (
                            model,
                            benchmark,
                            pregrouper,
                        )
                        == open_execution_audit_receipt.Q05_BOOLQ_WORD_CELL
                        else (
                            "not_comparable_prompt_version"
                            if benchmark == "race"
                            else "within_release_regression_tolerance"
                        )
                    ),
                    "structural_validation_status": "passed",
                }
                if benchmark != "race":
                    row.update(
                        {
                            "archive_attribution_pearson_r": 0.99,
                            "archive_attribution_mae": 0.1,
                            "archive_original_pearson_r": 0.999,
                            "archive_original_mae": 0.1,
                        }
                    )
                    is_qwen_word: bool = (
                        benchmark == "boolq"
                        and pregrouper == "word"
                        and model.startswith("qwen")
                    )
                    if is_qwen_word:
                        row.update(
                            {
                                "archive_attention_comparison": (
                                    "not_comparable_archive_sentence_contamination"
                                ),
                                "archive_attention_contaminated_coordinates": 546,
                                "archive_attention_max_direct_pearson_r": -0.1,
                                "archive_attention_max_sentence_contamination_pearson_r": 0.99,
                                "archive_attention_mean_direct_pearson_r": -0.1,
                                "archive_attention_mean_sentence_contamination_pearson_r": 0.99,
                                "archive_attention_rollout_direct_pearson_r": 0.1,
                                "archive_attention_rollout_sentence_contamination_pearson_r": 0.99,
                            }
                        )
                    else:
                        row.update(
                            {
                                "archive_attention_max_pearson_r": 0.99,
                                "archive_attention_mean_pearson_r": 0.99,
                                "archive_attention_rollout_pearson_r": 0.99,
                            }
                        )
                    if is_qwen_word and model == "qwen2.5-14b-instruct":
                        row["archive_representation_comparison"] = "not_available"
                    else:
                        row.update(
                            {
                                "archive_delta_norm_postnorm_pearson_r": 0.99,
                                "archive_representation_comparison": (
                                    "same_coordinates"
                                ),
                                "archive_w_dot_delta_z_postnorm_pearson_r": 0.99,
                                "archive_w_norm_mae": 0.0,
                            }
                        )
                execution_run: dict[str, Any] = self._execution_run(row)
                row["execution_run_sha256"] = self._write_run(
                    results_dir, row, execution_run, sealed=False
                )
                row["execution_run_metadata"] = execution_run
                self._write_run(results_dir, row, execution_run, sealed=True)
                if benchmark == "race":
                    row["archive_comparison"] = "not_comparable_prompt_version"
                    row["archive_race_w_norm_prompts"] = 4_934
                    row["archive_race_w_norm_max_abs_error"] = 0.0
                if benchmark != "lambada":
                    token_path: str = os.path.join(directory, f"{model}_tokens.tsv.gz")
                    with open(token_path, "wb") as output:
                        output.write(
                            f"tokens:{benchmark}:{pregrouper}:{model}".encode()
                        )
                    row["tokens_sha256"] = open_execution_audit_receipt._sha256(
                        token_path
                    )
                if is_special:
                    recheck_run: dict[str, Any] = {
                        **execution_run,
                        "parameters": {
                            "batch_size": 8,
                            "max_forward_passes": 10_000,
                            "max_samples": None,
                            "phase_attention_implementation": {"ablation": "sdpa"},
                            "phases": ["ablation"],
                            "seed": 42,
                        },
                    }
                    eager_run: dict[str, Any] = {
                        **recheck_run,
                        "parameters": {
                            **recheck_run["parameters"],
                            "phase_attention_implementation": {"ablation": "eager"},
                        },
                    }
                    scope_rows: dict[str, int] = {
                        "all": 10_000,
                        "system": 2_859,
                        "user": 7_141,
                    }
                    pairwise: dict[str, dict[str, dict[str, float | int]]] = {
                        peer: {
                            scope: {
                                "archive_pearson_r2": 0.2,
                                "archive_spearman": 0.3,
                                "current_pearson_r2": 0.201,
                                "current_spearman": 0.302,
                                "n_common": count,
                                "pearson_r2_delta": 0.001,
                                "spearman_delta": 0.002,
                            }
                            for scope, count in (
                                open_execution_audit_receipt.Q05_BOOLQ_WORD_PAIR_COMMON_ROWS[
                                    peer
                                ].items()
                            )
                        }
                        for peer in (
                            open_execution_audit_receipt.Q05_BOOLQ_WORD_PAIR_MODELS
                        )
                    }
                    projection_sha256: str = (
                        open_execution_audit_receipt._segment_projection_sha256(
                            segment_path
                        )
                    )
                    row.update(
                        {
                            "archive_comparison": (
                                "same_coordinates_component_agreement_"
                                "attribution_below_standard_threshold"
                            ),
                            "archive_ablated_pearson_r": 0.996,
                            "archive_ablated_rmse": 0.11,
                            "archive_attribution_mae": 0.12,
                            "archive_attribution_normalized_rmse": 0.4,
                            "archive_attribution_pearson_r": 0.936,
                            "archive_attribution_rmse": 0.16,
                            "archive_attribution_scope_diagnostics": {
                                scope: {
                                    "archive_std": 0.4,
                                    "mae": 0.12,
                                    "normalized_rmse": 0.4,
                                    "pearson_r": 0.936,
                                    "rmse": 0.16,
                                    "rows": count,
                                }
                                for scope, count in scope_rows.items()
                            },
                            "archive_attribution_std": 0.4,
                            "archive_attribution_validation": (
                                "new_artifact_not_archive_equivalence"
                            ),
                            "archive_attention_comparison": (
                                "not_comparable_archive_sentence_contamination"
                            ),
                            "archive_attention_contaminated_coordinates": 546,
                            "archive_attention_max_sentence_contamination_pearson_r": 0.99,
                            "archive_attention_mean_sentence_contamination_pearson_r": 0.99,
                            "archive_attention_rollout_sentence_contamination_pearson_r": 0.99,
                            "archive_cross_model_attribution_max_abs_pearson_r": 0.21,
                            "archive_cross_model_attribution_pearson_r": {
                                "llama-3.1-8b-instruct": 0.17,
                                "qwen2.5-14b-instruct": 0.20,
                                "qwen2.5-3b-instruct": 0.16,
                                "qwen2.5-7b-instruct": 0.21,
                            },
                            "archive_eager_attribution_mae": 0.2,
                            "archive_eager_attribution_pearson_r": 0.8,
                            "archive_delta_norm_postnorm_pearson_r": 0.99,
                            "archive_original_pearson_r": 0.996,
                            "archive_w_dot_delta_z_postnorm_pearson_r": 0.99,
                            "archive_w_norm_mae": 0.0,
                            "eager_backend_diagnostic": {
                                "completed_epoch": 23,
                                "diagnostic_wrapper_sha256": (
                                    open_execution_audit_receipt.Q05_BOOLQ_WORD_EAGER_DIAGNOSTIC_WRAPPER_SHA256
                                ),
                                "execution_run_metadata": eager_run,
                                "execution_run_sha256": (
                                    open_execution_audit_receipt._json_object_sha256(
                                        eager_run
                                    )
                                ),
                                "manifest_sha256": manifest_hash,
                                "schema_version": 1,
                                "segment_projection_sha256": "b" * 64,
                                "started_epoch": 22,
                                "tokens_sha256": "c" * 64,
                            },
                            "pairwise_f_attr_max_abs_delta": {
                                "pearson_r2": 0.001,
                                "spearman": 0.002,
                            },
                            "pairwise_f_attr_sensitivity": pairwise,
                            "sdpa_repeatability": {
                                "completed_epoch": 22,
                                "execution_run_metadata": recheck_run,
                                "execution_run_sha256": (
                                    open_execution_audit_receipt._json_object_sha256(
                                        recheck_run
                                    )
                                ),
                                "gold_segment_projection_sha256": (projection_sha256),
                                "manifest_sha256": manifest_hash,
                                "schema_version": 1,
                                "segment_projection_sha256": projection_sha256,
                                "started_epoch": 21,
                                "tokens_sha256": row["tokens_sha256"],
                            },
                        }
                    )
                rows.append(row)
        queues: dict[str, dict[str, Any]] = {}
        for (
            queue_name,
            queue_models,
        ) in open_execution_audit_receipt.QUEUE_MODELS.items():
            artifact_hashes: dict[str, str] = {}
            for row in rows:
                if row["model"] not in queue_models:
                    continue
                stem: str = f"{row['benchmark']}/{row['pregrouper']}/{row['model']}"
                artifact_hashes[f"{stem}_segment.tsv.gz"] = row[
                    "execution_segment_sha256"
                ]
                artifact_hashes[f"{stem}_run.json"] = row["execution_run_sha256"]
                if "tokens_sha256" in row:
                    artifact_hashes[f"{stem}_tokens.tsv.gz"] = row["tokens_sha256"]
            queues[queue_name] = {
                "artifact_sha256": artifact_hashes,
                "completed_epoch": 20,
                "models": list(queue_models),
                "phase_attention_implementation": {
                    "ablation": "sdpa",
                    "attention": "eager",
                },
                "started_epoch": 10,
            }
        return {
            "archive_comparison_role": (
                "historical_replication_diagnostic_not_scientific_correctness"
            ),
            "artifact_integrity_status": "passed",
            "execution_source_hash_timing": "run_completion",
            "model_artifact_hash_timing": "post_run_seal",
            "models": {
                model: {
                    "artifact_manifest_sha256": (
                        GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[model]
                    ),
                    "latest_selected_artifact_mtime_ns": 1,
                }
                for model in OPEN_MODELS
            },
            "numerical_cells": rows,
            "queues": queues,
            "schema_version": 1,
            "segment_packaging": {
                "name": "normalize_segment_outputs",
                "producer_to_release_evidence": (
                    "producer hashes plus audited deterministic public "
                    "transformation; not a cryptographic pre/post equivalence proof"
                ),
                "source_sha256": open_execution_audit_receipt._sha256(
                    os.path.join(
                        os.path.dirname(open_execution_audit_receipt.__file__),
                        "normalize_segment_outputs.py",
                    )
                ),
            },
            "source_files": {
                path: {"mtime_ns": 1, "sha256": digest}
                for path, digest in GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256.items()
            },
        }

    def test_receipt_binds_exact_grid_and_artifact_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            open_execution_audit_receipt.validate_receipt(receipt, results_dir)
            first: dict[str, Any] = receipt["numerical_cells"][0]
            path: str = open_execution_audit_receipt._artifact_path(
                results_dir, first, "segment_sha256"
            )
            frame: pd.DataFrame = pd.read_csv(path, sep="\t")
            frame.loc[0, "attention_mean"] = 42.0
            frame.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "release segment disagrees"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_distinguishes_producer_and_release_segment_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            for row in receipt["numerical_cells"]:
                self.assertNotEqual(
                    row["execution_segment_sha256"],
                    row["release_segment_sha256"],
                )

            open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_missing_grid_cell(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            receipt["numerical_cells"].pop()
            with self.assertRaisesRegex(ValueError, "exact 5x8 grid"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_unexpected_top_level_or_numerical_fields(self) -> None:
        for location in ("receipt", "row"):
            with (
                self.subTest(location=location),
                tempfile.TemporaryDirectory() as results_dir,
            ):
                receipt: dict[str, Any] = self._fixture(results_dir)
                if location == "receipt":
                    receipt["private_note"] = "must not be shipped"
                else:
                    receipt["numerical_cells"][0][
                        "private_note"
                    ] = "must not be shipped"
                with self.assertRaisesRegex(ValueError, "(metadata|row fields)"):
                    open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_unexpected_queue_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            receipt["queues"]["gpu0"]["artifact_sha256"]["private/path"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "queue artifacts disagree"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_invalid_normalized_release_segment(self) -> None:
        for mutation, message in (
            ("drop", "population"),
            ("identity", "identity"),
            ("metric", "metrics"),
        ):
            with (
                self.subTest(mutation=mutation),
                tempfile.TemporaryDirectory() as results_dir,
            ):
                receipt: dict[str, Any] = self._fixture(results_dir)
                row: dict[str, Any] = receipt["numerical_cells"][0]
                path: str = open_execution_audit_receipt._artifact_path(
                    results_dir, row, "segment_sha256"
                )
                frame: pd.DataFrame = pd.read_csv(path, sep="\t")
                if mutation == "drop":
                    frame = frame.iloc[0:0]
                elif mutation == "identity":
                    frame.loc[0, "segment_text"] = "changed"
                else:
                    frame.loc[0, "attention_mean"] = float("nan")
                frame.to_csv(path, sep="\t", index=False)
                row["release_segment_sha256"] = open_execution_audit_receipt._sha256(
                    path
                )
                row["release_segment_projection_sha256"] = (
                    open_execution_audit_receipt._release_equivalent_segment_projection_sha256(
                        path,
                        str(row["benchmark"]),
                    )
                )
                with self.assertRaisesRegex(ValueError, message):
                    open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_unsubstantiated_low_signal_exception(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            special: dict[str, Any] = self._special_row(receipt)
            special["sdpa_repeatability"]["tokens_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "artifact binding"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_each_eager_backend_improvement_signal(self) -> None:
        for field, value in (
            ("archive_eager_attribution_pearson_r", 0.99),
            ("archive_eager_attribution_mae", 0.01),
        ):
            with (
                self.subTest(field=field),
                tempfile.TemporaryDirectory() as results_dir,
            ):
                receipt: dict[str, Any] = self._fixture(results_dir)
                self._special_row(receipt)[field] = value
                with self.assertRaisesRegex(ValueError, "archive diagnostics"):
                    open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_treats_archive_similarity_as_a_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            ordinary: dict[str, Any] = next(
                row
                for row in receipt["numerical_cells"]
                if row["archive_comparison"] == "same_coordinates"
                and row is not self._special_row(receipt)
            )
            ordinary["archive_attribution_pearson_r"] = 0.94
            open_execution_audit_receipt.validate_receipt(receipt, results_dir)
            ordinary["archive_attribution_pearson_r"] = 1.01
            with self.assertRaisesRegex(ValueError, "correlation diagnostics"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_ordinary_archive_status_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            ordinary: dict[str, Any] = next(
                row
                for row in receipt["numerical_cells"]
                if row["archive_comparison"] == "same_coordinates"
            )
            ordinary["archive_comparison"] = "not_checked"
            with self.assertRaisesRegex(ValueError, "numerical status"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_requires_ordinary_representation_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            ordinary: dict[str, Any] = next(
                row
                for row in receipt["numerical_cells"]
                if row.get("archive_representation_comparison") == "same_coordinates"
                and row["archive_comparison"] == "same_coordinates"
            )
            ordinary.pop("archive_w_dot_delta_z_postnorm_pearson_r")
            with self.assertRaisesRegex(
                ValueError, "(row fields|correlation diagnostics)"
            ):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_changed_pairwise_common_population(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            special: dict[str, Any] = self._special_row(receipt)
            special["pairwise_f_attr_sensitivity"]["qwen2.5-14b-instruct"]["user"][
                "n_common"
            ] -= 1
            with self.assertRaisesRegex(ValueError, "pairwise diagnostic"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_unbound_recheck_source_evidence(self) -> None:
        for evidence, field in (
            ("sdpa_repeatability", "execution_run_sha256"),
            ("eager_backend_diagnostic", "diagnostic_wrapper_sha256"),
        ):
            with (
                self.subTest(evidence=evidence, field=field),
                tempfile.TemporaryDirectory() as results_dir,
            ):
                receipt: dict[str, Any] = self._fixture(results_dir)
                self._special_row(receipt)[evidence][field] = "0" * 64
                with self.assertRaisesRegex(ValueError, "(run hash|artifact binding)"):
                    open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_changed_sealed_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            first: dict[str, Any] = receipt["numerical_cells"][0]
            run_path: str = open_execution_audit_receipt._run_path(results_dir, first)
            with open(run_path, encoding="utf-8") as source:
                sealed: dict[str, Any] = json.load(source)
            sealed["parameters"]["phase_attention_implementation"]["attention"] = "sdpa"
            with open(run_path, "w", encoding="utf-8") as output:
                json.dump(sealed, output)
            with self.assertRaisesRegex(ValueError, "changed execution metadata"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_receipt_rejects_changed_embedded_execution_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            receipt: dict[str, Any] = self._fixture(results_dir)
            first: dict[str, Any] = receipt["numerical_cells"][0]
            first["execution_run_metadata"]["software"]["torch"] = "changed"
            with self.assertRaisesRegex(ValueError, "metadata hash disagrees"):
                open_execution_audit_receipt.validate_receipt(receipt, results_dir)

    def test_build_receipt_binds_queue_and_numerical_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_dir: str = os.path.join(temporary, "results")
            expected: dict[str, Any] = self._fixture(results_dir)
            observation_path: str = os.path.join(temporary, "observation.json")
            numerical_path: str = os.path.join(temporary, "numerical.jsonl")
            with open(observation_path, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "models": expected["models"],
                        "schema_version": 1,
                        "source_files": expected["source_files"],
                    },
                    output,
                )
            with open(numerical_path, "w", encoding="utf-8") as output:
                for row in expected["numerical_cells"]:
                    execution_run: dict[str, Any] = row["execution_run_metadata"]
                    run_sha256: str = self._write_run(
                        results_dir, row, execution_run, sealed=False
                    )
                    numerical_row: dict[str, Any] = dict(row)
                    numerical_row.pop("execution_run_metadata")
                    numerical_row.pop("execution_run_sha256")
                    numerical_row["segment_sha256"] = numerical_row.pop(
                        "execution_segment_sha256"
                    )
                    numerical_row.pop("release_segment_projection_sha256")
                    numerical_row.pop("release_segment_sha256")
                    numerical_row["status"] = numerical_row.pop(
                        "execution_numerical_audit_status"
                    )
                    numerical_row.pop("historical_replication_status")
                    numerical_row.pop("structural_validation_status")
                    numerical_row["run_sha256"] = run_sha256
                    if (
                        numerical_row["model"],
                        numerical_row["benchmark"],
                        numerical_row["pregrouper"],
                    ) == open_execution_audit_receipt.Q05_BOOLQ_WORD_CELL:
                        canonical_to_alias: dict[str, str] = {
                            canonical: alias
                            for alias, canonical in open_execution_audit_receipt.Q05_BOOLQ_WORD_CROSS_MODEL_ALIASES.items()
                        }
                        cross_model: dict[str, float] = numerical_row[
                            "archive_cross_model_attribution_pearson_r"
                        ]
                        numerical_row["archive_cross_model_attribution_pearson_r"] = {
                            canonical_to_alias[model]: value
                            for model, value in cross_model.items()
                        }
                    output.write(json.dumps(numerical_row) + "\n")
            queue_paths: list[str] = []
            for queue_name, queue in expected["queues"].items():
                queue_path: str = os.path.join(temporary, f"{queue_name}.json")
                with open(queue_path, "w", encoding="utf-8") as output:
                    json.dump(
                        {
                            **queue,
                            "cells_per_model": 8,
                            "queue": queue_name,
                            "phase_attention_implementation": {
                                "ablation": "sdpa",
                                "attention": "eager",
                            },
                            "schema_version": 2,
                            "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
                        },
                        output,
                    )
                queue_paths.append(queue_path)

            actual: dict[str, Any] = open_execution_audit_receipt.build_receipt(
                results_dir,
                observation_path,
                queue_paths,
                numerical_path,
            )
            self.assertEqual(40, len(actual["numerical_cells"]))
            self.assertNotIn("run_sha256", actual["numerical_cells"][0])
            self.assertIn("execution_run_sha256", actual["numerical_cells"][0])
            self.assertIn("execution_run_metadata", actual["numerical_cells"][0])
            self.assertEqual(
                set(
                    self._special_row(actual)[
                        "archive_cross_model_attribution_pearson_r"
                    ]
                ),
                open_execution_audit_receipt.Q05_BOOLQ_WORD_CROSS_MODELS,
            )


if __name__ == "__main__":
    unittest.main()
