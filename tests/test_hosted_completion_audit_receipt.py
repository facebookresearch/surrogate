# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
import tempfile
from typing import Any
from unittest import TestCase

import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.hosted_audit_receipt import (
    canonical_projection_digest,
    canonical_revision_digest,
)
from benchmark_scripts.hosted_completion_audit_receipt import (
    ARTIFACT_TYPE,
    AVAILABILITY_STATUSES,
    BENCHMARK,
    CANARY_POPULATION,
    COMPLETION_REQUEST_PARAMETERS,
    DATASET_SHA256,
    FULL_POPULATION,
    HOSTED_COMPLETION_MODELS,
    IDENTITY_FORMAT,
    LINEAGE_DETAIL,
    MODEL_SOURCE_POPULATION,
    POPULATION_COUNTS,
    POPULATION_IDENTITY_SHA256,
    PREGROUPER,
    PUBLIC_SOURCE_FORMATS,
    RAW_SOURCE_FORMATS,
    SCHEMA_VERSION,
    canonical_receipt_bytes,
    completion_payload_projection,
    completion_table_projection,
    compute_completion_dialog_identity,
    load_receipt,
    projection_summary,
    validate_receipt,
    verify_receipt_sha256,
    write_receipt,
)
from surrogate.model_types import make_dialog
from surrogate.text_augmentation import dialog_segments


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_receipt() -> dict[str, Any]:
    components: list[str] = sorted((_digest("producer"), _digest("scorer")))
    revision: str = canonical_revision_digest(components)
    populations: list[dict[str, Any]] = []
    for population in (FULL_POPULATION, CANARY_POPULATION):
        prompt_count, segment_count = POPULATION_COUNTS[population]
        coordinate, prompt, ablated = POPULATION_IDENTITY_SHA256[population]
        populations.append(
            {
                "name": population,
                "prompt_count": prompt_count,
                "segment_count": segment_count,
                "coordinate_sha256": coordinate,
                "prompt_identity_sha256": prompt,
                "ablated_dialog_identity_sha256": ablated,
            }
        )
    entries: list[dict[str, Any]] = []
    for model in HOSTED_COMPLETION_MODELS:
        population: str = MODEL_SOURCE_POPULATION[model]
        raw_prompts, raw_segments = POPULATION_COUNTS[population]
        supported: bool = population == FULL_POPULATION
        projection: str = _digest(model + ":projection")
        entries.append(
            {
                "benchmark": BENCHMARK,
                "pregrouper": PREGROUPER,
                "model": model,
                "source_population": population,
                "raw_source_format": RAW_SOURCE_FORMATS[population],
                "public_source_format": PUBLIC_SOURCE_FORMATS[population],
                "producer_revision_sha256": revision,
                "raw_artifact_sha256": _digest(model + ":raw"),
                "raw_artifact_size_bytes": 100,
                "raw_projection_sha256": projection,
                "public_artifact_sha256": _digest(model + ":public"),
                "public_artifact_size_bytes": 200,
                "public_projection_sha256": (
                    projection if supported else _digest(model + ":placeholder")
                ),
                "availability_status": AVAILABILITY_STATUSES[population],
                "raw_prompt_count": raw_prompts,
                "raw_segment_count": raw_segments,
                "public_prompt_count": POPULATION_COUNTS[FULL_POPULATION][0],
                "public_segment_count": POPULATION_COUNTS[FULL_POPULATION][1],
                "raw_original_available_count": raw_prompts if supported else 0,
                "raw_ablated_available_count": raw_segments if supported else 0,
                "raw_paired_attribution_available_count": (
                    raw_segments if supported else 0
                ),
                "public_original_available_count": (
                    POPULATION_COUNTS[FULL_POPULATION][0] if supported else 0
                ),
                "public_ablated_available_count": (
                    POPULATION_COUNTS[FULL_POPULATION][1] if supported else 0
                ),
                "public_paired_attribution_available_count": (
                    POPULATION_COUNTS[FULL_POPULATION][1] if supported else 0
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "lineage_detail": LINEAGE_DETAIL,
        "identity_format": IDENTITY_FORMAT,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": _digest("manifest"),
        "producer_component_sha256": components,
        "finalization_revision_sha256": revision,
        "request_parameters": dict(COMPLETION_REQUEST_PARAMETERS),
        "populations": populations,
        "entries": entries,
    }


class TestHostedCompletionAuditReceipt(TestCase):
    def test_validates_exact_inventory_and_source_semantics(self) -> None:
        validated = validate_receipt(_valid_receipt())
        self.assertEqual(
            [population["name"] for population in validated["populations"]],
            [FULL_POPULATION, CANARY_POPULATION],
        )
        self.assertEqual(
            [entry["model"] for entry in validated["entries"]],
            list(HOSTED_COMPLETION_MODELS),
        )

    def test_raw_and_public_projections_match_with_missing_scores(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "prompt_idx": 4,
                    "seg_idx": seg_idx,
                    "answer": "target",
                    "n_segments": 2,
                }
                for seg_idx in range(2)
            ]
        )
        payload: list[dict[str, Any]] = [
            {
                "prompt_idx": 4,
                "ablation_idx": 0,
                "answer": "target",
                "n_segments": 2,
                "orig_logprob": -1.25,
                "orig_status": "ok",
                "ablated_logprob": -2.5,
                "ablated_status": "ok",
            },
            {
                "prompt_idx": 4,
                "ablation_idx": 1,
                "answer": "target",
                "n_segments": 2,
                "orig_logprob": -1.25,
                "orig_status": "ok",
                "ablated_logprob": None,
                "ablated_status": "target_parse_unavailable",
            },
        ]
        table: list[dict[str, Any]] = [
            {
                "prompt_idx": 4,
                "seg_idx": 0,
                "orig_completion_logprob": -1.25,
                "original_result_status": "ok",
                "ablated_completion_logprob": -2.5,
                "segment_result_status": "ok",
            },
            {
                "prompt_idx": 4,
                "seg_idx": 1,
                "orig_completion_logprob": -1.25,
                "original_result_status": "ok",
                "ablated_completion_logprob": float("nan"),
                "segment_result_status": "target_parse_unavailable",
            },
        ]
        raw_projection = completion_payload_projection(payload, manifest)
        public_projection = completion_table_projection(table)
        self.assertEqual(raw_projection, public_projection)
        self.assertEqual(
            canonical_projection_digest(raw_projection),
            canonical_projection_digest(public_projection),
        )
        self.assertEqual(
            projection_summary(raw_projection),
            {
                "prompt_count": 1,
                "segment_count": 2,
                "original_available_count": 1,
                "ablated_available_count": 1,
                "paired_attribution_available_count": 1,
            },
        )

    def test_prompt_shaped_projection_requires_exact_population_order(self) -> None:
        manifest = pd.DataFrame(
            [
                {
                    "prompt_idx": 9,
                    "seg_idx": seg_idx,
                    "answer": "target",
                    "n_segments": 2,
                }
                for seg_idx in range(2)
            ]
        )
        payload: list[dict[str, Any]] = [
            {
                "prompt_idx": 9,
                "answer": "target",
                "n_segments": 2,
                "orig_logprob": None,
                "ablated_logprob": [None, -4.0],
            }
        ]
        projection = completion_payload_projection(
            payload, manifest, expected_prompt_indices=[9]
        )
        self.assertEqual(projection_summary(projection)["segment_count"], 2)
        with self.assertRaisesRegex(ValueError, "canonical expected order"):
            completion_payload_projection(
                payload, manifest, expected_prompt_indices=[10]
            )

    def test_dialog_identity_binds_target_prompt_coordinates_and_ablations(
        self,
    ) -> None:
        spec = BENCHMARKS[BENCHMARK]
        with tempfile.TemporaryDirectory() as temporary:
            dataset_path: str = os.path.join(temporary, "lambada.tsv")
            pd.DataFrame([{"context": "One short passage", "target": "word"}]).to_csv(
                dataset_path, sep="\t", index=False
            )
            dialog = make_dialog(spec.system_prompt_override or "", "One short passage")
            segments = dialog_segments(dialog, pregrouper_id=PREGROUPER)
            manifest = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": segment.segment_idx,
                        "answer": "word",
                        "message_idx": segment.message_idx,
                        "message_role": segment.message_role,
                        "message_seg_idx": segment.message_segment_idx,
                        "segment_text": segment.text,
                        "n_segments": len(segments),
                    }
                    for segment in segments
                ]
            )
            first = asyncio.run(
                compute_completion_dialog_identity(dataset_path, manifest)
            )
            second = asyncio.run(
                compute_completion_dialog_identity(dataset_path, manifest)
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertTrue(all(len(value) == 64 for value in first))
            changed = manifest.copy()
            changed.loc[0, "segment_text"] = "tampered"
            with self.assertRaisesRegex(ValueError, "Manifest identity mismatch"):
                asyncio.run(compute_completion_dialog_identity(dataset_path, changed))

    def test_rejects_tampering_and_private_or_reordered_inventory(self) -> None:
        mutations: list[tuple[str, Any]] = [
            (
                "private field",
                lambda receipt: receipt["entries"][0].update(
                    {"producer_path": "/private/source.py"}
                ),
            ),
            (
                "reordered model",
                lambda receipt: receipt["entries"].reverse(),
            ),
            (
                "identity digest",
                lambda receipt: receipt["populations"][0].update(
                    {"prompt_identity_sha256": "0" * 64}
                ),
            ),
            (
                "full projection",
                lambda receipt: receipt["entries"][0].update(
                    {"public_projection_sha256": "0" * 64}
                ),
            ),
            (
                "placeholder availability",
                lambda receipt: receipt["entries"][-1].update(
                    {"public_ablated_available_count": 1}
                ),
            ),
            (
                "truncated public grid",
                lambda receipt: receipt["entries"][-1].update(
                    {"public_segment_count": 1}
                ),
            ),
        ]
        for name, mutation in mutations:
            with self.subTest(name=name):
                receipt = _valid_receipt()
                mutation(receipt)
                with self.assertRaises(ValueError):
                    validate_receipt(receipt)

    def test_canonical_write_load_and_pinned_hash(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        self.assertEqual(
            canonical_receipt_bytes(receipt),
            canonical_receipt_bytes(copy.deepcopy(receipt)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path: str = os.path.join(temporary, "receipt.json")
            digest: str = write_receipt(path, receipt)
            self.assertEqual(load_receipt(path, digest), validate_receipt(receipt))
            self.assertEqual(verify_receipt_sha256(path, digest), digest)
            with self.assertRaisesRegex(ValueError, "SHA-256 disagrees"):
                verify_receipt_sha256(path, "0" * 64)
