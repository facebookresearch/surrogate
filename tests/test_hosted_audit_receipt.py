# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any
from unittest import TestCase

from benchmark_scripts.hosted_audit_receipt import (
    ARTIFACT_TYPE,
    CLASSIFICATION_CONFIGURATIONS,
    CLASSIFICATION_REQUEST_PARAMETERS,
    CONFIGURATION_COUNTS,
    CONFIGURATION_IDENTITY_SHA256,
    HOSTED_CLASSIFICATION_MODELS,
    IDENTITY_FORMAT,
    LINEAGE_DETAIL,
    SCHEMA_VERSION,
    canonical_projection_digest,
    canonical_receipt_bytes,
    canonical_revision_digest,
    classification_payload_projection,
    classification_table_projection,
    expected_entry_keys,
    load_receipt,
    validate_receipt,
    verify_receipt_sha256,
    write_receipt,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _valid_receipt() -> dict[str, Any]:
    components: list[str] = sorted(
        [_digest("runner"), _digest("scorer"), _digest("retry")]
    )
    revision: str = canonical_revision_digest(components)
    configurations: list[dict[str, Any]] = []
    for benchmark, pregrouper in CLASSIFICATION_CONFIGURATIONS:
        prompt_count, segment_count = CONFIGURATION_COUNTS[(benchmark, pregrouper)]
        prefix: str = f"{benchmark}:{pregrouper}"
        configurations.append(
            {
                "benchmark": benchmark,
                "pregrouper": pregrouper,
                "prompt_count": prompt_count,
                "segment_count": segment_count,
                "dataset_sha256": _digest(prefix + ":dataset"),
                "manifest_sha256": _digest(prefix + ":manifest"),
                "prompt_identity_sha256": CONFIGURATION_IDENTITY_SHA256[
                    (benchmark, pregrouper)
                ][0],
                "ablated_dialog_identity_sha256": CONFIGURATION_IDENTITY_SHA256[
                    (benchmark, pregrouper)
                ][1],
            }
        )
    entries: list[dict[str, Any]] = []
    for benchmark, pregrouper, model in expected_entry_keys():
        prompt_count, segment_count = CONFIGURATION_COUNTS[(benchmark, pregrouper)]
        identity: str = f"{benchmark}:{pregrouper}:{model}"
        entries.append(
            {
                "benchmark": benchmark,
                "pregrouper": pregrouper,
                "model": model,
                "producer_revision_sha256": revision,
                "raw_artifact_sha256": _digest(identity + ":raw"),
                "raw_artifact_size_bytes": 100,
                "projection_sha256": _digest(identity + ":projection"),
                "availability_status": "complete",
                "original_status_counts": {
                    "ok": prompt_count,
                    "content_filter": 0,
                    "transient_exhausted": 0,
                },
                "ablated_status_counts": {
                    "ok": segment_count,
                    "content_filter": 0,
                    "transient_exhausted": 0,
                },
                "original_result_available_count": prompt_count,
                "ablated_result_available_count": segment_count,
                "paired_attribution_available_count": segment_count,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "lineage_detail": LINEAGE_DETAIL,
        "identity_format": IDENTITY_FORMAT,
        "producer_component_sha256": components,
        "producer_revision_sha256": canonical_revision_digest(components),
        "request_parameters": dict(CLASSIFICATION_REQUEST_PARAMETERS),
        "configurations": configurations,
        "entries": entries,
    }


class TestHostedAuditReceipt(TestCase):
    def test_validates_exact_seven_by_seven_inventory(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        validated = validate_receipt(receipt)
        self.assertEqual(len(validated["configurations"]), 7)
        self.assertEqual(len(validated["entries"]), 49)
        self.assertEqual(
            [
                (row["benchmark"], row["pregrouper"])
                for row in validated["configurations"]
            ],
            list(CLASSIFICATION_CONFIGURATIONS),
        )
        self.assertEqual(
            [
                (row["benchmark"], row["pregrouper"], row["model"])
                for row in validated["entries"]
            ],
            list(expected_entry_keys()),
        )
        self.assertEqual(len(HOSTED_CLASSIFICATION_MODELS), 7)

    def test_revision_digest_is_order_independent_and_opaque(self) -> None:
        first: str = _digest("first")
        second: str = _digest("second")
        expected: str = hashlib.sha256(
            ("\n".join(sorted((first, second))) + "\n").encode("ascii")
        ).hexdigest()
        self.assertEqual(canonical_revision_digest([first, second]), expected)
        self.assertEqual(canonical_revision_digest([second, first]), expected)
        with self.assertRaises(ValueError):
            canonical_revision_digest([])
        with self.assertRaises(ValueError):
            canonical_revision_digest(["runner=/private/source.py"])

        receipt: dict[str, Any] = _valid_receipt()
        receipt["producer_component_sha256"].reverse()
        with self.assertRaisesRegex(ValueError, "canonically sorted"):
            validate_receipt(receipt)

        receipt = _valid_receipt()
        receipt["producer_revision_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "disagrees with component hashes"):
            validate_receipt(receipt)

        receipt = _valid_receipt()
        receipt["identity_format"] = "unspecified"
        with self.assertRaisesRegex(ValueError, "identity_format"):
            validate_receipt(receipt)

    def test_projection_digest_is_canonical_and_rejects_nonfinite(self) -> None:
        self.assertEqual(
            canonical_projection_digest({"b": [2, 1], "a": "value"}),
            canonical_projection_digest({"a": "value", "b": [2, 1]}),
        )
        with self.assertRaises(ValueError):
            canonical_projection_digest({"value": float("nan")})

    def test_raw_and_table_projections_match_with_censoring_and_failure(self) -> None:
        payload: list[dict[str, Any]] = [
            {
                "prompt_idx": 2,
                "n_segments": 2,
                "orig_label_logprobs": {"true": -1.0, "false": None},
                "original_request_status": "ok",
                "ablated_label_logprobs": [
                    {"true": -2.0, "false": -3.0},
                    None,
                ],
                "ablated_request_statuses": ["ok", "transient_exhausted"],
            }
        ]
        segment_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 2,
                "seg_idx": index,
                "original_request_status": "ok",
                "segment_request_status": status,
                "original_result_available": True,
                "segment_result_available": index == 0,
            }
            for index, status in [(0, "ok"), (1, "transient_exhausted")]
        ]
        token_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 2,
                "seg_idx": seg_idx,
                "kind": kind,
                "label": label,
                "token": label,
                "logprob": value,
                "logprob_granularity": "label_aggregate",
            }
            for seg_idx, kind, values in [
                (None, "orig", {"true": -1.0, "false": -float("inf")}),
                (0, "ablated", {"true": -2.0, "false": -3.0}),
                (1, "ablated", {"true": -float("inf"), "false": -float("inf")}),
            ]
            for label, value in values.items()
        ]
        raw_projection = classification_payload_projection(payload, ("true", "false"))
        table_projection = classification_table_projection(
            segment_rows, token_rows, ("true", "false")
        )
        self.assertEqual(raw_projection, table_projection)
        self.assertEqual(
            canonical_projection_digest(raw_projection),
            canonical_projection_digest(table_projection),
        )

    def test_successful_all_censored_response_is_not_a_request_failure(self) -> None:
        payload: list[dict[str, Any]] = [
            {
                "prompt_idx": 0,
                "n_segments": 1,
                "orig_label_logprobs": {},
                "original_request_status": "ok",
                "ablated_label_logprobs": [{}],
                "ablated_request_statuses": ["ok"],
            }
        ]
        segment_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 0,
                "seg_idx": 0,
                "original_request_status": "ok",
                "segment_request_status": "ok",
                "original_result_available": False,
                "segment_result_available": False,
            }
        ]
        token_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 0,
                "seg_idx": seg_idx,
                "kind": kind,
                "label": label,
                "token": label,
                "logprob": -float("inf"),
                "logprob_granularity": "label_aggregate",
            }
            for seg_idx, kind in [(None, "orig"), (0, "ablated")]
            for label in ("A", "B")
        ]
        self.assertEqual(
            classification_payload_projection(payload, ("A", "B")),
            classification_table_projection(segment_rows, token_rows, ("A", "B")),
        )

    def test_table_projection_rejects_token_semantics_tampering(self) -> None:
        segment_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 0,
                "seg_idx": 0,
                "original_request_status": "ok",
                "segment_request_status": "ok",
                "original_result_available": True,
                "segment_result_available": True,
            }
        ]
        token_rows: list[dict[str, Any]] = [
            {
                "prompt_idx": 0,
                "seg_idx": seg_idx,
                "kind": kind,
                "label": label,
                "token": label,
                "logprob": -1.0,
                "logprob_granularity": "label_aggregate",
            }
            for seg_idx, kind in [(None, "orig"), (0, "ablated")]
            for label in ("A", "B")
        ]
        token_rows[0]["token"] = "tampered"
        with self.assertRaisesRegex(ValueError, "token alias"):
            classification_table_projection(segment_rows, token_rows, ("A", "B"))
        token_rows[0]["token"] = "A"
        token_rows[0]["logprob_granularity"] = "token"
        with self.assertRaisesRegex(ValueError, "label_aggregate"):
            classification_table_projection(segment_rows, token_rows, ("A", "B"))

    def test_rejects_missing_extra_reordered_or_private_fields(self) -> None:
        for mutation in ("missing", "extra", "reordered", "private"):
            with self.subTest(mutation=mutation):
                receipt: dict[str, Any] = _valid_receipt()
                if mutation == "missing":
                    del receipt["entries"][0]["projection_sha256"]
                elif mutation == "extra":
                    receipt["entries"][0]["source_path"] = "raw.json"
                elif mutation == "reordered":
                    receipt["entries"][0], receipt["entries"][1] = (
                        receipt["entries"][1],
                        receipt["entries"][0],
                    )
                else:
                    receipt["private_path"] = "/private/audit"
                with self.assertRaises(ValueError):
                    validate_receipt(receipt)

    def test_rejects_identity_inventory_and_protocol_tampering(self) -> None:
        mutations: list[tuple[str, Any]] = [
            (
                "config hash",
                lambda value: value["configurations"][0].update(
                    dataset_sha256="x" * 64
                ),
            ),
            (
                "config count",
                lambda value: value["configurations"][0].update(prompt_count=1),
            ),
            (
                "model",
                lambda value: value["entries"][0].update(model="private-model-alias"),
            ),
            (
                "raw size",
                lambda value: value["entries"][0].update(raw_artifact_size_bytes=0),
            ),
            (
                "protocol",
                lambda value: value["request_parameters"].update(
                    initial_max_transient_attempts=4
                ),
            ),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                receipt: dict[str, Any] = _valid_receipt()
                mutate(receipt)
                with self.assertRaises(ValueError):
                    validate_receipt(receipt)

        receipt = _valid_receipt()
        receipt["request_parameters"]["max_tokens"] = True
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            validate_receipt(receipt)

    def test_rejects_status_and_availability_inconsistency(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        entry: dict[str, Any] = receipt["entries"][0]
        entry["ablated_status_counts"]["ok"] -= 1
        entry["ablated_status_counts"]["transient_exhausted"] = 1
        with self.assertRaisesRegex(ValueError, "availability_status"):
            validate_receipt(receipt)

        entry["availability_status"] = "complete_with_terminal_failures"
        entry["ablated_result_available_count"] -= 1
        entry["paired_attribution_available_count"] -= 1
        validate_receipt(receipt)

        entry["ablated_result_available_count"] = (
            entry["ablated_status_counts"]["ok"] + 1
        )
        with self.assertRaisesRegex(ValueError, "successful requests"):
            validate_receipt(receipt)

    def test_write_load_and_exact_receipt_sha_verification(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "receipt.json")
            digest: str = write_receipt(path, receipt)
            self.assertEqual(verify_receipt_sha256(path, digest), digest)
            self.assertEqual(load_receipt(path, expected_sha256=digest), receipt)
            with open(path, "rb") as source:
                self.assertEqual(source.read(), canonical_receipt_bytes(receipt))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                verify_receipt_sha256(path, "0" * 64)

            with open(path, "a", encoding="utf-8") as output:
                output.write(" ")
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_receipt(path, expected_sha256=digest)

            alternate: dict[str, Any] = _valid_receipt()
            alternate["entries"][0]["raw_artifact_sha256"] = "f" * 64
            write_receipt(path, alternate)
            validate_receipt(alternate)
            with self.assertRaisesRegex(ValueError, "disagrees"):
                load_receipt(path, expected_sha256=digest)

    def test_boolean_counts_and_unknown_statuses_are_rejected(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        receipt["entries"][0]["original_result_available_count"] = True
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            validate_receipt(receipt)

        receipt = _valid_receipt()
        counts: dict[str, int] = receipt["entries"][0]["original_status_counts"]
        counts["failed_call"] = counts.pop("content_filter")
        with self.assertRaisesRegex(ValueError, "fields disagree"):
            validate_receipt(receipt)

    def test_receipt_has_no_free_form_locator_fields(self) -> None:
        receipt: dict[str, Any] = _valid_receipt()
        serialized: str = json.dumps(receipt, sort_keys=True)
        for forbidden in ("path", "endpoint", "routing", "served_model", "attestation"):
            self.assertNotIn(f'"{forbidden}"', serialized)
