# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Validate portable receipts for hosted classification audits.

The receipt deliberately contains only public benchmark/model identifiers,
content digests, sizes, and aggregate counts.  Producer paths, routing aliases,
endpoint names, source filenames, and free-form attestations are not part of the
schema.  The raw hosted responses remain separate inputs identified by digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Any, Iterable, Mapping, TypedDict, cast

from benchmark_scripts.dialog_identity import IDENTITY_FORMAT
from surrogate.eval_constants import report_token_alias


SCHEMA_VERSION: int = 1
ARTIFACT_TYPE: str = "hosted_classification_audit_receipt"
LINEAGE_DETAIL: str = "final_state_with_posthoc_source_snapshot"

CLASSIFICATION_CONFIGURATIONS: tuple[tuple[str, str], ...] = (
    ("boolq", "sentence"),
    ("anli_r1", "sentence"),
    ("anli_r2", "sentence"),
    ("anli_r3", "sentence"),
    ("winogrande", "sentence"),
    ("race", "sentence"),
    ("boolq", "word"),
)

HOSTED_CLASSIFICATION_MODELS: tuple[str, ...] = (
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)

# Exact request populations in the frozen public segment manifests.
CONFIGURATION_COUNTS: dict[tuple[str, str], tuple[int, int]] = {
    ("boolq", "sentence"): (3_270, 27_516),
    ("anli_r1", "sentence"): (1_000, 8_181),
    ("anli_r2", "sentence"): (1_000, 8_163),
    ("anli_r3", "sentence"): (1_200, 10_028),
    ("winogrande", "sentence"): (1_267, 9_135),
    ("race", "sentence"): (4_934, 145_544),
    ("boolq", "word"): (3_028, 10_000),
}

# Canonical JSONL digests from an exhaustive comparison of the producer-side
# prompts/ablations with the frozen public prompt builders and segment grids.
CONFIGURATION_IDENTITY_SHA256: dict[tuple[str, str], tuple[str, str]] = {
    ("boolq", "sentence"): (
        "938f9bafe6ca520297c93130c33ce41f63e14cd396634da292ed9e76e9859d6d",
        "b64c4ce2582211eaa549f6c7c1611973b9fb7bc5e15c271ab25fdd0e4726f571",
    ),
    ("anli_r1", "sentence"): (
        "27e71314bf6f8986e62e917d85c99a5241b35140a73de222d175e7aede7cab0d",
        "f49e6554c841eb1820594d6d77fddb48327f85ca3e186e83cac765b7973a6804",
    ),
    ("anli_r2", "sentence"): (
        "7a8516aab625103bbb8adc1eb21d09c8724a03c43073f4d8e6eb2e0d96f73a55",
        "e159fa41beaea8d1e665449d2cf3158e7524768079953bae97e253d93ed8c80f",
    ),
    ("anli_r3", "sentence"): (
        "c982b57b5165dae9a371a0aae92d5131c6538d90b1eed57d7facaacea274073e",
        "52b229a8015dd1889d032a0a22ee2d22a111bd3fa74786740402a3b2f34629c6",
    ),
    ("winogrande", "sentence"): (
        "28ba18107712a1591cb6744b391730c42263e2048a2d3f3291c6d0aa8bca48ee",
        "72957a56ec740d23c11c177e5080432bd50f6d4a81529e9007457cfd24904616",
    ),
    ("race", "sentence"): (
        "8a6304df5ba98951c6959a0bd99354583ca405acf012640e7d19b45ed4b87c32",
        "70aebb614718419e55a2cfe30b41e4f37fd1e80dce34b165c7a7aef3977af539",
    ),
    ("boolq", "word"): (
        "3956afdc62bbb6792b14114e3af235b58adb9da599b8de9ee13dcd39a8047285",
        "610531a869676404b87b5912179e694570cef35a9c26f94500a7b7a2fa3559c8",
    ),
}

CLASSIFICATION_REQUEST_PARAMETERS: dict[str, Any] = {
    "max_tokens": 1,
    "top_logprobs": 19,
    "echo": False,
    "scoring": "first_generated_token_label_logprobs",
    "initial_max_transient_attempts": 5,
    "final_audit_attempts_per_failed_coordinate_per_round": 5,
    "final_audit_max_rounds": 3,
    "final_audit_rechecks_content_filters": True,
    "content_filter_policy": "explicit_missing_after_confirmation",
    "terminal_failure_policy": "explicit_nan_after_final_audit",
}

STATUS_NAMES: tuple[str, ...] = (
    "ok",
    "content_filter",
    "transient_exhausted",
)
AVAILABILITY_STATUSES: frozenset[str] = frozenset(
    {"complete", "complete_with_terminal_failures"}
)


class StatusCounts(TypedDict):
    """Exhaustive request-status counts for one request population."""

    ok: int
    content_filter: int
    transient_exhausted: int


class HostedAuditConfiguration(TypedDict):
    """Portable identity seals shared by all models in one configuration."""

    benchmark: str
    pregrouper: str
    prompt_count: int
    segment_count: int
    dataset_sha256: str
    manifest_sha256: str
    prompt_identity_sha256: str
    ablated_dialog_identity_sha256: str


class HostedAuditEntry(TypedDict):
    """Audit evidence for one hosted model/configuration pair."""

    benchmark: str
    pregrouper: str
    model: str
    producer_revision_sha256: str
    raw_artifact_sha256: str
    raw_artifact_size_bytes: int
    projection_sha256: str
    availability_status: str
    original_status_counts: StatusCounts
    ablated_status_counts: StatusCounts
    original_result_available_count: int
    ablated_result_available_count: int
    paired_attribution_available_count: int


class HostedAuditReceipt(TypedDict):
    """Complete sanitized receipt for all hosted classification artifacts."""

    schema_version: int
    artifact_type: str
    lineage_detail: str
    identity_format: str
    producer_component_sha256: list[str]
    producer_revision_sha256: str
    request_parameters: dict[str, Any]
    configurations: list[HostedAuditConfiguration]
    entries: list[HostedAuditEntry]


_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "artifact_type",
        "lineage_detail",
        "identity_format",
        "producer_component_sha256",
        "producer_revision_sha256",
        "request_parameters",
        "configurations",
        "entries",
    }
)
_CONFIGURATION_FIELDS: frozenset[str] = frozenset(
    HostedAuditConfiguration.__required_keys__
)
_ENTRY_FIELDS: frozenset[str] = frozenset(HostedAuditEntry.__required_keys__)
_STATUS_FIELDS: frozenset[str] = frozenset(StatusCounts.__required_keys__)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: Any, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return cast(str, value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonnegative_int(value: Any, field: str) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return cast(int, value)


def _require_exact_fields(
    value: Any, expected: frozenset[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual: set[Any] = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{field} fields disagree: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return cast(Mapping[str, Any], value)


def canonical_revision_digest(component_hashes: Iterable[str]) -> str:
    """Hash a sorted collection of opaque component hashes.

    Only hash values affect or appear in the digest; local component names and
    source paths never enter the public receipt. Duplicate values are retained.
    """

    if isinstance(component_hashes, (str, bytes)):
        raise TypeError("component_hashes must be an iterable of SHA-256 strings")
    components: list[str] = list(component_hashes)
    if not components:
        raise ValueError("component_hashes must not be empty")
    for index, digest in enumerate(components):
        _require_sha256(digest, f"component_hashes[{index}]")
    canonical: bytes = ("\n".join(sorted(components)) + "\n").encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def canonical_projection_digest(projection: Any) -> str:
    """Hash a JSON-compatible sanitized projection deterministically."""

    try:
        encoded: bytes = json.dumps(
            projection,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("projection must be finite JSON-compatible data") from error
    return hashlib.sha256(encoded).hexdigest()


def _projected_status(payload: Any, explicit_status: Any, field: str) -> str:
    status: str = (
        str(explicit_status)
        if explicit_status is not None
        else ("ok" if isinstance(payload, dict) else "")
    )
    if status not in STATUS_NAMES:
        raise ValueError(f"{field} has invalid request status {status!r}")
    if (status == "ok") != isinstance(payload, dict):
        raise ValueError(f"{field} request status disagrees with its payload")
    return status


def _projected_labels(
    payload: Any,
    status: str,
    expected_labels: tuple[str, ...],
    field: str,
) -> list[list[str]]:
    if status != "ok":
        return [[label, "unavailable"] for label in expected_labels]
    if not isinstance(payload, dict):
        raise ValueError(f"{field} successful response is not a label mapping")
    missing: set[str] = set(expected_labels) - set(payload)
    extra: set[str] = set(payload) - set(expected_labels)
    if extra or (payload and missing):
        raise ValueError(
            f"{field} label inventory disagrees: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    projected: list[list[str]] = []
    for label in expected_labels:
        value: Any = payload.get(label)
        if value is None:
            canonical_value: str = "-inf"
        else:
            numeric: float = float(value)
            if math.isnan(numeric) or numeric == float("inf"):
                raise ValueError(f"{field}.{label} is not a valid log-probability")
            canonical_value = "-inf" if numeric <= -1e300 else numeric.hex()
        projected.append([label, canonical_value])
    return projected


def classification_payload_projection(
    payload: list[dict[str, Any]], expected_labels: tuple[str, ...]
) -> list[list[Any]]:
    """Project raw classification JSON onto public response semantics."""

    records: list[list[Any]] = []
    seen_prompts: set[int] = set()
    for row in payload:
        prompt_idx: int = int(row["prompt_idx"])
        if prompt_idx in seen_prompts:
            raise ValueError(f"Duplicate prompt {prompt_idx} in classification payload")
        seen_prompts.add(prompt_idx)
        original_payload: Any = row.get("orig_label_logprobs")
        original_status: str = _projected_status(
            original_payload,
            row.get("original_request_status"),
            f"prompt {prompt_idx} original",
        )
        records.append(
            [
                prompt_idx,
                "orig",
                None,
                original_status,
                _projected_labels(
                    original_payload,
                    original_status,
                    expected_labels,
                    f"prompt {prompt_idx} original",
                ),
            ]
        )
        ablated_payloads: Any = row.get("ablated_label_logprobs")
        if not isinstance(ablated_payloads, list):
            raise ValueError(f"Prompt {prompt_idx} has no coordinate-level ablations")
        raw_indices: Any = row.get("ablation_indices")
        indices: list[int] = (
            [int(value) for value in raw_indices]
            if isinstance(raw_indices, list)
            else list(range(int(row["n_segments"])))
        )
        raw_statuses: Any = row.get("ablated_request_statuses")
        statuses: list[Any] = (
            list(raw_statuses)
            if isinstance(raw_statuses, list)
            else [None] * len(ablated_payloads)
        )
        if not (len(indices) == len(ablated_payloads) == len(statuses)):
            raise ValueError(f"Prompt {prompt_idx} ablation arrays disagree in length")
        for seg_idx, ablated_payload, explicit_status in zip(
            indices, ablated_payloads, statuses
        ):
            status: str = _projected_status(
                ablated_payload,
                explicit_status,
                f"prompt {prompt_idx} segment {seg_idx}",
            )
            records.append(
                [
                    prompt_idx,
                    "ablated",
                    seg_idx,
                    status,
                    _projected_labels(
                        ablated_payload,
                        status,
                        expected_labels,
                        f"prompt {prompt_idx} segment {seg_idx}",
                    ),
                ]
            )
    return sorted(
        records,
        key=lambda record: (
            int(record[0]),
            0 if record[1] == "orig" else 1,
            -1 if record[2] is None else int(record[2]),
        ),
    )


def classification_table_projection(
    segment_rows: list[dict[str, Any]],
    token_rows: list[dict[str, Any]],
    expected_labels: tuple[str, ...],
) -> list[list[Any]]:
    """Project shipped classification tables onto the same response semantics."""

    token_values: dict[tuple[int, str, int | None, str], Any] = {}
    for row in token_rows:
        label: str = str(row["label"])
        if row.get("logprob_granularity") != "label_aggregate":
            raise ValueError(
                "Hosted classification token rows must use label_aggregate "
                "granularity"
            )
        if row.get("token") != report_token_alias(label):
            raise ValueError(
                f"Hosted classification token alias disagrees for label {label!r}"
            )
        seg_idx_value: Any = row.get("seg_idx")
        seg_idx: int | None = (
            None
            if seg_idx_value is None
            or (isinstance(seg_idx_value, float) and math.isnan(seg_idx_value))
            else int(seg_idx_value)
        )
        key: tuple[int, str, int | None, str] = (
            int(row["prompt_idx"]),
            str(row["kind"]),
            seg_idx,
            label,
        )
        if key in token_values:
            raise ValueError(f"Duplicate projected token key {key}")
        token_values[key] = row.get("logprob")

    records: list[list[Any]] = []
    originals: dict[int, tuple[str, bool]] = {}
    for row in segment_rows:
        prompt_idx: int = int(row["prompt_idx"])
        original: tuple[str, bool] = (
            str(row["original_request_status"]),
            bool(row["original_result_available"]),
        )
        if prompt_idx in originals and originals[prompt_idx] != original:
            raise ValueError(f"Prompt {prompt_idx} has inconsistent original status")
        originals[prompt_idx] = original
        seg_idx: int = int(row["seg_idx"])
        status: str = str(row["segment_request_status"])
        available: bool = bool(row["segment_result_available"])
        if status != "ok" and available:
            raise ValueError(
                f"Prompt {prompt_idx} segment {seg_idx} status/availability disagree"
            )
        mapping: dict[str, Any] | None = (
            {
                label: token_values[(prompt_idx, "ablated", seg_idx, label)]
                for label in expected_labels
            }
            if status == "ok"
            else None
        )
        records.append(
            [
                prompt_idx,
                "ablated",
                seg_idx,
                status,
                _projected_labels(
                    mapping,
                    status,
                    expected_labels,
                    f"prompt {prompt_idx} segment {seg_idx}",
                ),
            ]
        )
    for prompt_idx, (status, available) in originals.items():
        if status != "ok" and available:
            raise ValueError(
                f"Prompt {prompt_idx} original status/availability disagree"
            )
        mapping = (
            {
                label: token_values[(prompt_idx, "orig", None, label)]
                for label in expected_labels
            }
            if status == "ok"
            else None
        )
        records.append(
            [
                prompt_idx,
                "orig",
                None,
                status,
                _projected_labels(
                    mapping,
                    status,
                    expected_labels,
                    f"prompt {prompt_idx} original",
                ),
            ]
        )
    return sorted(
        records,
        key=lambda record: (
            int(record[0]),
            0 if record[1] == "orig" else 1,
            -1 if record[2] is None else int(record[2]),
        ),
    )


def _validate_status_counts(
    value: Any, expected_total: int, field: str
) -> StatusCounts:
    counts: Mapping[str, Any] = _require_exact_fields(value, _STATUS_FIELDS, field)
    normalized: dict[str, int] = {
        status: _require_nonnegative_int(counts[status], f"{field}.{status}")
        for status in STATUS_NAMES
    }
    if sum(normalized.values()) != expected_total:
        raise ValueError(
            f"{field} must sum to {expected_total}, got {sum(normalized.values())}"
        )
    return cast(StatusCounts, normalized)


def _validate_request_parameters(value: Any) -> dict[str, Any]:
    parameters: Mapping[str, Any] = _require_exact_fields(
        value,
        frozenset(CLASSIFICATION_REQUEST_PARAMETERS),
        "request_parameters",
    )
    for field, expected in CLASSIFICATION_REQUEST_PARAMETERS.items():
        actual: Any = parameters[field]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"Classification request parameter {field!r} disagrees: "
                f"expected {expected!r}, got {actual!r}"
            )
    return dict(parameters)


def _validate_configuration(
    value: Any, expected_key: tuple[str, str]
) -> HostedAuditConfiguration:
    record: Mapping[str, Any] = _require_exact_fields(
        value, _CONFIGURATION_FIELDS, "configuration"
    )
    key: tuple[Any, Any] = (record["benchmark"], record["pregrouper"])
    if key != expected_key:
        raise ValueError(f"Configuration inventory/order disagrees at {expected_key}")
    expected_prompts, expected_segments = CONFIGURATION_COUNTS[expected_key]
    if record["prompt_count"] != expected_prompts or not _is_int(
        record["prompt_count"]
    ):
        raise ValueError(f"Incorrect prompt_count for {expected_key}")
    if record["segment_count"] != expected_segments or not _is_int(
        record["segment_count"]
    ):
        raise ValueError(f"Incorrect segment_count for {expected_key}")
    for field in (
        "dataset_sha256",
        "manifest_sha256",
        "prompt_identity_sha256",
        "ablated_dialog_identity_sha256",
    ):
        _require_sha256(record[field], f"configuration.{field}")
    if (
        record["prompt_identity_sha256"],
        record["ablated_dialog_identity_sha256"],
    ) != CONFIGURATION_IDENTITY_SHA256[expected_key]:
        raise ValueError(f"Identity digests disagree for {expected_key}")
    return cast(HostedAuditConfiguration, dict(record))


def _validate_entry(value: Any, expected_key: tuple[str, str, str]) -> HostedAuditEntry:
    record: Mapping[str, Any] = _require_exact_fields(value, _ENTRY_FIELDS, "entry")
    key: tuple[Any, Any, Any] = (
        record["benchmark"],
        record["pregrouper"],
        record["model"],
    )
    if key != expected_key:
        raise ValueError(f"Entry inventory/order disagrees at {expected_key}")
    config_key: tuple[str, str] = expected_key[:2]
    expected_prompts, expected_segments = CONFIGURATION_COUNTS[config_key]
    _require_sha256(
        record["producer_revision_sha256"], "entry.producer_revision_sha256"
    )
    _require_sha256(record["raw_artifact_sha256"], "entry.raw_artifact_sha256")
    _require_sha256(record["projection_sha256"], "entry.projection_sha256")
    raw_size: int = _require_nonnegative_int(
        record["raw_artifact_size_bytes"], "entry.raw_artifact_size_bytes"
    )
    if raw_size == 0:
        raise ValueError("entry.raw_artifact_size_bytes must be positive")
    original_counts: StatusCounts = _validate_status_counts(
        record["original_status_counts"], expected_prompts, "original_status_counts"
    )
    ablated_counts: StatusCounts = _validate_status_counts(
        record["ablated_status_counts"], expected_segments, "ablated_status_counts"
    )
    availability_status: Any = record["availability_status"]
    if availability_status not in AVAILABILITY_STATUSES:
        raise ValueError(f"Invalid availability_status {availability_status!r}")
    has_terminal_failures: bool = bool(
        original_counts["transient_exhausted"] or ablated_counts["transient_exhausted"]
    )
    if has_terminal_failures != (
        availability_status == "complete_with_terminal_failures"
    ):
        raise ValueError("availability_status disagrees with terminal-failure counts")

    original_available: int = _require_nonnegative_int(
        record["original_result_available_count"],
        "entry.original_result_available_count",
    )
    ablated_available: int = _require_nonnegative_int(
        record["ablated_result_available_count"],
        "entry.ablated_result_available_count",
    )
    paired_available: int = _require_nonnegative_int(
        record["paired_attribution_available_count"],
        "entry.paired_attribution_available_count",
    )
    if original_available > original_counts["ok"]:
        raise ValueError("Original result availability exceeds successful requests")
    if ablated_available > ablated_counts["ok"]:
        raise ValueError("Ablated result availability exceeds successful requests")
    if paired_available > ablated_available:
        raise ValueError("Paired attribution availability exceeds ablated availability")
    return cast(HostedAuditEntry, dict(record))


def expected_entry_keys() -> tuple[tuple[str, str, str], ...]:
    """Return the canonical seven-configuration by seven-model inventory."""

    return tuple(
        (benchmark, pregrouper, model)
        for benchmark, pregrouper in CLASSIFICATION_CONFIGURATIONS
        for model in HOSTED_CLASSIFICATION_MODELS
    )


def validate_receipt(value: Any) -> HostedAuditReceipt:
    """Validate and return one complete sanitized hosted audit receipt."""

    receipt: Mapping[str, Any] = _require_exact_fields(
        value, _TOP_LEVEL_FIELDS, "receipt"
    )
    if receipt["schema_version"] != SCHEMA_VERSION or not _is_int(
        receipt["schema_version"]
    ):
        raise ValueError(
            f"Unsupported receipt schema_version {receipt['schema_version']!r}"
        )
    if receipt["artifact_type"] != ARTIFACT_TYPE:
        raise ValueError(f"Invalid receipt artifact_type {receipt['artifact_type']!r}")
    if receipt["lineage_detail"] != LINEAGE_DETAIL:
        raise ValueError(
            f"Invalid receipt lineage_detail {receipt['lineage_detail']!r}"
        )
    if receipt["identity_format"] != IDENTITY_FORMAT:
        raise ValueError(
            f"Invalid receipt identity_format {receipt['identity_format']!r}"
        )
    raw_components: Any = receipt["producer_component_sha256"]
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("producer_component_sha256 must be a non-empty list")
    components: list[str] = [
        _require_sha256(value, f"producer_component_sha256[{index}]")
        for index, value in enumerate(raw_components)
    ]
    if components != sorted(components):
        raise ValueError("producer_component_sha256 must be canonically sorted")
    revision: str = _require_sha256(
        receipt["producer_revision_sha256"], "receipt.producer_revision_sha256"
    )
    if revision != canonical_revision_digest(components):
        raise ValueError("producer_revision_sha256 disagrees with component hashes")
    request_parameters: dict[str, Any] = _validate_request_parameters(
        receipt["request_parameters"]
    )

    configurations: Any = receipt["configurations"]
    if not isinstance(configurations, list) or len(configurations) != len(
        CLASSIFICATION_CONFIGURATIONS
    ):
        raise ValueError("Receipt must contain exactly seven configurations")
    validated_configurations: list[HostedAuditConfiguration] = [
        _validate_configuration(record, expected_key)
        for record, expected_key in zip(configurations, CLASSIFICATION_CONFIGURATIONS)
    ]

    entries: Any = receipt["entries"]
    keys: tuple[tuple[str, str, str], ...] = expected_entry_keys()
    if not isinstance(entries, list) or len(entries) != len(keys):
        raise ValueError(
            "Receipt must contain exactly 49 hosted classification entries"
        )
    validated_entries: list[HostedAuditEntry] = [
        _validate_entry(record, expected_key)
        for record, expected_key in zip(entries, keys)
    ]
    if any(
        entry["producer_revision_sha256"] != revision for entry in validated_entries
    ):
        raise ValueError(
            "Entry producer_revision_sha256 disagrees with the receipt revision"
        )
    return cast(
        HostedAuditReceipt,
        {
            **dict(receipt),
            "producer_component_sha256": components,
            "request_parameters": request_parameters,
            "configurations": validated_configurations,
            "entries": validated_entries,
        },
    )


def canonical_receipt_bytes(receipt: Any) -> bytes:
    """Validate a receipt and return its canonical UTF-8 JSON representation."""

    validated: HostedAuditReceipt = validate_receipt(receipt)
    return (
        json.dumps(
            validated,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: str) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_receipt_sha256(path: str, expected_sha256: str) -> str:
    """Verify an exact receipt-file digest and return the normalized digest."""

    expected: str = _require_sha256(expected_sha256, "expected receipt SHA-256")
    actual: str = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Receipt SHA-256 disagrees: expected {expected}, got {actual}"
        )
    return actual


def load_receipt(
    path: str, *, expected_sha256: str | None = None
) -> HostedAuditReceipt:
    """Load and validate a receipt, optionally verifying its exact file hash."""

    if expected_sha256 is not None:
        verify_receipt_sha256(path, expected_sha256)
    with open(path, encoding="utf-8") as source:
        value: Any = json.load(source)
    return validate_receipt(value)


def write_receipt(path: str, receipt: Any) -> str:
    """Write canonical validated receipt bytes and return their SHA-256 digest."""

    payload: bytes = canonical_receipt_bytes(receipt)
    parent: str = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = output.name
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return hashlib.sha256(payload).hexdigest()
