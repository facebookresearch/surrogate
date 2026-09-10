# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Portable audit receipts for hosted LAMBADA completion results.

The receipt contains only public model/configuration identifiers, canonical
content digests, sizes, and aggregate availability counts. Producer paths,
routing aliases, endpoint names, and source filenames are deliberately absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from typing import Any, Mapping, TypedDict, cast

import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.hosted_audit_receipt import (
    canonical_projection_digest,
    canonical_revision_digest,
    sha256_file,
)
from benchmark_scripts.hosted_completion import (
    LAMBADA_GOLD_CANARY,
    LAMBADA_GOLD_CANARY_ANSWERS,
    expand_prompt_completion_payload,
)
from surrogate.model_types import Dialog, make_dialog
from surrogate.text_augmentation import dialog_segments, segment_and_ablate


SCHEMA_VERSION: int = 1
ARTIFACT_TYPE: str = "hosted_completion_audit_receipt"
LINEAGE_DETAIL: str = "final_state_with_posthoc_source_snapshot"
IDENTITY_FORMAT: str = "canonical_jsonl_completion_target_and_full_dialog_v1"
BENCHMARK: str = "lambada"
PREGROUPER: str = "word"
DATASET_SHA256: str = "7bb96e6a12ea76dac59896d2acd9d29599e0b488afc9caedf7140800d8e665b1"

HOSTED_COMPLETION_MODELS: tuple[str, ...] = (
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)

FULL_POPULATION: str = "full_manifest"
CANARY_POPULATION: str = "fixed_canary"
POPULATION_COUNTS: dict[str, tuple[int, int]] = {
    FULL_POPULATION: (4_387, 10_000),
    CANARY_POPULATION: (10, 823),
}
MODEL_SOURCE_POPULATION: dict[str, str] = {
    "llama3.1-8b-instruct": FULL_POPULATION,
    "llama3.1-70b-instruct": FULL_POPULATION,
    "llama3.3-70b-instruct": FULL_POPULATION,
    "llama4-maverick-17b-128e-instruct": FULL_POPULATION,
    "gpt-4o": CANARY_POPULATION,
    "gpt-4-1": CANARY_POPULATION,
    "gemini-2-5-flash-lite-vertex": CANARY_POPULATION,
}
RAW_SOURCE_FORMATS: dict[str, str] = {
    FULL_POPULATION: "hosted_completion_logprob_json",
    CANARY_POPULATION: "hosted_completion_canary_logprob_json",
}
PUBLIC_SOURCE_FORMATS: dict[str, str] = {
    FULL_POPULATION: "hosted_completion_logprob_tsv",
    CANARY_POPULATION: "unsupported_completion_placeholder_after_canary",
}
AVAILABILITY_STATUSES: dict[str, str] = {
    FULL_POPULATION: "complete",
    CANARY_POPULATION: "unsupported_after_canary",
}
COMPLETION_REQUEST_PARAMETERS: dict[str, Any] = {
    "max_tokens": 0,
    "top_logprobs": 20,
    "echo": True,
    "scoring": "teacher_forced_echo_target_logprob_sum",
    "max_transient_attempts": 5,
}

# Filled from an exhaustive public reconstruction and independently checked
# against the producer implementation before a gold receipt is accepted.
POPULATION_IDENTITY_SHA256: dict[str, tuple[str, str, str]] = {
    FULL_POPULATION: (
        "d7f8f565ac80f527d1ced2b4f0a368373846f467f0fc1d1f4d722720f19467d1",
        "da3dc2e64d1a95261f4ed0c8c50f71adbb75c96771b4aedb8bf380d9f3cca089",
        "8505d5a13ced6ca216989d153732bdde913b9361cfaedfcb9cc013ddef4448d9",
    ),
    CANARY_POPULATION: (
        "e9c277e9086430e33a949e2a48ee10343238fb5914cc8e469c796a96c0eb3420",
        "110abbfe542b4dc921d90c3843e5fb634808338d143015901c2289e63a1fca72",
        "66ea712a223fca3b18b565cf75361cf9da99555266389f1f9bf59a41a69e5d91",
    ),
}

_COMPLETION_STATUSES: frozenset[str] = frozenset(
    {
        "ok",
        "unavailable",
        "application_error",
        "echo_response_unavailable",
        "target_parse_unavailable",
        "nonfinite_score",
        "unsupported_after_canary",
    }
)


class CompletionAuditPopulation(TypedDict):
    """Identity seals for one audited completion request population."""

    name: str
    prompt_count: int
    segment_count: int
    coordinate_sha256: str
    prompt_identity_sha256: str
    ablated_dialog_identity_sha256: str


class CompletionAuditEntry(TypedDict):
    """Audit evidence for one hosted LAMBADA model."""

    benchmark: str
    pregrouper: str
    model: str
    source_population: str
    raw_source_format: str
    public_source_format: str
    producer_revision_sha256: str
    raw_artifact_sha256: str
    raw_artifact_size_bytes: int
    raw_projection_sha256: str
    public_artifact_sha256: str
    public_artifact_size_bytes: int
    public_projection_sha256: str
    availability_status: str
    raw_prompt_count: int
    raw_segment_count: int
    public_prompt_count: int
    public_segment_count: int
    raw_original_available_count: int
    raw_ablated_available_count: int
    raw_paired_attribution_available_count: int
    public_original_available_count: int
    public_ablated_available_count: int
    public_paired_attribution_available_count: int


class HostedCompletionAuditReceipt(TypedDict):
    """Complete sanitized receipt for all hosted LAMBADA artifacts."""

    schema_version: int
    artifact_type: str
    lineage_detail: str
    identity_format: str
    dataset_sha256: str
    manifest_sha256: str
    producer_component_sha256: list[str]
    finalization_revision_sha256: str
    request_parameters: dict[str, Any]
    populations: list[CompletionAuditPopulation]
    entries: list[CompletionAuditEntry]


class CompletionProjectionSummary(TypedDict):
    """Availability summary of one canonical completion projection."""

    prompt_count: int
    segment_count: int
    original_available_count: int
    ablated_available_count: int
    paired_attribution_available_count: int


_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    HostedCompletionAuditReceipt.__required_keys__
)
_POPULATION_FIELDS: frozenset[str] = frozenset(
    CompletionAuditPopulation.__required_keys__
)
_ENTRY_FIELDS: frozenset[str] = frozenset(CompletionAuditEntry.__required_keys__)


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


def _nonnegative_int(value: Any, field: str) -> int:
    if not _is_int(value) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return cast(int, value)


def _exact_fields(
    value: Any, expected: frozenset[str], field: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual: set[Any] = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{field} fields disagree: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return cast(Mapping[str, Any], value)


def _score_projection(value: Any) -> list[str | None]:
    """Represent one score without non-standard JSON NaN or infinity values."""

    if value is None:
        return ["missing", None]
    if isinstance(value, bool):
        raise ValueError("Completion scores must be numeric, not boolean")
    numeric: float = float(value)
    if math.isnan(numeric):
        return ["missing", None]
    if math.isinf(numeric):
        return ["positive_inf" if numeric > 0 else "negative_inf", None]
    return ["finite", numeric.hex()]


def _normalized_status(value: Any, explicit: Any, field: str) -> str:
    state: str = _score_projection(value)[0] or ""
    explicit_missing: bool = explicit is None or (
        isinstance(explicit, float) and math.isnan(explicit)
    )
    status: str = (
        ("ok" if state == "finite" else "unavailable")
        if explicit_missing
        else str(explicit)
    )
    if status == "ok" and state != "finite":
        status = "nonfinite_score"
    if status not in _COMPLETION_STATUSES:
        raise ValueError(f"{field} has invalid completion status {status!r}")
    if status != "ok" and state == "finite":
        raise ValueError(f"{field} status disagrees with its finite score")
    return status


def _canonical_record(
    prompt_idx: int,
    kind: str,
    seg_idx: int | None,
    value: Any,
    explicit_status: Any,
) -> list[Any]:
    status: str = _normalized_status(
        value, explicit_status, f"prompt {prompt_idx}/{kind}/{seg_idx}"
    )
    return [prompt_idx, kind, seg_idx, status, _score_projection(value)]


def completion_payload_projection(
    payload: list[dict[str, Any]],
    manifest: pd.DataFrame,
    expected_prompt_indices: list[int] | None = None,
) -> list[list[Any]]:
    """Project a flat full run or prompt-shaped canary onto public semantics."""

    if not payload:
        raise ValueError("Completion payload must not be empty")
    flat: bool = all("ablation_idx" in row for row in payload)
    if flat != any("ablation_idx" in row for row in payload):
        raise ValueError("Completion payload mixes flat and prompt-shaped rows")
    rows: list[dict[str, Any]] = (
        payload
        if flat
        else expand_prompt_completion_payload(
            payload,
            manifest,
            expected_prompt_indices=expected_prompt_indices,
        )
    )
    expected_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in manifest[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError("Completion manifest contains duplicate coordinates")
    indexed_manifest: pd.DataFrame = manifest.set_index(["prompt_idx", "seg_idx"])
    seen_keys: set[tuple[int, int]] = set()
    originals: dict[int, list[Any]] = {}
    records: list[list[Any]] = []
    for row in rows:
        prompt_idx: int = int(row["prompt_idx"])
        seg_idx: int = int(row["ablation_idx"])
        key: tuple[int, int] = (prompt_idx, seg_idx)
        if key in seen_keys:
            raise ValueError(f"Duplicate completion coordinate {key}")
        seen_keys.add(key)
        if key not in expected_keys:
            raise ValueError(f"Completion coordinate {key} is outside the population")
        metadata: pd.Series = indexed_manifest.loc[key]
        if str(row.get("answer")) != str(metadata["answer"]) or int(
            row.get("n_segments", -1)
        ) != int(metadata["n_segments"]):
            raise ValueError(f"Completion coordinate {key} metadata disagrees")
        original: list[Any] = _canonical_record(
            prompt_idx,
            "orig",
            None,
            row.get("orig_logprob"),
            row.get("orig_status"),
        )
        if prompt_idx in originals and originals[prompt_idx] != original:
            raise ValueError(f"Prompt {prompt_idx} has inconsistent original result")
        originals[prompt_idx] = original
        records.append(
            _canonical_record(
                prompt_idx,
                "ablated",
                seg_idx,
                row.get("ablated_logprob"),
                row.get("ablated_status"),
            )
        )
    if seen_keys != expected_keys:
        raise ValueError(
            "Completion coordinates differ from the audited population: "
            f"{len(expected_keys - seen_keys)} missing, "
            f"{len(seen_keys - expected_keys)} extra"
        )
    records.extend(originals.values())
    return sorted(
        records,
        key=lambda record: (
            int(record[0]),
            0 if record[1] == "orig" else 1,
            -1 if record[2] is None else int(record[2]),
        ),
    )


def completion_table_projection(segment_rows: list[dict[str, Any]]) -> list[list[Any]]:
    """Project a shipped completion segment table onto receipt semantics."""

    if not segment_rows:
        raise ValueError("Completion segment table must not be empty")
    seen_keys: set[tuple[int, int]] = set()
    originals: dict[int, list[Any]] = {}
    records: list[list[Any]] = []
    for row in segment_rows:
        prompt_idx: int = int(row["prompt_idx"])
        seg_idx: int = int(row["seg_idx"])
        key: tuple[int, int] = (prompt_idx, seg_idx)
        if key in seen_keys:
            raise ValueError(f"Duplicate completion table coordinate {key}")
        seen_keys.add(key)
        original: list[Any] = _canonical_record(
            prompt_idx,
            "orig",
            None,
            row.get("orig_completion_logprob"),
            row.get("original_result_status"),
        )
        if prompt_idx in originals and originals[prompt_idx] != original:
            raise ValueError(f"Prompt {prompt_idx} has inconsistent public original")
        originals[prompt_idx] = original
        records.append(
            _canonical_record(
                prompt_idx,
                "ablated",
                seg_idx,
                row.get("ablated_completion_logprob"),
                row.get("segment_result_status"),
            )
        )
    records.extend(originals.values())
    return sorted(
        records,
        key=lambda record: (
            int(record[0]),
            0 if record[1] == "orig" else 1,
            -1 if record[2] is None else int(record[2]),
        ),
    )


def projection_summary(projection: list[list[Any]]) -> CompletionProjectionSummary:
    """Summarize finite original, ablated, and paired completion coverage."""

    originals: dict[int, bool] = {
        int(row[0]): row[4][0] == "finite" for row in projection if row[1] == "orig"
    }
    ablated: list[list[Any]] = [row for row in projection if row[1] == "ablated"]
    return {
        "prompt_count": len(originals),
        "segment_count": len(ablated),
        "original_available_count": sum(originals.values()),
        "ablated_available_count": sum(row[4][0] == "finite" for row in ablated),
        "paired_attribution_available_count": sum(
            originals[int(row[0])] and row[4][0] == "finite" for row in ablated
        ),
    }


def _messages(dialog: Dialog) -> list[list[str]]:
    return [[message.role, message.content] for message in dialog.messages]


def _line(value: list[Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


async def compute_completion_dialog_identity(
    dataset_path: str,
    manifest: pd.DataFrame,
) -> tuple[str, str, str]:
    """Hash coordinates, targets, prompts, and ablated dialogs for a population."""

    spec = BENCHMARKS[BENCHMARK]
    frame: pd.DataFrame = pd.read_csv(dataset_path, sep="\t")
    if spec.dataset_preprocessor is not None:
        frame = spec.dataset_preprocessor(frame.copy())
    grouped: dict[int, pd.DataFrame] = {
        int(prompt_idx): rows.sort_values("seg_idx")
        for prompt_idx, rows in manifest.groupby("prompt_idx", sort=False)
    }
    coordinate_hash = hashlib.sha256()
    prompt_hash = hashlib.sha256()
    ablated_hash = hashlib.sha256()
    for prompt_idx in sorted(grouped):
        if prompt_idx < 0 or prompt_idx >= len(frame):
            raise ValueError(f"Manifest prompt {prompt_idx} is outside LAMBADA")
        row: pd.Series = frame.iloc[prompt_idx]
        target: str = str(row[spec.target_column])
        system_prompt: str = spec.system_prompt_override or ""
        dialog: Dialog = make_dialog(system_prompt, spec.prompt_builder(row))
        prompt_hash.update(_line([prompt_idx, target, _messages(dialog)]))
        segments = dialog_segments(dialog, pregrouper_id=PREGROUPER)
        ablated: list[Dialog] = await segment_and_ablate(
            dialog, pregrouper_id=PREGROUPER
        )
        selected: pd.DataFrame = grouped[prompt_idx]
        declared_values: set[int] = set(selected["n_segments"].astype(int))
        if len(declared_values) != 1 or next(iter(declared_values)) != len(segments):
            raise ValueError(f"Segment count mismatch at LAMBADA prompt {prompt_idx}")
        if len(segments) != len(ablated):
            raise ValueError(f"Ablation count mismatch at LAMBADA prompt {prompt_idx}")
        for manifest_row in selected.itertuples(index=False):
            seg_idx: int = int(manifest_row.seg_idx)
            segment = segments[seg_idx]
            if (
                str(manifest_row.answer) != target
                or segment.message_idx != int(manifest_row.message_idx)
                or segment.message_role != str(manifest_row.message_role)
                or segment.message_segment_idx != int(manifest_row.message_seg_idx)
                or segment.text != str(manifest_row.segment_text)
            ):
                raise ValueError(
                    f"Manifest identity mismatch at LAMBADA/{prompt_idx}/{seg_idx}"
                )
            coordinate_hash.update(
                _line(
                    [
                        prompt_idx,
                        seg_idx,
                        target,
                        len(segments),
                        segment.message_idx,
                        segment.message_role,
                        segment.message_segment_idx,
                        segment.text,
                    ]
                )
            )
            ablated_hash.update(
                _line([prompt_idx, seg_idx, target, _messages(ablated[seg_idx])])
            )
    return (
        coordinate_hash.hexdigest(),
        prompt_hash.hexdigest(),
        ablated_hash.hexdigest(),
    )


def canary_manifest(dataset_path: str) -> pd.DataFrame:
    """Construct the audited 10-prompt, 823-segment LAMBADA canary manifest."""

    spec = BENCHMARKS[BENCHMARK]
    frame: pd.DataFrame = pd.read_csv(dataset_path, sep="\t")
    rows: list[dict[str, Any]] = []
    expected_counts: dict[int, int] = dict(LAMBADA_GOLD_CANARY)
    for prompt_idx, expected_count in LAMBADA_GOLD_CANARY:
        source: pd.Series = frame.iloc[prompt_idx]
        target: str = str(source[spec.target_column])
        if target != LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx]:
            raise ValueError(f"Canary target mismatch at LAMBADA prompt {prompt_idx}")
        dialog: Dialog = make_dialog(
            spec.system_prompt_override or "", spec.prompt_builder(source)
        )
        segments = dialog_segments(dialog, pregrouper_id=PREGROUPER)
        if len(segments) != expected_count:
            raise ValueError(f"Canary segment count mismatch at prompt {prompt_idx}")
        for segment in segments:
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "answer": target,
                    "seg_idx": segment.segment_idx,
                    "message_idx": segment.message_idx,
                    "message_role": segment.message_role,
                    "message_seg_idx": segment.message_segment_idx,
                    "segment_text": segment.text,
                    "n_segments": len(segments),
                }
            )
    manifest: pd.DataFrame = pd.DataFrame(rows)
    if len(manifest) != POPULATION_COUNTS[CANARY_POPULATION][1]:
        raise ValueError("Canary manifest does not contain exactly 823 segments")
    return manifest


def _validate_population(value: Any, expected_name: str) -> CompletionAuditPopulation:
    record: Mapping[str, Any] = _exact_fields(value, _POPULATION_FIELDS, "population")
    if record["name"] != expected_name:
        raise ValueError(f"Population inventory/order disagrees at {expected_name}")
    expected_prompts, expected_segments = POPULATION_COUNTS[expected_name]
    if (
        record["prompt_count"] != expected_prompts
        or not _is_int(record["prompt_count"])
        or record["segment_count"] != expected_segments
        or not _is_int(record["segment_count"])
    ):
        raise ValueError(f"Population counts disagree for {expected_name}")
    for field in (
        "coordinate_sha256",
        "prompt_identity_sha256",
        "ablated_dialog_identity_sha256",
    ):
        _require_sha256(record[field], f"population.{field}")
    expected_identity: tuple[str, str, str] = POPULATION_IDENTITY_SHA256[expected_name]
    actual_identity: tuple[str, str, str] = (
        str(record["coordinate_sha256"]),
        str(record["prompt_identity_sha256"]),
        str(record["ablated_dialog_identity_sha256"]),
    )
    if actual_identity != expected_identity:
        raise ValueError(f"Population identity digests disagree for {expected_name}")
    return cast(CompletionAuditPopulation, dict(record))


def _validate_entry(value: Any, expected_model: str) -> CompletionAuditEntry:
    record: Mapping[str, Any] = _exact_fields(value, _ENTRY_FIELDS, "entry")
    if (
        record["benchmark"] != BENCHMARK
        or record["pregrouper"] != PREGROUPER
        or record["model"] != expected_model
    ):
        raise ValueError(
            f"Completion entry inventory/order disagrees at {expected_model}"
        )
    population: str = MODEL_SOURCE_POPULATION[expected_model]
    if (
        record["source_population"] != population
        or record["raw_source_format"] != RAW_SOURCE_FORMATS[population]
        or record["public_source_format"] != PUBLIC_SOURCE_FORMATS[population]
        or record["availability_status"] != AVAILABILITY_STATUSES[population]
    ):
        raise ValueError(f"Completion source semantics disagree for {expected_model}")
    for field in (
        "producer_revision_sha256",
        "raw_artifact_sha256",
        "raw_projection_sha256",
        "public_artifact_sha256",
        "public_projection_sha256",
    ):
        _require_sha256(record[field], f"entry.{field}")
    if _nonnegative_int(record["raw_artifact_size_bytes"], "raw size") == 0:
        raise ValueError("entry.raw_artifact_size_bytes must be positive")
    if _nonnegative_int(record["public_artifact_size_bytes"], "public size") == 0:
        raise ValueError("entry.public_artifact_size_bytes must be positive")
    raw_prompts, raw_segments = POPULATION_COUNTS[population]
    if (
        _nonnegative_int(record["raw_prompt_count"], "entry.raw_prompt_count")
        != raw_prompts
        or _nonnegative_int(record["raw_segment_count"], "entry.raw_segment_count")
        != raw_segments
    ):
        raise ValueError(
            f"Completion entry population counts disagree for {expected_model}"
        )
    if (
        _nonnegative_int(record["public_prompt_count"], "entry.public_prompt_count")
        != POPULATION_COUNTS[FULL_POPULATION][0]
        or _nonnegative_int(
            record["public_segment_count"], "entry.public_segment_count"
        )
        != POPULATION_COUNTS[FULL_POPULATION][1]
    ):
        raise ValueError(f"Public completion grid counts disagree for {expected_model}")
    count_fields: tuple[tuple[str, int], ...] = (
        ("raw_original_available_count", raw_prompts),
        ("raw_ablated_available_count", raw_segments),
        ("raw_paired_attribution_available_count", raw_segments),
        ("public_original_available_count", POPULATION_COUNTS[FULL_POPULATION][0]),
        ("public_ablated_available_count", POPULATION_COUNTS[FULL_POPULATION][1]),
        (
            "public_paired_attribution_available_count",
            POPULATION_COUNTS[FULL_POPULATION][1],
        ),
    )
    for field, upper in count_fields:
        count: int = _nonnegative_int(record[field], f"entry.{field}")
        if count > upper:
            raise ValueError(f"entry.{field} exceeds its request population")
    if population == FULL_POPULATION:
        if record["raw_projection_sha256"] != record["public_projection_sha256"]:
            raise ValueError(
                f"Full completion projections disagree for {expected_model}"
            )
        thresholds: tuple[tuple[str, int], ...] = (
            ("public_original_available_count", POPULATION_COUNTS[FULL_POPULATION][0]),
            ("public_ablated_available_count", POPULATION_COUNTS[FULL_POPULATION][1]),
            (
                "public_paired_attribution_available_count",
                POPULATION_COUNTS[FULL_POPULATION][1],
            ),
        )
        for field, total in thresholds:
            if int(record[field]) / total < 0.8:
                raise ValueError(
                    f"Completion coverage is below 80% for {expected_model}"
                )
    elif any(
        int(record[field]) != 0
        for field in (
            "public_original_available_count",
            "public_ablated_available_count",
            "public_paired_attribution_available_count",
        )
    ):
        raise ValueError(
            f"Unsupported placeholder is not all-missing for {expected_model}"
        )
    return cast(CompletionAuditEntry, dict(record))


def validate_receipt(value: Any) -> HostedCompletionAuditReceipt:
    """Validate and return a complete sanitized hosted-completion receipt."""

    receipt: Mapping[str, Any] = _exact_fields(value, _TOP_LEVEL_FIELDS, "receipt")
    if receipt["schema_version"] != SCHEMA_VERSION or not _is_int(
        receipt["schema_version"]
    ):
        raise ValueError(f"Unsupported schema_version {receipt['schema_version']!r}")
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
    if receipt["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Completion receipt dataset SHA-256 disagrees")
    _require_sha256(receipt["manifest_sha256"], "manifest_sha256")
    raw_components: Any = receipt["producer_component_sha256"]
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("producer_component_sha256 must be a non-empty list")
    components: list[str] = [
        _require_sha256(component, f"producer_component_sha256[{index}]")
        for index, component in enumerate(raw_components)
    ]
    if components != sorted(components) or len(components) != len(set(components)):
        raise ValueError("producer_component_sha256 must be sorted and unique")
    revision: str = _require_sha256(
        receipt["finalization_revision_sha256"], "finalization_revision_sha256"
    )
    if revision != canonical_revision_digest(components):
        raise ValueError("finalization_revision_sha256 disagrees with components")
    parameters: Mapping[str, Any] = _exact_fields(
        receipt["request_parameters"],
        frozenset(COMPLETION_REQUEST_PARAMETERS),
        "request_parameters",
    )
    for field, expected in COMPLETION_REQUEST_PARAMETERS.items():
        if (
            type(parameters[field]) is not type(expected)
            or parameters[field] != expected
        ):
            raise ValueError(f"Completion request parameter {field!r} disagrees")
    raw_populations: Any = receipt["populations"]
    if not isinstance(raw_populations, list) or len(raw_populations) != 2:
        raise ValueError("Receipt must contain exactly two completion populations")
    populations: list[CompletionAuditPopulation] = [
        _validate_population(row, name)
        for row, name in zip(raw_populations, (FULL_POPULATION, CANARY_POPULATION))
    ]
    raw_entries: Any = receipt["entries"]
    if not isinstance(raw_entries, list) or len(raw_entries) != len(
        HOSTED_COMPLETION_MODELS
    ):
        raise ValueError("Receipt must contain exactly seven completion entries")
    entries: list[CompletionAuditEntry] = [
        _validate_entry(row, model)
        for row, model in zip(raw_entries, HOSTED_COMPLETION_MODELS)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "lineage_detail": LINEAGE_DETAIL,
        "identity_format": IDENTITY_FORMAT,
        "dataset_sha256": DATASET_SHA256,
        "manifest_sha256": str(receipt["manifest_sha256"]),
        "producer_component_sha256": components,
        "finalization_revision_sha256": revision,
        "request_parameters": dict(parameters),
        "populations": populations,
        "entries": entries,
    }


def canonical_receipt_bytes(value: Any) -> bytes:
    """Return validated canonical JSON bytes for one receipt."""

    receipt: HostedCompletionAuditReceipt = validate_receipt(value)
    return (
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_receipt(path: str, value: Any) -> str:
    """Atomically write a validated receipt and return its SHA-256."""

    payload: bytes = canonical_receipt_bytes(value)
    directory: str = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=directory, delete=False) as output:
        temporary: str = output.name
        output.write(payload)
    os.replace(temporary, path)
    return hashlib.sha256(payload).hexdigest()


def verify_receipt_sha256(path: str, expected_sha256: str) -> str:
    """Require exact receipt bytes to match a trusted SHA-256."""

    _require_sha256(expected_sha256, "expected_sha256")
    actual: str = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"Hosted completion receipt SHA-256 disagrees: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


def load_receipt(
    path: str, expected_sha256: str | None = None
) -> HostedCompletionAuditReceipt:
    """Load and validate a receipt, optionally requiring exact trusted bytes."""

    if expected_sha256 is not None:
        verify_receipt_sha256(path, expected_sha256)
    with open(path, encoding="utf-8") as source:
        loaded: Any = json.load(source)
    return validate_receipt(loaded)


__all__: list[str] = [
    "ARTIFACT_TYPE",
    "AVAILABILITY_STATUSES",
    "BENCHMARK",
    "CANARY_POPULATION",
    "COMPLETION_REQUEST_PARAMETERS",
    "DATASET_SHA256",
    "FULL_POPULATION",
    "HOSTED_COMPLETION_MODELS",
    "HostedCompletionAuditReceipt",
    "IDENTITY_FORMAT",
    "LINEAGE_DETAIL",
    "MODEL_SOURCE_POPULATION",
    "POPULATION_COUNTS",
    "POPULATION_IDENTITY_SHA256",
    "PREGROUPER",
    "PUBLIC_SOURCE_FORMATS",
    "RAW_SOURCE_FORMATS",
    "SCHEMA_VERSION",
    "canonical_projection_digest",
    "canonical_receipt_bytes",
    "canonical_revision_digest",
    "canary_manifest",
    "completion_payload_projection",
    "completion_table_projection",
    "compute_completion_dialog_identity",
    "load_receipt",
    "projection_summary",
    "sha256_file",
    "validate_receipt",
    "verify_receipt_sha256",
    "write_receipt",
]
