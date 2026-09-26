# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Import hosted-model JSON outputs into the portable public TSV schema.

This utility contains no service client. It converts already-produced result
JSON containing per-label log-probabilities or prompt-level completion scores
into the same tables consumed by the public postprocessing pipeline. Because
the legacy classification JSON stores label aggregates rather than individual
token variants, those rows are marked
``logprob_granularity=label_aggregate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from typing import Any

import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.hosted_audit_receipt import (
    canonical_projection_digest,
    classification_payload_projection,
    load_receipt,
    sha256_file,
)
from benchmark_scripts.hosted_completion import (
    LAMBADA_GOLD_CANARY,
    LAMBADA_GOLD_CANARY_ANSWERS,
    expand_prompt_completion_payload,
)
from benchmark_scripts.hosted_completion_audit_receipt import (
    CANARY_POPULATION,
    FULL_POPULATION,
    MODEL_SOURCE_POPULATION,
    RAW_SOURCE_FORMATS,
    canonical_projection_digest as canonical_completion_projection_digest,
    completion_payload_projection,
    load_receipt as load_completion_receipt,
)
from benchmark_scripts.provenance_sources import (
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    HOSTED_IMPORT_SOURCE_FILES,
)
from benchmark_scripts.record_hosted_provenance import _copy_and_summarize_canary
from surrogate.eval_constants import report_token_alias

GZIP_COMPRESSION: dict[str, Any] = {
    "method": "gzip",
    "compresslevel": 9,
    "mtime": 0,
}
CLASSIFICATION_REQUEST_STATUSES: set[str] = {
    "ok",
    "content_filter",
    "failed_call",
    "transient_exhausted",
}
AVAILABILITY_STATUSES: set[str] = {
    "complete",
    "complete_with_terminal_failures",
    "unsupported_after_canary",
}
HOSTED_AUDIT_RECEIPT_NAME: str = "hosted_classification_audit_receipt.json"
HOSTED_COMPLETION_AUDIT_RECEIPT_NAME: str = "hosted_completion_audit_receipt.json"


def prepare_producer_audit_receipt(
    receipt_path: str,
    input_path: str,
    manifest_path: str,
    output_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    producer_metadata: dict[str, Any],
    availability_status: str,
) -> dict[str, str]:
    """Verify, install, and reference one entry in a producer audit receipt."""

    receipt = load_receipt(
        receipt_path,
        expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    )
    configurations = {
        (row["benchmark"], row["pregrouper"]): row for row in receipt["configurations"]
    }
    entries = {
        (row["benchmark"], row["pregrouper"], row["model"]): row
        for row in receipt["entries"]
    }
    configuration = configurations[(benchmark, pregrouper)]
    entry = entries[(benchmark, pregrouper, model)]
    if configuration["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Producer audit receipt manifest hash disagrees")
    if entry["raw_artifact_sha256"] != _sha256(input_path) or entry[
        "raw_artifact_size_bytes"
    ] != os.path.getsize(input_path):
        raise ValueError("Producer audit receipt raw artifact identity disagrees")
    if entry["producer_revision_sha256"] != producer_metadata.get("producer_revision"):
        raise ValueError("Producer audit receipt revision disagrees")
    if receipt["request_parameters"] != producer_metadata.get("request_parameters"):
        raise ValueError("Producer audit receipt request protocol disagrees")
    if entry["availability_status"] != availability_status:
        raise ValueError("Producer audit receipt availability status disagrees")
    with open(input_path, encoding="utf-8") as source:
        raw_payload: Any = json.load(source)
    if not isinstance(raw_payload, list) or not all(
        isinstance(row, dict) for row in raw_payload
    ):
        raise ValueError("Producer audit receipt input is not a prompt-row list")
    eval_config = BENCHMARKS[benchmark].eval_config
    if eval_config is None:
        raise ValueError("Producer audit receipt is only valid for classification")
    projection_sha256: str = canonical_projection_digest(
        classification_payload_projection(
            raw_payload,
            tuple(eval_config.label_tokens),
        )
    )
    if entry["projection_sha256"] != projection_sha256:
        raise ValueError("Producer audit receipt response projection disagrees")

    receipt_sha256: str = GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256
    results_root: str = os.path.dirname(os.path.dirname(output_dir))
    installed_path: str = os.path.join(results_root, HOSTED_AUDIT_RECEIPT_NAME)
    if os.path.abspath(receipt_path) != os.path.abspath(installed_path):
        os.makedirs(results_root, exist_ok=True)
        if (
            os.path.exists(installed_path)
            and sha256_file(installed_path) != receipt_sha256
        ):
            raise ValueError("A different producer audit receipt is already installed")
        shutil.copyfile(receipt_path, installed_path)
    return {
        "path": HOSTED_AUDIT_RECEIPT_NAME,
        "sha256": receipt_sha256,
        "entry_id": f"{benchmark}/{pregrouper}/{model}",
    }


def prepare_completion_audit_receipt(
    receipt_path: str,
    input_path: str,
    manifest_path: str,
    output_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    producer_metadata: dict[str, Any],
    availability_status: str,
) -> dict[str, str]:
    """Verify, install, and reference one hosted-completion audit entry."""

    if (benchmark, pregrouper) != ("lambada", "word"):
        raise ValueError("Completion audit receipts apply only to lambada/word")
    receipt = load_completion_receipt(
        receipt_path,
        expected_sha256=GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    )
    entry = next(row for row in receipt["entries"] if row["model"] == model)
    if receipt["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError("Completion audit receipt manifest hash disagrees")
    if entry["raw_artifact_sha256"] != _sha256(input_path) or entry[
        "raw_artifact_size_bytes"
    ] != os.path.getsize(input_path):
        raise ValueError("Completion audit receipt raw artifact identity disagrees")
    if entry["producer_revision_sha256"] != producer_metadata.get("producer_revision"):
        raise ValueError("Completion audit receipt revision disagrees")
    if receipt["request_parameters"] != producer_metadata.get("request_parameters"):
        raise ValueError("Completion audit receipt request protocol disagrees")
    if entry["availability_status"] != availability_status:
        raise ValueError("Completion audit receipt availability status disagrees")
    population: str = MODEL_SOURCE_POPULATION[model]
    if (
        entry["source_population"] != population
        or entry["raw_source_format"] != RAW_SOURCE_FORMATS[population]
    ):
        raise ValueError("Completion audit receipt source population disagrees")

    with open(input_path, encoding="utf-8") as source:
        raw_payload: Any = json.load(source)
    if not isinstance(raw_payload, list) or not all(
        isinstance(row, dict) for row in raw_payload
    ):
        raise ValueError("Completion audit receipt input is not a result-row list")
    manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
    expected_prompt_indices: list[int] | None = None
    projection_manifest: pd.DataFrame = manifest
    if population == CANARY_POPULATION:
        expected_prompt_indices = [prompt_idx for prompt_idx, _ in LAMBADA_GOLD_CANARY]
        projection_manifest = pd.DataFrame(
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
    elif population != FULL_POPULATION:
        raise ValueError(f"Unknown completion population {population!r}")
    raw_projection_sha256: str = canonical_completion_projection_digest(
        completion_payload_projection(
            raw_payload,
            projection_manifest,
            expected_prompt_indices=expected_prompt_indices,
        )
    )
    if entry["raw_projection_sha256"] != raw_projection_sha256:
        raise ValueError("Completion audit receipt response projection disagrees")

    receipt_sha256: str = GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256
    results_root: str = os.path.dirname(os.path.dirname(output_dir))
    installed_path: str = os.path.join(
        results_root, HOSTED_COMPLETION_AUDIT_RECEIPT_NAME
    )
    if os.path.abspath(receipt_path) != os.path.abspath(installed_path):
        os.makedirs(results_root, exist_ok=True)
        if (
            os.path.exists(installed_path)
            and sha256_file(installed_path) != receipt_sha256
        ):
            raise ValueError(
                "A different completion audit receipt is already installed"
            )
        shutil.copyfile(receipt_path, installed_path)
    return {
        "path": HOSTED_COMPLETION_AUDIT_RECEIPT_NAME,
        "sha256": receipt_sha256,
        "entry_id": f"{benchmark}/{pregrouper}/{model}",
    }


def _classification_request_status(
    value: dict[str, float | None] | None,
    raw_status: Any,
) -> str:
    """Validate an optional producer status against the returned label map."""
    inferred: str = "ok" if value is not None else "failed_call"
    if raw_status is None:
        return inferred
    status: str = str(raw_status)
    if status not in CLASSIFICATION_REQUEST_STATUSES:
        raise ValueError(f"Unknown hosted classification request status {status!r}")
    if (value is not None) != (status == "ok"):
        raise ValueError(
            f"Hosted classification status {status!r} disagrees with its payload"
        )
    return status


def _transformation_source_hashes() -> dict[str, str]:
    """Hash every public source file used to normalize hosted JSON."""
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    return {
        relative_path: _sha256(os.path.join(repository_root, relative_path))
        for relative_path in HOSTED_IMPORT_SOURCE_FILES
    }


TOKEN_COLUMNS: tuple[str, ...] = (
    "prompt_idx",
    "seg_idx",
    "kind",
    "answer",
    "label",
    "token",
    "logprob",
    "logprob_granularity",
)


def _has_finite_label(
    label_logprobs: dict[str, float | None] | None,
) -> bool:
    return label_logprobs is not None and any(
        value is not None and math.isfinite(value) and value > -1e300
        for value in label_logprobs.values()
    )


def _has_finite_score(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _validate_canary_source(
    input_path: str,
    output_dir: str,
    canary_metadata: dict[str, Any],
) -> None:
    """Require the imported source to be the canary copied into the release."""

    artifact_name: Any = canary_metadata.get("artifact_path")
    if (
        not isinstance(artifact_name, str)
        or os.path.basename(artifact_name) != artifact_name
    ):
        raise ValueError("Unsupported canary artifact path must be a basename")
    artifact_path: str = os.path.join(output_dir, artifact_name)
    if not os.path.isfile(artifact_path) or _sha256(
        artifact_path
    ) != canary_metadata.get("artifact_sha256"):
        raise ValueError("Unsupported canary artifact identity disagrees")
    with open(input_path, encoding="utf-8") as source:
        source_payload: Any = json.load(source)
    with open(artifact_path, encoding="utf-8") as source:
        public_payload: Any = json.load(source)
    if not isinstance(source_payload, list) or not isinstance(public_payload, list):
        raise ValueError("Unsupported canary source must be a JSON list")
    public_fields: tuple[str, ...] = (
        "prompt_idx",
        "answer",
        "n_segments",
        "orig_logprob",
        "ablated_logprob",
        "ablation_indices",
    )
    sanitized: list[dict[str, Any]] = [
        {field: row[field] for field in public_fields if field in row}
        for row in source_payload
        if isinstance(row, dict)
    ]
    if len(sanitized) != len(source_payload) or sanitized != public_payload:
        raise ValueError(
            "Unsupported placeholder input must be the audited canary artifact"
        )


def _label_rows(
    label_logprobs: dict[str, float | None] | None,
    expected_labels: tuple[str, ...] | None,
    prompt_idx: int,
    seg_idx: int | None,
    kind: str,
    answer: Any,
) -> list[dict[str, Any]]:
    if label_logprobs is None and expected_labels is None:
        return []

    observed: dict[str, float | None] = label_logprobs or {}
    labels: tuple[str, ...] = expected_labels or tuple(observed)
    unexpected: set[str] = set(observed) - set(labels)
    if unexpected:
        raise ValueError(
            f"Hosted output contains unexpected labels {sorted(unexpected)}"
        )
    missing: set[str] = set(labels) - set(observed)
    # An empty mapping is the legacy representation for a successful response
    # in which none of the requested labels appeared in top-k. A non-empty
    # mapping must be complete so malformed partial payloads cannot masquerade
    # as ordinary top-k censoring.
    if observed and missing:
        raise ValueError(f"Hosted output is missing expected labels {sorted(missing)}")

    def _normalize_logprob(logprob: float) -> float:
        # Some legacy JSON serializers encoded -inf as -DBL_MAX.
        return -float("inf") if logprob <= -1e300 else logprob

    return [
        {
            "prompt_idx": prompt_idx,
            "seg_idx": seg_idx,
            "kind": kind,
            "answer": answer,
            "label": label,
            "token": report_token_alias(label),
            "logprob": (
                -float("inf") if logprob is None else _normalize_logprob(logprob)
            ),
            "logprob_granularity": "label_aggregate",
        }
        for label in labels
        for logprob in [observed.get(label)]
    ]


def import_results(
    input_path: str,
    manifest_path: str,
    output_dir: str,
    model_name: str,
    benchmark: str | None = None,
    pregrouper: str | None = None,
    identity_attestation: str | None = None,
    producer_metadata: dict[str, Any] | None = None,
    availability_status: str = "complete",
    canary_metadata: dict[str, Any] | None = None,
    producer_audit_receipt: dict[str, str] | None = None,
) -> None:
    """Convert one hosted-model JSON file to public segment/token TSVs."""
    with open(input_path, encoding="utf-8") as source:
        loaded: Any = json.load(source)
    if (
        not isinstance(loaded, list)
        or not loaded
        or not all(isinstance(row, dict) for row in loaded)
    ):
        raise ValueError("Hosted result JSON must be a non-empty list of objects")
    payload: list[dict[str, Any]] = loaded
    if (benchmark is not None or pregrouper is not None) and not (
        identity_attestation is not None and identity_attestation.strip()
    ):
        raise ValueError(
            "Hosted imports require a non-empty segment identity attestation"
        )
    if benchmark is not None or pregrouper is not None:
        required_producer_fields: set[str] = {
            "producer_revision",
            "generated_at",
            "served_model",
            "request_parameters",
        }
        missing_producer_fields: set[str] = required_producer_fields - set(
            producer_metadata or {}
        )
        if missing_producer_fields:
            raise ValueError(
                "Hosted imports require producer metadata fields "
                f"{sorted(missing_producer_fields)}"
            )
        if not isinstance((producer_metadata or {}).get("request_parameters"), dict):
            raise ValueError("request_parameters must be a JSON object")
    if availability_status not in AVAILABILITY_STATUSES:
        raise ValueError(f"Unsupported availability status {availability_status!r}")
    if availability_status == "complete_with_terminal_failures" and (
        benchmark == "lambada"
    ):
        raise ValueError(
            "Classification terminal-failure status is not valid for LAMBADA"
        )
    if availability_status == "unsupported_after_canary" and not canary_metadata:
        raise ValueError("Unsupported outputs require non-empty canary metadata")
    if availability_status == "unsupported_after_canary":
        if canary_metadata is None:
            raise ValueError("Unsupported outputs require non-empty canary metadata")
        _validate_canary_source(input_path, output_dir, canary_metadata)
    if producer_audit_receipt is not None and set(producer_audit_receipt) != {
        "path",
        "sha256",
        "entry_id",
    }:
        raise ValueError(
            "producer_audit_receipt must contain exactly path, sha256, and entry_id"
        )
    manifest: pd.DataFrame = pd.read_csv(
        manifest_path,
        sep="\t",
        keep_default_na=False,
        na_values=[""],
    )
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError("Segment manifest contains duplicate keys")
    manifest_by_prompt: dict[int, pd.DataFrame] = {
        int(prompt_idx): rows.sort_values("seg_idx")
        for prompt_idx, rows in manifest.groupby("prompt_idx", sort=False)
    }
    expected_labels: tuple[str, ...] | None = None
    if benchmark is not None:
        if benchmark not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark {benchmark!r}")
        eval_config = BENCHMARKS[benchmark].eval_config
        if eval_config is not None:
            expected_labels = tuple(eval_config.label_tokens)

    if availability_status == "unsupported_after_canary":
        if benchmark != "lambada" or pregrouper != "word":
            raise ValueError(
                "Unsupported canary placeholders are only valid for lambada/word"
            )
        placeholder_payload: list[dict[str, Any]] = [
            {
                "prompt_idx": int(row.prompt_idx),
                "ablation_idx": int(row.seg_idx),
                "answer": row.answer,
                "n_segments": int(row.n_segments),
                "orig_logprob": None,
                "ablated_logprob": None,
                "orig_status": "unsupported_after_canary",
                "ablated_status": "unsupported_after_canary",
            }
            for row in manifest.itertuples(index=False)
        ]
        _import_completion_results(
            placeholder_payload,
            manifest,
            input_path,
            output_dir,
            model_name,
            benchmark,
            pregrouper,
            manifest_path,
            identity_attestation,
            producer_metadata,
            availability_status,
            canary_metadata,
            source_format="unsupported_completion_placeholder_after_canary",
            producer_audit_receipt=producer_audit_receipt,
        )
        return

    flat_completion: bool = any("ablation_idx" in row for row in payload)
    completion_scores: bool = any(
        "orig_logprob" in row or "ablated_logprob" in row for row in payload
    )
    if flat_completion or completion_scores:
        completion_payload: list[dict[str, Any]] = payload
        if flat_completion and not all("ablation_idx" in row for row in payload):
            raise ValueError("Hosted completion payload mixes flat and prompt rows")
        if not flat_completion:
            completion_payload = expand_prompt_completion_payload(payload, manifest)
        _import_completion_results(
            completion_payload,
            manifest,
            input_path,
            output_dir,
            model_name,
            benchmark,
            pregrouper,
            manifest_path,
            identity_attestation,
            producer_metadata,
            availability_status,
            canary_metadata,
            source_format="hosted_completion_logprob_json",
            producer_audit_receipt=producer_audit_receipt,
        )
        return

    segment_frames: list[pd.DataFrame] = []
    token_rows: list[dict[str, Any]] = []
    seen_prompts: set[int] = set()
    for result in payload:
        prompt_idx: int = int(result["prompt_idx"])
        if prompt_idx in seen_prompts:
            raise ValueError(f"Duplicate prompt {prompt_idx} in {input_path}")
        seen_prompts.add(prompt_idx)
        if prompt_idx not in manifest_by_prompt:
            raise ValueError(f"Prompt {prompt_idx} is absent from segment manifest")
        prompt_segments: pd.DataFrame = manifest_by_prompt[prompt_idx].copy()
        manifest_answers: set[str] = set(prompt_segments["answer"].map(str))
        if manifest_answers != {str(result["answer"])}:
            raise ValueError(
                f"Prompt {prompt_idx}: answer {result['answer']!r} disagrees "
                f"with manifest answer(s) {sorted(manifest_answers)}"
            )
        n_segments: int = int(result["n_segments"])
        declared_counts: set[int] = set(prompt_segments["n_segments"].astype(int))
        if declared_counts != {n_segments}:
            raise ValueError(
                f"Prompt {prompt_idx}: JSON n_segments={n_segments} disagrees "
                f"with manifest values {sorted(declared_counts)}"
            )
        prompt_segments["answer"] = result["answer"]
        original_label_logprobs: dict[str, float | None] | None = result.get(
            "orig_label_logprobs"
        )
        prompt_segments["original_request_status"] = _classification_request_status(
            original_label_logprobs,
            result.get("original_request_status"),
        )
        prompt_segments["original_result_available"] = _has_finite_label(
            original_label_logprobs
        )
        token_rows.extend(
            _label_rows(
                original_label_logprobs,
                expected_labels,
                prompt_idx,
                None,
                "orig",
                result["answer"],
            )
        )
        ablated: list[dict[str, float | None] | None] | None = result.get(
            "ablated_label_logprobs"
        )
        ablation_indices_value: Any = result.get("ablation_indices")
        ablation_indices: list[int] = (
            [int(value) for value in ablation_indices_value]
            if ablation_indices_value is not None
            else list(range(n_segments))
        )
        manifest_indices: list[int] = prompt_segments["seg_idx"].astype(int).tolist()
        if ablation_indices != manifest_indices:
            raise ValueError(
                f"Prompt {prompt_idx}: ablation indices disagree with manifest"
            )
        if ablated is None or len(ablated) != len(ablation_indices):
            raise ValueError(
                f"Prompt {prompt_idx} lacks {len(ablation_indices)} ablated "
                "label mappings"
            )
        raw_segment_statuses: Any = result.get("ablated_request_statuses")
        if raw_segment_statuses is not None and (
            not isinstance(raw_segment_statuses, list)
            or len(raw_segment_statuses) != len(ablated)
        ):
            raise ValueError(
                f"Prompt {prompt_idx}: ablated request statuses disagree with "
                "the ablation payload"
            )
        prompt_segments["segment_request_status"] = [
            _classification_request_status(
                label_logprobs,
                (
                    raw_segment_statuses[index]
                    if isinstance(raw_segment_statuses, list)
                    else None
                ),
            )
            for index, label_logprobs in enumerate(ablated)
        ]
        prompt_segments["segment_result_available"] = [
            _has_finite_label(label_logprobs) for label_logprobs in ablated
        ]
        segment_frames.append(prompt_segments)
        for seg_idx, label_logprobs in zip(ablation_indices, ablated):
            token_rows.extend(
                _label_rows(
                    label_logprobs,
                    expected_labels,
                    prompt_idx,
                    seg_idx,
                    "ablated",
                    result["answer"],
                )
            )

    expected_prompts: set[int] = set(manifest_by_prompt)
    if seen_prompts != expected_prompts:
        raise ValueError(
            f"Hosted output prompt coverage differs from manifest: "
            f"{len(expected_prompts - seen_prompts)} missing, "
            f"{len(seen_prompts - expected_prompts)} extra"
        )
    os.makedirs(output_dir, exist_ok=True)
    segment_output: str = os.path.join(output_dir, f"{model_name}_segment.tsv.gz")
    token_output: str = os.path.join(output_dir, f"{model_name}_tokens.tsv.gz")
    pd.concat(segment_frames, ignore_index=True).to_csv(
        segment_output,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )
    pd.DataFrame(token_rows, columns=TOKEN_COLUMNS).to_csv(
        token_output,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "source_sha256": _sha256(input_path),
        "source_size_bytes": os.path.getsize(input_path),
        "transformation_source_sha256": _transformation_source_hashes(),
        "source_format": "hosted_label_logprob_json",
        "logprob_granularity": "label_aggregate",
        "segmentation_scope": "full_dialog_in_message_order",
        "prompts": len(seen_prompts),
        "segments": len(manifest),
        "manifest_sha256": _sha256(manifest_path),
        "identity_attestation": identity_attestation,
        "producer": producer_metadata,
        "availability_status": availability_status,
        "canary": canary_metadata,
        "producer_audit_receipt": producer_audit_receipt,
    }
    metadata["artifact_sha256"] = {
        "segment": _sha256(segment_output),
        "tokens": _sha256(token_output),
    }
    metadata_path: str = os.path.join(output_dir, f"{model_name}_run.json")
    with open(metadata_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def _import_completion_results(
    payload: list[dict[str, Any]],
    manifest: pd.DataFrame,
    input_path: str,
    output_dir: str,
    model_name: str,
    benchmark: str | None,
    pregrouper: str | None,
    manifest_path: str,
    identity_attestation: str | None,
    producer_metadata: dict[str, Any] | None,
    availability_status: str,
    canary_metadata: dict[str, Any] | None,
    source_format: str,
    producer_audit_receipt: dict[str, str] | None,
) -> None:
    """Import flat, forward-pass-budgeted completion ablation records."""
    manifest_indexed: pd.DataFrame = manifest.set_index(["prompt_idx", "seg_idx"])
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[int, int]] = set()
    for result in payload:
        key: tuple[int, int] = (
            int(result["prompt_idx"]),
            int(result["ablation_idx"]),
        )
        if key in seen_keys:
            raise ValueError(f"Duplicate completion key {key} in {input_path}")
        seen_keys.add(key)
        if key not in manifest_indexed.index:
            raise ValueError(f"Completion key {key} is absent from segment manifest")
        metadata: dict[str, Any] = manifest_indexed.loc[key].to_dict()
        if str(result["answer"]) != str(metadata["answer"]):
            raise ValueError(f"Completion key {key} has an answer mismatch")
        if int(result["n_segments"]) != int(metadata["n_segments"]):
            raise ValueError(f"Completion key {key} has inconsistent n_segments")
        original_available: bool = _has_finite_score(result.get("orig_logprob"))
        segment_available: bool = _has_finite_score(result.get("ablated_logprob"))
        original_status: Any = result.get(
            "orig_status", "ok" if original_available else "unavailable"
        )
        segment_status: Any = result.get(
            "ablated_status", "ok" if segment_available else "unavailable"
        )
        if original_status == "ok" and not original_available:
            original_status = "nonfinite_score"
        if segment_status == "ok" and not segment_available:
            segment_status = "nonfinite_score"
        rows.append(
            {
                "prompt_idx": key[0],
                "seg_idx": key[1],
                **metadata,
                "answer": result["answer"],
                "orig_completion_logprob": result.get("orig_logprob"),
                "ablated_completion_logprob": result.get("ablated_logprob"),
                "original_result_available": original_available,
                "segment_result_available": segment_available,
                "original_result_status": original_status,
                "segment_result_status": segment_status,
            }
        )
    expected_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in manifest[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }
    if seen_keys != expected_keys:
        raise ValueError(
            f"Completion keys differ from manifest: "
            f"{len(expected_keys - seen_keys)} missing, "
            f"{len(seen_keys - expected_keys)} extra"
        )
    os.makedirs(output_dir, exist_ok=True)
    segment_output: str = os.path.join(output_dir, f"{model_name}_segment.tsv.gz")
    pd.DataFrame(rows).to_csv(
        segment_output,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )
    metadata_output: dict[str, Any] = {
        "schema_version": 1,
        "model": model_name,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "source_sha256": _sha256(input_path),
        "source_size_bytes": os.path.getsize(input_path),
        "transformation_source_sha256": _transformation_source_hashes(),
        "source_format": source_format,
        "segmentation_scope": "full_dialog_in_message_order",
        "segments": len(rows),
        "manifest_sha256": _sha256(manifest_path),
        "identity_attestation": identity_attestation,
        "producer": producer_metadata,
        "availability_status": availability_status,
        "canary": canary_metadata,
        "producer_audit_receipt": producer_audit_receipt,
    }
    metadata_output["artifact_sha256"] = {
        "segment": _sha256(segment_output),
    }
    metadata_path: str = os.path.join(output_dir, f"{model_name}_run.json")
    with open(metadata_path, "w", encoding="utf-8") as output:
        json.dump(metadata_output, output, indent=2, sort_keys=True)
        output.write("\n")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pregrouper", choices=["sentence", "word"], required=True)
    parser.add_argument(
        "--identity-attestation",
        required=True,
        help=(
            "Human-auditable statement describing how the producing runner's "
            "segment identities were matched to the manifest."
        ),
    )
    parser.add_argument("--producer-revision", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--request-parameters-json",
        required=True,
        help="JSON object containing generation parameters such as top_logprobs.",
    )
    parser.add_argument(
        "--availability-status",
        choices=sorted(AVAILABILITY_STATUSES),
        default="complete",
    )
    parser.add_argument(
        "--producer-audit-receipt",
        default=None,
        help=(
            "Sanitized producer-audit receipt. Required by gold validation for "
            "classification and completion imports."
        ),
    )
    parser.add_argument("--canary-file", default=None)
    parser.add_argument("--canary-seed", type=int, default=None)
    args: argparse.Namespace = parser.parse_args()
    request_parameters: Any = json.loads(args.request_parameters_json)
    if not isinstance(request_parameters, dict):
        raise ValueError("--request-parameters-json must decode to an object")
    if args.availability_status == "unsupported_after_canary" and (
        args.canary_file is None or args.canary_seed is None
    ):
        raise ValueError(
            "--canary-file and --canary-seed are required for unsupported output"
        )
    canary_metadata: dict[str, Any] | None = (
        _copy_and_summarize_canary(
            args.canary_file,
            args.output_dir,
            args.model,
            args.canary_seed,
        )
        if args.canary_file is not None and args.canary_seed is not None
        else None
    )
    producer_metadata: dict[str, Any] = {
        "producer_revision": args.producer_revision,
        "generated_at": args.generated_at,
        "served_model": args.served_model,
        "request_parameters": request_parameters,
    }
    producer_audit_receipt: dict[str, str] | None = None
    if args.producer_audit_receipt is not None:
        if args.benchmark == "lambada":
            producer_audit_receipt = prepare_completion_audit_receipt(
                args.producer_audit_receipt,
                args.input,
                args.manifest,
                args.output_dir,
                args.benchmark,
                args.pregrouper,
                args.model,
                producer_metadata,
                args.availability_status,
            )
        else:
            producer_audit_receipt = prepare_producer_audit_receipt(
                args.producer_audit_receipt,
                args.input,
                args.manifest,
                args.output_dir,
                args.benchmark,
                args.pregrouper,
                args.model,
                producer_metadata,
                args.availability_status,
            )
    import_results(
        args.input,
        args.manifest,
        args.output_dir,
        args.model,
        args.benchmark,
        args.pregrouper,
        args.identity_attestation,
        producer_metadata,
        args.availability_status,
        canary_metadata,
        producer_audit_receipt,
    )


if __name__ == "__main__":
    main()
