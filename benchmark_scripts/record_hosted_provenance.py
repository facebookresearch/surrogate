# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Record provenance for precomputed hosted-model TSV artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import pandas as pd

from benchmark_scripts.hosted_completion import summarize_completion_canary
from benchmark_scripts.provenance_sources import HOSTED_RECORD_SOURCE_FILES


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _transformation_source_hashes() -> dict[str, str]:
    """Hash every public source file used to normalize and record legacy TSVs."""
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    return {
        relative_path: _sha256(os.path.join(repository_root, relative_path))
        for relative_path in HOSTED_RECORD_SOURCE_FILES
    }


def _copy_and_summarize_canary(
    source_path: str,
    directory: str,
    model: str,
    seed: int,
) -> dict[str, Any]:
    """Copy a completion canary beside the artifact and summarize coverage."""
    with open(source_path, encoding="utf-8") as source:
        payload: Any = json.load(source)
    if not isinstance(payload, list):
        raise ValueError("The unsupported-model canary must be a JSON list")
    summary: dict[str, Any] = summarize_completion_canary(payload, seed)
    destination_name: str = f"{model}_canary.json"
    destination: str = os.path.join(directory, destination_name)
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
        for row in payload
    ]
    with open(destination, "w", encoding="utf-8") as output:
        json.dump(sanitized, output, indent=2, sort_keys=True)
        output.write("\n")
    return {
        "artifact_path": destination_name,
        "artifact_sha256": _sha256(destination),
        **summary,
    }


def record(
    directory: str,
    model: str,
    benchmark: str,
    pregrouper: str,
    source_revision: str,
    source_segment: str,
    source_tokens: str | None,
    identity_attestation: str,
    generated_at: str,
    served_model: str,
    request_parameters: dict[str, Any],
) -> None:
    """Write a portable provenance record for one hosted-model result."""
    normalized_segment: str = os.path.join(directory, f"{model}_segment.tsv.gz")
    manifest_path: str = os.path.join(directory, "segments.tsv.gz")
    if not os.path.exists(normalized_segment):
        raise FileNotFoundError(normalized_segment)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)
    if not identity_attestation.strip():
        raise ValueError("identity_attestation must be non-empty")
    if not source_revision.strip():
        raise ValueError("source_revision must be non-empty")
    segment_frame: pd.DataFrame = pd.read_csv(normalized_segment, sep="\t")
    availability: float = (
        float(segment_frame["segment_result_available"].mean())
        if "segment_result_available" in segment_frame.columns
        else 1.0
    )
    source_hashes: dict[str, str] = {"segment": _sha256(source_segment)}
    if source_tokens is not None:
        source_hashes["tokens"] = _sha256(source_tokens)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "model": model,
        "segmentation_scope": "full_dialog_in_message_order",
        "source_format": "precomputed_portable_tsv",
        "source_revision": source_revision,
        "source_sha256": source_hashes,
        "transformation_source_sha256": _transformation_source_hashes(),
        "manifest_sha256": _sha256(manifest_path),
        "artifact_sha256": {
            "segment": _sha256(normalized_segment),
            **(
                {"tokens": _sha256(os.path.join(directory, f"{model}_tokens.tsv.gz"))}
                if os.path.exists(os.path.join(directory, f"{model}_tokens.tsv.gz"))
                else {}
            ),
        },
        "normalization": "canonical_manifest_join_with_explicit_missingness",
        "identity_attestation": identity_attestation,
        "producer": {
            "producer_revision": source_revision,
            "generated_at": generated_at,
            "served_model": served_model,
            "request_parameters": request_parameters,
        },
        "availability_status": "complete",
        "canary": None,
        "segment_result_coverage": availability,
    }
    output_path: str = os.path.join(directory, f"{model}_run.json")
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pregrouper", choices=["sentence", "word"], required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-segment", required=True)
    parser.add_argument("--source-tokens")
    parser.add_argument(
        "--identity-attestation",
        required=True,
        help="Human-auditable statement describing how legacy keys were verified.",
    )
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--request-parameters-json", required=True)
    args: argparse.Namespace = parser.parse_args()
    request_parameters: Any = json.loads(args.request_parameters_json)
    if not isinstance(request_parameters, dict):
        raise ValueError("--request-parameters-json must decode to an object")
    record(
        args.directory,
        args.model,
        args.benchmark,
        args.pregrouper,
        args.source_revision,
        args.source_segment,
        args.source_tokens,
        args.identity_attestation,
        args.generated_at,
        args.served_model,
        request_parameters,
    )


if __name__ == "__main__":
    main()
