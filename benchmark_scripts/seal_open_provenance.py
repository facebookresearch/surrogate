# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Seal completed open-model artifacts with content and release-source hashes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from typing import Any

import torch
import transformers

from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
    OPEN_COMPLETE_SOURCE_FILES,
    OPEN_EXECUTION_SOURCE_FILES,
    OPEN_MODEL_IDENTITY_FILENAMES,
    canonical_file_hash_manifest_sha256,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_hashes(model_path: str) -> dict[str, str]:
    """Hash all weight and identity files needed to identify local weights."""
    names: list[str] = sorted(
        name
        for name in os.listdir(model_path)
        if name.endswith((".safetensors", ".bin", ".json", ".model", ".txt"))
        and os.path.isfile(os.path.join(model_path, name))
    )
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"No model weight files found in {model_path}")
    return {name: _sha256(os.path.join(model_path, name)) for name in names}


def _release_source_corrections(
    execution_hashes: Any,
    release_hashes: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Accept either the gold execution snapshot or an exact release rerun."""
    release_execution_hashes: dict[str, str] = {
        path: release_hashes[path] for path in OPEN_EXECUTION_SOURCE_FILES
    }
    if execution_hashes == GOLD_OPEN_EXECUTION_SOURCE_SHA256:
        expected_corrections: dict[str, dict[str, str]] = (
            GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS
        )
    elif execution_hashes == release_execution_hashes:
        expected_corrections = {}
    else:
        raise ValueError("Execution source hashes are neither gold nor current release")
    actual_corrections: dict[str, dict[str, str]] = {}
    for path, execution_sha256 in execution_hashes.items():
        release_sha256: str = release_hashes[path]
        if release_sha256 == execution_sha256:
            continue
        expected: dict[str, str] | None = expected_corrections.get(path)
        if (
            expected is None
            or expected.get("execution_sha256") != execution_sha256
            or expected.get("release_sha256") != release_sha256
        ):
            raise ValueError(f"Unapproved execution/release source change: {path}")
        actual_corrections[path] = expected
    if actual_corrections != expected_corrections:
        raise ValueError("Expected execution/release source correction is absent")
    return actual_corrections


def seal(results_dir: str, model: str, model_path: str) -> int:
    """Add verifiable output, manifest, weight, and runtime hashes to run JSONs."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(model_path)
    if model not in GOLD_OPEN_MODEL_REVISIONS:
        raise ValueError(f"No pinned public model identity for {model}")
    model_hashes: dict[str, str] = _model_artifact_hashes(model_path)
    model_manifest_sha256: str = canonical_file_hash_manifest_sha256(model_hashes)
    expected_manifest_sha256: str = GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[model]
    if model_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"Model artifact manifest disagrees for {model}: expected "
            f"{expected_manifest_sha256}, got {model_manifest_sha256}"
        )
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    source_hashes: dict[str, str] = {
        path: _sha256(os.path.join(repository_root, path))
        for path in OPEN_COMPLETE_SOURCE_FILES
    }
    metadata_paths: list[str] = sorted(
        glob.glob(os.path.join(results_dir, "*", "*", f"{model}_run.json"))
    )
    if not metadata_paths:
        raise FileNotFoundError(f"No run metadata found for {model} in {results_dir}")
    metadata_records: list[tuple[str, dict[str, Any]]] = []
    for metadata_path in metadata_paths:
        with open(metadata_path, encoding="utf-8") as source:
            metadata: dict[str, Any] = json.load(source)
        if metadata.get("model") != model:
            raise ValueError(f"Model mismatch in {metadata_path}")
        execution_hashes: Any = metadata.get("source_sha256")
        execution_model_source: Any = metadata.get(
            "execution_model_source", metadata.get("model_source")
        )
        release_execution_hashes: dict[str, str] = {
            path: source_hashes[path] for path in OPEN_EXECUTION_SOURCE_FILES
        }
        expected_execution_model_source: str = (
            GOLD_OPEN_EXECUTION_MODEL_SOURCES[model]
            if execution_hashes == GOLD_OPEN_EXECUTION_SOURCE_SHA256
            else GOLD_OPEN_MODEL_REPOSITORIES[model]
        )
        if (
            execution_hashes
            not in (GOLD_OPEN_EXECUTION_SOURCE_SHA256, release_execution_hashes)
            or execution_model_source != expected_execution_model_source
        ):
            raise ValueError(
                f"Execution model source disagrees in {metadata_path}: "
                f"{execution_model_source!r}"
            )
        execution_identity: Any = metadata.get("model_identity_files_sha256")
        expected_execution_identity: dict[str, str] = {
            filename: digest
            for filename, digest in model_hashes.items()
            if filename in OPEN_MODEL_IDENTITY_FILENAMES
        }
        if not expected_execution_identity or execution_identity != (
            expected_execution_identity
        ):
            raise ValueError(
                f"Execution-time and sealing-time model identities disagree in "
                f"{metadata_path}"
            )
        metadata["release_source_corrections"] = _release_source_corrections(
            execution_hashes, source_hashes
        )
        metadata["execution_model_source"] = execution_model_source
        metadata_records.append((metadata_path, metadata))

    for metadata_path, metadata in metadata_records:
        directory: str = os.path.dirname(metadata_path)
        manifest_path: str = os.path.join(directory, "segments.tsv.gz")
        segment_path: str = os.path.join(directory, f"{model}_segment.tsv.gz")
        token_path: str = os.path.join(directory, f"{model}_tokens.tsv.gz")
        if not os.path.exists(manifest_path) or not os.path.exists(segment_path):
            raise FileNotFoundError(
                f"Incomplete artifact beside provenance file {metadata_path}"
            )
        artifact_hashes: dict[str, str] = {"segment": _sha256(segment_path)}
        if os.path.exists(token_path):
            artifact_hashes["tokens"] = _sha256(token_path)
        metadata["artifact_sha256"] = artifact_hashes
        metadata["manifest_sha256"] = _sha256(manifest_path)
        metadata["schema_version"] = 3
        metadata["model_source"] = GOLD_OPEN_MODEL_REPOSITORIES[model]
        metadata["model_revision"] = GOLD_OPEN_MODEL_REVISIONS[model]
        metadata["model_artifact_sha256"] = model_hashes
        metadata["model_artifact_manifest_sha256"] = model_manifest_sha256
        metadata["execution_source_hash_timing"] = (
            "run_completion"
            if metadata["source_sha256"] == GOLD_OPEN_EXECUTION_SOURCE_SHA256
            else "run_start"
        )
        metadata["model_artifact_hash_timing"] = "post_run_seal"
        metadata.pop("complete_source_sha256", None)
        metadata["release_source_sha256"] = source_hashes
        software: dict[str, Any] = dict(metadata.get("software", {}))
        software["transformers"] = transformers.__version__
        software["cuda_runtime"] = torch.version.cuda
        metadata["software"] = software
        metadata["provenance_seal_sha256"] = _sha256(__file__)
        with open(metadata_path, "w", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
    return len(metadata_paths)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-path", required=True)
    args: argparse.Namespace = parser.parse_args()
    count: int = seal(args.results_dir, args.model, args.model_path)
    print(f"Sealed {count} run records for {args.model}")


if __name__ == "__main__":
    main()
