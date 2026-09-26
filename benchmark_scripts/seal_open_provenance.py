# Copyright (c) 2025 The Authors
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

from transformers import AutoTokenizer

from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_POST_RERUN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    OPEN_COMPLETE_SOURCE_FILES,
    OPEN_EXECUTION_SOURCE_FILES,
    OPEN_MODEL_IDENTITY_FILENAMES,
    build_release_source_corrections,
    canonical_file_hash_manifest_sha256,
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_source_corrections(
    execution_dependency_hashes: Any,
    release_hashes: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Build the release corrections recorded in an open-model sidecar."""
    return build_release_source_corrections(
        execution_dependency_hashes,
        release_hashes,
    )


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


def _verify_rendered_chat_tokenization(
    model: str,
    model_path: str,
    rendered_add_special_tokens: bool,
) -> dict[str, Any]:
    """Verify the pinned tokenizer does not duplicate rendered control tokens."""
    tokenizer: Any = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    rendered: Any = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError(f"Pinned tokenizer produced no rendered probe for {model}")
    without_specials: list[int] = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=False)
    ]
    with_specials: list[int] = [
        int(value) for value in tokenizer.encode(rendered, add_special_tokens=True)
    ]
    literal_ids: list[int] = (
        with_specials if rendered_add_special_tokens else without_specials
    )
    bos_token_id: Any = tokenizer.bos_token_id
    effective_bos_count: int = (
        0
        if bos_token_id is None
        else sum(token_id == int(bos_token_id) for token_id in literal_ids)
    )
    if model.startswith("qwen2.5-"):
        if without_specials != with_specials or effective_bos_count != 0:
            raise ValueError(
                f"Pinned Qwen tokenizer no-op special-token check failed for {model}"
            )
        method: str = "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false"
    elif model == "llama-3.1-8b-instruct":
        if (
            rendered_add_special_tokens
            or not isinstance(bos_token_id, int)
            or not literal_ids
            or literal_ids[0] != bos_token_id
            or effective_bos_count != 1
        ):
            raise ValueError(
                f"Pinned Llama tokenizer single-BOS check failed for {model}"
            )
        method = "explicit_no_special_tokens_after_chat_template"
    else:
        raise ValueError(f"No rendered-tokenization policy for {model}")
    return {
        "effective_bos_count": effective_bos_count,
        "no_duplicate_special_tokens": True,
        "verification_method": method,
    }


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
    tokenization_verification_by_flag: dict[bool, dict[str, Any]] = {}
    for metadata_path in metadata_paths:
        with open(metadata_path, encoding="utf-8") as source:
            metadata: dict[str, Any] = json.load(source)
        if metadata.get("model") != model:
            raise ValueError(f"Model mismatch in {metadata_path}")
        execution_hashes: Any = metadata.get("source_sha256")
        legacy_execution: bool = execution_hashes == GOLD_OPEN_EXECUTION_SOURCE_SHA256
        if legacy_execution and model == "llama-3.1-8b-instruct":
            raise ValueError(
                "Refusing to seal the legacy duplicated-BOS Llama execution; "
                "rerun it with the single-BOS policy"
            )
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
            not in (
                GOLD_OPEN_EXECUTION_SOURCE_SHA256,
                GOLD_OPEN_POST_RERUN_EXECUTION_SOURCE_SHA256,
                release_execution_hashes,
            )
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
        parameters: Any = metadata.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Missing execution parameters in {metadata_path}")
        if legacy_execution:
            # The historical code used the tokenizer default (True). Qwen's
            # rendered prompt token IDs are identical under True and False, so
            # record the literal execution behavior separately. Llama is
            # rejected above because its tokenizer would add a second BOS.
            parameters["rendered_chat_add_special_tokens"] = True
            execution_dependency_hashes: dict[str, str] = dict(
                GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
            )
            execution_dependency_hash_timing: str = (
                "post_run_reconstruction_not_execution_attested"
            )
        elif parameters.get("rendered_chat_add_special_tokens") is not False:
            raise ValueError(
                f"Rendered-chat tokenization policy is absent in {metadata_path}"
            )
        else:
            execution_dependency_hashes = dict(execution_hashes)
            execution_dependency_hash_timing = "run_start"
        rendered_add_special_tokens: bool = bool(
            parameters["rendered_chat_add_special_tokens"]
        )
        if rendered_add_special_tokens not in tokenization_verification_by_flag:
            tokenization_verification_by_flag[rendered_add_special_tokens] = (
                _verify_rendered_chat_tokenization(
                    model,
                    model_path,
                    rendered_add_special_tokens,
                )
            )
        tokenization_verification: dict[str, Any] = dict(
            tokenization_verification_by_flag[rendered_add_special_tokens]
        )
        metadata["release_source_corrections"] = _release_source_corrections(
            execution_dependency_hashes, source_hashes
        )
        metadata["execution_model_source"] = execution_model_source
        metadata["execution_dependency_sha256"] = execution_dependency_hashes
        metadata["execution_dependency_hash_timing"] = execution_dependency_hash_timing
        metadata["tokenization_verification"] = tokenization_verification
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
        metadata["schema_version"] = 5
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
        software: Any = metadata.get("software")
        if not isinstance(software, dict) or not {
            "numpy",
            "pandas",
            "torch",
        }.issubset(software):
            raise ValueError(f"Missing execution software in {metadata_path}")
        metadata["software"] = {
            name: software[name] for name in ("numpy", "pandas", "torch")
        }
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
