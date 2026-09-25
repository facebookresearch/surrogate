# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Validate a complete public surrogate-fidelity result artifact.

Each model's segment rows must use exactly the canonical
``(prompt_idx, seg_idx)`` grid recorded by the model-independent segment
manifest. Token-logprob rows may be sparse for hosted APIs, but their keys must
be a subset of that grid and their coverage is explicit in ``coverage.tsv``.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import hashlib
import itertools
import json
import os
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.derived_provenance import (
    DERIVED_SUPPORTING_SOURCE_FILES,
    PROVENANCE_SCHEMA_VERSION,
    collect_result_inputs,
)
from benchmark_scripts.dialog_identity import compute_dialog_identity
from benchmark_scripts.f_table import (
    API_MODELS,
    DEFAULT_BENCHMARK_CONFIGS,
    OPEN_MODELS,
    PAPER_MODELS,
    _contrast_metadata,
    _derived_input_paths as _f_table_derived_input_paths,
    _resolved_scope,
    _unsupported_model_components,
)
from benchmark_scripts.hosted_audit_receipt import (
    CLASSIFICATION_CONFIGURATIONS,
    CLASSIFICATION_REQUEST_PARAMETERS as HOSTED_CLASSIFICATION_REQUEST_PARAMETERS,
    CONFIGURATION_IDENTITY_SHA256,
    HOSTED_CLASSIFICATION_MODELS,
    HostedAuditReceipt,
    canonical_projection_digest,
    classification_table_projection,
    load_receipt,
)
from benchmark_scripts.hosted_completion import (
    LAMBADA_GOLD_CANARY,
    LAMBADA_GOLD_CANARY_ANSWERS,
    summarize_completion_canary,
)
from benchmark_scripts.hosted_completion_audit_receipt import (
    CANARY_POPULATION,
    FULL_POPULATION,
    HOSTED_COMPLETION_MODELS,
    POPULATION_IDENTITY_SHA256,
    HostedCompletionAuditReceipt,
    canary_manifest,
    canonical_projection_digest as canonical_completion_projection_digest,
    completion_payload_projection,
    completion_table_projection,
    compute_completion_dialog_identity,
    load_receipt as load_completion_receipt,
    projection_summary as completion_projection_summary,
)
from benchmark_scripts.layerwise_fidelity import (
    DEFAULT_CONFIGS as LAYERWISE_CONFIGS,
    _contrast_labels as _layer_contrast_labels,
    _validate_layer_frame,
    _validate_manifest_coverage as _validate_layer_manifest_coverage,
)
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_IDENTITY_ATTESTATION,
    GOLD_HOSTED_IMPORT_SOURCE_SHA256,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
    HOSTED_IMPORT_SOURCE_FILES,
    HOSTED_RECORD_SOURCE_FILES,
    LAYER_EXECUTION_SOURCE_FILES,
    OPEN_COMPLETE_SOURCE_FILES,
    OPEN_EXECUTION_SOURCE_FILES,
    OPEN_MODEL_IDENTITY_FILENAMES,
    canonical_file_hash_manifest_sha256,
)
from surrogate.eval_constants import label_column_alias

ARTIFACT_CONFIGS: list[tuple[str, str]] = [
    *DEFAULT_BENCHMARK_CONFIGS,
    ("race", "sentence"),
]

RELEASE_PAPER_DERIVED_TABLES: tuple[str, ...] = (
    "f_table.tsv",
    "f_table_finite_extreme_sensitivity.tsv",
    "race_rv.tsv",
)
RELEASE_OPEN_DERIVED_TABLES: tuple[str, ...] = (
    "f_table_open.tsv",
    "f_table_finite_extreme_sensitivity_open.tsv",
    "race_rv_open.tsv",
)
RELEASE_LAYERWISE_DERIVED_TABLES: tuple[str, ...] = ("layerwise_fidelity.tsv",)
RELEASE_LAMBADA_CANARY_MODELS: tuple[str, ...] = (
    "gemini-2-5-flash-lite-vertex",
    "gpt-4-1",
    "gpt-4o",
)
OPEN_RUN_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_sha256",
        "benchmark",
        "dataset",
        "execution_model_source",
        "execution_dependency_hash_timing",
        "execution_dependency_sha256",
        "execution_source_hash_timing",
        "manifest_sha256",
        "model",
        "model_artifact_sha256",
        "model_artifact_manifest_sha256",
        "model_artifact_hash_timing",
        "model_identity_files_sha256",
        "model_revision",
        "model_source",
        "parameters",
        "pregrouper",
        "provenance_seal_sha256",
        "release_source_sha256",
        "release_source_corrections",
        "schema_version",
        "segmentation_scope",
        "software",
        "source_sha256",
        "tokenization_verification",
    }
)
HOSTED_RUN_COMMON_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_sha256",
        "availability_status",
        "benchmark",
        "canary",
        "identity_attestation",
        "manifest_sha256",
        "model",
        "pregrouper",
        "producer",
        "schema_version",
        "segmentation_scope",
        "source_format",
        "source_sha256",
        "transformation_source_sha256",
    }
)
HOSTED_CLASSIFICATION_RUN_FIELDS: frozenset[str] = HOSTED_RUN_COMMON_FIELDS | {
    "logprob_granularity",
    "producer_audit_receipt",
    "prompts",
    "segments",
    "source_size_bytes",
}
HOSTED_COMPLETION_RUN_FIELDS: frozenset[str] = HOSTED_RUN_COMMON_FIELDS | {
    "producer_audit_receipt",
    "segments",
    "source_size_bytes",
}
HOSTED_PRECOMPUTED_RUN_FIELDS: frozenset[str] = HOSTED_RUN_COMMON_FIELDS | {
    "normalization",
    "segment_result_coverage",
    "source_revision",
}
OPEN_DATASET_FIELDS: frozenset[str] = frozenset(
    {
        "hf_name",
        "hf_path",
        "hf_split",
        "normalized_frame_sha256",
        "rows",
        "snapshot_filename",
        "snapshot_sha256",
    }
)
OPEN_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "batch_size",
        "max_forward_passes",
        "max_samples",
        "phase_attention_implementation",
        "phases",
        "rendered_chat_add_special_tokens",
        "seed",
    }
)
OPEN_SOFTWARE_FIELDS: frozenset[str] = frozenset({"numpy", "pandas", "torch"})
OPEN_MODEL_METADATA_FILENAMES: frozenset[str] = OPEN_MODEL_IDENTITY_FILENAMES | {
    "added_tokens.json",
    "merges.txt",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
}


def _expected_open_tokenization_verification(model: str) -> dict[str, Any]:
    """Return the verified rendered-chat tokenization invariant by model."""
    if model.startswith("qwen2.5-"):
        return {
            "effective_bos_count": 0,
            "no_duplicate_special_tokens": True,
            "verification_method": (
                "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false"
            ),
        }
    if model == "llama-3.1-8b-instruct":
        return {
            "effective_bos_count": 1,
            "no_duplicate_special_tokens": True,
            "verification_method": "explicit_no_special_tokens_after_chat_template",
        }
    raise ValueError(f"No rendered-tokenization policy for {model}")


DERIVED_PROVENANCE_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_type",
        "generator",
        "inputs",
        "output",
        "parameters",
        "schema_version",
        "software",
        "supporting_sources",
    }
)
DERIVED_SOFTWARE_FIELDS: frozenset[str] = frozenset(
    {"numpy", "pandas", "python", "python_implementation", "scipy", "tqdm"}
)

# Exact canonical grids generated from the frozen dataset snapshots documented
# in results/README.md. Hashes are over decompressed UTF-8 TSV bytes, so gzip
# container metadata cannot affect validation.
GOLD_MANIFESTS: dict[tuple[str, str], tuple[int, int, str]] = {
    ("boolq", "sentence"): (
        27_516,
        3_270,
        "73b8b7638ae6956e229fac63ce2f08f7b8688e0e6482106c15e76573a7102f1c",
    ),
    ("anli_r1", "sentence"): (
        8_181,
        1_000,
        "df79c7ec45311f3a80cadfde755ffe00479c8e526bd01c12c47162c942bf20fc",
    ),
    ("anli_r2", "sentence"): (
        8_163,
        1_000,
        "a43b52119b91ecd198b82b184f8dc610105efb81d3190079aeca90b4b62e68d0",
    ),
    ("anli_r3", "sentence"): (
        10_028,
        1_200,
        "b1403b6d586c4c9e0045c33cc4d41c33ccfda8399388d9b0d232bb7419255765",
    ),
    ("winogrande", "sentence"): (
        9_135,
        1_267,
        "1440c85a5aa61add180199a27bc02db6db45870ae0237a678d0e4bb7b8609099",
    ),
    ("race", "sentence"): (
        145_544,
        4_934,
        "fab3fae5c02ceb915ea13d62ae2eb56361a46777dc21ffa95c3b22ec0ba43950",
    ),
    ("boolq", "word"): (
        10_000,
        3_028,
        "e225ff1288652d42486e5e4bc23320244ede489eee0dff4fdf220553ef94c55d",
    ),
    ("lambada", "word"): (
        10_000,
        4_387,
        "495bab828480139ec57e07dec49e5cea2d841d717e562902195135b29748b60f",
    ),
}
GOLD_DATASET_SHA256: dict[str, str] = {
    "boolq": "80040aa10f18e5b01082386dae3bdde48931a0311e807f6cb10f7173995f346a",
    "anli_r1": "b87837e172a9a98c70217677fb4897fd6dd3a32497dface9ad3621ea2c85839d",
    "anli_r2": "10ebd3593714d7fe106a5e2ed7f22ea47a34100aad713d8fde4d190d896479ef",
    "anli_r3": "4b08122b488ad0ed1e3d18ce50ef67ba0d686aaa29a140c621151b8326a36e8f",
    "winogrande": "cd158f8e0699aecbc0c2090d89004514ec515ec7f2a2238dd93d4b4457505ca4",
    "race": "ce70066c14b0f4da95bd5d577a8f0d0208228c04fcff1b11d2e362371a9c6289",
    "lambada": "7bb96e6a12ea76dac59896d2acd9d29599e0b488afc9caedf7140800d8e665b1",
}
GOLD_DATASET_FILES: dict[str, str] = {
    "boolq": "google_boolq_validation.tsv",
    "anli_r1": "facebook_anli_test_r1.tsv",
    "anli_r2": "facebook_anli_test_r2.tsv",
    "anli_r3": "facebook_anli_test_r3.tsv",
    "winogrande": "allenai_winogrande_validation.tsv",
    "race": "race_test.tsv",
    "lambada": "lambada_test.tsv",
}
OPEN_SEGMENT_COLUMNS: tuple[str, ...] = (
    "attention_mean",
    "attention_max",
    "attention_rollout",
    "w_norm",
    "delta_norm_prenorm",
    "delta_norm_postnorm",
    "cossim_prenorm",
    "cossim_postnorm",
    "w_dot_delta_z_prenorm",
    "w_dot_delta_z_postnorm",
    "z_orig_norm_prenorm",
    "z_pert_norm_prenorm",
    "z_orig_norm_postnorm",
    "z_pert_norm_postnorm",
    "w_dot_z_orig_prenorm",
    "w_dot_z_pert_prenorm",
    "w_dot_z_orig_postnorm",
    "w_dot_z_pert_postnorm",
)
MIN_HOSTED_RESULT_COVERAGE: float = 0.8
HOSTED_AUDIT_RECEIPT_NAME: str = "hosted_classification_audit_receipt.json"
HOSTED_COMPLETION_AUDIT_RECEIPT_NAME: str = "hosted_completion_audit_receipt.json"
GOLD_HOSTED_TERMINAL_FAILURE_COUNTS: dict[tuple[str, str, str], tuple[int, int]] = {
    ("race", "sentence", "gpt-4o"): (0, 110),
    ("race", "sentence", "gpt-4-1"): (0, 890),
}
LOWERCASE_SHA256_RE: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
RFC3339_RE: re.Pattern[str] = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
)
ANALYSIS_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/compute_logodds.py",
    "benchmark_scripts/consolidate_results.py",
    "benchmark_scripts/derived_provenance.py",
    "benchmark_scripts/dialog_identity.py",
    "benchmark_scripts/f_table.py",
    "benchmark_scripts/hosted_audit_receipt.py",
    "benchmark_scripts/hosted_completion.py",
    "benchmark_scripts/hosted_completion_audit_receipt.py",
    "benchmark_scripts/import_hosted_results.py",
    "benchmark_scripts/layerwise_fidelity.py",
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/race_rv.py",
    "benchmark_scripts/record_hosted_provenance.py",
    "benchmark_scripts/seal_open_provenance.py",
    "benchmark_scripts/validate_results.py",
    "benchmark_scripts/run_all_benchmarks.sh",
    "benchmark_scripts/run_layerwise.py",
    "surrogate/eval_constants.py",
    "surrogate/layerwise_scoring.py",
    "surrogate/model_types.py",
    "surrogate/text_augmentation.py",
    "surrogate/transformers_model.py",
    "surrogate/utils.py",
)

LAYER_RUN_FIELDS: frozenset[str] = frozenset(
    {
        "alignment",
        "artifact",
        "benchmark",
        "dataset",
        "label_score_definition",
        "labels",
        "layer_slots",
        "manifest_sha256",
        "model",
        "model_artifact_hash_timing",
        "model_artifact_manifest_sha256",
        "model_artifact_sha256",
        "model_identity_files_sha256",
        "model_revision",
        "model_source",
        "parameters",
        "pregrouper",
        "schema_version",
        "segmentation_scope",
        "software",
        "source_hash_timing",
        "source_sha256",
    }
)
LAYER_PARAMETER_FIELDS: frozenset[str] = frozenset(
    {
        "attention_implementation",
        "batch_size",
        "canary",
        "device_map",
        "max_samples",
        "rendered_chat_add_special_tokens",
        "seed",
        "torch_dtype",
    }
)
MAX_LAYER_FINAL_CONTRAST_ABS_ERROR: float = 1e-4
# Spearman is discontinuous at ties: independent BF16 batches can swap nearly
# equal values even when every final-layer contrast passes the 1e-4 raw-value
# gate. Keep a separate, still-small bound for that derived rank statistic.
MAX_LAYER_FINAL_SPEARMAN_ABS_ERROR: float = 1e-3
LAYER_DATASET_FIELDS: frozenset[str] = frozenset(
    {
        "hf_name",
        "hf_path",
        "hf_split",
        "normalized_frame_sha256",
        "prompts",
        "snapshot_filename",
        "snapshot_sha256",
    }
)
LAYER_SOFTWARE_FIELDS: frozenset[str] = frozenset(
    {"cuda_runtime", "numpy", "pandas", "torch", "transformers"}
)
LAYER_DERIVED_SUPPORTING_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/derived_provenance.py",
    "benchmark_scripts/f_table.py",
    "surrogate/eval_constants.py",
)
F_TABLE_DERIVED_SUPPORTING_SOURCE_FILES: tuple[str, ...] = (
    *DERIVED_SUPPORTING_SOURCE_FILES,
    "benchmark_scripts/layerwise_fidelity.py",
)


@dataclass(frozen=True)
class ValidationSummary:
    """Summary counts for one model and benchmark configuration."""

    benchmark: str
    pregrouper: str
    model: str
    expected_segments: int
    segment_rows: int
    missing_segment_rows: int
    extra_segment_rows: int
    original_prompt_coverage: float | None
    ablated_segment_coverage: float | None
    paired_attribution_coverage: float | None
    label_cell_coverage: float | None
    original_content_filter_count: int | None
    ablated_content_filter_count: int | None
    original_terminal_failure_count: int | None
    ablated_terminal_failure_count: int | None


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        dtype={"label": str, "token": str, "kind": str},
        float_precision="round_trip",
        keep_default_na=False,
        na_values=[""],
    )


def _model_from_path(path: str, suffix: str) -> str:
    base: str = os.path.basename(path)
    for extension in (suffix + ".gz", suffix):
        if base.endswith(extension):
            return base[: -len(extension)]
    raise ValueError(f"Unexpected result filename: {path}")


def _keys(frame: pd.DataFrame) -> set[tuple[int, int]]:
    return {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in frame[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }


def _validate_manifest(
    path: str,
    require_complete_prompts: bool = True,
) -> tuple[pd.DataFrame, set[tuple[int, int]]]:
    manifest: pd.DataFrame = pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
        na_values=[""],
    )
    required: set[str] = {
        "prompt_idx",
        "answer",
        "seg_idx",
        "message_idx",
        "message_role",
        "message_seg_idx",
        "segment_text",
        "n_segments",
    }
    missing: set[str] = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError(f"{path} has duplicate segment keys")
    roles: set[str] = set(manifest["message_role"].dropna().astype(str))
    if not roles.issubset({"system", "user", "assistant"}):
        raise ValueError(f"{path} has unexpected message roles {sorted(roles)}")
    if manifest[["message_role", "segment_text"]].isna().any().any():
        raise ValueError(f"{path} has blank segment identity metadata")
    for _, prompt_rows in manifest.groupby("prompt_idx"):
        declared: set[int] = set(prompt_rows["n_segments"].astype(int))
        if len(declared) != 1:
            raise ValueError(f"{path} has inconsistent n_segments values")
        if int(prompt_rows["seg_idx"].max()) >= next(iter(declared)):
            raise ValueError(f"{path} has seg_idx outside n_segments")
        if require_complete_prompts:
            expected_indices: set[int] = set(range(next(iter(declared))))
            actual_indices: set[int] = set(prompt_rows["seg_idx"].astype(int))
            if actual_indices != expected_indices:
                raise ValueError(f"{path} omits segments from a full prompt")
    return manifest, _keys(manifest)


def _validate_segment_file(
    path: str,
    manifest: pd.DataFrame,
    manifest_keys: set[tuple[int, int]],
) -> tuple[int, int, int]:
    frame: pd.DataFrame = _read_tsv(path)
    required: set[str] = {
        "prompt_idx",
        "answer",
        "seg_idx",
        "message_idx",
        "message_role",
        "message_seg_idx",
        "segment_text",
        "n_segments",
        "segment_result_available",
    }
    missing_columns: set[str] = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns {sorted(missing_columns)}")
    if frame.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError(f"{path} has duplicate segment keys")
    frame_keys: set[tuple[int, int]] = _keys(frame)
    missing_keys: set[tuple[int, int]] = manifest_keys - frame_keys
    extra_keys: set[tuple[int, int]] = frame_keys - manifest_keys
    if missing_keys or extra_keys:
        raise ValueError(
            f"{path} does not match its segment manifest: "
            f"{len(missing_keys)} missing, {len(extra_keys)} extra keys"
        )
    expected_counts: pd.Series = manifest.set_index(["prompt_idx", "seg_idx"])[
        "n_segments"
    ]
    actual_counts: pd.Series = frame.set_index(["prompt_idx", "seg_idx"])["n_segments"]
    if not expected_counts.astype(int).equals(actual_counts.astype(int)):
        raise ValueError(f"{path} has n_segments values that disagree with manifest")
    availability: pd.Series = frame["segment_result_available"]
    if availability.isna().any() or not availability.isin([True, False]).all():
        raise ValueError(f"{path} has invalid segment_result_available values")
    if "original_result_available" in frame.columns:
        original_availability: pd.Series = frame["original_result_available"]
        if (
            original_availability.isna().any()
            or not original_availability.isin([True, False]).all()
            or frame.groupby("prompt_idx")["original_result_available"]
            .nunique(dropna=False)
            .gt(1)
            .any()
        ):
            raise ValueError(
                f"{path} has invalid or inconsistent original-result availability"
            )
    shared_metadata: list[str] = [
        column
        for column in (
            "answer",
            "message_idx",
            "message_role",
            "message_seg_idx",
            "segment_text",
        )
        if column in frame.columns
    ]
    for column in shared_metadata:
        expected: pd.Series = manifest.set_index(["prompt_idx", "seg_idx"])[column]
        actual: pd.Series = frame.set_index(["prompt_idx", "seg_idx"])[column]
        equal: pd.Series = expected.eq(actual) | (expected.isna() & actual.isna())
        if not equal.all():
            raise ValueError(f"{path} has incorrect {column!r} metadata")
    return len(frame_keys), len(missing_keys), len(extra_keys)


def _sha256_decompressed(path: str) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset_snapshots(dataset_dir: str) -> None:
    """Require every frozen source TSV to match its audited content hash."""
    for benchmark, filename in GOLD_DATASET_FILES.items():
        path: str = os.path.join(dataset_dir, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing frozen dataset: {path}")
        actual: str = _sha256(path)
        expected: str = GOLD_DATASET_SHA256[benchmark]
        if actual != expected:
            raise ValueError(
                f"Frozen dataset hash disagrees for {benchmark}: "
                f"expected {expected}, got {actual}"
            )


def _validate_hosted_dialog_identities(results_dir: str, dataset_dir: str) -> None:
    """Recompute every public-side identity digest in the hosted receipt."""

    async def _validate() -> None:
        receipt_path: str = os.path.join(results_dir, HOSTED_AUDIT_RECEIPT_NAME)
        receipt: HostedAuditReceipt = load_receipt(
            receipt_path,
            expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
        )
        configurations = {
            (row["benchmark"], row["pregrouper"]): row
            for row in receipt["configurations"]
        }
        for benchmark, pregrouper in CLASSIFICATION_CONFIGURATIONS:
            actual: tuple[str, str] = await compute_dialog_identity(
                benchmark,
                pregrouper,
                os.path.join(dataset_dir, GOLD_DATASET_FILES[benchmark]),
                os.path.join(results_dir, benchmark, pregrouper, "segments.tsv.gz"),
            )
            expected: tuple[str, str] = CONFIGURATION_IDENTITY_SHA256[
                (benchmark, pregrouper)
            ]
            configuration = configurations[(benchmark, pregrouper)]
            receipt_identity: tuple[str, str] = (
                configuration["prompt_identity_sha256"],
                configuration["ablated_dialog_identity_sha256"],
            )
            if actual != expected or receipt_identity != expected:
                raise ValueError(
                    f"Hosted dialog identity disagrees for {benchmark}/{pregrouper}"
                )

    asyncio.run(_validate())


def _validate_hosted_completion_dialog_identities(
    results_dir: str, dataset_dir: str
) -> None:
    """Reconstruct both LAMBADA request populations and their dialog identities."""

    async def _validate() -> None:
        receipt_path: str = os.path.join(
            results_dir, HOSTED_COMPLETION_AUDIT_RECEIPT_NAME
        )
        receipt: HostedCompletionAuditReceipt = load_completion_receipt(
            receipt_path,
            expected_sha256=GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
        )
        manifest_path: str = os.path.join(
            results_dir, "lambada", "word", "segments.tsv.gz"
        )
        dataset_path: str = os.path.join(dataset_dir, GOLD_DATASET_FILES["lambada"])
        if receipt["dataset_sha256"] != GOLD_DATASET_SHA256["lambada"] or receipt[
            "manifest_sha256"
        ] != _sha256(manifest_path):
            raise ValueError("Hosted completion dataset or manifest seal disagrees")
        full_manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
        populations: dict[str, Any] = {
            row["name"]: row for row in receipt["populations"]
        }
        for name, manifest in (
            (FULL_POPULATION, full_manifest),
            (CANARY_POPULATION, canary_manifest(dataset_path)),
        ):
            actual: tuple[str, str, str] = await compute_completion_dialog_identity(
                dataset_path, manifest
            )
            population: Any = populations[name]
            recorded: tuple[str, str, str] = (
                population["coordinate_sha256"],
                population["prompt_identity_sha256"],
                population["ablated_dialog_identity_sha256"],
            )
            expected: tuple[str, str, str] = POPULATION_IDENTITY_SHA256[name]
            if actual != expected or recorded != expected:
                raise ValueError(
                    f"Hosted completion dialog identity disagrees for {name}"
                )

    asyncio.run(_validate())


def _validate_manifests_only(results_dir: str, require_gold: bool) -> None:
    """Validate every canonical coordinate grid without requiring model runs."""
    for benchmark, pregrouper in ARTIFACT_CONFIGS:
        path: str = os.path.join(
            results_dir,
            benchmark,
            pregrouper,
            "segments.tsv.gz",
        )
        manifest, _ = _validate_manifest(
            path,
            require_complete_prompts=pregrouper == "sentence",
        )
        if require_gold:
            _validate_gold_manifest(benchmark, pregrouper, path, manifest)


def _validate_gold_manifest(
    benchmark: str,
    pregrouper: str,
    path: str,
    manifest: pd.DataFrame,
) -> None:
    """Require the exact frozen-data segment grid for a gold artifact."""
    expected_rows, expected_prompts, expected_sha256 = GOLD_MANIFESTS[
        (benchmark, pregrouper)
    ]
    actual: tuple[int, int, str] = (
        len(manifest),
        int(manifest["prompt_idx"].nunique()),
        _sha256_decompressed(path),
    )
    expected: tuple[int, int, str] = (
        expected_rows,
        expected_prompts,
        expected_sha256,
    )
    if actual != expected:
        raise ValueError(
            f"{benchmark}/{pregrouper} does not match the frozen gold segment "
            f"grid: expected {expected}, got {actual}"
        )
    roles: set[str] = set(manifest["message_role"].astype(str))
    if not {"system", "user"}.issubset(roles):
        raise ValueError(
            f"{benchmark}/{pregrouper} does not contain both system and user segments"
        )


def _token_coverage(
    path: str,
    manifest_keys: set[tuple[int, int]],
    expected_prompts: set[int],
    expected_labels: set[str],
    segment_frame: pd.DataFrame | None = None,
) -> tuple[float, float, float]:
    frame: pd.DataFrame = _read_tsv(path)
    required: set[str] = {
        "prompt_idx",
        "seg_idx",
        "kind",
        "answer",
        "label",
        "token",
        "logprob",
    }
    missing: set[str] = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if frame.duplicated(["prompt_idx", "seg_idx", "kind", "label", "token"]).any():
        raise ValueError(f"{path} has duplicate token-logprob keys")
    kinds: set[str] = set(frame["kind"].dropna().astype(str))
    if not kinds.issubset({"orig", "ablated"}):
        raise ValueError(f"{path} has unexpected kinds {sorted(kinds)}")
    labels: set[str] = set(frame["label"].dropna().astype(str))
    if labels != expected_labels:
        raise ValueError(
            f"{path} has labels {sorted(labels)}; expected {sorted(expected_labels)}"
        )
    original_rows: pd.Series = frame["kind"] == "orig"
    ablated_rows: pd.Series = frame["kind"] == "ablated"
    if frame.loc[original_rows, "seg_idx"].notna().any():
        raise ValueError(f"{path} has original rows with a segment index")
    if frame.loc[ablated_rows, "seg_idx"].isna().any():
        raise ValueError(f"{path} has ablated rows without a segment index")
    outside_prompts: set[int] = (
        set(frame.loc[original_rows, "prompt_idx"].astype(int)) - expected_prompts
    )
    if outside_prompts:
        raise ValueError(f"{path} has original rows outside the manifest prompts")
    if segment_frame is not None:
        expected_answers: pd.Series = segment_frame.groupby("prompt_idx")[
            "answer"
        ].first()
        observed_answers: pd.Series = frame["prompt_idx"].map(expected_answers)
        if (
            observed_answers.isna().any()
            or not observed_answers.astype(str).eq(frame["answer"].astype(str)).all()
        ):
            raise ValueError(f"{path} has answers that disagree with the manifest")

    finite_logprob: pd.Series = np.isfinite(
        pd.to_numeric(frame["logprob"], errors="coerce")
    )
    finite: pd.DataFrame = frame[finite_logprob]
    original: pd.DataFrame = finite[finite["kind"] == "orig"]
    ablated: pd.DataFrame = finite[finite["kind"] == "ablated"]
    original_prompts: set[int] = set(original["prompt_idx"].astype(int))
    ablated_keys: set[tuple[int, int]] = _keys(ablated)
    if not ablated_keys.issubset(manifest_keys):
        raise ValueError(f"{path} has ablated token rows outside the manifest")
    if segment_frame is not None:
        available_keys: set[tuple[int, int]] = _keys(
            segment_frame[segment_frame["segment_result_available"].eq(True)]
        )
        if available_keys != ablated_keys:
            raise ValueError(
                f"{path} finite ablation keys disagree with segment availability"
            )
        if "original_result_available" in segment_frame.columns:
            available_prompts: set[int] = set(
                segment_frame.loc[
                    segment_frame["original_result_available"].eq(True),
                    "prompt_idx",
                ].astype(int)
            )
            if available_prompts != original_prompts:
                raise ValueError(
                    f"{path} finite original prompts disagree with availability"
                )

    original_coverage: float = len(original_prompts & expected_prompts) / len(
        expected_prompts
    )
    ablated_coverage: float = len(ablated_keys) / len(manifest_keys)
    observed_cells: int = len(
        pd.concat(
            [
                original[["prompt_idx", "label"]]
                .drop_duplicates()
                .assign(observation="orig"),
                ablated[["prompt_idx", "seg_idx", "label"]]
                .drop_duplicates()
                .assign(observation="ablated"),
            ],
            ignore_index=True,
        )
    )
    expected_cells: int = len(expected_labels) * (
        len(expected_prompts) + len(manifest_keys)
    )
    return (
        original_coverage,
        ablated_coverage,
        observed_cells / expected_cells,
    )


def _validate_hosted_audit_binding(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    manifest_path: str,
    segment_frame: pd.DataFrame,
    token_path: str,
    provenance: dict[str, Any],
) -> None:
    """Bind one shipped classification result to the producer audit receipt."""

    if tuple(API_MODELS) != HOSTED_CLASSIFICATION_MODELS:
        raise ValueError("Hosted audit and analysis model inventories disagree")
    receipt_path: str = os.path.join(results_dir, HOSTED_AUDIT_RECEIPT_NAME)
    receipt: HostedAuditReceipt = load_receipt(
        receipt_path,
        expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    )
    expected_reference: dict[str, str] = {
        "path": HOSTED_AUDIT_RECEIPT_NAME,
        "sha256": GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
        "entry_id": f"{benchmark}/{pregrouper}/{model}",
    }
    if provenance.get("producer_audit_receipt") != expected_reference:
        raise ValueError(f"Producer audit receipt reference disagrees in {model}")

    configuration = next(
        row
        for row in receipt["configurations"]
        if (row["benchmark"], row["pregrouper"]) == (benchmark, pregrouper)
    )
    _, _, expected_manifest_content_sha256 = GOLD_MANIFESTS[(benchmark, pregrouper)]
    if (
        configuration["dataset_sha256"] != GOLD_DATASET_SHA256[benchmark]
        or configuration["manifest_sha256"] != _sha256(manifest_path)
        or _sha256_decompressed(manifest_path) != expected_manifest_content_sha256
    ):
        raise ValueError(f"Producer audit configuration identity disagrees for {model}")

    entry = next(
        row
        for row in receipt["entries"]
        if (row["benchmark"], row["pregrouper"], row["model"])
        == (benchmark, pregrouper, model)
    )
    producer: Any = provenance.get("producer")
    if (
        entry["raw_artifact_sha256"] != provenance.get("source_sha256")
        or entry["raw_artifact_size_bytes"] != provenance.get("source_size_bytes")
        or entry["producer_revision_sha256"]
        != (producer.get("producer_revision") if isinstance(producer, dict) else None)
        or receipt["request_parameters"]
        != (producer.get("request_parameters") if isinstance(producer, dict) else None)
        or entry["availability_status"] != provenance.get("availability_status")
    ):
        raise ValueError(f"Producer audit entry metadata disagrees for {model}")

    original_rows: pd.DataFrame = segment_frame.drop_duplicates("prompt_idx")
    original_status_counts: dict[str, int] = {
        status: int(original_rows["original_request_status"].eq(status).sum())
        for status in ("ok", "content_filter", "transient_exhausted")
    }
    ablated_status_counts: dict[str, int] = {
        status: int(segment_frame["segment_request_status"].eq(status).sum())
        for status in ("ok", "content_filter", "transient_exhausted")
    }
    if (
        entry["original_status_counts"] != original_status_counts
        or entry["ablated_status_counts"] != ablated_status_counts
        or entry["original_result_available_count"]
        != int(original_rows["original_result_available"].eq(True).sum())
        or entry["ablated_result_available_count"]
        != int(segment_frame["segment_result_available"].eq(True).sum())
        or entry["paired_attribution_available_count"]
        != int(
            (
                segment_frame["original_result_available"].eq(True)
                & segment_frame["segment_result_available"].eq(True)
            ).sum()
        )
    ):
        raise ValueError(f"Producer audit status summary disagrees for {model}")

    token_frame: pd.DataFrame = pd.read_csv(
        token_path,
        sep="\t",
        dtype={"kind": str, "label": str, "token": str},
        float_precision="round_trip",
    )
    eval_config = BENCHMARKS[benchmark].eval_config
    if eval_config is None:
        raise ValueError(f"Producer audit binding requires labels for {benchmark}")
    projection_sha256: str = canonical_projection_digest(
        classification_table_projection(
            segment_frame.to_dict("records"),
            token_frame.to_dict("records"),
            tuple(eval_config.label_tokens),
        )
    )
    if entry["projection_sha256"] != projection_sha256:
        raise ValueError(f"Producer audit response projection disagrees for {model}")


def _validate_hosted_completion_audit_binding(
    results_dir: str,
    model: str,
    manifest_path: str,
    segment_path: str,
    segment_frame: pd.DataFrame,
    provenance: dict[str, Any],
) -> None:
    """Bind one shipped LAMBADA result to its producer completion receipt."""

    if tuple(API_MODELS) != HOSTED_COMPLETION_MODELS:
        raise ValueError("Hosted completion and analysis inventories disagree")
    receipt_path: str = os.path.join(results_dir, HOSTED_COMPLETION_AUDIT_RECEIPT_NAME)
    receipt: HostedCompletionAuditReceipt = load_completion_receipt(
        receipt_path,
        expected_sha256=GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    )
    expected_reference: dict[str, str] = {
        "path": HOSTED_COMPLETION_AUDIT_RECEIPT_NAME,
        "sha256": GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
        "entry_id": f"lambada/word/{model}",
    }
    if provenance.get("producer_audit_receipt") != expected_reference:
        raise ValueError(f"Completion audit receipt reference disagrees in {model}")
    if receipt["manifest_sha256"] != _sha256(manifest_path):
        raise ValueError(f"Completion audit manifest identity disagrees for {model}")

    entry = next(row for row in receipt["entries"] if row["model"] == model)
    producer: Any = provenance.get("producer")
    expected_source_format: str = (
        entry["raw_source_format"]
        if entry["source_population"] == FULL_POPULATION
        else entry["public_source_format"]
    )
    if (
        entry["raw_artifact_sha256"] != provenance.get("source_sha256")
        or entry["raw_artifact_size_bytes"] != provenance.get("source_size_bytes")
        or entry["producer_revision_sha256"]
        != (producer.get("producer_revision") if isinstance(producer, dict) else None)
        or receipt["request_parameters"]
        != (producer.get("request_parameters") if isinstance(producer, dict) else None)
        or entry["availability_status"] != provenance.get("availability_status")
        or expected_source_format != provenance.get("source_format")
        or entry["public_artifact_sha256"] != _sha256(segment_path)
        or entry["public_artifact_size_bytes"] != os.path.getsize(segment_path)
    ):
        raise ValueError(f"Completion audit entry metadata disagrees for {model}")

    projection: list[list[Any]] = completion_table_projection(
        segment_frame.to_dict("records")
    )
    summary = completion_projection_summary(projection)
    if (
        entry["public_projection_sha256"]
        != canonical_completion_projection_digest(projection)
        or entry["public_prompt_count"] != summary["prompt_count"]
        or entry["public_segment_count"] != summary["segment_count"]
        or entry["public_original_available_count"]
        != summary["original_available_count"]
        or entry["public_ablated_available_count"] != summary["ablated_available_count"]
        or entry["public_paired_attribution_available_count"]
        != summary["paired_attribution_available_count"]
    ):
        raise ValueError(f"Completion audit public projection disagrees for {model}")

    if entry["source_population"] == CANARY_POPULATION:
        canary: Any = provenance.get("canary")
        if not isinstance(canary, dict):
            raise ValueError(f"Completion audit canary metadata is missing for {model}")
        canary_name: str = str(canary.get("artifact_path", ""))
        canary_path: str = os.path.join(os.path.dirname(segment_path), canary_name)
        if (
            os.path.basename(canary_name) != canary_name
            or not os.path.isfile(canary_path)
            or _sha256(canary_path) != canary.get("artifact_sha256")
        ):
            raise ValueError(f"Completion audit canary artifact disagrees for {model}")
        with open(canary_path, encoding="utf-8") as source:
            canary_payload: Any = json.load(source)
        if not isinstance(canary_payload, list) or not all(
            isinstance(row, dict) for row in canary_payload
        ):
            raise ValueError(f"Completion audit canary payload is invalid for {model}")
        prompt_indices: list[int] = [
            prompt_idx for prompt_idx, _ in LAMBADA_GOLD_CANARY
        ]
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
        raw_projection: list[list[Any]] = completion_payload_projection(
            canary_payload,
            fixed_manifest,
            expected_prompt_indices=prompt_indices,
        )
        raw_summary = completion_projection_summary(raw_projection)
        if (
            entry["raw_projection_sha256"]
            != canonical_completion_projection_digest(raw_projection)
            or entry["raw_prompt_count"] != raw_summary["prompt_count"]
            or entry["raw_segment_count"] != raw_summary["segment_count"]
            or entry["raw_original_available_count"]
            != raw_summary["original_available_count"]
            or entry["raw_ablated_available_count"]
            != raw_summary["ablated_available_count"]
            or entry["raw_paired_attribution_available_count"]
            != raw_summary["paired_attribution_available_count"]
        ):
            raise ValueError(
                f"Completion audit canary projection disagrees for {model}"
            )


def _validate_gold_terminal_failure_counts(
    benchmark: str,
    pregrouper: str,
    model: str,
    original_count: int,
    ablated_count: int,
) -> None:
    """Enforce exact terminal-failure counts stated in the correction record."""

    expected: tuple[int, int] | None = GOLD_HOSTED_TERMINAL_FAILURE_COUNTS.get(
        (benchmark, pregrouper, model)
    )
    actual: tuple[int, int] = (original_count, ablated_count)
    if expected is not None and actual != expected:
        raise ValueError(
            f"Gold terminal-failure count disagrees for "
            f"{benchmark}/{pregrouper}/{model}: expected {expected}, got {actual}"
        )


def _validate_open_segment_metrics(frame: pd.DataFrame, model: str) -> None:
    """Require every released open-model representation value to be finite."""
    missing_metrics: set[str] = set(OPEN_SEGMENT_COLUMNS) - set(frame.columns)
    if missing_metrics:
        raise ValueError(
            f"Open-model segment metrics are missing for {model}: "
            f"{sorted(missing_metrics)}"
        )
    for column in OPEN_SEGMENT_COLUMNS:
        values: pd.Series = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values).all():
            raise ValueError(f"Open-model metric {column!r} is incomplete for {model}")


def _validate_gold_hosted_metadata(
    provenance: dict[str, Any], producer: dict[str, Any], model: str, path: str
) -> None:
    """Reject free-form or nonportable producer identity in a gold artifact."""
    if provenance.get("identity_attestation") != GOLD_HOSTED_IDENTITY_ATTESTATION:
        raise ValueError(f"Gold hosted identity attestation disagrees in {path}")
    revision: str = str(producer.get("producer_revision", ""))
    if LOWERCASE_SHA256_RE.fullmatch(revision) is None:
        raise ValueError(f"Gold hosted producer revision is not SHA-256 in {path}")
    if producer.get("served_model") != model:
        raise ValueError(f"Gold hosted served-model identity disagrees in {path}")
    generated_at: str = str(producer.get("generated_at", ""))
    if RFC3339_RE.fullmatch(generated_at) is None:
        raise ValueError(f"Gold hosted generation timestamp is not RFC3339 in {path}")


def _validate_gold_run_metadata_schema(
    provenance: dict[str, Any],
    model: str,
    benchmark: str,
    source_format: str,
    path: str,
) -> None:
    """Reject undeclared fields and nonportable values in gold run metadata."""
    if model not in OPEN_MODELS:
        expected_hosted_formats: set[str] = (
            {
                "hosted_completion_logprob_json",
                "unsupported_completion_placeholder_after_canary",
            }
            if benchmark == "lambada"
            else {"hosted_label_logprob_json"}
        )
        if source_format not in expected_hosted_formats:
            raise ValueError(
                f"Gold hosted source format is invalid for {benchmark} in {path}: "
                f"{source_format!r}"
            )
    if model in OPEN_MODELS:
        expected_fields: frozenset[str] = OPEN_RUN_FIELDS
    elif source_format == "hosted_label_logprob_json":
        expected_fields = HOSTED_CLASSIFICATION_RUN_FIELDS
    elif source_format in {
        "hosted_completion_logprob_json",
        "unsupported_completion_placeholder_after_canary",
    }:
        expected_fields = HOSTED_COMPLETION_RUN_FIELDS
    elif source_format == "precomputed_portable_tsv":
        expected_fields = HOSTED_PRECOMPUTED_RUN_FIELDS
    else:
        raise ValueError(f"Unknown gold source format {source_format!r} in {path}")
    if set(provenance) != expected_fields:
        raise ValueError(
            f"Gold run metadata fields disagree in {path}: "
            f"missing={sorted(expected_fields - set(provenance))}, "
            f"extra={sorted(set(provenance) - expected_fields)}"
        )
    if model not in OPEN_MODELS:
        if provenance.get("schema_version") != 1:
            raise ValueError(f"Gold hosted schema version disagrees in {path}")
        canary: Any = provenance.get("canary")
        if source_format == "unsupported_completion_placeholder_after_canary":
            if not isinstance(canary, dict):
                raise ValueError(f"Gold hosted canary metadata is missing in {path}")
        elif canary is not None:
            raise ValueError(f"Gold hosted canary metadata must be null in {path}")
        configuration: tuple[str, str] = (
            benchmark,
            str(provenance.get("pregrouper", "")),
        )
        if configuration not in GOLD_MANIFESTS:
            raise ValueError(f"Gold hosted configuration is unknown in {path}")
        expected_segments, expected_prompts, _ = GOLD_MANIFESTS[configuration]
        if source_format in {
            "hosted_label_logprob_json",
            "hosted_completion_logprob_json",
            "unsupported_completion_placeholder_after_canary",
        } and (
            LOWERCASE_SHA256_RE.fullmatch(str(provenance.get("source_sha256", "")))
            is None
            or type(provenance.get("source_size_bytes")) is not int
            or int(provenance["source_size_bytes"]) <= 0
            or type(provenance.get("segments")) is not int
            or int(provenance["segments"]) != expected_segments
        ):
            raise ValueError(f"Gold hosted source summary disagrees in {path}")
        if source_format == "hosted_label_logprob_json" and (
            type(provenance.get("prompts")) is not int
            or int(provenance["prompts"]) != expected_prompts
        ):
            raise ValueError(f"Gold hosted prompt count disagrees in {path}")
        return

    if provenance.get("schema_version") != 5:
        raise ValueError(f"Gold open schema version disagrees in {path}")
    if provenance.get("model_source") != GOLD_OPEN_MODEL_REPOSITORIES.get(model):
        raise ValueError(f"Gold open model source disagrees in {path}")
    if provenance.get("model_artifact_hash_timing") != "post_run_seal":
        raise ValueError(f"Gold open model artifact timing disagrees in {path}")
    dataset: Any = provenance.get("dataset")
    parameters: Any = provenance.get("parameters")
    software: Any = provenance.get("software")
    tokenization: Any = provenance.get("tokenization_verification")
    if not isinstance(dataset, dict) or set(dataset) != OPEN_DATASET_FIELDS:
        raise ValueError(f"Gold open dataset metadata fields disagree in {path}")
    if (
        not isinstance(parameters, dict)
        or set(parameters) != OPEN_PARAMETER_FIELDS
        or not isinstance(parameters.get("rendered_chat_add_special_tokens"), bool)
    ):
        raise ValueError(f"Gold open parameter fields disagree in {path}")
    if not isinstance(software, dict) or set(software) != OPEN_SOFTWARE_FIELDS:
        raise ValueError(f"Gold open software fields disagree in {path}")
    if any(
        not isinstance(value, str) or "/" in value or "\\" in value
        for value in software.values()
        if value is not None
    ):
        raise ValueError(f"Gold open software metadata is nonportable in {path}")
    if tokenization != _expected_open_tokenization_verification(model):
        raise ValueError(f"Gold open tokenization verification disagrees in {path}")
    spec = BENCHMARKS[benchmark]
    expected_dataset_identity: dict[str, Any] = {
        "hf_path": spec.hf_dataset_path,
        "hf_name": spec.hf_dataset_name,
        "hf_split": spec.hf_split,
        "snapshot_filename": GOLD_DATASET_FILES[benchmark],
        "snapshot_sha256": GOLD_DATASET_SHA256[benchmark],
    }
    if any(
        dataset.get(field) != value
        for field, value in expected_dataset_identity.items()
    ):
        raise ValueError(f"Gold open dataset identity disagrees in {path}")
    if (
        not isinstance(dataset.get("rows"), int)
        or int(dataset["rows"]) <= 0
        or LOWERCASE_SHA256_RE.fullmatch(
            str(dataset.get("normalized_frame_sha256", ""))
        )
        is None
    ):
        raise ValueError(f"Gold open dataset summary disagrees in {path}")


def _validate_open_model_identity(
    provenance: dict[str, Any], model: str, path: str
) -> dict[str, Any]:
    """Validate one open run against the immutable public model snapshot."""

    if provenance.get("model_source") != GOLD_OPEN_MODEL_REPOSITORIES.get(
        model
    ) or provenance.get("model_revision") != GOLD_OPEN_MODEL_REVISIONS.get(model):
        raise ValueError(f"Pinned open model revision disagrees in {path}")
    model_hashes: Any = provenance.get("model_artifact_sha256")
    if not isinstance(model_hashes, dict) or not model_hashes:
        raise ValueError(f"Gold open model-artifact metadata is missing in {path}")
    for filename, digest in model_hashes.items():
        if (
            filename not in OPEN_MODEL_METADATA_FILENAMES
            and re.fullmatch(
                r"(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)",
                str(filename),
            )
            is None
        ) or LOWERCASE_SHA256_RE.fullmatch(str(digest)) is None:
            raise ValueError(f"Gold open model-artifact metadata disagrees in {path}")
    actual_manifest_sha256: str = canonical_file_hash_manifest_sha256(model_hashes)
    if provenance.get(
        "model_artifact_manifest_sha256"
    ) != actual_manifest_sha256 or actual_manifest_sha256 != GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256.get(
        model
    ):
        raise ValueError(f"Pinned open model artifact manifest disagrees in {path}")

    identity_hashes: Any = provenance.get("model_identity_files_sha256")
    expected_identity_hashes: dict[str, str] = {
        filename: digest
        for filename, digest in model_hashes.items()
        if filename in OPEN_MODEL_IDENTITY_FILENAMES
    }
    if not expected_identity_hashes or identity_hashes != expected_identity_hashes:
        raise ValueError(
            f"Execution-time and sealing-time model identities disagree in {path}"
        )
    return {
        "execution_model_source": provenance["execution_model_source"],
        "model_source": provenance["model_source"],
        "model_revision": provenance["model_revision"],
        "model_artifact_manifest_sha256": provenance["model_artifact_manifest_sha256"],
        "model_artifact_sha256": model_hashes,
        "model_identity_files_sha256": identity_hashes,
    }


def validate_configuration(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    required_models: tuple[str, ...] = PAPER_MODELS,
    require_gold_manifest: bool = True,
    require_hosted_audit: bool = False,
) -> list[ValidationSummary]:
    """Validate one benchmark/pregrouper directory and return coverage rows."""
    config_dir: str = os.path.join(results_dir, benchmark, pregrouper)
    manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing segment manifest: {manifest_path}")
    manifest, manifest_keys = _validate_manifest(
        manifest_path,
        require_complete_prompts=pregrouper == "sentence",
    )
    if require_gold_manifest:
        _validate_gold_manifest(benchmark, pregrouper, manifest_path, manifest)
    if pregrouper == "word" and len(manifest) != 10_000:
        raise ValueError(
            f"{benchmark}/{pregrouper} should contain the fixed 10,000-pair sample"
        )
    expected_prompts: set[int] = set(manifest["prompt_idx"].astype(int))
    spec = BENCHMARKS[benchmark]
    expected_labels: set[str] = (
        set(spec.eval_config.label_tokens) if spec.eval_config is not None else set()
    )

    segment_paths: dict[str, str] = {}
    for path in sorted(
        set(
            glob.glob(os.path.join(config_dir, "*_segment.tsv"))
            + glob.glob(os.path.join(config_dir, "*_segment.tsv.gz"))
        )
    ):
        model: str = _model_from_path(path, "_segment.tsv")
        if model in segment_paths:
            raise ValueError(
                f"Duplicate compressed/uncompressed segment output for {model}"
            )
        segment_paths[model] = path
    effective_required_models: set[str] = set(required_models)
    if require_hosted_audit and expected_labels:
        effective_required_models.update(HOSTED_CLASSIFICATION_MODELS)
    missing_models: set[str] = effective_required_models - set(segment_paths)
    if missing_models:
        raise ValueError(
            f"{benchmark}/{pregrouper} is missing models {sorted(missing_models)}"
        )
    allowed_models: tuple[str, ...] = OPEN_MODELS + API_MODELS
    unexpected_models: set[str] = set(segment_paths) - set(allowed_models)
    if unexpected_models:
        raise ValueError(
            f"{benchmark}/{pregrouper} has unexpected models "
            f"{sorted(unexpected_models)}"
        )
    models_to_validate: list[str] = [
        model for model in allowed_models if model in segment_paths
    ]

    summaries: list[ValidationSummary] = []
    for model in models_to_validate:
        row_count, missing_count, extra_count = _validate_segment_file(
            segment_paths[model], manifest, manifest_keys
        )
        model_frame: pd.DataFrame = _read_tsv(segment_paths[model])
        if model in OPEN_MODELS:
            if not model_frame["segment_result_available"].eq(True).all():
                raise ValueError(
                    f"Open-model segment coverage is incomplete for {model}"
                )
            _validate_open_segment_metrics(model_frame, model)
        original_coverage: float | None = None
        ablated_coverage: float | None = None
        paired_attribution_coverage: float | None = None
        label_cell_coverage: float | None = None
        original_content_filter_count: int | None = None
        ablated_content_filter_count: int | None = None
        original_terminal_failure_count: int | None = None
        ablated_terminal_failure_count: int | None = None
        if expected_labels:
            token_base: str = os.path.join(config_dir, f"{model}_tokens.tsv")
            token_path: str | None = next(
                (
                    candidate
                    for candidate in (token_base + ".gz", token_base)
                    if os.path.exists(candidate)
                ),
                None,
            )
            if token_path is None:
                raise FileNotFoundError(f"Missing token output for {model}")
            (
                original_coverage,
                ablated_coverage,
                label_cell_coverage,
            ) = _token_coverage(
                token_path,
                manifest_keys,
                expected_prompts,
                expected_labels,
                model_frame,
            )
            if model in OPEN_MODELS and (
                original_coverage != 1.0
                or ablated_coverage != 1.0
                or label_cell_coverage != 1.0
            ):
                raise ValueError(f"Open-model token coverage is incomplete for {model}")
            if "original_result_available" not in model_frame.columns:
                raise ValueError(
                    f"{segment_paths[model]} lacks original_result_available"
                )
            paired_attribution_coverage = float(
                (
                    model_frame["original_result_available"].eq(True)
                    & model_frame["segment_result_available"].eq(True)
                ).mean()
            )
        else:
            completion: pd.DataFrame = _read_tsv(segment_paths[model])
            required_completion: set[str] = {
                "orig_completion_logprob",
                "ablated_completion_logprob",
            }
            missing_completion: set[str] = required_completion - set(completion.columns)
            if missing_completion:
                raise ValueError(
                    f"{segment_paths[model]} is missing completion columns "
                    f"{sorted(missing_completion)}"
                )
            finite_orig: pd.Series = np.isfinite(
                pd.to_numeric(completion["orig_completion_logprob"], errors="coerce")
            )
            finite_ablated: pd.Series = np.isfinite(
                pd.to_numeric(completion["ablated_completion_logprob"], errors="coerce")
            )
            if "original_result_available" not in completion.columns:
                raise ValueError(
                    f"{segment_paths[model]} lacks original_result_available"
                )
            if (
                completion.groupby("prompt_idx")["orig_completion_logprob"]
                .nunique(dropna=False)
                .gt(1)
                .any()
            ):
                raise ValueError(
                    f"{segment_paths[model]} has inconsistent original completion "
                    "scores within a prompt"
                )
            if (
                not completion["segment_result_available"].eq(finite_ablated).all()
                or not completion["original_result_available"].eq(finite_orig).all()
            ):
                raise ValueError(
                    f"{segment_paths[model]} has incorrect completion availability flags"
                )
            original_coverage = float(
                completion.loc[finite_orig, "prompt_idx"].nunique()
                / len(expected_prompts)
            )
            ablated_coverage = float(finite_ablated.mean())
            paired_attribution_coverage = float((finite_orig & finite_ablated).mean())
            if model in OPEN_MODELS and (
                original_coverage != 1.0 or ablated_coverage != 1.0
            ):
                raise ValueError(
                    f"Open-model completion coverage is incomplete for {model}"
                )
        run_metadata: str = os.path.join(config_dir, f"{model}_run.json")
        if not os.path.exists(run_metadata):
            raise FileNotFoundError(f"Missing run provenance: {run_metadata}")
        with open(run_metadata, encoding="utf-8") as source:
            provenance: dict[str, Any] = json.load(source)
        expected_provenance: dict[str, str] = {
            "model": model,
            "benchmark": benchmark,
            "pregrouper": pregrouper,
        }
        for field, expected in expected_provenance.items():
            if provenance.get(field) != expected:
                raise ValueError(
                    f"Invalid {field} in {run_metadata}: "
                    f"expected {expected!r}, got {provenance.get(field)!r}"
                )
        if provenance.get("segmentation_scope") != "full_dialog_in_message_order":
            raise ValueError(f"Invalid segmentation scope in {run_metadata}")
        if not provenance.get("source_sha256"):
            raise ValueError(f"Missing source hashes in {run_metadata}")
        source_format: str = str(provenance.get("source_format", ""))
        if require_gold_manifest:
            _validate_gold_run_metadata_schema(
                provenance,
                model,
                benchmark,
                source_format,
                run_metadata,
            )
        artifact_hashes: Any = provenance.get("artifact_sha256")
        expected_artifact_hashes: dict[str, str] = {
            "segment": _sha256(segment_paths[model])
        }
        token_candidate: str = os.path.join(config_dir, f"{model}_tokens.tsv.gz")
        if os.path.exists(token_candidate):
            expected_artifact_hashes["tokens"] = _sha256(token_candidate)
        if model in API_MODELS:
            allowed_hosted_formats: set[str] = {
                "hosted_label_logprob_json",
                "hosted_completion_logprob_json",
                "unsupported_completion_placeholder_after_canary",
                "precomputed_portable_tsv",
            }
            if source_format not in allowed_hosted_formats:
                raise ValueError(
                    f"Invalid hosted source format {source_format!r} in {run_metadata}"
                )
            transformation_files: tuple[str, ...] = (
                HOSTED_RECORD_SOURCE_FILES
                if source_format == "precomputed_portable_tsv"
                else HOSTED_IMPORT_SOURCE_FILES
            )
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            expected_transformation_hashes: dict[str, str] = (
                GOLD_HOSTED_IMPORT_SOURCE_SHA256
                if require_gold_manifest
                else {
                    relative_path: _sha256(os.path.join(repository_root, relative_path))
                    for relative_path in transformation_files
                }
            )
            if (
                provenance.get("transformation_source_sha256")
                != expected_transformation_hashes
            ):
                raise ValueError(
                    f"Hosted transformation source hashes disagree in {run_metadata}"
                )
            if not isinstance(artifact_hashes, dict):
                raise ValueError(f"Missing artifact hashes in {run_metadata}")
            if artifact_hashes != expected_artifact_hashes:
                raise ValueError(f"Artifact hashes disagree in {run_metadata}")
            if provenance.get("manifest_sha256") != _sha256(manifest_path):
                raise ValueError(f"Manifest hash disagrees in {run_metadata}")
            if not str(provenance.get("identity_attestation", "")).strip():
                raise ValueError(f"Missing identity attestation in {run_metadata}")
            producer: Any = provenance.get("producer")
            required_producer_fields: set[str] = {
                "producer_revision",
                "generated_at",
                "served_model",
                "request_parameters",
            }
            if (
                not isinstance(producer, dict)
                or set(producer) != required_producer_fields
            ):
                raise ValueError(f"Missing producer metadata in {run_metadata}")
            if (
                not all(
                    str(producer[field]).strip()
                    for field in ("producer_revision", "generated_at", "served_model")
                )
                or not isinstance(producer["request_parameters"], dict)
                or not producer["request_parameters"]
            ):
                raise ValueError(f"Invalid producer metadata in {run_metadata}")
            if require_gold_manifest:
                _validate_gold_hosted_metadata(
                    provenance, cast(dict[str, Any], producer), model, run_metadata
                )
            availability_status: str = str(provenance.get("availability_status", ""))
            if availability_status not in {
                "complete",
                "complete_with_terminal_failures",
                "unsupported_after_canary",
            }:
                raise ValueError(f"Invalid availability status in {run_metadata}")
            if availability_status == "complete_with_terminal_failures" and (
                benchmark == "lambada" or source_format != "hosted_label_logprob_json"
            ):
                raise ValueError(
                    "Terminal-failure availability is valid only for hosted "
                    f"classification JSON in {run_metadata}"
                )
            if benchmark != "lambada" and source_format == "hosted_label_logprob_json":
                if (
                    producer["request_parameters"]
                    != HOSTED_CLASSIFICATION_REQUEST_PARAMETERS
                ):
                    raise ValueError(
                        f"Hosted classification protocol disagrees in {run_metadata}"
                    )
                required_request_statuses: set[str] = {
                    "original_request_status",
                    "segment_request_status",
                }
                if not required_request_statuses.issubset(model_frame.columns):
                    raise ValueError(
                        f"Hosted request status columns are missing in "
                        f"{segment_paths[model]}"
                    )
                allowed_request_statuses: set[str] = {
                    "ok",
                    "content_filter",
                    "transient_exhausted",
                }
                terminal_failure_statuses: set[str] = {"transient_exhausted"}
                original_request_status: pd.Series = model_frame[
                    "original_request_status"
                ].map(str)
                segment_request_status: pd.Series = model_frame[
                    "segment_request_status"
                ].map(str)
                original_content_filter_count = int(
                    model_frame.loc[
                        original_request_status.eq("content_filter"), "prompt_idx"
                    ].nunique()
                )
                ablated_content_filter_count = int(
                    segment_request_status.eq("content_filter").sum()
                )
                original_terminal_failure_count = int(
                    model_frame.loc[
                        original_request_status.isin(terminal_failure_statuses),
                        "prompt_idx",
                    ].nunique()
                )
                ablated_terminal_failure_count = int(
                    segment_request_status.isin(terminal_failure_statuses).sum()
                )
                if (
                    model_frame.assign(original_request_status=original_request_status)
                    .groupby("prompt_idx")["original_request_status"]
                    .nunique(dropna=False)
                    .gt(1)
                    .any()
                ):
                    raise ValueError(
                        f"Hosted classification original request statuses disagree "
                        f"within a prompt in {segment_paths[model]}"
                    )
                if not set(original_request_status).issubset(
                    allowed_request_statuses
                ) or not set(segment_request_status).issubset(allowed_request_statuses):
                    raise ValueError(
                        f"Hosted classification run contains unclassified failed calls "
                        f"in {segment_paths[model]}"
                    )
                has_terminal_failures: bool = bool(
                    original_terminal_failure_count or ablated_terminal_failure_count
                )
                if has_terminal_failures != (
                    availability_status == "complete_with_terminal_failures"
                ):
                    raise ValueError(
                        "Hosted classification availability status disagrees with "
                        f"terminal failed calls in {segment_paths[model]}"
                    )
                if require_hosted_audit:
                    _validate_gold_terminal_failure_counts(
                        benchmark,
                        pregrouper,
                        model,
                        original_terminal_failure_count,
                        ablated_terminal_failure_count,
                    )
                if (
                    model_frame.loc[
                        original_request_status.ne("ok"),
                        "original_result_available",
                    ]
                    .eq(True)
                    .any()
                    or model_frame.loc[
                        segment_request_status.ne("ok"),
                        "segment_result_available",
                    ]
                    .eq(True)
                    .any()
                ):
                    raise ValueError(
                        f"Hosted failed-call status disagrees with availability "
                        f"in {segment_paths[model]}"
                    )
                if require_hosted_audit:
                    if token_path is None:
                        raise FileNotFoundError(f"Missing token output for {model}")
                    _validate_hosted_audit_binding(
                        results_dir,
                        benchmark,
                        pregrouper,
                        model,
                        manifest_path,
                        model_frame,
                        token_path,
                        provenance,
                    )
            if source_format == "precomputed_portable_tsv":
                raise ValueError(
                    f"Corrected hosted gold requires raw-JSON request status "
                    f"evidence, not a precomputed TSV, in {run_metadata}"
                )
            if benchmark == "lambada" and source_format in {
                "hosted_completion_logprob_json",
                "unsupported_completion_placeholder_after_canary",
            }:
                required_status_columns: set[str] = {
                    "original_result_status",
                    "segment_result_status",
                }
                if not required_status_columns.issubset(completion.columns):
                    raise ValueError(
                        f"Hosted completion status columns are missing in "
                        f"{segment_paths[model]}"
                    )
                original_status: pd.Series = completion["original_result_status"].map(
                    str
                )
                segment_status: pd.Series = completion["segment_result_status"].map(str)
                if (
                    completion.groupby("prompt_idx")["original_result_status"]
                    .nunique(dropna=False)
                    .gt(1)
                    .any()
                ):
                    raise ValueError(
                        f"Hosted completion original statuses disagree within a "
                        f"prompt in {segment_paths[model]}"
                    )
                if not original_status.eq("ok").eq(finite_orig).all() or not (
                    segment_status.eq("ok").eq(finite_ablated).all()
                ):
                    raise ValueError(
                        f"Hosted completion statuses disagree with scores in "
                        f"{segment_paths[model]}"
                    )
                allowed_completion_statuses: set[str] = {
                    "ok",
                    "application_error",
                    "echo_response_unavailable",
                    "nonfinite_score",
                    "target_parse_unavailable",
                }
                if source_format == "hosted_completion_logprob_json" and not (
                    set(original_status).issubset(allowed_completion_statuses)
                    and set(segment_status).issubset(allowed_completion_statuses)
                ):
                    raise ValueError(
                        f"Hosted completion contains a failed-call status in "
                        f"{segment_paths[model]}"
                    )
            if (
                benchmark == "lambada"
                and source_format == "hosted_completion_logprob_json"
                and producer["request_parameters"]
                != {
                    "max_tokens": 0,
                    "echo": True,
                    "scoring": "teacher_forced_echo_target_logprob_sum",
                    "top_logprobs": 20,
                    "max_transient_attempts": 5,
                }
            ):
                raise ValueError(
                    f"Hosted completion protocol disagrees in {run_metadata}"
                )
            if availability_status == "unsupported_after_canary":
                if (
                    source_format != "unsupported_completion_placeholder_after_canary"
                    or original_coverage != 0.0
                    or ablated_coverage != 0.0
                    or paired_attribution_coverage != 0.0
                    or not original_status.eq("unsupported_after_canary").all()
                    or not segment_status.eq("unsupported_after_canary").all()
                ):
                    raise ValueError(
                        f"Unsupported output is not an all-missing placeholder in "
                        f"{run_metadata}"
                    )
                canary: Any = provenance.get("canary")
                required_canary_fields: set[str] = {
                    "artifact_path",
                    "artifact_sha256",
                    "seed",
                    "selection",
                    "sample_size",
                    "prompt_indices",
                    "expected_segments",
                    "original_coverage",
                    "ablated_coverage",
                    "paired_attribution_coverage",
                }
                if (
                    benchmark != "lambada"
                    or not isinstance(canary, dict)
                    or set(canary) != required_canary_fields
                ):
                    raise ValueError(
                        f"Invalid unsupported-canary evidence in {run_metadata}"
                    )
                try:
                    sample_size: int = int(canary["sample_size"])
                    canary_seed: int = int(canary["seed"])
                    canary_original: float = float(canary["original_coverage"])
                    canary_paired: float = float(canary["paired_attribution_coverage"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid unsupported-canary values in {run_metadata}"
                    ) from error
                if (
                    sample_size != 10
                    or canary_seed != 42
                    or not 0.0 <= canary_original <= 1.0
                    or not 0.0 <= canary_paired <= 1.0
                    or min(canary_original, canary_paired) >= MIN_HOSTED_RESULT_COVERAGE
                ):
                    raise ValueError(
                        f"Unsupported-canary evidence does not justify skipping "
                        f"generation in {run_metadata}"
                    )
                request_parameters: Any = producer["request_parameters"]
                if request_parameters != {
                    "max_tokens": 0,
                    "echo": True,
                    "scoring": "teacher_forced_echo_target_logprob_sum",
                    "top_logprobs": 20,
                    "max_transient_attempts": 5,
                }:
                    raise ValueError(
                        f"Unsupported canary protocol disagrees in {run_metadata}"
                    )
                canary_relative_path: str = str(canary["artifact_path"])
                if os.path.basename(canary_relative_path) != canary_relative_path:
                    raise ValueError(f"Invalid canary path in {run_metadata}")
                canary_path: str = os.path.join(
                    os.path.dirname(run_metadata), canary_relative_path
                )
                if not os.path.isfile(canary_path) or _sha256(canary_path) != str(
                    canary["artifact_sha256"]
                ):
                    raise ValueError(
                        f"Canary artifact hash disagrees in {run_metadata}"
                    )
                with open(canary_path, encoding="utf-8") as canary_source:
                    canary_payload: Any = json.load(canary_source)
                canary_row_fields: set[str] = {
                    "prompt_idx",
                    "answer",
                    "n_segments",
                    "orig_logprob",
                    "ablated_logprob",
                    "ablation_indices",
                }
                required_canary_row_fields: set[str] = canary_row_fields - {
                    "ablation_indices"
                }
                if not isinstance(canary_payload, list) or any(
                    not isinstance(row, dict)
                    or not required_canary_row_fields.issubset(row)
                    or not set(row).issubset(canary_row_fields)
                    for row in canary_payload
                ):
                    raise ValueError(
                        f"Canary artifact contents disagree in {run_metadata}"
                    )
                recomputed_canary: dict[str, Any] = summarize_completion_canary(
                    canary_payload,
                    canary_seed,
                    sample_size,
                )
                for field, recomputed_value in recomputed_canary.items():
                    recorded_value: Any = canary[field]
                    if isinstance(recomputed_value, float):
                        agrees: bool = np.isclose(
                            float(recorded_value), recomputed_value
                        )
                    else:
                        agrees = recorded_value == recomputed_value
                    if not agrees:
                        raise ValueError(
                            f"Canary field {field!r} disagrees with its artifact "
                            f"in {run_metadata}"
                        )
                if (
                    original_coverage is None
                    or paired_attribution_coverage is None
                    or min(original_coverage, paired_attribution_coverage)
                    >= MIN_HOSTED_RESULT_COVERAGE
                ):
                    raise ValueError(
                        f"Unsupported final artifact exceeds the coverage threshold "
                        f"in {run_metadata}"
                    )
            elif source_format == "unsupported_completion_placeholder_after_canary":
                raise ValueError(
                    f"Canary placeholder is not marked unsupported in {run_metadata}"
                )
            elif benchmark == "lambada" and (
                original_coverage is None
                or ablated_coverage is None
                or paired_attribution_coverage is None
                or original_coverage < MIN_HOSTED_RESULT_COVERAGE
                or ablated_coverage < MIN_HOSTED_RESULT_COVERAGE
                or paired_attribution_coverage < MIN_HOSTED_RESULT_COVERAGE
            ):
                raise ValueError(
                    f"Complete hosted run is below the "
                    f"{MIN_HOSTED_RESULT_COVERAGE:.0%} coverage floor in "
                    f"{run_metadata}"
                )
            if require_hosted_audit and benchmark == "lambada":
                _validate_hosted_completion_audit_binding(
                    results_dir,
                    model,
                    manifest_path,
                    segment_paths[model],
                    completion,
                    provenance,
                )
            if (
                source_format == "precomputed_portable_tsv"
                and not str(provenance.get("source_revision", "")).strip()
            ):
                raise ValueError(f"Missing source revision in {run_metadata}")
        if model in OPEN_MODELS:
            _validate_open_model_identity(provenance, model, run_metadata)
            parameters: Any = provenance.get("parameters")
            expected_phase_backends: dict[str, str] = {
                "ablation": "sdpa",
                "attention": "eager",
            }
            if (
                provenance.get("schema_version") != 5
                or not isinstance(parameters, dict)
                or parameters.get("phases") != ["ablation", "attention"]
                or parameters.get("phase_attention_implementation")
                != expected_phase_backends
            ):
                raise ValueError(
                    f"Open phase/backend/tokenization provenance disagrees in "
                    f"{run_metadata}"
                )
            execution_hashes: Any = provenance.get("source_sha256")
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            gold_execution_hashes: dict[str, str] = dict(
                GOLD_OPEN_EXECUTION_SOURCE_SHA256
            )
            if artifact_hashes != expected_artifact_hashes:
                raise ValueError(f"Open artifact hashes disagree in {run_metadata}")
            if provenance.get("manifest_sha256") != _sha256(manifest_path):
                raise ValueError(f"Open manifest hash disagrees in {run_metadata}")
            if require_gold_manifest:
                dataset_metadata: Any = provenance.get("dataset")
                if (
                    not isinstance(dataset_metadata, dict)
                    or dataset_metadata.get("snapshot_sha256")
                    != GOLD_DATASET_SHA256[benchmark]
                ):
                    raise ValueError(f"Frozen dataset hash disagrees in {run_metadata}")
            model_hashes: Any = provenance.get("model_artifact_sha256")
            if not isinstance(model_hashes, dict) or not model_hashes:
                raise ValueError(f"Missing model artifact hashes in {run_metadata}")
            seal_path: str = os.path.join(
                os.path.dirname(__file__), "seal_open_provenance.py"
            )
            if provenance.get("provenance_seal_sha256") != _sha256(seal_path):
                raise ValueError(f"Invalid provenance seal in {run_metadata}")
            release_hashes: Any = provenance.get("release_source_sha256")
            if not isinstance(release_hashes, dict) or set(release_hashes) != set(
                OPEN_COMPLETE_SOURCE_FILES
            ):
                raise ValueError(f"Missing release source hashes in {run_metadata}")
            for relative_path, expected_hash in release_hashes.items():
                source_path: str = os.path.join(repository_root, str(relative_path))
                if (
                    not os.path.isfile(source_path)
                    or _sha256(source_path) != expected_hash
                ):
                    raise ValueError(
                        f"Release source hash disagrees for {relative_path} in "
                        f"{run_metadata}"
                    )
            release_execution_hashes: dict[str, str] = {
                path: release_hashes[path] for path in OPEN_EXECUTION_SOURCE_FILES
            }
            if execution_hashes == gold_execution_hashes:
                if model == "llama-3.1-8b-instruct":
                    raise ValueError(
                        f"Legacy duplicated-BOS Llama execution is forbidden in "
                        f"{run_metadata}"
                    )
                expected_corrections: dict[str, dict[str, str]] = (
                    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS
                )
                expected_execution_model_source: str | None = (
                    GOLD_OPEN_EXECUTION_MODEL_SOURCES.get(model)
                )
                expected_source_hash_timing: str = "run_completion"
                expected_dependency_hashes: dict[str, str] = dict(
                    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
                )
                expected_dependency_hash_timing: str = (
                    "post_run_reconstruction_not_execution_attested"
                )
                expected_add_special_tokens: bool = True
            elif execution_hashes == release_execution_hashes:
                expected_corrections = {}
                expected_execution_model_source = GOLD_OPEN_MODEL_REPOSITORIES.get(
                    model
                )
                expected_source_hash_timing = "run_start"
                expected_dependency_hashes = release_execution_hashes
                expected_dependency_hash_timing = "run_start"
                expected_add_special_tokens = False
            else:
                raise ValueError(
                    f"Execution source hashes are neither gold nor current release "
                    f"in {run_metadata}"
                )
            if (
                provenance.get("execution_model_source")
                != expected_execution_model_source
            ):
                raise ValueError(
                    f"Gold open execution model source disagrees in {run_metadata}"
                )
            if (
                provenance.get("execution_source_hash_timing")
                != expected_source_hash_timing
            ):
                raise ValueError(
                    f"Gold open execution source timing disagrees in {run_metadata}"
                )
            tokenization: Any = provenance.get("tokenization_verification")
            if (
                parameters.get("rendered_chat_add_special_tokens")
                is not expected_add_special_tokens
                or provenance.get("execution_dependency_sha256")
                != expected_dependency_hashes
                or provenance.get("execution_dependency_hash_timing")
                != expected_dependency_hash_timing
                or tokenization != _expected_open_tokenization_verification(model)
            ):
                raise ValueError(
                    f"Open execution tokenization/dependency provenance disagrees "
                    f"in {run_metadata}"
                )
            actual_corrections: dict[str, dict[str, str]] = {}
            for relative_path, execution_sha256 in expected_dependency_hashes.items():
                release_sha256: str | None = release_hashes.get(relative_path)
                if release_sha256 == execution_sha256:
                    continue
                expected: dict[str, str] | None = expected_corrections.get(
                    relative_path
                )
                if (
                    expected is None
                    or expected.get("execution_sha256") != execution_sha256
                    or expected.get("release_sha256") != release_sha256
                ):
                    raise ValueError(
                        f"Unapproved execution/release source change for "
                        f"{relative_path} in {run_metadata}"
                    )
                actual_corrections[relative_path] = expected
            if (
                actual_corrections != expected_corrections
                or provenance.get("release_source_corrections") != expected_corrections
            ):
                raise ValueError(
                    f"Execution/release source corrections disagree in {run_metadata}"
                )
        summaries.append(
            ValidationSummary(
                benchmark=benchmark,
                pregrouper=pregrouper,
                model=model,
                expected_segments=len(manifest_keys),
                segment_rows=row_count,
                missing_segment_rows=missing_count,
                extra_segment_rows=extra_count,
                original_prompt_coverage=original_coverage,
                ablated_segment_coverage=ablated_coverage,
                paired_attribution_coverage=paired_attribution_coverage,
                label_cell_coverage=label_cell_coverage,
                original_content_filter_count=original_content_filter_count,
                ablated_content_filter_count=ablated_content_filter_count,
                original_terminal_failure_count=original_terminal_failure_count,
                ablated_terminal_failure_count=ablated_terminal_failure_count,
            )
        )
    return summaries


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_open_model_identity_consistency(results_dir: str) -> None:
    """Require every benchmark cell for a model to name identical model bytes."""

    for model in OPEN_MODELS:
        baseline: dict[str, Any] | None = None
        baseline_path: str | None = None
        for benchmark, pregrouper in ARTIFACT_CONFIGS:
            path: str = os.path.join(
                results_dir, benchmark, pregrouper, f"{model}_run.json"
            )
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8") as source:
                provenance: Any = json.load(source)
            if not isinstance(provenance, dict):
                raise ValueError(f"Invalid open run metadata in {path}")
            signature: dict[str, Any] = {
                field: provenance.get(field)
                for field in (
                    "execution_dependency_hash_timing",
                    "execution_dependency_sha256",
                    "execution_model_source",
                    "execution_source_hash_timing",
                    "model_source",
                    "model_revision",
                    "model_artifact_manifest_sha256",
                    "model_artifact_sha256",
                    "model_identity_files_sha256",
                    "source_sha256",
                    "tokenization_verification",
                )
            }
            if baseline is None:
                baseline = signature
                baseline_path = path
            elif signature != baseline:
                raise ValueError(
                    f"Open model identity differs between {baseline_path} and {path}"
                )


def _layer_artifact_columns(benchmark: str) -> set[str]:
    """Return the exact scalar-only schema for one layer artifact."""
    eval_config = BENCHMARKS[benchmark].eval_config
    if eval_config is None:
        raise ValueError(f"Layerwise benchmark {benchmark!r} has no label config")
    labels: tuple[str, ...] = tuple(eval_config.label_tokens)
    columns: set[str] = {
        "prompt_idx",
        "seg_idx",
        "kind",
        "answer",
        "layer_slot",
        "layer_kind",
        "block_idx",
        "delta_norm_postnorm",
        *(f"label_score_{label_column_alias(label)}" for label in labels),
    }
    for first, second in itertools.combinations(labels, 2):
        suffix: str = f"{label_column_alias(first)}_vs_{label_column_alias(second)}"
        columns.add(f"w_dot_delta_z_postnorm_{suffix}")
        columns.add(f"w_norm_{suffix}")
    return columns


def _canonical_layer_aliases(benchmark: str) -> dict[str, dict[str, str]]:
    """Return each configured layer label's exact report-alias surface map."""
    eval_config = BENCHMARKS[benchmark].eval_config
    if eval_config is None:
        raise ValueError(f"Layerwise benchmark {benchmark!r} has no label config")
    aliases: dict[str, dict[str, str]] = {}
    for label in eval_config.label_tokens:
        report_tokens = eval_config.report_tokens.get(label)
        if not report_tokens:
            raise ValueError(
                f"Layerwise label {label!r} has no canonical report tokens"
            )
        alias_map: dict[str, str] = {
            token.alias: token.surface for token in report_tokens
        }
        if len(alias_map) != len(report_tokens):
            raise ValueError(
                f"Layerwise label {label!r} has duplicate report-token aliases"
            )
        aliases[label] = alias_map
    return aliases


def _is_token_id(value: Any) -> bool:
    """Return whether *value* is an integer token ID rather than a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_layer_alias_metadata(
    label_metadata: Any,
    benchmark: str,
    path: str,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Validate the exact tokenizer-result partition of configured aliases.

    Canonical release artifacts reject deduplicated aliases. Although merging
    aliases with the same token ID is harmless for a set-valued query, it makes
    a nominal alias-weighted estimator silently model-dependent.

    Returns:
        Accepted and rejected alias sets keyed by label.
    """
    canonical: dict[str, dict[str, str]] = _canonical_layer_aliases(benchmark)
    if not isinstance(label_metadata, dict) or set(label_metadata) != set(canonical):
        raise ValueError(f"Layer label metadata disagrees in {path}")

    expected_detail_fields: frozenset[str] = frozenset(
        {
            "accepted_single_token_aliases",
            "deduplicated_single_token_aliases",
            "rejected_multitoken_aliases",
        }
    )
    accepted_by_label: dict[str, set[str]] = {}
    rejected_by_label: dict[str, set[str]] = {}
    accepted_token_owner: dict[int, tuple[str, str]] = {}
    has_deduplicated_aliases: bool = False
    for label, expected_aliases in canonical.items():
        details: Any = label_metadata.get(label)
        if not isinstance(details, dict) or set(details) != expected_detail_fields:
            raise ValueError(f"Layer alias metadata fields disagree in {path}")
        accepted: Any = details["accepted_single_token_aliases"]
        deduplicated: Any = details["deduplicated_single_token_aliases"]
        rejected: Any = details["rejected_multitoken_aliases"]
        if not all(
            isinstance(values, list) for values in (accepted, deduplicated, rejected)
        ):
            raise ValueError(f"Layer alias metadata partition is malformed in {path}")
        if not accepted:
            raise ValueError(f"Layer label {label!r} has no accepted aliases in {path}")

        observed_aliases: dict[str, tuple[str, str]] = {}
        accepted_aliases: set[str] = set()
        accepted_ids: dict[str, int] = {}
        for item in accepted:
            if not isinstance(item, dict) or set(item) != {
                "alias",
                "surface",
                "token_id",
            }:
                raise ValueError(f"Accepted layer aliases are malformed in {path}")
            alias: Any = item.get("alias")
            surface: Any = item.get("surface")
            token_id: Any = item.get("token_id")
            if (
                not isinstance(alias, str)
                or not isinstance(surface, str)
                or not _is_token_id(token_id)
                or alias in observed_aliases
            ):
                raise ValueError(f"Accepted layer aliases are malformed in {path}")
            observed_aliases[alias] = (surface, "accepted")
            accepted_aliases.add(alias)
            accepted_ids[alias] = token_id
            previous_owner: tuple[str, str] | None = accepted_token_owner.get(token_id)
            if previous_owner is not None:
                raise ValueError(
                    f"Accepted layer token ID {token_id} is shared by "
                    f"{previous_owner} and {(label, alias)} in {path}"
                )
            accepted_token_owner[token_id] = (label, alias)

        for item in deduplicated:
            if not isinstance(item, dict) or set(item) != {
                "alias",
                "surface",
                "token_id",
                "duplicate_of_alias",
            }:
                raise ValueError(f"Deduplicated layer aliases are malformed in {path}")
            alias = item.get("alias")
            surface = item.get("surface")
            token_id = item.get("token_id")
            duplicate_of: Any = item.get("duplicate_of_alias")
            if (
                not isinstance(alias, str)
                or not isinstance(surface, str)
                or not _is_token_id(token_id)
                or not isinstance(duplicate_of, str)
                or duplicate_of not in accepted_ids
                or token_id != accepted_ids[duplicate_of]
                or alias in observed_aliases
            ):
                raise ValueError(f"Deduplicated layer aliases are malformed in {path}")
            observed_aliases[alias] = (surface, "deduplicated")
            has_deduplicated_aliases = True

        rejected_aliases: set[str] = set()
        for item in rejected:
            if not isinstance(item, dict) or set(item) != {
                "alias",
                "surface",
                "token_ids",
            }:
                raise ValueError(f"Rejected layer aliases are malformed in {path}")
            alias = item.get("alias")
            surface = item.get("surface")
            token_ids: Any = item.get("token_ids")
            if (
                not isinstance(alias, str)
                or not isinstance(surface, str)
                or not isinstance(token_ids, list)
                or len(token_ids) == 1
                or not all(_is_token_id(token_id) for token_id in token_ids)
                or alias in observed_aliases
            ):
                raise ValueError(f"Rejected layer aliases are malformed in {path}")
            observed_aliases[alias] = (surface, "rejected")
            rejected_aliases.add(alias)

        observed_surfaces: dict[str, str] = {
            alias: surface for alias, (surface, _status) in observed_aliases.items()
        }
        if observed_surfaces != expected_aliases:
            raise ValueError(f"Layer alias metadata partition disagrees in {path}")
        accepted_by_label[label] = accepted_aliases
        rejected_by_label[label] = rejected_aliases

    if has_deduplicated_aliases:
        raise ValueError(
            f"Canonical layer release cannot contain deduplicated aliases in {path}"
        )
    return accepted_by_label, rejected_by_label


def _validate_ordinary_layer_token_grid(
    tokens: pd.DataFrame,
    final: pd.DataFrame,
    benchmark: str,
    accepted_by_label: dict[str, set[str]],
    rejected_by_label: dict[str, set[str]],
    path: str,
) -> pd.DataFrame:
    """Require the exact observation-by-canonical-alias ordinary token grid."""
    canonical: dict[str, dict[str, str]] = _canonical_layer_aliases(benchmark)
    checked: pd.DataFrame = tokens.copy()
    checked["prompt_idx"] = pd.to_numeric(checked["prompt_idx"], errors="raise")
    if (checked["prompt_idx"] % 1 != 0).any():
        raise ValueError(f"Ordinary token prompt indices are invalid in {path}")
    checked["prompt_idx"] = checked["prompt_idx"].astype(int)
    numeric_segments: pd.Series = pd.to_numeric(checked["seg_idx"], errors="coerce")
    if (numeric_segments.dropna() % 1 != 0).any():
        raise ValueError(f"Ordinary token segment indices are invalid in {path}")
    checked["seg_key"] = numeric_segments.fillna(-1).astype(int)
    if checked[["kind", "label", "token"]].isna().any().any():
        raise ValueError(f"Ordinary token identities are missing in {path}")
    checked["logprob"] = pd.to_numeric(checked["logprob"], errors="raise")

    identity: list[str] = ["prompt_idx", "seg_key", "kind"]
    alias_identity: list[str] = [*identity, "label", "token"]
    if checked.duplicated(alias_identity).any():
        raise ValueError(f"Ordinary token grid has duplicate cells in {path}")
    canonical_pairs: set[tuple[str, str]] = {
        (label, alias) for label, aliases in canonical.items() for alias in aliases
    }
    observed_pairs: set[tuple[str, str]] = set(
        zip(checked["label"].astype(str), checked["token"].astype(str))
    )
    if observed_pairs != canonical_pairs:
        raise ValueError(f"Ordinary token alias grid disagrees in {path}")

    final_observations: set[tuple[int, int, str]] = set(
        zip(
            final["prompt_idx"].astype(int),
            final["seg_key"].astype(int),
            final["kind"].astype(str),
        )
    )
    if len(final_observations) != len(final):
        raise ValueError(f"Final layer has duplicate observations for {path}")
    token_observations: set[tuple[int, int, str]] = set(
        zip(
            checked["prompt_idx"],
            checked["seg_key"],
            checked["kind"].astype(str),
        )
    )
    counts: pd.Series = checked.groupby(identity, dropna=False).size()
    if (
        token_observations != final_observations
        or len(counts) != len(final_observations)
        or not counts.eq(len(canonical_pairs)).all()
    ):
        raise ValueError(f"Ordinary token observation grid disagrees in {path}")

    accepted_pairs: set[tuple[str, str]] = {
        (label, alias)
        for label, aliases in accepted_by_label.items()
        for alias in aliases
    }
    rejected_pairs: set[tuple[str, str]] = {
        (label, alias)
        for label, aliases in rejected_by_label.items()
        for alias in aliases
    }
    row_pairs: list[tuple[str, str]] = list(
        zip(checked["label"].astype(str), checked["token"].astype(str))
    )
    accepted_mask: np.ndarray = np.asarray(
        [pair in accepted_pairs for pair in row_pairs], dtype=bool
    )
    rejected_mask: np.ndarray = np.asarray(
        [pair in rejected_pairs for pair in row_pairs], dtype=bool
    )
    if not np.isfinite(
        checked.loc[accepted_mask, "logprob"].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"Accepted ordinary alias logprobs must be finite in {path}")
    if not checked.loc[rejected_mask, "logprob"].isna().all():
        raise ValueError(f"Rejected ordinary alias logprobs must be missing in {path}")
    return checked


def _validate_layer_run_sidecar(
    path: str,
    layer_path: str,
    manifest_path: str,
    frame: pd.DataFrame,
    benchmark: str,
    pregrouper: str,
    model: str,
) -> None:
    """Validate one canonical layer-run record and all bound identities."""
    with open(path, encoding="utf-8") as source:
        metadata: Any = json.load(source)
    if not isinstance(metadata, dict) or set(metadata) != LAYER_RUN_FIELDS:
        raise ValueError(f"Layer run metadata fields disagree in {path}")
    expected_identity: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "model": model,
        "segmentation_scope": "full_dialog_in_message_order",
        "source_hash_timing": "run_start",
        "model_artifact_hash_timing": "pre_model_load",
    }
    if any(metadata.get(field) != value for field, value in expected_identity.items()):
        raise ValueError(f"Layer run identity disagrees in {path}")

    artifact: Any = metadata.get("artifact")
    if not isinstance(artifact, dict) or artifact != {
        "filename": os.path.basename(layer_path),
        "rows": len(frame),
        "sha256": _sha256(layer_path),
    }:
        raise ValueError(f"Layer artifact identity disagrees in {path}")
    if metadata.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError(f"Layer manifest identity disagrees in {path}")

    parameters: Any = metadata.get("parameters")
    if (
        not isinstance(parameters, dict)
        or set(parameters) != LAYER_PARAMETER_FIELDS
        or parameters.get("rendered_chat_add_special_tokens") is not False
        or parameters.get("attention_implementation") != "sdpa"
        or parameters.get("canary") is not False
        or parameters.get("device_map") != "auto"
        or parameters.get("max_samples") is not None
        or parameters.get("seed") != 42
        or parameters.get("batch_size") != 32
        or parameters.get("torch_dtype") != "bfloat16"
    ):
        raise ValueError(f"Layer run parameters disagree in {path}")

    identity_payload: dict[str, Any] = dict(metadata)
    identity_payload["execution_model_source"] = metadata.get("model_source")
    _validate_open_model_identity(identity_payload, model, path)
    ordinary_path: str = os.path.join(os.path.dirname(path), f"{model}_run.json")
    if not os.path.isfile(ordinary_path):
        raise FileNotFoundError(
            f"Missing ordinary run identity for layer artifact: {ordinary_path}"
        )
    with open(ordinary_path, encoding="utf-8") as source:
        ordinary: Any = json.load(source)
    identity_fields: tuple[str, ...] = (
        "model_source",
        "model_revision",
        "model_artifact_manifest_sha256",
        "model_artifact_sha256",
        "model_identity_files_sha256",
    )
    if not isinstance(ordinary, dict) or any(
        ordinary.get(field) != metadata.get(field) for field in identity_fields
    ):
        raise ValueError(f"Layer and ordinary model identities disagree in {path}")

    spec = BENCHMARKS[benchmark]
    expected_prompts: int = int(
        frame.loc[frame["kind"] == "orig", "prompt_idx"].nunique()
    )
    dataset: Any = metadata.get("dataset")
    if (
        not isinstance(dataset, dict)
        or set(dataset) != LAYER_DATASET_FIELDS
        or dataset.get("hf_path") != spec.hf_dataset_path
        or dataset.get("hf_name") != spec.hf_dataset_name
        or dataset.get("hf_split") != spec.hf_split
        or dataset.get("snapshot_filename") != GOLD_DATASET_FILES[benchmark]
        or dataset.get("snapshot_sha256") != GOLD_DATASET_SHA256[benchmark]
        or dataset.get("prompts") != expected_prompts
        or LOWERCASE_SHA256_RE.fullmatch(
            str(dataset.get("normalized_frame_sha256", ""))
        )
        is None
    ):
        raise ValueError(f"Layer dataset identity disagrees in {path}")

    slots: Any = metadata.get("layer_slots")
    if not isinstance(slots, dict) or slots != {
        "count": int(frame["layer_slot"].nunique()),
        "convention": (
            "slot 0 is the embedding output; slot k+1 is decoder block k "
            "output; final norm is applied before all scores"
        ),
    }:
        raise ValueError(f"Layer-slot metadata disagrees in {path}")
    label_metadata: Any = metadata.get("labels")
    _validate_layer_alias_metadata(label_metadata, benchmark, path)
    if metadata.get("alignment") != {
        "direction": (
            "uniform sum of accepted label unembedding rows; each unordered "
            "contrast follows configured label order"
        ),
        "multi_alias_status": (
            "diagnostic approximation to grouped-logsumexp attribution"
        ),
    } or metadata.get("label_score_definition") != (
        "intermediate slots store logsumexp of accepted alias logits; the final "
        "slot stores logsumexp of the model's native-dtype full-head "
        "log-probabilities to match ordinary outputs; pairwise differences are "
        "grouped-label log-probability contrasts"
    ):
        raise ValueError(f"Layer scoring metadata disagrees in {path}")

    software: Any = metadata.get("software")
    if (
        not isinstance(software, dict)
        or set(software) != LAYER_SOFTWARE_FIELDS
        or any(
            value is not None
            and (
                not isinstance(value, str) or not value or "/" in value or "\\" in value
            )
            for value in software.values()
        )
    ):
        raise ValueError(f"Layer software metadata disagrees in {path}")
    source_hashes: Any = metadata.get("source_sha256")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    expected_source_hashes: dict[str, str] = {
        relative: _sha256(os.path.join(repository_root, relative))
        for relative in LAYER_EXECUTION_SOURCE_FILES
    }
    if source_hashes != expected_source_hashes:
        raise ValueError(f"Layer execution source hashes disagree in {path}")


def _validate_layer_artifact(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
) -> None:
    """Validate one layer artifact against its canonical segment grid."""
    config_dir: str = os.path.join(results_dir, benchmark, pregrouper)
    layer_path: str = os.path.join(config_dir, f"{model}_layers.tsv.gz")
    run_path: str = os.path.join(config_dir, f"{model}_layers_run.json")
    manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
    for required_path in (layer_path, run_path, manifest_path):
        if not os.path.isfile(required_path):
            raise FileNotFoundError(
                f"Missing canonical layer artifact: {required_path}"
            )
    frame: pd.DataFrame = pd.read_csv(layer_path, sep="\t")
    expected_columns: set[str] = _layer_artifact_columns(benchmark)
    if set(frame.columns) != expected_columns:
        raise ValueError(
            f"Layer artifact columns disagree in {layer_path}: "
            f"missing={sorted(expected_columns - set(frame.columns))}, "
            f"extra={sorted(set(frame.columns) - expected_columns)}"
        )
    eval_config = BENCHMARKS[benchmark].eval_config
    if eval_config is None:
        raise ValueError(f"Layerwise benchmark {benchmark!r} has no label config")
    labels: tuple[str, ...] = tuple(eval_config.label_tokens)
    positive, negative = _layer_contrast_labels(
        benchmark,
        "entailment_contradiction" if benchmark.startswith("anli_") else "canonical",
    )
    checked: pd.DataFrame = _validate_layer_frame(frame, layer_path, positive, negative)
    manifest, manifest_keys = _validate_manifest(manifest_path)
    _validate_gold_manifest(benchmark, pregrouper, manifest_path, manifest)
    _validate_layer_manifest_coverage(
        checked,
        pd.MultiIndex.from_frame(manifest[["prompt_idx", "seg_idx"]]),
        layer_path,
    )
    slot_count: int = int(checked["layer_slot"].nunique())
    expected_rows: int = slot_count * (
        int(manifest["prompt_idx"].nunique()) + len(manifest_keys)
    )
    if len(checked) != expected_rows:
        raise ValueError(f"Layer artifact row matrix is incomplete in {layer_path}")

    scalar_columns: list[str] = [
        f"label_score_{label_column_alias(label)}" for label in labels
    ]
    for first, second in itertools.combinations(labels, 2):
        suffix: str = f"{label_column_alias(first)}_vs_{label_column_alias(second)}"
        scalar_columns.extend(
            [
                f"w_dot_delta_z_postnorm_{suffix}",
                f"w_norm_{suffix}",
            ]
        )
    ablated: pd.Series = checked["kind"].eq("ablated")
    originals: pd.Series = ~ablated
    label_columns: list[str] = [
        column for column in scalar_columns if column.startswith("label_score_")
    ]
    delta_columns: list[str] = [
        "delta_norm_postnorm",
        *(column for column in scalar_columns if column.startswith("w_dot_delta_")),
    ]
    norm_columns: list[str] = [
        column for column in scalar_columns if column.startswith("w_norm_")
    ]
    if (
        not np.isfinite(checked[label_columns].to_numpy(dtype=float)).all()
        or not np.isfinite(
            checked.loc[ablated, delta_columns].to_numpy(dtype=float)
        ).all()
        or checked.loc[originals, delta_columns].notna().any().any()
        or not np.isfinite(checked[norm_columns].to_numpy(dtype=float)).all()
        or (checked.loc[ablated, "delta_norm_postnorm"] < 0).any()
        or (checked[norm_columns] <= 0).any().any()
        or any(checked[column].nunique(dropna=False) != 1 for column in norm_columns)
    ):
        raise ValueError(
            f"Layer scalar values are incomplete or invalid in {layer_path}"
        )

    expected_answers: pd.Series = checked["prompt_idx"].map(
        manifest.groupby("prompt_idx")["answer"].first()
    )
    if not checked["answer"].eq(expected_answers).all():
        raise ValueError(f"Layer answers disagree with the manifest in {layer_path}")
    _validate_layer_run_sidecar(
        run_path,
        layer_path,
        manifest_path,
        checked,
        benchmark,
        pregrouper,
        model,
    )


def _finite_logsumexp(values: pd.Series) -> float:
    """Compute a stable log-sum-exp and reject unavailable open-model cells."""
    array: np.ndarray = values.to_numpy(dtype=float)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("final-layer comparison requires finite token logprobs")
    maximum: float = float(array.max())
    return maximum + float(np.log(np.exp(array - maximum).sum()))


def _validate_layer_final_readout(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
) -> dict[str, float]:
    """Require every final-block label contrast to match ordinary scoring.

    The two paths use the same rendered inputs, ordering, model,
    full-vocabulary output head, and native-dtype log-softmax. A small tolerance
    covers independent BF16 batches, grouped reduction, and TSV round-tripping.
    """
    directory: str = os.path.join(results_dir, benchmark, pregrouper)
    layer_path: str = os.path.join(directory, f"{model}_layers.tsv.gz")
    run_path: str = os.path.join(directory, f"{model}_layers_run.json")
    token_path: str = os.path.join(directory, f"{model}_tokens.tsv.gz")
    for path in (layer_path, run_path, token_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    layer: pd.DataFrame = pd.read_csv(layer_path, sep="\t")
    tokens: pd.DataFrame = pd.read_csv(
        token_path,
        sep="\t",
        dtype={"kind": str, "label": str, "token": str},
    )
    with open(run_path, encoding="utf-8") as source:
        metadata: Any = json.load(source)
    label_metadata: Any = metadata.get("labels") if isinstance(metadata, dict) else None
    aliases_by_label, rejected_by_label = _validate_layer_alias_metadata(
        label_metadata, benchmark, run_path
    )

    final_slot: int = int(pd.to_numeric(layer["layer_slot"], errors="raise").max())
    final: pd.DataFrame = layer[layer["layer_slot"] == final_slot].copy()
    final["seg_key"] = final["seg_idx"].fillna(-1).astype(int)
    required_token_columns: set[str] = {
        "prompt_idx",
        "seg_idx",
        "kind",
        "label",
        "token",
        "logprob",
    }
    if not required_token_columns.issubset(tokens.columns):
        raise ValueError(f"Ordinary token columns are incomplete in {token_path}")
    tokens = _validate_ordinary_layer_token_grid(
        tokens,
        final,
        benchmark,
        aliases_by_label,
        rejected_by_label,
        token_path,
    )

    selected_parts: list[pd.DataFrame] = []
    for label, aliases in aliases_by_label.items():
        selected_parts.append(
            tokens[(tokens["label"] == label) & tokens["token"].isin(aliases)]
        )
    selected: pd.DataFrame = pd.concat(selected_parts, ignore_index=True)
    selected["logprob"] = pd.to_numeric(selected["logprob"], errors="raise")
    selected["seg_key"] = selected["seg_idx"].fillna(-1).astype(int)
    grouped: pd.DataFrame = (
        selected.groupby(
            ["prompt_idx", "seg_key", "kind", "label"],
            sort=False,
        )["logprob"]
        .apply(_finite_logsumexp)
        .unstack("label")
        .reset_index()
    )
    errors: dict[str, float] = {}
    identity: list[str] = ["prompt_idx", "seg_key", "kind"]
    for positive, negative in itertools.combinations(aliases_by_label, 2):
        if positive not in grouped or negative not in grouped:
            raise ValueError(
                f"Ordinary labels for {positive}/{negative} are incomplete in {token_path}"
            )
        positive_column: str = f"label_score_{label_column_alias(positive)}"
        negative_column: str = f"label_score_{label_column_alias(negative)}"
        if positive_column not in final or negative_column not in final:
            raise ValueError(f"Final layer contrast is missing in {layer_path}")
        layer_signal: pd.DataFrame = final[
            [*identity, positive_column, negative_column]
        ].copy()
        layer_signal["layer_contrast"] = (
            layer_signal[positive_column] - layer_signal[negative_column]
        )
        reference: pd.DataFrame = grouped[[*identity, positive, negative]].copy()
        reference["ordinary_contrast"] = reference[positive] - reference[negative]
        compared: pd.DataFrame = layer_signal.merge(
            reference[[*identity, "ordinary_contrast"]],
            on=identity,
            how="inner",
            validate="one_to_one",
        )
        if len(compared) != len(final):
            raise ValueError(
                f"Final-layer/ordinary coverage mismatch in {layer_path}: "
                f"{len(compared)} != {len(final)}"
            )
        difference: np.ndarray = np.abs(
            compared["layer_contrast"].to_numpy(dtype=float)
            - compared["ordinary_contrast"].to_numpy(dtype=float)
        )
        maximum: float = float(difference.max())
        name: str = f"{positive}_minus_{negative}"
        errors[name] = maximum
        if not np.isfinite(maximum) or maximum > MAX_LAYER_FINAL_CONTRAST_ABS_ERROR:
            raise ValueError(
                f"Final-layer contrast {name} disagrees with ordinary scoring in "
                f"{layer_path}: max_abs_error={maximum:.9g}, "
                f"limit={MAX_LAYER_FINAL_CONTRAST_ABS_ERROR:.9g}"
            )
    return errors


def _validate_layer_artifacts(results_dir: str) -> None:
    """Require the complete canonical configuration-by-open-model matrix."""
    if not set(LAYERWISE_CONFIGS).issubset(set(DEFAULT_BENCHMARK_CONFIGS)):
        raise ValueError("Layerwise configurations are outside the canonical grid")
    for benchmark, pregrouper in LAYERWISE_CONFIGS:
        for model in OPEN_MODELS:
            _validate_layer_artifact(results_dir, benchmark, pregrouper, model)
            _validate_layer_final_readout(results_dir, benchmark, pregrouper, model)


def _allowed_release_files(cohort: str) -> set[str]:
    """Return the complete portable result-artifact path allowlist."""
    if cohort not in {"open", "paper"}:
        raise ValueError(f"Unsupported release cohort: {cohort}")
    allowed: set[str] = {
        "README.md",
        "artifact_manifest.json",
        "coverage.tsv",
    }
    if cohort == "paper":
        allowed.update(
            {
                HOSTED_AUDIT_RECEIPT_NAME,
                HOSTED_COMPLETION_AUDIT_RECEIPT_NAME,
            }
        )
        release_models: tuple[str, ...] = tuple(dict.fromkeys(OPEN_MODELS + API_MODELS))
        derived_tables: tuple[str, ...] = RELEASE_PAPER_DERIVED_TABLES
    else:
        release_models = OPEN_MODELS
        derived_tables = RELEASE_OPEN_DERIVED_TABLES
    for benchmark, pregrouper in ARTIFACT_CONFIGS:
        prefix: str = f"{benchmark}/{pregrouper}"
        allowed.add(f"{prefix}/segments.tsv.gz")
        for model in release_models:
            allowed.add(f"{prefix}/{model}_run.json")
            allowed.add(f"{prefix}/{model}_segment.tsv.gz")
            if benchmark != "lambada":
                allowed.add(f"{prefix}/{model}_tokens.tsv.gz")
        if cohort == "paper" and benchmark == "lambada":
            for model in RELEASE_LAMBADA_CANARY_MODELS:
                allowed.add(f"{prefix}/{model}_canary.json")
    for benchmark, pregrouper in LAYERWISE_CONFIGS:
        prefix = f"{benchmark}/{pregrouper}"
        for model in OPEN_MODELS:
            allowed.add(f"{prefix}/{model}_layers.tsv.gz")
            allowed.add(f"{prefix}/{model}_layers_run.json")
    for table in (*derived_tables, *RELEASE_LAYERWISE_DERIVED_TABLES):
        allowed.add(table)
        allowed.add(f"{table}.provenance.json")
    return allowed


def _root_scratch_files() -> set[str]:
    """Return non-release consolidated tables created during finalization."""
    scratch: set[str] = set()
    for benchmark, pregrouper in ARTIFACT_CONFIGS:
        stem: str = f"{benchmark}_{pregrouper}"
        scratch.add(f"{stem}_segments.tsv")
        scratch.add(f"{stem}_tokens.tsv")
        if benchmark != "lambada":
            scratch.add(f"{stem}_logodds.tsv")
    return scratch


def _validate_release_inventory(
    results_dir: str,
    cohort: str,
    *,
    require_complete: bool = True,
    require_manifest: bool = False,
) -> None:
    """Require an exact, link-free public release inventory."""
    if os.path.islink(results_dir):
        raise ValueError(f"Release results root must not be a symlink: {results_dir}")
    allowed: set[str] = _allowed_release_files(cohort)
    scratch: set[str] = _root_scratch_files()
    unexpected: list[str] = []
    actual_release: set[str] = set()
    allowed_directories: set[str] = {"."}
    for relative in allowed | scratch:
        parent: str = os.path.dirname(relative)
        while parent:
            allowed_directories.add(parent)
            parent = os.path.dirname(parent)
    for root, directories, names in os.walk(results_dir):
        for directory in directories:
            directory_path: str = os.path.join(root, directory)
            relative_directory: str = os.path.relpath(directory_path, results_dir)
            if (
                os.path.islink(directory_path)
                or relative_directory not in allowed_directories
            ):
                unexpected.append(relative_directory)
        for name in names:
            path: str = os.path.join(root, name)
            relative: str = os.path.relpath(path, results_dir)
            if os.path.islink(path) or not os.path.isfile(path):
                unexpected.append(relative)
            elif relative not in allowed and relative not in scratch:
                unexpected.append(relative)
            elif relative in allowed:
                actual_release.add(relative)
    if unexpected:
        raise ValueError(
            "Unexpected files or links in release results: "
            f"{sorted(set(unexpected))}"
        )
    required: set[str] = set(allowed)
    if not require_manifest:
        required.remove("artifact_manifest.json")
    if cohort == "open" or not require_manifest:
        # Standalone rerun directories do not need a copy of the repository's
        # release documentation.
        required.remove("README.md")
    if cohort == "paper" and not require_manifest:
        required.difference_update(
            f"lambada/word/{model}_canary.json"
            for model in RELEASE_LAMBADA_CANARY_MODELS
        )
    if require_complete:
        missing: set[str] = required - actual_release
        if missing:
            raise ValueError(f"Missing files from release results: {sorted(missing)}")


def _write_artifact_manifest(results_dir: str) -> None:
    files: list[dict[str, Any]] = []
    for root, _, names in os.walk(results_dir):
        for name in sorted(names):
            path: str = os.path.join(root, name)
            relative: str = os.path.relpath(path, results_dir)
            if relative == "artifact_manifest.json":
                continue
            if os.path.dirname(relative) == "" and relative.endswith(
                ("_segments.tsv", "_tokens.tsv", "_logodds.tsv")
            ):
                continue
            files.append(
                {
                    "path": relative,
                    "bytes": os.path.getsize(path),
                    "sha256": _sha256(path),
                }
            )
    output_path: str = os.path.join(results_dir, "artifact_manifest.json")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    source_hashes: dict[str, str] = {
        path: _sha256(os.path.join(repository_root, path))
        for path in ANALYSIS_SOURCE_FILES
    }
    with open(output_path, "w", encoding="utf-8") as output:
        json.dump(
            {
                "schema_version": 2,
                "analysis_source_sha256": source_hashes,
                "files": sorted(files, key=lambda x: x["path"]),
            },
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")


def _validate_artifact_manifest(results_dir: str) -> None:
    """Verify an existing release manifest without rewriting trusted hashes."""
    manifest_path: str = os.path.join(results_dir, "artifact_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as source:
        recorded: Any = json.load(source)
    if (
        not isinstance(recorded, dict)
        or set(recorded) != {"schema_version", "analysis_source_sha256", "files"}
        or recorded.get("schema_version") != 2
    ):
        raise ValueError(f"Invalid artifact manifest schema in {manifest_path}")

    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    expected_source_hashes: dict[str, str] = {
        path: _sha256(os.path.join(repository_root, path))
        for path in ANALYSIS_SOURCE_FILES
    }
    if recorded.get("analysis_source_sha256") != expected_source_hashes:
        raise ValueError(f"Analysis source hashes disagree in {manifest_path}")

    expected_files: list[dict[str, Any]] = []
    for root, _, names in os.walk(results_dir):
        for name in sorted(names):
            path: str = os.path.join(root, name)
            relative: str = os.path.relpath(path, results_dir)
            if relative == "artifact_manifest.json":
                continue
            if os.path.dirname(relative) == "" and relative.endswith(
                ("_segments.tsv", "_tokens.tsv", "_logodds.tsv")
            ):
                continue
            expected_files.append(
                {
                    "path": relative,
                    "bytes": os.path.getsize(path),
                    "sha256": _sha256(path),
                }
            )
    expected_files.sort(key=lambda item: str(item["path"]))
    if recorded.get("files") != expected_files:
        raise ValueError(
            f"Artifact file inventory or hashes disagree in {manifest_path}"
        )


def _read_derived(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing derived result table: {path}")
    frame: pd.DataFrame = pd.read_csv(path, sep="\t")
    required: set[str] = {
        "benchmark",
        "pregrouper",
        "scope",
        "model_s",
        "model_t",
        "metric",
        "statistic",
        "aggregation",
        "n_observations",
        "f_point",
        "f_lo",
        "f_hi",
    }
    missing: set[str] = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing derived columns {sorted(missing)}")
    key_columns: list[str] = [
        column
        for column in (
            "benchmark",
            "pregrouper",
            "scope",
            "contrast",
            "representation",
            "aggregation",
            "model_s",
            "model_t",
            "metric",
            "statistic",
        )
        if column in frame.columns
    ]
    if frame.duplicated(key_columns).any():
        raise ValueError(f"{path} contains duplicate derived result keys")
    return frame


def _validate_derived_sidecar(
    results_dir: str,
    output_path: str,
    generator_module: str,
    benchmark_configs: list[tuple[str, str]],
    models: tuple[str, ...],
    *,
    expected_input_paths: dict[str, str] | None = None,
    supporting_source_files: tuple[str, ...] = DERIVED_SUPPORTING_SOURCE_FILES,
) -> dict[str, Any]:
    """Verify that a derived table is sealed to all shipped raw inputs."""
    sidecar_path: str = f"{output_path}.provenance.json"
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(f"Missing derived provenance: {sidecar_path}")
    with open(sidecar_path, encoding="utf-8") as source:
        payload: Any = json.load(source)
    if (
        not isinstance(payload, dict)
        or set(payload) != DERIVED_PROVENANCE_FIELDS
        or payload.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or payload.get("artifact_type") != "derived_table"
    ):
        raise ValueError(f"Invalid derived provenance schema in {sidecar_path}")

    output: Any = payload.get("output")
    expected_output_path: str = os.path.relpath(output_path, results_dir).replace(
        os.sep, "/"
    )
    if not isinstance(output, dict) or output != {
        "path": expected_output_path,
        "sha256": _sha256(output_path),
        "size_bytes": os.path.getsize(output_path),
    }:
        raise ValueError(f"Derived output seal disagrees in {sidecar_path}")

    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    generator_relative_path: str = generator_module.replace(".", "/") + ".py"
    generator: Any = payload.get("generator")
    provenance_writer_path: str = os.path.join(
        repository_root, "benchmark_scripts/derived_provenance.py"
    )
    if not isinstance(generator, dict) or generator != {
        "module": generator_module,
        "source_sha256": _sha256(
            os.path.join(repository_root, generator_relative_path)
        ),
        "provenance_writer_sha256": _sha256(provenance_writer_path),
    }:
        raise ValueError(f"Derived generator seal disagrees in {sidecar_path}")

    supporting: Any = payload.get("supporting_sources")
    expected_supporting: dict[str, dict[str, str | int]] = {
        relative_path: {
            "sha256": _sha256(os.path.join(repository_root, relative_path)),
            "size_bytes": os.path.getsize(os.path.join(repository_root, relative_path)),
        }
        for relative_path in supporting_source_files
    }
    if supporting != expected_supporting:
        raise ValueError(f"Derived supporting-source seal disagrees in {sidecar_path}")

    if expected_input_paths is None:
        expected_input_paths = {}
        for benchmark, pregrouper in benchmark_configs:
            expected_input_paths.update(
                collect_result_inputs(results_dir, benchmark, pregrouper, {}, models)
            )
    inputs: Any = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_input_paths):
        raise ValueError(f"Derived input set disagrees in {sidecar_path}")
    for identifier, input_path in expected_input_paths.items():
        relative_path: str = os.path.relpath(input_path, results_dir).replace(
            os.sep, "/"
        )
        expected_input: dict[str, str | int] = {
            "path": relative_path,
            "sha256": _sha256(input_path),
            "size_bytes": os.path.getsize(input_path),
        }
        if inputs[identifier] != expected_input:
            raise ValueError(
                f"Derived input seal disagrees for {identifier} in {sidecar_path}"
            )
    software: Any = payload.get("software")
    if (
        not isinstance(software, dict)
        or set(software) != DERIVED_SOFTWARE_FIELDS
        or any(
            not isinstance(value, str) or not value or "/" in value or "\\" in value
            for value in software.values()
        )
    ):
        raise ValueError(f"Invalid derived software metadata in {sidecar_path}")
    parameters: Any = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Missing derived parameters in {sidecar_path}")
    return parameters


def _validate_f_table_parameters(
    parameters: dict[str, Any],
    expected_configs: set[tuple[str, str, str, str]],
    models: tuple[str, ...],
    api_infinity_policy: str,
    transfer_aggregation: str,
    scalar_only: bool,
    path: str,
    anli_contrast: str | None = None,
) -> None:
    """Require the exact public F-table analysis parameters."""
    expected_benchmarks: set[tuple[str, str]] = {
        (benchmark, pregrouper) for benchmark, pregrouper, _, _ in expected_configs
    }
    recorded_benchmarks: set[tuple[str, str]] = {
        (str(item["benchmark"]), str(item["pregrouper"]))
        for item in parameters.get("benchmark_configs", [])
    }
    resolved_jobs_value: Any = parameters.get("resolved_jobs")
    resolved_jobs: list[tuple[str, str, str, str]] = (
        [
            (
                str(item["benchmark"]),
                str(item["pregrouper"]),
                str(item["scope"]),
                str(item["contrast"]),
            )
            for item in resolved_jobs_value
        ]
        if isinstance(resolved_jobs_value, list)
        and all(
            isinstance(item, dict)
            and set(item) == {"benchmark", "pregrouper", "scope", "contrast"}
            for item in resolved_jobs_value
        )
        else []
    )
    expected_scopes: set[str] = {scope for _, _, scope, _ in expected_configs}
    expected_contrasts: set[str] = (
        {"canonical"}
        if anli_contrast is not None
        else {contrast for _, _, _, contrast in expected_configs}
    )
    expected_metrics: list[str] | None = ["F_attr", "F_pred"] if scalar_only else None
    expected_values: dict[str, Any] = {
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
        "seed": 42,
        "bootstrap_rng": "sha256_cell_key_v1",
        "cohort": "open" if models == OPEN_MODELS else "paper",
        "requested_models": list(models),
        "output_models": sorted(models),
        "api_infinity_policy": api_infinity_policy,
        "transfer_aggregation": transfer_aggregation,
        "missingness_policy": (
            "pair_specific_complete_case"
            if api_infinity_policy == "pairwise_complete"
            else "model_specific_finite_extreme_replacement"
        ),
        "metrics": expected_metrics,
        "anli_contrast": anli_contrast,
    }
    expected_parameter_fields: set[str] = {
        "benchmark_configs",
        "contrasts",
        "resolved_jobs",
        "scopes",
        *expected_values,
    }
    if (
        set(parameters) != expected_parameter_fields
        or recorded_benchmarks != expected_benchmarks
        or len(resolved_jobs) != len(expected_configs)
        or set(resolved_jobs) != expected_configs
        or set(parameters.get("scopes", [])) != expected_scopes
        or set(parameters.get("contrasts", [])) != expected_contrasts
        or any(parameters.get(key) != value for key, value in expected_values.items())
    ):
        raise ValueError(f"Incorrect F-table derivation parameters in {path}")


def _require_pair_grid(
    frame: pd.DataFrame,
    expected_configs: set[tuple[str, str, str, str]],
    models: tuple[str, ...],
    path: str,
    api_infinity_policy: str,
    transfer_aggregation: str,
    unsupported_prediction_configs: set[tuple[str, str, str]],
    unsupported_attribution_configs: set[tuple[str, str, str]],
    scalar_only: bool = False,
) -> None:
    resolved_columns: set[str] = {
        "cohort",
        "pair_population",
        "availability_status",
        "unavailable_reason",
        "requested_scope",
        "resolved_scope",
        "requested_contrast",
        "resolved_source_contrast",
        "resolved_target_contrast",
        "readout_contrast",
    }
    missing_resolved_columns: set[str] = resolved_columns - set(frame.columns)
    if missing_resolved_columns:
        raise ValueError(
            f"{path} lacks resolved estimand metadata "
            f"{sorted(missing_resolved_columns)}"
        )
    expected_cohort: str = "open" if models == OPEN_MODELS else "paper"
    if set(frame["cohort"].astype(str)) != {expected_cohort}:
        raise ValueError(f"{path} has incorrect cohort metadata")
    for row in frame.itertuples(index=False):
        requested_scope: str = str(row.scope)
        requested_contrast: str = str(row.contrast)
        metric: str = str(row.metric)
        source_contrast, target_contrast, readout_contrast = _contrast_metadata(
            str(row.benchmark), requested_contrast, metric
        )
        if (
            str(row.requested_scope) != requested_scope
            or str(row.resolved_scope) != _resolved_scope(metric, requested_scope)
            or str(row.requested_contrast) != requested_contrast
            or str(row.resolved_source_contrast) != source_contrast
            or str(row.resolved_target_contrast) != target_contrast
            or str(row.readout_contrast) != readout_contrast
            or str(row.pair_population)
            != (
                "open_open"
                if str(row.model_s) in OPEN_MODELS and str(row.model_t) in OPEN_MODELS
                else (
                    "open_hosted"
                    if str(row.model_s) in OPEN_MODELS
                    or str(row.model_t) in OPEN_MODELS
                    else "hosted_hosted"
                )
            )
        ):
            raise ValueError(f"{path} has incorrect resolved estimand metadata")
    required_policy: set[str] = set(frame["api_infinity_policy"].astype(str))
    if required_policy != {api_infinity_policy}:
        raise ValueError(
            f"{path} has API infinity policies {sorted(required_policy)}; "
            f"expected only {api_infinity_policy!r}"
        )
    transfer_rows: pd.Series = frame["metric"].astype(str).str.endswith("_to_attr")
    expected_aggregations: pd.Series = pd.Series(
        np.where(transfer_rows, transfer_aggregation, "row_pooled"),
        index=frame.index,
    )
    if not frame["aggregation"].astype(str).eq(expected_aggregations).all():
        raise ValueError(f"{path} has incorrect aggregation metadata")
    symmetric_pairs: set[tuple[str, str]] = set(itertools.combinations(models, 2))
    open_models: tuple[str, ...] = tuple(
        model for model in OPEN_MODELS if model in models
    )
    open_pairs: set[tuple[str, str]] = set(itertools.combinations(open_models, 2))
    transfer_pairs: set[tuple[str, str]] = {
        (source, target)
        for source in open_models
        for target in models
        if source != target
    }
    expected_keys: set[tuple[str, str, str, str, str, str, str, str]] = set()
    for benchmark, pregrouper, scope, contrast in expected_configs:
        symmetric_metrics: tuple[str, ...] = ("F_pred", "F_attr")
        if not scalar_only:
            symmetric_metrics += (
                "F_attn_rollout",
                "F_attn_mean",
                "F_attn_max",
                "F_mag",
            )
            symmetric_metrics += ("F_align",)
        for statistic in ("spearman", "pearson_r", "pearson_r2"):
            for metric in symmetric_metrics:
                pairs: set[tuple[str, str]] = (
                    symmetric_pairs if metric in {"F_pred", "F_attr"} else open_pairs
                )
                expected_keys.update(
                    (
                        benchmark,
                        pregrouper,
                        scope,
                        contrast,
                        model_s,
                        model_t,
                        metric,
                        statistic,
                    )
                    for model_s, model_t in pairs
                )
            if not scalar_only:
                transfer_metrics: tuple[str, ...] = (
                    "F_mag_to_attr",
                    "F_attn_rollout_to_attr",
                    "F_attn_mean_to_attr",
                    "F_attn_max_to_attr",
                )
                transfer_metrics += ("F_align_to_attr",)
                expected_keys.update(
                    (
                        benchmark,
                        pregrouper,
                        scope,
                        contrast,
                        model_s,
                        model_t,
                        metric,
                        statistic,
                    )
                    for metric in transfer_metrics
                    for model_s, model_t in transfer_pairs
                )
    key_columns: list[str] = [
        "benchmark",
        "pregrouper",
        "scope",
        "contrast",
        "model_s",
        "model_t",
        "metric",
        "statistic",
    ]
    actual_keys: set[tuple[str, str, str, str, str, str, str, str]] = set(
        frame[key_columns].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise ValueError(
            f"{path} has an incorrect semantic result grid: "
            f"{len(expected_keys - actual_keys)} missing and "
            f"{len(actual_keys - expected_keys)} extra keys"
        )
    coverage_columns: set[str] = {
        "expected_observations",
        "observation_coverage",
        "expected_prompts",
        "prompt_coverage",
    }
    if not coverage_columns.issubset(frame.columns):
        raise ValueError(f"{path} omits pair-specific coverage metadata")
    observations: pd.Series = pd.to_numeric(frame["n_observations"], errors="coerce")
    expected_observations: pd.Series = pd.to_numeric(
        frame["expected_observations"], errors="coerce"
    )
    observation_coverage: pd.Series = pd.to_numeric(
        frame["observation_coverage"], errors="coerce"
    )
    prompts: pd.Series = pd.to_numeric(frame["n_prompts"], errors="coerce")
    expected_prompts: pd.Series = pd.to_numeric(
        frame["expected_prompts"], errors="coerce"
    )
    prompt_coverage: pd.Series = pd.to_numeric(
        frame["prompt_coverage"], errors="coerce"
    )
    if (
        observations.isna().any()
        or expected_observations.isna().any()
        or prompts.isna().any()
        or expected_prompts.isna().any()
    ):
        raise ValueError(f"{path} has invalid result counts")

    def _unsupported_row(row: pd.Series) -> bool:
        config_s: tuple[str, str, str] = (
            str(row["benchmark"]),
            str(row["pregrouper"]),
            str(row["model_s"]),
        )
        config_t: tuple[str, str, str] = (
            str(row["benchmark"]),
            str(row["pregrouper"]),
            str(row["model_t"]),
        )
        metric: str = str(row["metric"])
        if metric == "F_pred":
            return (
                config_s in unsupported_prediction_configs
                or config_t in unsupported_prediction_configs
            )
        if metric == "F_attr":
            return (
                config_s in unsupported_attribution_configs
                or config_t in unsupported_attribution_configs
            )
        if metric.endswith("_to_attr"):
            return config_t in unsupported_attribution_configs
        return False

    unsupported_rows: pd.Series = frame.apply(_unsupported_row, axis=1)
    points: pd.Series = pd.to_numeric(frame["f_point"], errors="coerce")
    lows: pd.Series = pd.to_numeric(frame["f_lo"], errors="coerce")
    highs: pd.Series = pd.to_numeric(frame["f_hi"], errors="coerce")
    unavailable_rows: pd.Series = unsupported_rows
    if not (
        points[unavailable_rows].isna().all()
        and lows[unavailable_rows].isna().all()
        and highs[unavailable_rows].isna().all()
        and observations[unavailable_rows].eq(0).all()
        and prompts[unavailable_rows].eq(0).all()
        and frame.loc[unavailable_rows, "availability_status"]
        .astype(str)
        .eq("unavailable")
        .all()
    ):
        raise ValueError(f"{path} reports estimates for unsupported model outputs")
    estimable: pd.Series = ~unavailable_rows
    if not (
        observation_coverage[estimable].between(0.0, 1.0).all()
        and prompt_coverage[estimable].between(0.0, 1.0).all()
        and np.allclose(
            observation_coverage[estimable],
            observations[estimable] / expected_observations[estimable],
        )
        and np.allclose(
            prompt_coverage[estimable],
            prompts[estimable] / expected_prompts[estimable],
        )
    ):
        raise ValueError(f"{path} has inconsistent pair-specific coverage")
    if not (
        np.isfinite(points[estimable]).all()
        and np.isfinite(lows[estimable]).all()
        and np.isfinite(highs[estimable]).all()
        and frame.loc[estimable, "availability_status"]
        .astype(str)
        .eq("available")
        .all()
        and frame.loc[estimable, "unavailable_reason"].fillna("").eq("").all()
    ):
        raise ValueError(f"{path} has non-finite estimates with sufficient data")

    if len(expected_configs) > 1 and {config[2] for config in expected_configs} == {
        "all",
        "system",
        "user",
    }:
        prediction_rows: pd.DataFrame = frame[frame["metric"] == "F_pred"].copy()
        comparison_columns: list[str] = [
            "benchmark",
            "pregrouper",
            "contrast",
            "model_s",
            "model_t",
            "statistic",
        ]
        for _, group in prediction_rows.groupby(comparison_columns, dropna=False):
            if len(group) != 3:
                raise ValueError(f"{path} has an incomplete F_pred scope group")
            values: list[tuple[float, float, float, int, int]] = [
                (
                    float(row.f_point),
                    float(row.f_lo),
                    float(row.f_hi),
                    int(row.n_observations),
                    int(row.n_prompts),
                )
                for row in group.itertuples()
            ]
            first: tuple[float, float, float, int, int] = values[0]
            if any(
                not all(
                    (np.isnan(a) and np.isnan(b)) or a == b
                    for a, b in zip(first, value)
                )
                for value in values[1:]
            ):
                raise ValueError(f"{path} changes prompt-level F_pred by scope")


def _layerwise_derived_inputs(results_dir: str) -> dict[str, str]:
    """Return the exact raw inputs sealed by the canonical layer analysis."""
    inputs: dict[str, str] = {}
    for benchmark, pregrouper in LAYERWISE_CONFIGS:
        prefix: str = f"{benchmark}/{pregrouper}"
        config_dir: str = os.path.join(results_dir, benchmark, pregrouper)
        inputs[f"{prefix}/segments"] = os.path.join(config_dir, "segments.tsv.gz")
        for model in OPEN_MODELS:
            inputs[f"{prefix}/{model}/layers"] = os.path.join(
                config_dir, f"{model}_layers.tsv.gz"
            )
            inputs[f"{prefix}/{model}/layers_run"] = os.path.join(
                config_dir, f"{model}_layers_run.json"
            )
    return inputs


def _validate_layerwise_endpoint_consistency(
    layerwise: pd.DataFrame,
    ordinary: pd.DataFrame,
    layerwise_path: str,
    ordinary_path: str,
) -> None:
    """Require final-depth layer estimates to reproduce the ordinary F-table."""
    identity: list[str] = [
        "benchmark",
        "pregrouper",
        "scope",
        "contrast",
        "model_s",
        "model_t",
        "metric",
        "statistic",
    ]
    endpoint: pd.DataFrame = layerwise[
        pd.to_numeric(layerwise["relative_depth"], errors="coerce").eq(1.0)
    ].copy()
    layer_configs: set[tuple[str, str]] = set(LAYERWISE_CONFIGS)
    ordinary_configs: pd.Series = pd.Series(
        list(zip(ordinary["benchmark"], ordinary["pregrouper"])),
        index=ordinary.index,
    ).isin(layer_configs)
    ordinary_open: pd.DataFrame = ordinary[
        ordinary_configs
        & ordinary["scope"].eq("user")
        & ordinary["metric"].isin(("F_pred", "F_attr"))
        & ordinary["model_s"].isin(OPEN_MODELS)
        & ordinary["model_t"].isin(OPEN_MODELS)
    ].copy()
    if endpoint.empty or ordinary_open.empty:
        raise ValueError(
            f"Layerwise/F-table endpoint comparison is empty for "
            f"{layerwise_path} and {ordinary_path}"
        )
    if endpoint.duplicated(identity).any() or ordinary_open.duplicated(identity).any():
        raise ValueError(
            f"Layerwise/F-table endpoint comparison has duplicate keys in "
            f"{layerwise_path} or {ordinary_path}"
        )
    endpoint_keys: set[tuple[str, ...]] = set(
        endpoint[identity].astype(str).itertuples(index=False, name=None)
    )
    ordinary_keys: set[tuple[str, ...]] = set(
        ordinary_open[identity].astype(str).itertuples(index=False, name=None)
    )
    if endpoint_keys != ordinary_keys:
        raise ValueError(
            f"Layerwise/F-table endpoint keys disagree between {layerwise_path} "
            f"and {ordinary_path}"
        )

    comparison_columns: tuple[str, ...] = (
        "f_point",
        "n_observations",
        "expected_observations",
        "observation_coverage",
        "n_prompts",
        "expected_prompts",
        "prompt_coverage",
    )
    compared: pd.DataFrame = endpoint[[*identity, *comparison_columns]].merge(
        ordinary_open[[*identity, *comparison_columns]],
        on=identity,
        how="inner",
        suffixes=("_layerwise", "_ordinary"),
        validate="one_to_one",
    )
    count_columns: tuple[str, ...] = (
        "n_observations",
        "expected_observations",
        "n_prompts",
        "expected_prompts",
    )
    for column in count_columns:
        layer_values: pd.Series = pd.to_numeric(
            compared[f"{column}_layerwise"], errors="coerce"
        )
        ordinary_values: pd.Series = pd.to_numeric(
            compared[f"{column}_ordinary"], errors="coerce"
        )
        if layer_values.isna().any() or not layer_values.eq(ordinary_values).all():
            raise ValueError(
                f"Layerwise endpoint {column} disagrees with {ordinary_path}"
            )
    for column in ("observation_coverage", "prompt_coverage"):
        layer_values = pd.to_numeric(compared[f"{column}_layerwise"], errors="coerce")
        ordinary_values = pd.to_numeric(compared[f"{column}_ordinary"], errors="coerce")
        if not np.array_equal(
            layer_values.to_numpy(dtype=float), ordinary_values.to_numpy(dtype=float)
        ):
            raise ValueError(
                f"Layerwise endpoint {column} disagrees with {ordinary_path}"
            )
    layer_points: np.ndarray = pd.to_numeric(
        compared["f_point_layerwise"], errors="coerce"
    ).to_numpy(dtype=float)
    ordinary_points: np.ndarray = pd.to_numeric(
        compared["f_point_ordinary"], errors="coerce"
    ).to_numpy(dtype=float)
    absolute_errors: np.ndarray = np.abs(layer_points - ordinary_points)
    limits: np.ndarray = np.where(
        compared["statistic"].eq("spearman").to_numpy(),
        MAX_LAYER_FINAL_SPEARMAN_ABS_ERROR,
        MAX_LAYER_FINAL_CONTRAST_ABS_ERROR,
    )
    invalid: np.ndarray = (
        ~np.isfinite(layer_points)
        | ~np.isfinite(ordinary_points)
        | (absolute_errors > limits)
    )
    if invalid.any():
        invalid_indices: np.ndarray = np.flatnonzero(invalid)
        worst_index: int = int(
            invalid_indices[
                np.argmax(absolute_errors[invalid_indices] / limits[invalid_indices])
            ]
        )
        raise ValueError(
            f"Layerwise endpoint point estimates disagree with {ordinary_path}: "
            f"statistic={compared.iloc[worst_index]['statistic']}, "
            f"abs_error={absolute_errors[worst_index]:.9g}, "
            f"limit={limits[worst_index]:.9g}"
        )


def _validate_layerwise_derived_output(
    results_dir: str, ordinary_f_table_path: str
) -> None:
    """Validate the canonical matched-relative-depth fidelity table."""
    path: str = os.path.join(results_dir, RELEASE_LAYERWISE_DERIVED_TABLES[0])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing derived result table: {path}")
    frame: pd.DataFrame = pd.read_csv(path, sep="\t")
    required_columns: set[str] = {
        "aggregation",
        "availability_status",
        "benchmark",
        "cohort",
        "contrast",
        "depth_alignment",
        "embedding_slot_included",
        "expected_observations",
        "expected_prompts",
        "f_hi",
        "f_lo",
        "f_point",
        "metric",
        "model_s",
        "model_t",
        "n_observations",
        "n_prompts",
        "observation_coverage",
        "pair_population",
        "pregrouper",
        "prompt_coverage",
        "readout_contrast",
        "relative_depth",
        "relative_depth_grid_size",
        "relative_depth_index",
        "requested_contrast",
        "requested_scope",
        "resolved_scope",
        "resolved_source_contrast",
        "resolved_target_contrast",
        "scope",
        "source_num_blocks",
        "statistic",
        "target_num_blocks",
        "unavailable_reason",
    }
    if set(frame.columns) != required_columns:
        raise ValueError(f"{path} has an incorrect layerwise result schema")

    depth_grid_size: int = 21
    expected_keys: set[tuple[str, str, str, str, str, str, int]] = {
        (
            benchmark,
            pregrouper,
            source,
            target,
            metric,
            statistic,
            depth_index,
        )
        for benchmark, pregrouper in LAYERWISE_CONFIGS
        for source, target in itertools.combinations(OPEN_MODELS, 2)
        for metric in ("F_pred", "F_attr")
        for statistic in ("spearman", "pearson_r", "pearson_r2")
        for depth_index in range(depth_grid_size)
    }
    actual_keys: set[tuple[str, str, str, str, str, str, int]] = {
        (
            str(row.benchmark),
            str(row.pregrouper),
            str(row.model_s),
            str(row.model_t),
            str(row.metric),
            str(row.statistic),
            int(row.relative_depth_index),
        )
        for row in frame.itertuples()
    }
    if len(frame) != len(expected_keys) or actual_keys != expected_keys:
        raise ValueError(f"{path} has an incomplete canonical layerwise grid")

    expected_depth: pd.Series = pd.to_numeric(
        frame["relative_depth_index"], errors="coerce"
    ) / (depth_grid_size - 1)
    numeric_columns: tuple[str, ...] = (
        "f_point",
        "f_lo",
        "f_hi",
        "n_observations",
        "expected_observations",
        "observation_coverage",
        "n_prompts",
        "expected_prompts",
        "prompt_coverage",
        "source_num_blocks",
        "target_num_blocks",
    )
    numeric: pd.DataFrame = frame[list(numeric_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        not np.isfinite(numeric.to_numpy(dtype=float)).all()
        or not np.allclose(
            pd.to_numeric(frame["relative_depth"], errors="coerce"),
            expected_depth,
        )
        or not frame["relative_depth_grid_size"].eq(depth_grid_size).all()
        or not frame["cohort"].eq("open").all()
        or not frame["pair_population"].eq("open_open").all()
        or not frame["scope"].eq("user").all()
        or not frame["requested_scope"].eq("user").all()
        or not frame["availability_status"].eq("available").all()
        or not frame["unavailable_reason"].fillna("").eq("").all()
        or not frame["aggregation"].eq("row_pooled").all()
        or not frame["depth_alignment"]
        .eq("linear_interpolation_block_outputs_only")
        .all()
        or not frame["embedding_slot_included"].eq(False).all()
        or not numeric["observation_coverage"].eq(1.0).all()
        or not numeric["prompt_coverage"].eq(1.0).all()
        or not numeric["n_observations"].eq(numeric["expected_observations"]).all()
        or not numeric["n_prompts"].eq(numeric["expected_prompts"]).all()
        or (numeric[["source_num_blocks", "target_num_blocks"]] < 2).any().any()
    ):
        raise ValueError(f"{path} has invalid layerwise values or method metadata")
    for benchmark, rows in frame.groupby("benchmark"):
        expected_contrast: str = (
            "entailment_contradiction"
            if str(benchmark).startswith("anli_")
            else "canonical"
        )
        if (
            not rows["contrast"].eq(expected_contrast).all()
            or not rows["requested_contrast"].eq(expected_contrast).all()
        ):
            raise ValueError(f"{path} has an incorrect contrast for {benchmark}")
        for metric, metric_rows in rows.groupby("metric"):
            source_contrast, target_contrast, readout = _contrast_metadata(
                str(benchmark), expected_contrast, str(metric)
            )
            if (
                not metric_rows["resolved_scope"]
                .eq(_resolved_scope(str(metric), "user"))
                .all()
                or not metric_rows["resolved_source_contrast"].eq(source_contrast).all()
                or not metric_rows["resolved_target_contrast"].eq(target_contrast).all()
                or not metric_rows["readout_contrast"].eq(readout).all()
            ):
                raise ValueError(f"{path} has incorrect resolved estimand metadata")

    parameters: dict[str, Any] = _validate_derived_sidecar(
        results_dir,
        path,
        "benchmark_scripts.layerwise_fidelity",
        list(LAYERWISE_CONFIGS),
        OPEN_MODELS,
        expected_input_paths=_layerwise_derived_inputs(results_dir),
        supporting_source_files=LAYER_DERIVED_SUPPORTING_SOURCE_FILES,
    )
    expected_parameters: dict[str, Any] = {
        "benchmark_configs": [
            {"benchmark": benchmark, "pregrouper": pregrouper}
            for benchmark, pregrouper in LAYERWISE_CONFIGS
        ],
        "scopes": ["user"],
        "anli_contrast": "entailment_contradiction",
        "models": list(OPEN_MODELS),
        "depth_grid_size": depth_grid_size,
        "depth_alignment": "linear_interpolation",
        "depth_definition": "first_block_0_last_block_1_embedding_excluded",
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
        "seed": 42,
        "bootstrap_rng": "sha256_cell_key_v1",
        "missingness_policy": "pair_specific_complete_case",
    }
    if parameters != expected_parameters:
        raise ValueError(f"Incorrect layerwise derivation parameters in {path}")
    _validate_layerwise_endpoint_consistency(
        frame,
        _read_derived(ordinary_f_table_path),
        path,
        ordinary_f_table_path,
    )


def _validate_derived_outputs(
    results_dir: str,
    cohort_name: str,
    models: tuple[str, ...],
) -> None:
    """Validate the canonical scalar and multivariate result tables."""
    unsupported_prediction_configs: set[tuple[str, str, str]] = set()
    unsupported_attribution_configs: set[tuple[str, str, str]] = set()
    for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS:
        unsupported_prediction, unsupported_attribution = _unsupported_model_components(
            results_dir, benchmark, pregrouper
        )
        unsupported_prediction_configs.update(
            (benchmark, pregrouper, model) for model in unsupported_prediction
        )
        unsupported_attribution_configs.update(
            (benchmark, pregrouper, model) for model in unsupported_attribution
        )

    suffix: str = "_open" if cohort_name == "open" else ""
    expected_configs: set[tuple[str, str, str, str]] = {
        (
            benchmark,
            pregrouper,
            "user",
            (
                "entailment_contradiction"
                if benchmark.startswith("anli_")
                else "canonical"
            ),
        )
        for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
    }
    for filename, infinity_policy in (
        (f"f_table{suffix}.tsv", "pairwise_complete"),
        (f"f_table_finite_extreme_sensitivity{suffix}.tsv", "finite_extreme"),
    ):
        path: str = os.path.join(results_dir, filename)
        frame: pd.DataFrame = _read_derived(path)
        parameters: dict[str, Any] = _validate_derived_sidecar(
            results_dir,
            path,
            "benchmark_scripts.f_table",
            list(DEFAULT_BENCHMARK_CONFIGS),
            models,
            expected_input_paths=_f_table_derived_input_paths(
                results_dir,
                list(DEFAULT_BENCHMARK_CONFIGS),
                models,
                {
                    (benchmark, pregrouper)
                    for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
                    if benchmark.startswith("anli_")
                },
            ),
            supporting_source_files=F_TABLE_DERIVED_SUPPORTING_SOURCE_FILES,
        )
        _validate_f_table_parameters(
            parameters,
            expected_configs,
            models,
            infinity_policy,
            "row_pooled",
            False,
            path,
            "entailment_contradiction",
        )
        actual_configs: set[tuple[str, str, str, str]] = set(
            frame[["benchmark", "pregrouper", "scope", "contrast"]].itertuples(
                index=False, name=None
            )
        )
        if actual_configs != expected_configs:
            raise ValueError(f"{path} has an incomplete configuration grid")
        _require_pair_grid(
            frame,
            expected_configs,
            models,
            path,
            infinity_policy,
            "row_pooled",
            unsupported_prediction_configs,
            unsupported_attribution_configs,
        )

    race_filename: str = "race_rv_open.tsv" if cohort_name == "open" else "race_rv.tsv"
    race_path: str = os.path.join(results_dir, race_filename)
    race: pd.DataFrame = _read_derived(race_path)
    parameters = _validate_derived_sidecar(
        results_dir,
        race_path,
        "benchmark_scripts.race_rv",
        [("race", "sentence")],
        models,
    )
    expected_race_parameters: dict[str, Any] = {
        "scopes": ["user"],
        "representations": ["all_pairs", "anchor_a"],
        "scalar_representation": "canonical_scalar",
        "scalar_attribution": "correct_vs_rest",
        "bootstrap_resamples": 1000,
        "confidence_level": 0.95,
        "seed": 42,
        "cohort": cohort_name,
        "requested_models": list(models),
        "output_models": sorted(models),
        "missingness_policy": "pair_specific_complete_case",
    }
    if parameters != expected_race_parameters:
        raise ValueError(f"Incorrect RACE RV derivation parameters in {race_path}")
    required_columns: set[str] = {
        "representation",
        "missingness_policy",
        "expected_observations",
        "observation_coverage",
        "expected_prompts",
        "prompt_coverage",
    }
    if not required_columns.issubset(race.columns):
        raise ValueError(f"{race_path} omits RACE coverage metadata")
    expected_race_keys: set[tuple[str, str, str, str, str]] = {
        ("user", representation, model_s, model_t, metric)
        for representation in ("all_pairs", "anchor_a")
        for model_s, model_t in itertools.combinations(models, 2)
        for metric in ("F_pred_rv", "F_attr_rv")
    }
    race_open_models: tuple[str, ...] = tuple(
        model for model in OPEN_MODELS if model in models
    )
    scalar_metrics: tuple[str, ...] = (
        "F_attn_mean_rv",
        "F_attn_max_rv",
        "F_attn_rollout_rv",
        "F_mag_rv",
        "F_align_rv",
    )
    cross_metrics: tuple[str, ...] = (
        "F_attn_mean_to_attr_rv",
        "F_attn_max_to_attr_rv",
        "F_attn_rollout_to_attr_rv",
        "F_mag_to_attr_rv",
        "F_align_to_attr_rv",
    )
    expected_race_keys.update(
        ("user", "canonical_scalar", model_s, model_t, metric)
        for model_s, model_t in itertools.combinations(race_open_models, 2)
        for metric in scalar_metrics
    )
    expected_race_keys.update(
        ("user", "canonical_scalar", model_s, model_t, metric)
        for model_s in race_open_models
        for model_t in models
        if model_s != model_t
        for metric in cross_metrics
    )
    actual_race_keys: set[tuple[str, str, str, str, str]] = set(
        race[["scope", "representation", "model_s", "model_t", "metric"]].itertuples(
            index=False, name=None
        )
    )
    if actual_race_keys != expected_race_keys:
        raise ValueError(f"{race_path} has an incorrect RACE result grid")
    if (
        set(race["statistic"].astype(str)) != {"rv"}
        or set(race["missingness_policy"].astype(str))
        != {"pair_specific_complete_case"}
        or set(race["aggregation"].astype(str)) != {"row_pooled"}
    ):
        raise ValueError(f"{race_path} has incorrect RACE method metadata")

    observed: pd.Series = pd.to_numeric(race["n_observations"], errors="coerce")
    expected: pd.Series = pd.to_numeric(race["expected_observations"], errors="coerce")
    coverage: pd.Series = pd.to_numeric(race["observation_coverage"], errors="coerce")
    observed_prompts: pd.Series = pd.to_numeric(race["n_prompts"], errors="coerce")
    expected_prompts: pd.Series = pd.to_numeric(
        race["expected_prompts"], errors="coerce"
    )
    prompt_coverage: pd.Series = pd.to_numeric(race["prompt_coverage"], errors="coerce")
    if (
        observed.isna().any()
        or expected.isna().any()
        or observed_prompts.isna().any()
        or expected_prompts.isna().any()
        or not coverage.between(0.0, 1.0).all()
        or not prompt_coverage.between(0.0, 1.0).all()
        or not np.allclose(coverage, observed / expected)
        or not np.allclose(prompt_coverage, observed_prompts / expected_prompts)
    ):
        raise ValueError(f"{race_path} has inconsistent RACE coverage")
    points: pd.Series = pd.to_numeric(race["f_point"], errors="coerce")
    lows: pd.Series = pd.to_numeric(race["f_lo"], errors="coerce")
    highs: pd.Series = pd.to_numeric(race["f_hi"], errors="coerce")
    estimable: pd.Series = observed.ge(3)
    if not (
        np.isfinite(points[estimable]).all()
        and np.isfinite(lows[estimable]).all()
        and np.isfinite(highs[estimable]).all()
    ):
        raise ValueError(f"{race_path} has non-finite estimable RACE results")

    _validate_layerwise_derived_output(
        results_dir, os.path.join(results_dir, f"f_table{suffix}.tsv")
    )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--skip-manifest",
        action="store_true",
        help="Validate only; do not rewrite artifact_manifest.json.",
    )
    parser.add_argument(
        "--verify-manifest",
        action="store_true",
        help="Verify the existing artifact_manifest.json instead of rewriting it.",
    )
    parser.add_argument(
        "--require-derived",
        action="store_true",
        help="Also require complete F-table and RACE derived outputs.",
    )
    parser.add_argument(
        "--allow-non-gold-manifests",
        action="store_true",
        help="Validate structure without requiring frozen gold manifest hashes.",
    )
    parser.add_argument(
        "--cohort",
        choices=["open", "paper"],
        default="paper",
        help="Required model cohort (default: complete 11-model paper cohort).",
    )
    parser.add_argument(
        "--manifests-only",
        action="store_true",
        help="Validate canonical grids and exit before requiring model outputs.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Also verify exact frozen source TSV hashes in this directory.",
    )
    args: argparse.Namespace = parser.parse_args()
    if args.skip_manifest and args.verify_manifest:
        parser.error("--skip-manifest and --verify-manifest are mutually exclusive")
    if args.dataset_dir is not None:
        _validate_dataset_snapshots(args.dataset_dir)
    if args.manifests_only:
        _validate_manifests_only(
            args.results_dir,
            require_gold=not args.allow_non_gold_manifests,
        )
        print(f"Validated {len(ARTIFACT_CONFIGS)} segment manifests")
        return

    required_models: tuple[str, ...] = (
        OPEN_MODELS if args.cohort == "open" else PAPER_MODELS
    )
    summaries: list[ValidationSummary] = []
    for benchmark, pregrouper in ARTIFACT_CONFIGS:
        summaries.extend(
            validate_configuration(
                args.results_dir,
                benchmark,
                pregrouper,
                required_models=required_models,
                require_gold_manifest=not args.allow_non_gold_manifests,
                require_hosted_audit=(
                    args.cohort == "paper" and not args.allow_non_gold_manifests
                ),
            )
        )
    if not args.allow_non_gold_manifests:
        _validate_layer_artifacts(args.results_dir)
    _validate_open_model_identity_consistency(args.results_dir)
    if (
        args.dataset_dir is not None
        and args.cohort == "paper"
        and not args.allow_non_gold_manifests
    ):
        _validate_hosted_dialog_identities(args.results_dir, args.dataset_dir)
        _validate_hosted_completion_dialog_identities(
            args.results_dir, args.dataset_dir
        )
    coverage_path: str = os.path.join(args.results_dir, "coverage.tsv")
    expected_coverage: pd.DataFrame = pd.DataFrame(
        [summary.__dict__ for summary in summaries]
    )
    if args.verify_manifest:
        if not os.path.isfile(coverage_path):
            raise FileNotFoundError(f"Missing coverage table: {coverage_path}")
        recorded_coverage: pd.DataFrame = pd.read_csv(coverage_path, sep="\t")
        try:
            pd.testing.assert_frame_equal(
                recorded_coverage,
                expected_coverage,
                check_dtype=False,
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        except AssertionError as error:
            raise ValueError(
                f"Recorded coverage disagrees with raw artifacts: {coverage_path}"
            ) from error
    else:
        expected_coverage.to_csv(coverage_path, sep="\t", index=False)
    if args.require_derived:
        _validate_derived_outputs(
            args.results_dir,
            args.cohort,
            required_models,
        )
    if args.verify_manifest:
        _validate_release_inventory(
            args.results_dir,
            args.cohort,
            require_manifest=True,
        )
        _validate_artifact_manifest(args.results_dir)
    elif not args.skip_manifest:
        _validate_release_inventory(
            args.results_dir,
            args.cohort,
        )
        _write_artifact_manifest(args.results_dir)
    print(f"Validated {len(summaries)} model/configuration outputs")


if __name__ == "__main__":
    main()
