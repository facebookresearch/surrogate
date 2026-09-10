# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Validate a complete public surrogate-fidelity result artifact.

The validator fails on the indexing bug that motivated this correction: each
model's segment rows must use exactly the canonical ``(prompt_idx, seg_idx)``
grid recorded by the model-independent segment manifest. Token-logprob rows
may be sparse for hosted APIs, but their keys must be a subset of that grid and
their coverage is made explicit in ``coverage.tsv``.
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
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from benchmark_scripts.attention_replay_receipt import (
    validate_receipt as validate_attention_replay_receipt,
)
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
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_ATTENTION_REPLAY_RECEIPT_SHA256,
    GOLD_OPEN_EXECUTION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256,
    GOLD_HOSTED_IDENTITY_ATTESTATION,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
    HOSTED_IMPORT_SOURCE_FILES,
    HOSTED_RECORD_SOURCE_FILES,
    OPEN_COMPLETE_SOURCE_FILES,
    OPEN_EXECUTION_SOURCE_FILES,
    OPEN_MODEL_IDENTITY_FILENAMES,
    RECONCILIATION_ARCHIVE_CHECKS_SHA256,
    RECONCILIATION_ARCHIVE_LEDGER_SHA256,
    RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
    RECONCILIATION_ARCHIVE_SHA256,
    RECONCILIATION_ARCHIVE_SIZE_BYTES,
    RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
    RECONCILIATION_RACE_RAW_SHA256,
    RECONCILIATION_RACE_RAW_SIZE_BYTES,
    canonical_file_hash_manifest_sha256,
)
from benchmark_scripts.open_execution_audit_receipt import (
    validate_receipt as validate_open_execution_audit_receipt,
)
from benchmark_scripts.reconcile_results import (
    BOOLQ_WORD_ATTENTION_NOTE,
    BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS,
    CLAIM_COLUMNS,
    LEDGER_COLUMNS,
    MANUSCRIPT_DISPOSITION_COLUMNS,
    MODEL_IDENTITY_NOTE,
    PAPER_CLAIM_CANONICALIZATION,
    PAPER_IDENTITY_STATUS,
    PAPER_REFERENCE_ID,
    RACE_NOTE,
    SCHEMA_VERSION as RECONCILIATION_SCHEMA_VERSION,
    _attach_comparisons,
    _corrected_rows,
    _published_rows,
    archive_checks_sha256,
    archive_ledger_sha256,
    audited_claim_set_sha256,
    estimand_id,
    historical_claims_sha256,
    method_id,
    manuscript_dispositions,
)

ARTIFACT_CONFIGS: list[tuple[str, str]] = [
    *DEFAULT_BENCHMARK_CONFIGS,
    ("race", "sentence"),
]

RELEASE_PAPER_DERIVED_TABLES: tuple[str, ...] = (
    "f_table.tsv",
    "f_table_finite_extreme_sensitivity.tsv",
    "f_table_prompt_equal_transfer.tsv",
    "f_table_message_scopes.tsv",
    "f_table_anli_contrasts.tsv",
    "f_table_revision_candidate_pairwise_drop.tsv",
    "f_table_revision_candidate_finite_extreme.tsv",
    "race_rv_open.tsv",
    "race_rv_paper_complete_case.tsv",
    "race_multiclass_open.tsv",
    "race_multiclass_paper_global_complete_case.tsv",
    "race_scalar_paper_compatibility.tsv",
)
RELEASE_OPEN_DERIVED_TABLES: tuple[str, ...] = (
    "f_table_open.tsv",
    "f_table_finite_extreme_sensitivity_open.tsv",
    "f_table_prompt_equal_transfer_open.tsv",
    "f_table_message_scopes_open.tsv",
    "f_table_anli_contrasts_open.tsv",
    "race_rv_open.tsv",
    "race_multiclass_open.tsv",
    "race_scalar_paper_compatibility_open.tsv",
)
RELEASE_RECONCILIATION_FILES: tuple[str, ...] = (
    "historical_claims.tsv",
    "manuscript_disposition.tsv",
    "reconciliation.tsv",
    "reconciliation_checks.json",
    "reconciliation_manifest.json",
)
RELEASE_LAMBADA_CANARY_MODELS: tuple[str, ...] = (
    "gemini-2-5-flash-lite-vertex",
    "gpt-4-1",
    "gpt-4o",
)
OPEN_EXECUTION_AUDIT_RECEIPT_NAME: str = "open_execution_audit_receipt.json"
OPEN_ATTENTION_REPLAY_RECEIPT_NAME: str = "open_attention_replay_receipt.json"
OPEN_RUN_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_sha256",
        "benchmark",
        "dataset",
        "execution_model_source",
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
        "seed",
    }
)
OPEN_SOFTWARE_FIELDS: frozenset[str] = frozenset(
    {"cuda_runtime", "numpy", "pandas", "torch", "transformers"}
)
OPEN_MODEL_METADATA_FILENAMES: frozenset[str] = OPEN_MODEL_IDENTITY_FILENAMES | {
    "added_tokens.json",
    "merges.txt",
    "pytorch_model.bin.index.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
}
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
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/race_multiclass.py",
    "benchmark_scripts/race_rv.py",
    "benchmark_scripts/reconcile_results.py",
    "benchmark_scripts/record_hosted_provenance.py",
    "benchmark_scripts/seal_open_provenance.py",
    "benchmark_scripts/validate_results.py",
    "benchmark_scripts/run_all_benchmarks.sh",
    "surrogate/eval_constants.py",
    "surrogate/utils.py",
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
    """Require every published open-model representation value to be finite."""
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

    if provenance.get("schema_version") != 3:
        raise ValueError(f"Gold open schema version disagrees in {path}")
    if provenance.get("model_source") != GOLD_OPEN_MODEL_REPOSITORIES.get(model):
        raise ValueError(f"Gold open model source disagrees in {path}")
    if provenance.get("model_artifact_hash_timing") != "post_run_seal":
        raise ValueError(f"Gold open model artifact timing disagrees in {path}")
    dataset: Any = provenance.get("dataset")
    parameters: Any = provenance.get("parameters")
    software: Any = provenance.get("software")
    if not isinstance(dataset, dict) or set(dataset) != OPEN_DATASET_FIELDS:
        raise ValueError(f"Gold open dataset metadata fields disagree in {path}")
    if not isinstance(parameters, dict) or set(parameters) != OPEN_PARAMETER_FIELDS:
        raise ValueError(f"Gold open parameter fields disagree in {path}")
    if not isinstance(software, dict) or set(software) != OPEN_SOFTWARE_FIELDS:
        raise ValueError(f"Gold open software fields disagree in {path}")
    if any(
        not isinstance(value, str) or "/" in value or "\\" in value
        for value in software.values()
        if value is not None
    ):
        raise ValueError(f"Gold open software metadata is nonportable in {path}")
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
            expected_transformation_hashes: dict[str, str] = {
                relative_path: _sha256(os.path.join(repository_root, relative_path))
                for relative_path in transformation_files
            }
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
                provenance.get("schema_version") != 3
                or not isinstance(parameters, dict)
                or parameters.get("phases") != ["ablation", "attention"]
                or parameters.get("phase_attention_implementation")
                != expected_phase_backends
            ):
                raise ValueError(
                    f"Open phase/backend provenance disagrees in {run_metadata}"
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
                expected_corrections: dict[str, dict[str, str]] = (
                    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS
                )
                expected_execution_model_source: str | None = (
                    GOLD_OPEN_EXECUTION_MODEL_SOURCES.get(model)
                )
                expected_source_hash_timing: str = "run_completion"
            elif execution_hashes == release_execution_hashes:
                expected_corrections = {}
                expected_execution_model_source = GOLD_OPEN_MODEL_REPOSITORIES.get(
                    model
                )
                expected_source_hash_timing = "run_start"
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
            actual_corrections: dict[str, dict[str, str]] = {}
            for relative_path, execution_sha256 in execution_hashes.items():
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
                    "execution_model_source",
                    "model_source",
                    "model_revision",
                    "model_artifact_manifest_sha256",
                    "model_artifact_sha256",
                    "model_identity_files_sha256",
                )
            }
            if baseline is None:
                baseline = signature
                baseline_path = path
            elif signature != baseline:
                raise ValueError(
                    f"Open model identity differs between {baseline_path} and {path}"
                )


def _validate_open_audit_receipts(results_dir: str) -> None:
    """Verify the two portable receipts unique to the corrected gold run."""
    specifications: tuple[
        tuple[str, str, Callable[[dict[str, Any], str], None]], ...
    ] = (
        (
            OPEN_EXECUTION_AUDIT_RECEIPT_NAME,
            GOLD_OPEN_EXECUTION_AUDIT_RECEIPT_SHA256,
            validate_open_execution_audit_receipt,
        ),
        (
            OPEN_ATTENTION_REPLAY_RECEIPT_NAME,
            GOLD_OPEN_ATTENTION_REPLAY_RECEIPT_SHA256,
            validate_attention_replay_receipt,
        ),
    )
    for filename, expected_sha256, validate in specifications:
        path: str = os.path.join(results_dir, filename)
        if not os.path.isfile(path) or _sha256(path) != expected_sha256:
            raise ValueError(f"Open audit receipt seal disagrees: {path}")
        with open(path, encoding="utf-8") as source:
            receipt: Any = json.load(source)
        if not isinstance(receipt, dict):
            raise ValueError(f"Open audit receipt is not an object: {path}")
        validate(receipt, results_dir)


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
                OPEN_ATTENTION_REPLAY_RECEIPT_NAME,
                OPEN_EXECUTION_AUDIT_RECEIPT_NAME,
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
    for table in derived_tables:
        allowed.add(table)
        allowed.add(f"{table}.provenance.json")
    if cohort == "paper":
        for filename in RELEASE_RECONCILIATION_FILES:
            allowed.add(f"reconciliation/{filename}")
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
    require_reconciliation: bool = False,
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
    if cohort == "open" or not require_reconciliation:
        # Standalone rerun directories do not need a copy of the repository's
        # release documentation.
        required.remove("README.md")
    if cohort == "paper" and not require_reconciliation:
        required.difference_update(
            f"reconciliation/{filename}" for filename in RELEASE_RECONCILIATION_FILES
        )
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


def _validate_reconciliation_outputs(results_dir: str) -> None:
    """Verify the paper/archive/corrected reconciliation artifact and its seals."""
    directory: str = os.path.join(results_dir, "reconciliation")
    table_path: str = os.path.join(directory, "reconciliation.tsv")
    claims_path: str = os.path.join(directory, "historical_claims.tsv")
    disposition_path: str = os.path.join(directory, "manuscript_disposition.tsv")
    checks_path: str = os.path.join(directory, "reconciliation_checks.json")
    manifest_path: str = os.path.join(directory, "reconciliation_manifest.json")
    for path in (
        table_path,
        claims_path,
        disposition_path,
        checks_path,
        manifest_path,
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing reconciliation artifact: {path}")

    table: pd.DataFrame = pd.read_csv(
        table_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if tuple(table.columns) != LEDGER_COLUMNS or table.empty:
        raise ValueError(f"Invalid reconciliation ledger schema in {table_path}")
    if archive_ledger_sha256(table_path) != RECONCILIATION_ARCHIVE_LEDGER_SHA256:
        raise ValueError(
            f"Reconciliation archive ledger digest disagrees in {table_path}"
        )
    claims: pd.DataFrame = pd.read_csv(
        claims_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if tuple(claims.columns) != CLAIM_COLUMNS or claims.empty:
        raise ValueError(f"Invalid historical-claims schema in {claims_path}")
    if historical_claims_sha256(claims_path) != RECONCILIATION_HISTORICAL_CLAIMS_SHA256:
        raise ValueError(f"Historical-claims digest disagrees in {claims_path}")
    claim_ids: list[str] = claims["claim_id"].astype(str).tolist()
    if not all(claim_ids) or len(claim_ids) != len(set(claim_ids)):
        raise ValueError(f"Duplicate or blank historical claim IDs in {claims_path}")
    dispositions: pd.DataFrame = pd.read_csv(
        disposition_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    expected_dispositions: pd.DataFrame = pd.DataFrame(
        manuscript_dispositions(), columns=list(MANUSCRIPT_DISPOSITION_COLUMNS)
    ).sort_values("issue_id", kind="stable")
    if tuple(
        dispositions.columns
    ) != MANUSCRIPT_DISPOSITION_COLUMNS or not dispositions.equals(
        expected_dispositions.reset_index(drop=True)
    ):
        raise ValueError(f"Invalid manuscript dispositions in {disposition_path}")
    for claim in claims.to_dict(orient="records"):
        expected_value: float = float(claim["expected_value"])
        actual_value: float = float(claim["actual_value"])
        tolerance: float = float(claim["tolerance"])
        relation: str = str(claim["expected_relation"])
        difference: float = abs(actual_value - expected_value)
        if (
            not np.isfinite(expected_value)
            or not np.isfinite(actual_value)
            or tolerance < 0.0
            or relation not in {"within_tolerance", "outside_tolerance"}
            or (
                difference > tolerance
                if relation == "within_tolerance"
                else difference <= tolerance
            )
            or claim["passed"] != "True"
        ):
            raise ValueError(f"Historical claim failed in {claims_path}")
        try:
            locators: Any = json.loads(str(claim["input_locators"]))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid historical claim locators in {claims_path}"
            ) from error
        if not isinstance(locators, list) or not locators:
            raise ValueError(f"Historical claim has no inputs in {claims_path}")
        for locator in locators:
            normalized_locator: str = os.path.normpath(str(locator))
            if (
                not isinstance(locator, str)
                or not locator
                or os.path.isabs(locator)
                or normalized_locator == os.pardir
                or normalized_locator.startswith(os.pardir + os.sep)
                or "\\" in locator
                or ":" in locator
            ):
                raise ValueError(
                    f"Non-portable historical claim locator in {claims_path}"
                )
    records: list[dict[str, Any]] = table.to_dict(orient="records")
    required_states: set[str] = {"published", "archive", "corrected"}
    if set(table["source_state"].astype(str)) != required_states:
        raise ValueError(f"Incorrect source states in {table_path}")
    for locator in table["source_locator"].astype(str).unique():
        normalized: str = os.path.normpath(locator)
        if (
            not locator
            or os.path.isabs(locator)
            or normalized == os.pardir
            or normalized.startswith(os.pardir + os.sep)
            or "\\" in locator
            or ":" in locator
        ):
            raise ValueError(f"Non-portable source locator in {table_path}")

    expected_published_rows: list[dict[str, Any]] = _published_rows()
    expected_ids: set[str] = {
        str(row["reconciliation_id"]) for row in expected_published_rows
    }
    published: list[dict[str, Any]] = [
        row for row in records if row["source_state"] == "published"
    ]
    archived: list[dict[str, Any]] = [
        row for row in records if row["source_state"] == "archive"
    ]
    corrected: list[dict[str, Any]] = [
        row for row in records if row["source_state"] == "corrected"
    ]
    if (
        len(published) != len(expected_ids)
        or {str(row["reconciliation_id"]) for row in published} != expected_ids
        or len(archived) != len(expected_ids)
        or {str(row["reconciliation_id"]) for row in archived} != expected_ids
    ):
        raise ValueError(f"Incomplete published/archive cell inventory in {table_path}")
    recomputed_corrected, corrected_checks, _ = _corrected_rows(results_dir)
    if any(not bool(check["passed"]) for check in corrected_checks):
        raise ValueError("Corrected reconciliation inputs failed provenance checks")
    expected_corrected_keys: set[tuple[str, str]] = {
        (str(row["reconciliation_id"]), str(row["estimand_id"]))
        for row in recomputed_corrected
    }
    actual_corrected_keys: set[tuple[str, str]] = {
        (str(row["reconciliation_id"]), str(row["estimand_id"])) for row in corrected
    }
    if (
        len(corrected) != len(recomputed_corrected)
        or actual_corrected_keys != expected_corrected_keys
    ):
        raise ValueError(f"Incomplete corrected cell inventory in {table_path}")

    row_keys: list[tuple[str, str, str]] = [
        (
            str(row["source_state"]),
            str(row["reconciliation_id"]),
            str(row["estimand_id"]),
        )
        for row in records
    ]
    if len(row_keys) != len(set(row_keys)):
        raise ValueError(f"Duplicate reconciliation ledger rows in {table_path}")
    for row in records:
        expected_reconciliation_id: str = (
            f"{row['paper_location']}:{row['benchmark']}:{row['pregrouper']}:"
            f"{row['metric']}:{row['pair_set']}:{row['value_component']}"
        )
        if row["reconciliation_id"] != expected_reconciliation_id:
            raise ValueError(f"Invalid reconciliation cell ID in {table_path}")
        if row["method_id"] != method_id(row) or row["estimand_id"] != estimand_id(row):
            raise ValueError(f"Invalid reconciliation estimand ID in {table_path}")

    expected_publication: dict[str, dict[str, Any]] = {
        str(row["reconciliation_id"]): row for row in expected_published_rows
    }
    for row in published:
        expected: dict[str, Any] = expected_publication[str(row["reconciliation_id"])]
        for column in LEDGER_COLUMNS:
            actual_value: Any = row[column]
            expected_value: Any = expected[column]
            if column in {"value", "delta_from_published"}:
                actual_number: float | None = (
                    None if actual_value in ("", None) else float(actual_value)
                )
                expected_number: float | None = (
                    None if expected_value in ("", None) else float(expected_value)
                )
                if (actual_number is None) != (expected_number is None) or (
                    actual_number is not None
                    and expected_number is not None
                    and actual_number != expected_number
                ):
                    raise ValueError(f"Published cell value disagrees in {table_path}")
            elif str(actual_value) != str(expected_value):
                raise ValueError(f"Published cell metadata disagrees in {table_path}")

    published_by_id: dict[str, dict[str, Any]] = {
        str(row["reconciliation_id"]): row for row in published
    }
    for row in archived:
        reference: dict[str, Any] = published_by_id[str(row["reconciliation_id"])]
        expected_data_status: str = (
            "reproduced_invalid_sentence_attention_contamination"
            if row["benchmark"] == "boolq"
            and row["pregrouper"] == "word"
            and row["metric"] in BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
            else (
                "reproduced_metric_mislabeled"
                if row["benchmark"] == "race" and row["metric"] in {"F_pred", "F_attr"}
                else "complete"
            )
        )
        if (
            row["method_id"] != reference["method_id"]
            or row["estimand_id"] != reference["estimand_id"]
            or row["data_status"] != expected_data_status
        ):
            raise ValueError(
                f"Archive estimand does not match the paper in {table_path}"
            )

    expected_corrected_rows: list[dict[str, Any]] = [
        dict(row) for row in expected_published_rows + recomputed_corrected
    ]
    _attach_comparisons(expected_corrected_rows)
    recomputed_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["reconciliation_id"]), str(row["estimand_id"])): row
        for row in expected_corrected_rows
        if row["source_state"] == "corrected"
    }
    for row in corrected:
        key: tuple[str, str] = (
            str(row["reconciliation_id"]),
            str(row["estimand_id"]),
        )
        expected = recomputed_by_key[key]
        for column in LEDGER_COLUMNS:
            actual_value: Any = row[column]
            expected_value: Any = expected[column]
            if column in {"value", "delta_from_published"}:
                actual_number: float | None = (
                    None if actual_value in ("", None) else float(actual_value)
                )
                expected_number: float | None = (
                    None if expected_value in ("", None) else float(expected_value)
                )
                if (actual_number is None) != (expected_number is None) or (
                    actual_number is not None
                    and expected_number is not None
                    and not np.isclose(
                        actual_number, expected_number, rtol=0.0, atol=1e-15
                    )
                ):
                    raise ValueError(
                        f"Corrected reconciliation value disagrees in {table_path}"
                    )
            elif str(actual_value) != str(expected_value):
                raise ValueError(
                    f"Corrected reconciliation metadata disagrees in {table_path}"
                )

    expected_comparisons: list[dict[str, Any]] = [dict(row) for row in records]
    for row in expected_comparisons:
        if row["source_state"] != "published":
            row["comparison_status"] = "pending"
            row["delta_from_published"] = None
    _attach_comparisons(expected_comparisons)
    for actual, expected in zip(records, expected_comparisons):
        if actual["comparison_status"] != expected["comparison_status"]:
            raise ValueError(f"Incorrect comparison status in {table_path}")
        actual_delta: float | None = (
            None
            if actual["delta_from_published"] == ""
            else float(actual["delta_from_published"])
        )
        expected_delta_value: Any = expected["delta_from_published"]
        expected_delta: float | None = (
            None if expected_delta_value in ("", None) else float(expected_delta_value)
        )
        if (actual_delta is None) != (expected_delta is None) or (
            actual_delta is not None
            and expected_delta is not None
            and not np.isclose(actual_delta, expected_delta, rtol=0.0, atol=1e-15)
        ):
            raise ValueError(f"Incorrect published delta in {table_path}")

    with open(checks_path, encoding="utf-8") as source:
        checks: Any = json.load(source)
    check_rows: Any = checks.get("checks") if isinstance(checks, dict) else None
    if (
        not isinstance(checks, dict)
        or set(checks) != {"schema_version", "artifact_type", "all_passed", "checks"}
        or checks.get("schema_version") != RECONCILIATION_SCHEMA_VERSION
        or checks.get("artifact_type") != "reconciliation_checks"
        or checks.get("all_passed") is not True
        or not isinstance(check_rows, list)
        or not check_rows
        or any(
            not isinstance(check, dict) or check.get("passed") is not True
            for check in check_rows
        )
    ):
        raise ValueError(f"Reconciliation checks did not all pass in {checks_path}")
    check_ids: list[str] = [str(check.get("check_id", "")) for check in check_rows]
    if not all(check_ids) or len(check_ids) != len(set(check_ids)):
        raise ValueError(
            f"Duplicate or blank reconciliation check IDs in {checks_path}"
        )
    if archive_checks_sha256(check_rows) != RECONCILIATION_ARCHIVE_CHECKS_SHA256:
        raise ValueError(
            f"Reconciliation archive checks digest disagrees in {checks_path}"
        )
    required_check_ids: set[str] = {
        f"published_rounding:{row['reconciliation_id']}" for row in archived
    }
    for row in archived:
        if row["paper_location"] == "Table 1":
            required_check_ids.add(
                f"archive_pair_grid:Table1:{row['benchmark']}:{row['pregrouper']}:"
                f"{row['metric']}:{row['pair_set']}:{row['value_component']}"
            )
        else:
            required_check_ids.add(
                f"archive_pair_grid:Table2:{row['benchmark']}:{row['pregrouper']}:"
                f"{row['metric']}:all"
            )
    required_check_ids.update(
        {
            "archive_byte_sha256:reconciliation_models",
            "historical_numeric:boolq:sentence:F_pred",
            "historical_numeric:boolq:sentence:F_attr",
            "historical_numeric:race:sentence:F_pred",
            "historical_numeric:race:sentence:F_attr",
            "historical_numeric:boolq:qwen7b_qwen14b:F_attr",
            "archive_ledger_sha256",
            "audited_claim_set_sha256",
            "historical_claims_sha256",
            "nonmatching_estimands_have_no_delta",
            *{
                f"archive_byte_sha256:{benchmark}:{pregrouper}"
                for benchmark, pregrouper in (
                    *DEFAULT_BENCHMARK_CONFIGS,
                    ("race", "sentence"),
                )
            },
            *{
                f"archive_rows:{benchmark}:{pregrouper}"
                for benchmark, pregrouper in (
                    *DEFAULT_BENCHMARK_CONFIGS,
                    ("race", "sentence"),
                )
            },
        }
    )
    required_check_ids.update(f"historical_claim:{claim_id}" for claim_id in claim_ids)
    required_check_ids.update(
        {
            f"archive_byte_sha256:{locator.removeprefix('archive/')}"
            for locator in RECONCILIATION_RACE_RAW_SHA256
        }
    )
    required_check_ids.update(
        {
            f"archive_size_bytes:{locator.removeprefix('archive/')}"
            for locator in RECONCILIATION_RACE_RAW_SIZE_BYTES
        }
    )
    required_check_ids.update(str(check["check_id"]) for check in corrected_checks)
    if set(check_ids) != required_check_ids:
        raise ValueError(f"Incomplete reconciliation check inventory in {checks_path}")
    allowed_check_keys: set[str] = {
        "check_id",
        "kind",
        "locator",
        "expected",
        "actual",
        "tolerance",
        "passed",
    }
    for check in check_rows:
        if not set(check).issubset(allowed_check_keys | {"bypassed"}):
            raise ValueError(f"Unexpected reconciliation check fields in {checks_path}")
        locator_value: str = str(check.get("locator", ""))
        if (
            not locator_value
            or os.path.isabs(locator_value)
            or "\\" in locator_value
            or ":" in locator_value
        ):
            raise ValueError(
                f"Non-portable reconciliation check locator in {checks_path}"
            )
    actual_checks_by_id: dict[str, dict[str, Any]] = {
        str(check["check_id"]): check for check in check_rows
    }
    for claim in claims.to_dict(orient="records"):
        claim_check: dict[str, Any] = actual_checks_by_id[
            f"historical_claim:{claim['claim_id']}"
        ]
        if (
            float(claim_check["expected"]) != float(claim["expected_value"])
            or float(claim_check["actual"]) != float(claim["actual_value"])
            or float(claim_check["tolerance"]) != float(claim["tolerance"])
            or claim_check["passed"] is not True
        ):
            raise ValueError(
                f"Historical claim check disagrees for {claim['claim_id']}"
            )
    for expected_check in corrected_checks:
        check_id: str = str(expected_check["check_id"])
        if actual_checks_by_id[check_id] != expected_check:
            raise ValueError(
                f"Corrected provenance check disagrees for {check_id} in {checks_path}"
            )

    with open(manifest_path, encoding="utf-8") as source:
        manifest: Any = json.load(source)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != RECONCILIATION_SCHEMA_VERSION
        or manifest.get("artifact_type") != "result_reconciliation"
        or set(manifest)
        != {
            "schema_version",
            "artifact_type",
            "generator",
            "inputs",
            "outputs",
            "historical_estimand",
            "paper_reference",
            "archive_provenance",
            "race_caveat",
            "boolq_word_attention_caveat",
            "model_identity_caveat",
            "software",
        }
    ):
        raise ValueError(f"Invalid reconciliation manifest in {manifest_path}")
    if (
        manifest.get("historical_estimand")
        != {
            "statistic": "pearson_r2",
            "missingness_policy": "pairwise_drop_nonfinite",
            "prediction_scope": "not_applicable",
            "segment_level_declared_scope": "user",
            "segment_level_executed_scope": "all",
            "anli_declared_contrast": "entailment_contradiction",
            "anli_executed_contrast": "entailment_neutral",
        }
        or manifest.get("race_caveat") != RACE_NOTE
        or manifest.get("boolq_word_attention_caveat") != BOOLQ_WORD_ATTENTION_NOTE
        or manifest.get("model_identity_caveat") != MODEL_IDENTITY_NOTE
    ):
        raise ValueError(f"Reconciliation method metadata disagrees in {manifest_path}")
    paper_reference: Any = manifest.get("paper_reference")
    expected_claim_digest: str = audited_claim_set_sha256(
        expected_published_rows,
        claims.to_dict(orient="records"),
    )
    if (
        paper_reference
        != {
            "reference_id": PAPER_REFERENCE_ID,
            "work_id": "arXiv:2606.32008",
            "version": None,
            "pdf_sha256": None,
            "identity_status": PAPER_IDENTITY_STATUS,
            "claim_canonicalization": PAPER_CLAIM_CANONICALIZATION,
            "audited_claim_set_sha256": expected_claim_digest,
            "claim_scope": (
                "Tables 1-2 plus explicitly enumerated Figure 13 and "
                "main-text numerical claims"
            ),
        }
        or expected_claim_digest != RECONCILIATION_AUDITED_CLAIM_SET_SHA256
    ):
        raise ValueError(f"Paper claim-set identity disagrees in {manifest_path}")
    software: Any = manifest.get("software")
    if (
        not isinstance(software, dict)
        or set(software)
        != {
            "python",
            "python_implementation",
            "numpy",
            "pandas",
        }
        or any(
            not isinstance(value, str)
            or not value
            or len(value) > 100
            or any(
                character
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.+-_"
                for character in value
            )
            for value in software.values()
        )
    ):
        raise ValueError(f"Invalid reconciliation software metadata in {manifest_path}")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    generator_path: str = os.path.join(
        repository_root, "benchmark_scripts/reconcile_results.py"
    )
    if manifest.get("generator") != {
        "locator": "benchmark_scripts/reconcile_results.py",
        "sha256": _sha256(generator_path),
    }:
        raise ValueError(f"Reconciliation generator seal disagrees in {manifest_path}")
    expected_outputs: dict[str, dict[str, str | int]] = {
        "reconciliation.tsv": {
            "sha256": _sha256(table_path),
            "size_bytes": os.path.getsize(table_path),
        },
        "historical_claims.tsv": {
            "sha256": _sha256(claims_path),
            "size_bytes": os.path.getsize(claims_path),
        },
        "manuscript_disposition.tsv": {
            "sha256": _sha256(disposition_path),
            "size_bytes": os.path.getsize(disposition_path),
        },
        "reconciliation_checks.json": {
            "sha256": _sha256(checks_path),
            "size_bytes": os.path.getsize(checks_path),
        },
    }
    if manifest.get("outputs") != expected_outputs:
        raise ValueError(f"Reconciliation output seals disagree in {manifest_path}")

    inputs: Any = manifest.get("inputs")
    required_inputs: set[str] = {
        "archive/reconciliation_models.json",
        *{
            f"archive/{benchmark}_{pregrouper}_consolidated.tsv"
            for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
        },
        "archive/race_sentence_consolidated.tsv",
        "corrected/f_table.tsv",
        "corrected/f_table.tsv.provenance.json",
        "corrected/f_table_prompt_equal_transfer.tsv",
        "corrected/f_table_prompt_equal_transfer.tsv.provenance.json",
        "corrected/f_table_revision_candidate_pairwise_drop.tsv",
        "corrected/f_table_revision_candidate_pairwise_drop.tsv.provenance.json",
        "corrected/f_table_revision_candidate_finite_extreme.tsv",
        "corrected/f_table_revision_candidate_finite_extreme.tsv.provenance.json",
        "corrected/race_scalar_paper_compatibility.tsv",
        "corrected/race_scalar_paper_compatibility.tsv.provenance.json",
        "corrected/race/sentence/segments.tsv.gz",
        *{f"corrected/race/sentence/{model}_segment.tsv.gz" for model in OPEN_MODELS},
        *RECONCILIATION_RACE_RAW_SHA256,
    }
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise ValueError(f"Incomplete reconciliation inputs in {manifest_path}")
    archive_provenance: Any = manifest.get("archive_provenance")
    if archive_provenance != {
        "kind": "raw_archive_snapshot_bytes",
        "enforced": True,
        "trusted_sha256": dict(sorted(RECONCILIATION_ARCHIVE_SHA256.items())),
        "trusted_size_bytes": dict(sorted(RECONCILIATION_ARCHIVE_SIZE_BYTES.items())),
        "trusted_race_raw_sha256": dict(sorted(RECONCILIATION_RACE_RAW_SHA256.items())),
        "trusted_race_raw_size_bytes": dict(
            sorted(RECONCILIATION_RACE_RAW_SIZE_BYTES.items())
        ),
        "sanitized_ledger_sha256": RECONCILIATION_ARCHIVE_LEDGER_SHA256,
        "sanitized_checks_sha256": RECONCILIATION_ARCHIVE_CHECKS_SHA256,
        "historical_claims_sha256": RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
        "audited_claim_set_sha256": RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
    }:
        raise ValueError(f"Untrusted reconciliation archive policy in {manifest_path}")
    for locator, expected_digest in RECONCILIATION_ARCHIVE_SHA256.items():
        record: Any = inputs.get(locator)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != expected_digest
            or record.get("size_bytes") != RECONCILIATION_ARCHIVE_SIZE_BYTES[locator]
        ):
            raise ValueError(
                f"Reconciliation archive digest disagrees for {locator} in "
                f"{manifest_path}"
            )
    for locator, expected_digest in RECONCILIATION_RACE_RAW_SHA256.items():
        record = inputs.get(locator)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != expected_digest
            or record.get("size_bytes") != RECONCILIATION_RACE_RAW_SIZE_BYTES[locator]
        ):
            raise ValueError(
                f"Reconciliation RACE raw digest disagrees for {locator} in "
                f"{manifest_path}"
            )
    for locator, record in inputs.items():
        normalized = os.path.normpath(str(locator))
        digest: Any = record.get("sha256") if isinstance(record, dict) else None
        size: Any = record.get("size_bytes") if isinstance(record, dict) else None
        if (
            not isinstance(record, dict)
            or set(record) != {"sha256", "size_bytes"}
            or not str(locator)
            or os.path.isabs(str(locator))
            or normalized == os.pardir
            or normalized.startswith(os.pardir + os.sep)
            or "\\" in str(locator)
            or ":" in str(locator)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid reconciliation input seal in {manifest_path}")
        if str(locator).startswith("corrected/"):
            corrected_path: str = os.path.join(
                results_dir, str(locator).removeprefix("corrected/")
            )
            if (
                not os.path.isfile(corrected_path)
                or _sha256(corrected_path) != digest
                or os.path.getsize(corrected_path) != size
            ):
                raise ValueError(
                    f"Corrected reconciliation input disagrees in {manifest_path}"
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
        for relative_path in DERIVED_SUPPORTING_SOURCE_FILES
    }
    if supporting != expected_supporting:
        raise ValueError(f"Derived supporting-source seal disagrees in {sidecar_path}")

    expected_input_paths: dict[str, str] = {}
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
    candidate_id: str | None = None,
) -> None:
    """Require the exact public F-table analysis parameters."""
    expected_benchmarks: set[tuple[str, str]] = {
        (benchmark, pregrouper) for benchmark, pregrouper, _, _ in expected_configs
    }
    recorded_benchmarks: set[tuple[str, str]] = {
        (str(item["benchmark"]), str(item["pregrouper"]))
        for item in parameters.get("benchmark_configs", [])
    }
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
            "pair_specific_finite_row_deletion"
            if api_infinity_policy == "drop"
            else "model_specific_finite_extreme_replacement"
        ),
        "metrics": expected_metrics,
        "anli_contrast": anli_contrast,
        "artifact_role": "revision_candidate" if candidate_id else "analysis",
        "candidate_id": candidate_id,
        "selection_status": (
            "author_decision_required" if candidate_id else "not_applicable"
        ),
    }
    expected_parameter_fields: set[str] = {
        "benchmark_configs",
        "contrasts",
        "scopes",
        *expected_values,
    }
    if (
        set(parameters) != expected_parameter_fields
        or recorded_benchmarks != expected_benchmarks
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
        "artifact_role",
        "candidate_id",
        "cohort",
        "pair_population",
        "selection_status",
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
    expected_artifact_role: str = (
        "revision_candidate"
        if frame["candidate_id"].astype(str).ne("not_applicable").any()
        else "analysis"
    )
    expected_selection_status: str = (
        "author_decision_required"
        if expected_artifact_role == "revision_candidate"
        else "not_applicable"
    )
    if (
        set(frame["cohort"].astype(str)) != {expected_cohort}
        or set(frame["artifact_role"].astype(str)) != {expected_artifact_role}
        or set(frame["selection_status"].astype(str)) != {expected_selection_status}
    ):
        raise ValueError(f"{path} has incorrect artifact-role or cohort metadata")
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
        for statistic in ("spearman", "pearson_r2"):
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
    observations: pd.Series = pd.to_numeric(frame["n_observations"], errors="coerce")
    prompts: pd.Series = pd.to_numeric(frame["n_prompts"], errors="coerce")
    if observations.isna().any() or prompts.isna().any():
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

    readout_mismatch_rows: pd.Series = frame["contrast"].eq(
        "entailment_contradiction"
    ) & frame["metric"].isin({"F_align", "F_align_to_attr"})
    unsupported_rows: pd.Series = frame.apply(_unsupported_row, axis=1)
    points: pd.Series = pd.to_numeric(frame["f_point"], errors="coerce")
    lows: pd.Series = pd.to_numeric(frame["f_lo"], errors="coerce")
    highs: pd.Series = pd.to_numeric(frame["f_hi"], errors="coerce")
    unavailable_rows: pd.Series = unsupported_rows | readout_mismatch_rows
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
    if (
        not frame.loc[readout_mismatch_rows, "unavailable_reason"]
        .astype(str)
        .eq("readout_contrast_mismatch")
        .all()
    ):
        raise ValueError(f"{path} does not label readout-contrast mismatches")
    estimable: pd.Series = ~unavailable_rows
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


def _validate_derived_outputs(
    results_dir: str,
    cohort_name: str,
    models: tuple[str, ...],
) -> None:
    """Validate complete derived-table grids and explicit RACE missingness."""
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
    specifications: list[
        tuple[
            str,
            set[tuple[str, str, str, str]],
            str,
            str,
            bool,
            str | None,
            str | None,
        ]
    ] = [
        (
            f"f_table{suffix}.tsv",
            {
                (benchmark, pregrouper, "all", "canonical")
                for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
            },
            "drop",
            "row_pooled",
            False,
            None,
            None,
        ),
        (
            f"f_table_finite_extreme_sensitivity{suffix}.tsv",
            {
                (benchmark, pregrouper, "all", "canonical")
                for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
            },
            "finite_extreme",
            "row_pooled",
            False,
            None,
            None,
        ),
        (
            f"f_table_prompt_equal_transfer{suffix}.tsv",
            {
                (benchmark, pregrouper, "all", "canonical")
                for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
            },
            "drop",
            "prompt_equal_mean_r2",
            False,
            None,
            None,
        ),
        (
            f"f_table_message_scopes{suffix}.tsv",
            {
                (benchmark, pregrouper, scope, "canonical")
                for benchmark, pregrouper in DEFAULT_BENCHMARK_CONFIGS
                for scope in ("all", "system", "user")
            },
            "drop",
            "row_pooled",
            False,
            None,
            None,
        ),
        (
            f"f_table_anli_contrasts{suffix}.tsv",
            {
                (benchmark, "sentence", scope, contrast)
                for benchmark in ("anli_r1", "anli_r2", "anli_r3")
                for scope in ("all", "system", "user")
                for contrast in (
                    "entailment_neutral",
                    "entailment_contradiction",
                )
            },
            "drop",
            "row_pooled",
            False,
            None,
            None,
        ),
        (
            f"race_scalar_paper_compatibility{suffix}.tsv",
            {
                ("race", "sentence", scope, "canonical")
                for scope in ("all", "system", "user")
            },
            "drop",
            "row_pooled",
            True,
            None,
            None,
        ),
    ]
    if cohort_name == "paper":
        revision_configs: set[tuple[str, str, str, str]] = {
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
        specifications.extend(
            [
                (
                    "f_table_revision_candidate_pairwise_drop.tsv",
                    revision_configs,
                    "drop",
                    "row_pooled",
                    False,
                    "entailment_contradiction",
                    "declared_method_pairwise_drop",
                ),
                (
                    "f_table_revision_candidate_finite_extreme.tsv",
                    revision_configs,
                    "finite_extreme",
                    "row_pooled",
                    False,
                    "entailment_contradiction",
                    "declared_method_finite_extreme",
                ),
            ]
        )
    candidate_frames: dict[str, pd.DataFrame] = {}
    for (
        filename,
        expected_configs,
        infinity_policy,
        transfer_aggregation,
        scalar_only,
        anli_contrast,
        candidate_id,
    ) in specifications:
        path: str = os.path.join(results_dir, filename)
        frame: pd.DataFrame = _read_derived(path)
        benchmark_configs: list[tuple[str, str]] = sorted(
            {
                (benchmark, pregrouper)
                for benchmark, pregrouper, _, _ in expected_configs
            }
        )
        parameters: dict[str, Any] = _validate_derived_sidecar(
            results_dir,
            path,
            "benchmark_scripts.f_table",
            benchmark_configs,
            models,
        )
        _validate_f_table_parameters(
            parameters,
            expected_configs,
            models,
            infinity_policy,
            transfer_aggregation,
            scalar_only,
            path,
            anli_contrast,
            candidate_id,
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
            transfer_aggregation,
            unsupported_prediction_configs,
            unsupported_attribution_configs,
            scalar_only,
        )

        expected_candidate_value: str = candidate_id or "not_applicable"
        if set(frame["candidate_id"].astype(str)) != {expected_candidate_value}:
            raise ValueError(f"{path} has an incorrect candidate ID")
        if candidate_id is not None:
            candidate_frames[candidate_id] = frame

    if cohort_name == "paper":
        if set(candidate_frames) != {
            "declared_method_pairwise_drop",
            "declared_method_finite_extreme",
        }:
            raise ValueError("Revision-candidate table inventory is incomplete")
        invariant_columns: list[str] = [
            column
            for column in candidate_frames[
                "declared_method_pairwise_drop"
            ].columns.tolist()
            if column not in {"api_infinity_policy", "candidate_id"}
        ]
        sort_columns: list[str] = [
            "benchmark",
            "pregrouper",
            "requested_scope",
            "requested_contrast",
            "model_s",
            "model_t",
            "metric",
            "statistic",
        ]
        invariant_views: list[pd.DataFrame] = []
        for candidate_id in (
            "declared_method_pairwise_drop",
            "declared_method_finite_extreme",
        ):
            candidate: pd.DataFrame = candidate_frames[candidate_id]
            view: pd.DataFrame = candidate[
                candidate["pair_population"].eq("open_open")
            ][invariant_columns].sort_values(sort_columns)
            invariant_views.append(view.reset_index(drop=True))
        if not invariant_views[0].equals(invariant_views[1]):
            raise ValueError(
                "Revision candidates differ outside the hosted non-finite policy"
            )

    race_files: list[tuple[str, int, bool]] = [
        ("race_rv_open.tsv", len(OPEN_MODELS), True)
    ]
    if cohort_name == "paper":
        race_files.append(("race_rv_paper_complete_case.tsv", len(PAPER_MODELS), False))
    for filename, race_models, require_full_coverage in race_files:
        path = os.path.join(results_dir, filename)
        frame = _read_derived(path)
        race_cohort: tuple[str, ...] = (
            OPEN_MODELS if race_models == len(OPEN_MODELS) else PAPER_MODELS
        )
        parameters = _validate_derived_sidecar(
            results_dir,
            path,
            "benchmark_scripts.race_rv",
            [("race", "sentence")],
            race_cohort,
        )
        expected_rv_parameters: dict[str, Any] = {
            "scopes": ["all", "system", "user"],
            "representations": ["all_pairs", "anchor_a"],
            "bootstrap_resamples": 1000,
            "confidence_level": 0.95,
            "seed": 42,
            "cohort": "open" if require_full_coverage else "paper",
            "requested_models": list(race_cohort),
            "output_models": sorted(race_cohort),
            "missingness_policy": "pair_specific_complete_case",
        }
        if parameters != expected_rv_parameters:
            raise ValueError(f"Incorrect RACE RV derivation parameters in {path}")
        extra_required: set[str] = {
            "representation",
            "missingness_policy",
            "expected_observations",
            "observation_coverage",
        }
        if not extra_required.issubset(frame.columns):
            raise ValueError(f"{path} omits RACE coverage metadata")
        expected_race_keys: set[tuple[str, str, str, str, str]] = {
            (scope, representation, model_s, model_t, metric)
            for scope in ("all", "system", "user")
            for representation in ("all_pairs", "anchor_a")
            for model_s, model_t in itertools.combinations(race_cohort, 2)
            for metric in ("F_pred_rv", "F_attr_rv")
        }
        actual_race_keys: set[tuple[str, str, str, str, str]] = set(
            frame[
                ["scope", "representation", "model_s", "model_t", "metric"]
            ].itertuples(index=False, name=None)
        )
        if actual_race_keys != expected_race_keys:
            raise ValueError(f"{path} has an incorrect RACE semantic result grid")
        if (
            set(frame["statistic"].astype(str)) != {"rv"}
            or set(frame["missingness_policy"].astype(str))
            != {"pair_specific_complete_case"}
            or set(frame["aggregation"].astype(str)) != {"row_pooled"}
        ):
            raise ValueError(f"{path} has incorrect RACE method metadata")
        coverage: pd.Series = pd.to_numeric(
            frame["observation_coverage"], errors="coerce"
        )
        if coverage.isna().any() or not coverage.between(0.0, 1.0).all():
            raise ValueError(f"{path} has invalid observation coverage")
        observed: pd.Series = pd.to_numeric(frame["n_observations"], errors="coerce")
        expected: pd.Series = pd.to_numeric(
            frame["expected_observations"], errors="coerce"
        )
        if (
            observed.isna().any()
            or expected.isna().any()
            or not np.allclose(coverage, observed / expected)
        ):
            raise ValueError(f"{path} has inconsistent RACE coverage counts")
        points = pd.to_numeric(frame["f_point"], errors="coerce")
        lows = pd.to_numeric(frame["f_lo"], errors="coerce")
        highs = pd.to_numeric(frame["f_hi"], errors="coerce")
        estimable = observed.ge(3)
        if not (
            np.isfinite(points[estimable]).all()
            and np.isfinite(lows[estimable]).all()
            and np.isfinite(highs[estimable]).all()
        ):
            raise ValueError(f"{path} has non-finite estimable RACE results")
        if require_full_coverage and not coverage.eq(1.0).all():
            raise ValueError(f"{path} has incomplete open-model RACE coverage")

    multiclass_files: list[tuple[str, tuple[str, ...], str]] = [
        ("race_multiclass_open.tsv", OPEN_MODELS, "strict_complete")
    ]
    if cohort_name == "paper":
        multiclass_files.append(
            (
                "race_multiclass_paper_global_complete_case.tsv",
                PAPER_MODELS,
                "global_complete_case_mnar",
            )
        )
    vector_metrics: tuple[str, ...] = (
        "signed_frobenius_r",
        "direction_cosine",
        "linear_cka",
    )
    magnitude_metrics: tuple[str, ...] = (
        "aitchison_magnitude_r",
        "fisher_rao_magnitude_r",
        "fisher_local_magnitude_r",
    )
    for filename, race_cohort, missingness_policy in multiclass_files:
        path = os.path.join(results_dir, filename)
        frame = _read_derived(path)
        parameters = _validate_derived_sidecar(
            results_dir,
            path,
            "benchmark_scripts.race_multiclass",
            [("race", "sentence")],
            race_cohort,
        )
        expected_multiclass_parameters: dict[str, Any] = {
            "scopes": ["all", "system", "user"],
            "coordinate_system": "helmert_ilr_a_b_c_d",
            "aggregations": ["row_pooled", "prompt_equal"],
            "bootstrap_resamples": 500,
            "confidence_level": 0.95,
            "seed": 20260717,
            "cohort": "open" if race_cohort == OPEN_MODELS else "paper",
            "requested_models": list(race_cohort),
            "missingness_policy": missingness_policy,
        }
        if parameters != expected_multiclass_parameters:
            raise ValueError(
                f"Incorrect RACE multiclass derivation parameters in {path}"
            )
        required_columns: set[str] = {
            "cohort",
            "aggregation",
            "missingness_policy",
            "expected_observations",
            "observation_coverage",
            "n_bootstrap",
            "n_bootstrap_valid",
            "aligned_key_sha256",
        }
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"{path} omits multiclass audit metadata")
        expected_keys: set[tuple[str, str, str, str, str]] = set()
        for model_s, model_t in itertools.combinations(race_cohort, 2):
            expected_keys.update(
                (
                    "prediction",
                    "row_pooled",
                    model_s,
                    model_t,
                    metric,
                )
                for metric in vector_metrics
            )
            expected_keys.update(
                (
                    scope,
                    aggregation,
                    model_s,
                    model_t,
                    metric,
                )
                for scope in ("all", "system", "user")
                for aggregation in ("row_pooled", "prompt_equal")
                for metric in (*vector_metrics, *magnitude_metrics)
            )
        actual_keys: set[tuple[str, str, str, str, str]] = set(
            frame[["scope", "aggregation", "model_s", "model_t", "metric"]].itertuples(
                index=False, name=None
            )
        )
        if actual_keys != expected_keys:
            raise ValueError(f"{path} has an incorrect multiclass result grid")
        expected_cohort_name: str = "open" if race_cohort == OPEN_MODELS else "paper"
        if set(frame["cohort"].astype(str)) != {expected_cohort_name} or set(
            frame["missingness_policy"].astype(str)
        ) != {missingness_policy}:
            raise ValueError(f"{path} has incorrect multiclass cohort metadata")
        observed = pd.to_numeric(frame["n_observations"], errors="coerce")
        expected = pd.to_numeric(frame["expected_observations"], errors="coerce")
        coverage = pd.to_numeric(frame["observation_coverage"], errors="coerce")
        points = pd.to_numeric(frame["f_point"], errors="coerce")
        lows = pd.to_numeric(frame["f_lo"], errors="coerce")
        highs = pd.to_numeric(frame["f_hi"], errors="coerce")
        bootstrap = pd.to_numeric(frame["n_bootstrap"], errors="coerce")
        bootstrap_valid = pd.to_numeric(frame["n_bootstrap_valid"], errors="coerce")
        if (
            observed.isna().any()
            or expected.isna().any()
            or coverage.isna().any()
            or not np.isfinite(points).all()
            or not np.isfinite(lows).all()
            or not np.isfinite(highs).all()
            or not np.allclose(coverage, observed / expected)
            or not bootstrap.eq(500).all()
            or not bootstrap_valid.ge(0.9 * bootstrap).all()
            or frame["aligned_key_sha256"].astype(str).str.len().ne(64).any()
        ):
            raise ValueError(f"{path} has invalid multiclass estimates or metadata")
        if missingness_policy == "strict_complete" and not coverage.eq(1.0).all():
            raise ValueError(f"{path} has incomplete strict multiclass coverage")
        if missingness_policy == "global_complete_case_mnar":
            for _, group in frame.groupby("scope"):
                if (
                    group["n_observations"].nunique() != 1
                    or group["n_prompts"].nunique() != 1
                    or group["aligned_key_sha256"].nunique() != 1
                ):
                    raise ValueError(
                        f"{path} does not use one cohort-global complete-case set"
                    )

    rv_open: pd.DataFrame = _read_derived(os.path.join(results_dir, "race_rv_open.tsv"))
    multiclass_open: pd.DataFrame = _read_derived(
        os.path.join(results_dir, "race_multiclass_open.tsv")
    )
    rv_comparison: pd.DataFrame = rv_open[
        (rv_open["representation"] == "all_pairs")
        & (
            (rv_open["metric"] == "F_attr_rv")
            | ((rv_open["metric"] == "F_pred_rv") & (rv_open["scope"] == "all"))
        )
    ][["scope", "model_s", "model_t", "metric", "f_point"]].copy()
    rv_comparison.loc[rv_comparison["metric"] == "F_pred_rv", "scope"] = "prediction"
    multiclass_comparison: pd.DataFrame = multiclass_open[
        (multiclass_open["metric"] == "linear_cka")
        & (multiclass_open["aggregation"] == "row_pooled")
    ][["scope", "model_s", "model_t", "f_point"]].copy()
    comparison: pd.DataFrame = rv_comparison.merge(
        multiclass_comparison,
        on=["scope", "model_s", "model_t"],
        suffixes=("_rv", "_multiclass"),
        validate="one_to_one",
    )
    if len(comparison) != len(multiclass_comparison) or not np.allclose(
        comparison["f_point_rv"],
        comparison["f_point_multiclass"],
        rtol=1e-10,
        atol=1e-12,
    ):
        raise ValueError("Open RACE all-pairs RV and CLR-basis linear CKA do not agree")


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
        "--require-reconciliation",
        action="store_true",
        help="Also require and verify the paper/archive reconciliation bundle.",
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
    if args.require_reconciliation and args.cohort != "paper":
        parser.error("--require-reconciliation requires --cohort paper")

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
    _validate_open_model_identity_consistency(args.results_dir)
    if args.cohort == "paper" and not args.allow_non_gold_manifests:
        _validate_open_audit_receipts(args.results_dir)
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
    reconciliation_directory: str = os.path.join(args.results_dir, "reconciliation")
    reconciliation_present: bool = any(
        os.path.exists(os.path.join(reconciliation_directory, filename))
        for filename in RELEASE_RECONCILIATION_FILES
    )
    if args.require_reconciliation or reconciliation_present:
        _validate_reconciliation_outputs(args.results_dir)
    if args.verify_manifest:
        _validate_release_inventory(
            args.results_dir,
            args.cohort,
            require_manifest=True,
            require_reconciliation=args.require_reconciliation,
        )
        _validate_artifact_manifest(args.results_dir)
    elif not args.skip_manifest:
        _validate_release_inventory(
            args.results_dir,
            args.cohort,
            require_reconciliation=args.require_reconciliation,
        )
        _write_artifact_manifest(args.results_dir)
    print(f"Validated {len(summaries)} model/configuration outputs")


if __name__ == "__main__":
    main()
