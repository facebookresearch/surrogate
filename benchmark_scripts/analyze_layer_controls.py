# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Analyze BoolQ random controls for cross-model layerwise fidelity.

The large per-observation control matrices remain raw execution artifacts.
This module validates them and emits a compact relative-depth summary for
prediction fidelity, signed attribution fidelity, and matched attribution
controls. Randomization bands describe variation over control directions,
while prompt-cluster bootstrap intervals describe uncertainty in the observed
target curves.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import tempfile
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from benchmark_scripts.build_layer_control_spec import (
    validate_production_control_spec,
)
from benchmark_scripts.derived_provenance import sha256_file, write_derived_provenance
from benchmark_scripts.f_table import OPEN_MODELS
from benchmark_scripts.layerwise_fidelity import _load_validated_layer_frame
from benchmark_scripts.run_layer_controls import _parse_control_spec

logger: logging.Logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
ARTIFACT_TYPE: str = "layer_random_control_projections"
PROJECTION_DEFINITION: str = "signed_per_segment_postnorm_delta_dot_unit_direction"
ROW_ORDER: str = "segments_manifest_file_order"
DEFAULT_DEPTH_GRID_SIZE: int = 64
DEFAULT_BOOTSTRAP_RESAMPLES: int = 1000
DEFAULT_CONFIDENCE_LEVEL: float = 0.95
DEFAULT_SEED: int = 42
DEFAULT_OBSERVATION_PERMUTATION_DRAWS: int = 64
SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")

# This inventory is intentionally local to the control producer.  Adding it to
# provenance_sources.py would change a file frozen into the existing layer-run
# records.
CONTROL_EXECUTION_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/run_layer_controls.py",
    "benchmark_scripts/run_layerwise.py",
    "surrogate/eval_constants.py",
    "surrogate/layer_control_scoring.py",
    "surrogate/layerwise_scoring.py",
    "surrogate/model_types.py",
    "surrogate/text_augmentation.py",
    "surrogate/transformers_model.py",
    "surrogate/utils.py",
)

CONTROL_FAMILIES: tuple[str, ...] = (
    "shared_single_token_pair",
    "shuffled_shared_single_token_pair",
    "grouped_9v8_pseudo_label",
    "shuffled_grouped_9v8_pseudo_label",
    "independent_isotropic",
)
OBSERVATION_PERMUTATION: str = "observation_pair_permutation"
ANALYSIS_CONTROL_FAMILIES: tuple[str, ...] = (
    *CONTROL_FAMILIES,
    OBSERVATION_PERMUTATION,
)
TARGET_CURVES: tuple[str, ...] = (
    "grouped_logsumexp_prediction",
    "grouped_logsumexp_attribution",
    "single_token_true_false_attribution_diagnostic",
    "grouped_alias_linear_projection_diagnostic",
)
GAP_CURVE: str = "prediction_minus_attribution_gap"
CONTROL_COMPARISON_TARGET: dict[str, str] = {
    "shared_single_token_pair": TARGET_CURVES[2],
    "shuffled_shared_single_token_pair": TARGET_CURVES[2],
    "grouped_9v8_pseudo_label": TARGET_CURVES[1],
    "shuffled_grouped_9v8_pseudo_label": TARGET_CURVES[1],
    "independent_isotropic": TARGET_CURVES[2],
    OBSERVATION_PERMUTATION: TARGET_CURVES[1],
}


@dataclass(frozen=True)
class ControlArtifact:
    """Validated memory-mapped control projections and their run record."""

    model: str
    path: str
    sidecar_path: str
    values: np.ndarray
    metadata: dict[str, Any]
    num_draws: int


@dataclass(frozen=True)
class ModelInputs:
    """Validated control and existing target inputs for one model."""

    control: ControlArtifact
    grouped_logsumexp_prediction: np.ndarray
    grouped_logsumexp_attribution: np.ndarray
    grouped_alias_projection: np.ndarray
    layer_path: str
    layer_sidecar_path: str


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _load_json_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        value: Any = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_control_spec(path: str) -> tuple[dict[str, Any], str]:
    payload: dict[str, Any] = _load_json_object(path)
    # Reuse the producer's complete production-contract parser. The additional
    # checks below bind model-specific resolved IDs to this validated input.
    _parse_control_spec(payload, canary=False)
    validate_production_control_spec(payload)
    return payload, sha256_file(path)


def _control_paths(
    control_results_dir: str, benchmark: str, pregrouper: str, model: str
) -> tuple[str, str]:
    directory: str = os.path.join(control_results_dir, benchmark, pregrouper)
    return (
        os.path.join(directory, f"{model}_layer_controls.npy"),
        os.path.join(directory, f"{model}_layer_controls_run.json"),
    )


def _manifest_path(results_dir: str, benchmark: str, pregrouper: str) -> str:
    base: str = os.path.join(results_dir, benchmark, pregrouper, "segments.tsv")
    for path in (base + ".gz", base):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(base + "[.gz]")


def _load_manifest(path: str) -> pd.DataFrame:
    manifest: pd.DataFrame = pd.read_csv(path, sep="\t")
    required: set[str] = {"prompt_idx", "seg_idx", "message_role"}
    missing: set[str] = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns {sorted(missing)}")
    for column in ("prompt_idx", "seg_idx"):
        numeric: pd.Series = pd.to_numeric(manifest[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"{path} has non-integer {column}")
        manifest[column] = numeric.astype(int)
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError(f"{path} has duplicate segment identities")
    roles: set[str] = set(manifest["message_role"].astype(str))
    if not roles.issubset({"system", "user"}):
        raise ValueError(f"{path} contains unsupported message roles {sorted(roles)}")
    if manifest.empty:
        raise ValueError(f"{path} contains no segments")
    return manifest


def _source_hashes_are_current(source_hashes: Any, sidecar_path: str) -> None:
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(
        CONTROL_EXECUTION_SOURCE_FILES
    ):
        raise ValueError(f"{sidecar_path} has an incomplete execution source set")
    root: str = os.path.dirname(os.path.dirname(__file__))
    for relative_path in CONTROL_EXECUTION_SOURCE_FILES:
        recorded: str = _require_sha256(
            source_hashes.get(relative_path), f"source_sha256.{relative_path}"
        )
        path: str = os.path.join(root, relative_path)
        if recorded != sha256_file(path):
            raise ValueError(f"{sidecar_path} execution source SHA disagrees")


def _alias_id(layer_metadata: Mapping[str, Any], label: str, surface: str) -> int:
    labels: Any = layer_metadata.get("labels")
    if not isinstance(labels, dict) or not isinstance(labels.get(label), dict):
        raise ValueError(f"main layer sidecar lacks label metadata for {label}")
    accepted: Any = labels[label].get("accepted_single_token_aliases")
    if not isinstance(accepted, list):
        raise ValueError(f"main layer sidecar lacks accepted aliases for {label}")
    ids: list[int] = [
        int(entry["token_id"])
        for entry in accepted
        if isinstance(entry, dict)
        and entry.get("surface") == surface
        and isinstance(entry.get("token_id"), int)
    ]
    if len(ids) != 1:
        raise ValueError(
            f"main layer sidecar has no unique {surface!r} alias for {label!r}"
        )
    return ids[0]


def _validate_canonical_direction(
    value: Any, layer_metadata: Mapping[str, Any], sidecar_path: str
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{sidecar_path} lacks canonical_direction")
    expected: dict[str, Any] = {
        "positive_label": "true",
        "negative_label": "false",
        "positive_surface": " true",
        "negative_surface": " false",
        "positive_token_id": _alias_id(layer_metadata, "true", " true"),
        "negative_token_id": _alias_id(layer_metadata, "false", " false"),
    }
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            raise ValueError(f"{sidecar_path} canonical_direction disagrees at {field}")
    norm: Any = value.get("pre_normalization_norm")
    if (
        not isinstance(norm, (int, float))
        or not math.isfinite(float(norm))
        or norm <= 0
    ):
        raise ValueError(f"{sidecar_path} has invalid canonical direction norm")


def _validate_shared_pairs(value: Any, num_draws: int, sidecar_path: str) -> None:
    if not isinstance(value, list) or len(value) != num_draws:
        raise ValueError(f"{sidecar_path} shared token-pair count disagrees")
    seen_surfaces: set[tuple[str, str]] = set()
    for draw_idx, entry in enumerate(value):
        if not isinstance(entry, dict) or entry.get("draw_idx") != draw_idx:
            raise ValueError(f"{sidecar_path} shared token-pair indices disagree")
        positive: Any = entry.get("positive_surface")
        negative: Any = entry.get("negative_surface")
        positive_id: Any = entry.get("positive_token_id")
        negative_id: Any = entry.get("negative_token_id")
        if (
            not isinstance(positive, str)
            or not positive
            or not isinstance(negative, str)
            or not negative
            or positive == negative
            or not isinstance(positive_id, int)
            or not isinstance(negative_id, int)
            or positive_id < 0
            or negative_id < 0
            or positive_id == negative_id
        ):
            raise ValueError(f"{sidecar_path} has an invalid shared token pair")
        surface_pair: tuple[str, str] = (positive, negative)
        if surface_pair in seen_surfaces:
            raise ValueError(f"{sidecar_path} repeats a shared token pair")
        seen_surfaces.add(surface_pair)
        norm: Any = entry.get("pre_normalization_norm")
        if (
            not isinstance(norm, (int, float))
            or not math.isfinite(float(norm))
            or norm <= 0
        ):
            raise ValueError(f"{sidecar_path} has an invalid shared direction norm")


def _validate_grouped_pairs(value: Any, num_draws: int, sidecar_path: str) -> None:
    if not isinstance(value, list) or len(value) != num_draws:
        raise ValueError(f"{sidecar_path} grouped pseudo-label count disagrees")
    seen_groups: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for draw_idx, entry in enumerate(value):
        if not isinstance(entry, dict) or entry.get("draw_idx") != draw_idx:
            raise ValueError(f"{sidecar_path} grouped pseudo-label indices disagree")
        positive: Any = entry.get("positive_surfaces")
        negative: Any = entry.get("negative_surfaces")
        positive_ids: Any = entry.get("positive_token_ids")
        negative_ids: Any = entry.get("negative_token_ids")
        if (
            not isinstance(positive, list)
            or not isinstance(negative, list)
            or not isinstance(positive_ids, list)
            or not isinstance(negative_ids, list)
            or len(positive) != 9
            or len(negative) != 8
            or len(positive_ids) != 9
            or len(negative_ids) != 8
            or not all(isinstance(item, str) and item for item in positive + negative)
            or not all(
                isinstance(item, int) and item >= 0
                for item in positive_ids + negative_ids
            )
            or len(set(positive + negative)) != 17
            or len(set(positive_ids + negative_ids)) != 17
        ):
            raise ValueError(f"{sidecar_path} has an invalid 9-vs-8 pseudo-label pair")
        groups: tuple[tuple[str, ...], tuple[str, ...]] = (
            tuple(positive),
            tuple(negative),
        )
        if groups in seen_groups:
            raise ValueError(f"{sidecar_path} repeats a grouped pseudo-label draw")
        seen_groups.add(groups)
        norm: Any = entry.get("pre_normalization_norm")
        if (
            not isinstance(norm, (int, float))
            or not math.isfinite(float(norm))
            or norm <= 0
        ):
            raise ValueError(f"{sidecar_path} has an invalid grouped direction norm")


def _validated_basis_pool(
    value: Any, width: int, expected_size: int, field: str, sidecar_path: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(value, list) or len(value) != expected_size:
        raise ValueError(f"{sidecar_path} has invalid {field} size")
    by_base: dict[str, Mapping[str, Any]] = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"{sidecar_path} has a non-object {field} entry")
        base: Any = entry.get("base_surface")
        variants: Any = entry.get("variants")
        ids_by_model: Any = entry.get("token_ids_by_model")
        if (
            not isinstance(base, str)
            or not base
            or base in by_base
            or not isinstance(variants, list)
            or len(variants) != width
            or len(set(variants)) != width
            or not all(isinstance(surface, str) and surface for surface in variants)
            or not isinstance(ids_by_model, dict)
            or set(ids_by_model) != set(OPEN_MODELS)
        ):
            raise ValueError(f"{sidecar_path} has an invalid {field} entry")
        for model, token_ids in ids_by_model.items():
            if (
                not isinstance(model, str)
                or not isinstance(token_ids, list)
                or len(token_ids) != width
                or len(set(token_ids)) != width
                or not all(
                    isinstance(token_id, int) and token_id >= 0
                    for token_id in token_ids
                )
            ):
                raise ValueError(f"{sidecar_path} has invalid IDs in {field}")
        by_base[base] = entry
    return by_base


def _validate_basis_selection(
    resolved_spec: Mapping[str, Any],
    input_spec: Mapping[str, Any],
    num_draws: int,
    model: str,
    sidecar_path: str,
) -> None:
    p9: dict[str, Mapping[str, Any]] = _validated_basis_pool(
        input_spec.get("p9_basis_pool"), 9, 278, "p9_basis_pool", sidecar_path
    )
    p8: dict[str, Mapping[str, Any]] = _validated_basis_pool(
        input_spec.get("p8_basis_pool"), 8, 399, "p8_basis_pool", sidecar_path
    )
    selected: Any = input_spec.get("selected_paired_bases")
    if not isinstance(selected, list) or len(selected) != num_draws:
        raise ValueError(f"{sidecar_path} has invalid selected_paired_bases")
    positive_bases: set[str] = set()
    negative_bases: set[str] = set()
    shared: Sequence[Mapping[str, Any]] = resolved_spec["shared_token_pairs"]
    grouped: Sequence[Mapping[str, Any]] = resolved_spec["grouped_pseudo_label_pairs"]
    for draw_idx, entry in enumerate(selected):
        if not isinstance(entry, dict) or entry.get("draw_idx") != draw_idx:
            raise ValueError(f"{sidecar_path} has invalid selected basis indices")
        positive: Any = entry.get("positive_base_surface")
        negative: Any = entry.get("negative_base_surface")
        if (
            not isinstance(positive, str)
            or positive not in p9
            or positive in positive_bases
            or not isinstance(negative, str)
            or negative not in p8
            or negative in negative_bases
        ):
            raise ValueError(f"{sidecar_path} violates no-replacement basis sampling")
        positive_bases.add(positive)
        negative_bases.add(negative)
        expected_positive_surface: str = " " + positive.lower()
        expected_negative_surface: str = " " + negative.lower()
        positive_variants: Sequence[str] = p9[positive]["variants"]
        negative_variants: Sequence[str] = p8[negative]["variants"]
        positive_ids: Sequence[int] = p9[positive]["token_ids_by_model"][model]
        negative_ids: Sequence[int] = p8[negative]["token_ids_by_model"][model]
        if (
            shared[draw_idx]["positive_surface"] != expected_positive_surface
            or shared[draw_idx]["negative_surface"] != expected_negative_surface
            or expected_positive_surface not in positive_variants
            or expected_negative_surface not in negative_variants
            or shared[draw_idx]["positive_token_id"]
            != positive_ids[positive_variants.index(expected_positive_surface)]
            or shared[draw_idx]["negative_token_id"]
            != negative_ids[negative_variants.index(expected_negative_surface)]
            or grouped[draw_idx]["positive_surfaces"] != positive_variants
            or grouped[draw_idx]["negative_surfaces"] != negative_variants
            or grouped[draw_idx]["positive_token_ids"] != positive_ids
            or grouped[draw_idx]["negative_token_ids"] != negative_ids
        ):
            raise ValueError(f"{sidecar_path} resolved controls disagree with bases")


def _validate_isotropic(value: Any, num_draws: int, sidecar_path: str) -> None:
    if not isinstance(value, list) or len(value) != num_draws:
        raise ValueError(f"{sidecar_path} isotropic direction count disagrees")
    hashes: set[str] = set()
    seeds: set[int] = set()
    for draw_idx, entry in enumerate(value):
        if not isinstance(entry, dict) or entry.get("draw_idx") != draw_idx:
            raise ValueError(f"{sidecar_path} isotropic direction indices disagree")
        seed: Any = entry.get("seed")
        digest: str = _require_sha256(
            entry.get("vector_sha256"), f"isotropic[{draw_idx}].vector_sha256"
        )
        if not isinstance(seed, int) or seed < 0 or seed in seeds or digest in hashes:
            raise ValueError(f"{sidecar_path} repeats an isotropic direction")
        seeds.add(seed)
        hashes.add(digest)


def _validate_control_artifact(
    path: str,
    sidecar_path: str,
    manifest_path: str,
    manifest_rows: int,
    model: str,
    layer_metadata: Mapping[str, Any],
    layer_path: str,
    layer_sidecar_path: str,
    input_spec: Mapping[str, Any],
    _input_spec_path: str,
    input_spec_sha256: str,
) -> ControlArtifact:
    """Load one NPY only after validating its complete public run record."""
    metadata: dict[str, Any] = _load_json_object(sidecar_path)
    for field, expected in {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "benchmark": "boolq",
        "pregrouper": "sentence",
        "model": model,
        "segmentation_scope": "full_dialog_in_message_order",
    }.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{sidecar_path} has invalid {field}")
    if _require_sha256(
        metadata.get("manifest_sha256"), "manifest_sha256"
    ) != sha256_file(manifest_path):
        raise ValueError(f"{sidecar_path} manifest SHA disagrees")
    for field in (
        "model_source",
        "model_revision",
        "model_artifact_manifest_sha256",
        "model_identity_files_sha256",
        "layer_slots",
    ):
        if metadata.get(field) != layer_metadata.get(field):
            raise ValueError(f"{sidecar_path} disagrees with main layer run at {field}")
    input_models: Any = input_spec.get("models")
    input_model: Any = (
        input_models.get(model) if isinstance(input_models, dict) else None
    )
    if not isinstance(input_model, dict):
        raise ValueError(f"{sidecar_path} model is absent from input control spec")
    for field in (
        "model_source",
        "model_revision",
        "model_artifact_manifest_sha256",
    ):
        if input_model.get(field) != layer_metadata.get(field):
            raise ValueError(f"{sidecar_path} input spec model identity disagrees")
    tokenizer_hashes: Any = input_model.get("tokenizer_files_sha256")
    artifact_hashes: Any = layer_metadata.get("model_artifact_sha256")
    if (
        not isinstance(tokenizer_hashes, dict)
        or not isinstance(artifact_hashes, dict)
        or any(
            artifact_hashes.get(name) != digest
            for name, digest in tokenizer_hashes.items()
        )
    ):
        raise ValueError(f"{sidecar_path} input tokenizer identity disagrees")
    expected_sources: tuple[tuple[str, str], ...] = (
        ("source_layer_artifact", layer_path),
        ("source_layer_run", layer_sidecar_path),
    )
    for field, source_path in expected_sources:
        source: Any = metadata.get(field)
        if (
            not isinstance(source, dict)
            or source.get("filename") != os.path.basename(source_path)
            or _require_sha256(source.get("sha256"), f"{field}.sha256")
            != sha256_file(source_path)
        ):
            raise ValueError(f"{sidecar_path} has invalid {field}")
    parameters: Any = metadata.get("parameters")
    if (
        not isinstance(parameters, dict)
        or parameters.get("attention_implementation") != "sdpa"
        or parameters.get("batch_size") != 32
        or parameters.get("canary") is not False
        or parameters.get("device_map") != "auto"
        or parameters.get("max_samples") is not None
        or parameters.get("rendered_chat_add_special_tokens") is not False
        or parameters.get("torch_dtype") != "bfloat16"
    ):
        raise ValueError(f"{sidecar_path} is not a complete production control run")
    if metadata.get("source_hash_timing") != "run_start":
        raise ValueError(f"{sidecar_path} does not attest run-start source hashes")
    _source_hashes_are_current(metadata.get("source_sha256"), sidecar_path)

    spec: Any = metadata.get("control_spec")
    if not isinstance(spec, dict):
        raise ValueError(f"{sidecar_path} lacks control_spec")
    num_draws: Any = spec.get("num_draws")
    if not isinstance(num_draws, int) or num_draws < len(OPEN_MODELS):
        raise ValueError(f"{sidecar_path} has invalid num_draws")
    expected_layout: dict[str, int] = {
        "legacy_true_false_linear": 0,
        "shared_token_pair_start": 1,
        "grouped_pseudo_label_start": 1 + num_draws,
        "isotropic_start": 1 + 2 * num_draws,
        "total": 1 + 3 * num_draws,
    }
    if spec.get("column_layout") != expected_layout:
        raise ValueError(f"{sidecar_path} has invalid column_layout")
    if spec.get("projection_definition") != PROJECTION_DEFINITION:
        raise ValueError(f"{sidecar_path} has invalid projection_definition")
    if spec.get("grouped_projection_definition") != (
        "signed_original_minus_ablated_grouped_logsumexp_contrast"
    ):
        raise ValueError(f"{sidecar_path} has invalid grouped projection definition")
    if spec.get("absolute_attribution") is not False:
        raise ValueError(f"{sidecar_path} must store signed attribution values")
    if spec.get("row_order") != ROW_ORDER:
        raise ValueError(f"{sidecar_path} does not bind NPY rows to manifest order")
    if (
        not isinstance(spec.get("input_spec_filename"), str)
        or not spec.get("input_spec_filename")
        or _require_sha256(spec.get("input_spec_sha256"), "input_spec_sha256")
        != input_spec_sha256
        or spec.get("namespace") != input_spec.get("namespace")
        or spec.get("selection_algorithm") != input_spec.get("selection_algorithm")
    ):
        raise ValueError(f"{sidecar_path} input control specification disagrees")
    _validate_canonical_direction(
        spec.get("canonical_direction"), layer_metadata, sidecar_path
    )
    _validate_shared_pairs(spec.get("shared_token_pairs"), num_draws, sidecar_path)
    _validate_grouped_pairs(
        spec.get("grouped_pseudo_label_pairs"), num_draws, sidecar_path
    )
    _validate_isotropic(spec.get("isotropic"), num_draws, sidecar_path)
    _validate_basis_selection(spec, input_spec, num_draws, model, sidecar_path)
    expected_seeds: Any = input_spec.get("isotropic_seeds_by_model", {}).get(model)
    actual_seeds: list[Any] = [entry["seed"] for entry in spec["isotropic"]]
    if actual_seeds != expected_seeds:
        raise ValueError(f"{sidecar_path} isotropic seeds disagree with input spec")

    artifact: Any = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"{sidecar_path} lacks artifact metadata")
    expected_shape: list[int] = [
        manifest_rows,
        int(layer_metadata["layer_slots"]["count"]),
        1 + 3 * num_draws,
    ]
    if (
        artifact.get("filename") != os.path.basename(path)
        or artifact.get("shape") != expected_shape
        or artifact.get("dtype") != "float32"
        or artifact.get("byte_size") != os.path.getsize(path)
        or _require_sha256(artifact.get("sha256"), "artifact.sha256")
        != sha256_file(path)
    ):
        raise ValueError(f"{sidecar_path} artifact metadata disagrees")
    values: np.ndarray = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(values.shape) != expected_shape
        or str(values.dtype) != "float32"
        or values.dtype.str != "<f4"
    ):
        raise ValueError(f"{path} array shape or dtype disagrees")
    for start in range(0, manifest_rows, 1024):
        if not np.isfinite(values[start : start + 1024]).all():
            raise ValueError(f"{path} contains non-finite projections")
    return ControlArtifact(model, path, sidecar_path, values, metadata, num_draws)


def _shared_surface_signature(artifact: ControlArtifact) -> tuple[Any, ...]:
    spec: Mapping[str, Any] = artifact.metadata["control_spec"]
    shared: Sequence[Mapping[str, Any]] = spec["shared_token_pairs"]
    grouped: Sequence[Mapping[str, Any]] = spec["grouped_pseudo_label_pairs"]
    return (
        tuple(
            (entry["draw_idx"], entry["positive_surface"], entry["negative_surface"])
            for entry in shared
        ),
        tuple(
            (
                entry["draw_idx"],
                tuple(entry["positive_surfaces"]),
                tuple(entry["negative_surfaces"]),
            )
            for entry in grouped
        ),
    )


def _validate_cross_model_specs(artifacts: Sequence[ControlArtifact]) -> None:
    if tuple(artifact.model for artifact in artifacts) != OPEN_MODELS:
        raise ValueError("control artifacts must use the canonical five-model order")
    num_draws: set[int] = {artifact.num_draws for artifact in artifacts}
    if len(num_draws) != 1:
        raise ValueError("control artifacts disagree on num_draws")
    signature: tuple[Any, ...] = _shared_surface_signature(artifacts[0])
    if any(
        _shared_surface_signature(artifact) != signature for artifact in artifacts[1:]
    ):
        raise ValueError("shared control surfaces or pseudo-label bases disagree")
    for draw_idx in range(artifacts[0].num_draws):
        hashes: list[str] = [
            artifact.metadata["control_spec"]["isotropic"][draw_idx]["vector_sha256"]
            for artifact in artifacts
        ]
        if len(set(hashes)) != len(hashes):
            raise ValueError("isotropic directions are not independent across models")


def _load_existing_targets(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, dict[str, Any]]:
    frame, layer_path, _manifest_path_unused, layer_sidecar_path = (
        _load_validated_layer_frame(
            results_dir, benchmark, pregrouper, model, "true", "false"
        )
    )
    layer_metadata: dict[str, Any] = _load_json_object(layer_sidecar_path)
    ablated: pd.DataFrame = frame.loc[
        frame["kind"] == "ablated",
        [
            "prompt_idx",
            "seg_idx",
            "layer_slot",
            "w_dot_delta_z_postnorm_true_vs_false",
            "label_score_true",
            "label_score_false",
        ],
    ].copy()
    ablated[["prompt_idx", "seg_idx", "layer_slot"]] = ablated[
        ["prompt_idx", "seg_idx", "layer_slot"]
    ].astype(int)
    manifest_index: pd.MultiIndex = pd.MultiIndex.from_frame(
        manifest[["prompt_idx", "seg_idx"]]
    )
    slots: int = int(layer_metadata["layer_slots"]["count"])

    originals: pd.DataFrame = frame.loc[
        frame["kind"] == "orig",
        ["prompt_idx", "layer_slot", "label_score_true", "label_score_false"],
    ].copy()
    originals[["prompt_idx", "layer_slot"]] = originals[
        ["prompt_idx", "layer_slot"]
    ].astype(int)
    originals["original_contrast"] = (
        originals["label_score_true"] - originals["label_score_false"]
    )
    ablated["ablated_contrast"] = (
        ablated["label_score_true"] - ablated["label_score_false"]
    )
    ablated = ablated.merge(
        originals[["prompt_idx", "layer_slot", "original_contrast"]],
        on=["prompt_idx", "layer_slot"],
        validate="many_to_one",
    )
    ablated["grouped_logsumexp_attribution"] = (
        ablated["original_contrast"] - ablated["ablated_contrast"]
    )
    prompt_index: pd.Index = pd.Index(
        manifest["prompt_idx"].drop_duplicates().to_numpy(dtype=int),
        name="prompt_idx",
    )
    prediction_pivot: pd.DataFrame = originals.pivot(
        index="prompt_idx", columns="layer_slot", values="original_contrast"
    ).reindex(index=prompt_index, columns=range(slots))
    prediction: np.ndarray = prediction_pivot.to_numpy(dtype=np.float64)
    if (
        prediction.shape != (len(prompt_index), slots)
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(f"{layer_path} has incomplete or non-finite prediction")
    arrays: list[np.ndarray] = []
    for column in (
        "grouped_logsumexp_attribution",
        "w_dot_delta_z_postnorm_true_vs_false",
    ):
        pivot: pd.DataFrame = ablated.pivot(
            index=["prompt_idx", "seg_idx"], columns="layer_slot", values=column
        ).reindex(index=manifest_index, columns=range(slots))
        values: np.ndarray = pivot.to_numpy(dtype=np.float64)
        if values.shape != (len(manifest), slots) or not np.isfinite(values).all():
            raise ValueError(f"{layer_path} has incomplete or non-finite {column}")
        arrays.append(values)
    return (
        prediction,
        arrays[0],
        arrays[1],
        layer_path,
        layer_sidecar_path,
        layer_metadata,
    )


def _scope_indices(manifest: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "all":
        mask: np.ndarray = np.ones(len(manifest), dtype=bool)
    elif scope in {"system", "user"}:
        mask = manifest["message_role"].astype(str).eq(scope).to_numpy()
    else:
        raise ValueError(f"unknown message scope {scope!r}")
    indices: np.ndarray = np.flatnonzero(mask)
    if len(indices) < 2:
        raise ValueError(f"scope {scope!r} has fewer than two segments")
    return indices


def _interpolate_at_depth(
    values: np.ndarray,
    row_indices: np.ndarray,
    relative_depth: float,
    include_embedding: bool = False,
) -> np.ndarray:
    """Interpolate stored slots over the selected relative-depth convention."""
    if values.ndim not in {2, 3} or values.shape[1] < 3:
        raise ValueError("layer values must contain embedding plus at least two blocks")
    first_slot: int = 0 if include_embedding else 1
    num_positions: int = values.shape[1] - first_slot
    position: float = relative_depth * (num_positions - 1)
    lower: int = int(math.floor(position))
    upper: int = int(math.ceil(position))
    weight: float = position - lower
    lower_values: np.ndarray = np.asarray(
        values[row_indices, lower + first_slot], dtype=np.float64
    )
    if upper == lower:
        return lower_values
    upper_values: np.ndarray = np.asarray(
        values[row_indices, upper + first_slot], dtype=np.float64
    )
    return lower_values * (1.0 - weight) + upper_values * weight


def _pearson_r2(x: np.ndarray, y: np.ndarray) -> float:
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 2:
        raise ValueError("Pearson inputs must be equal one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Pearson inputs must be finite")
    centered_x: np.ndarray = x - x.mean(dtype=np.float64)
    centered_y: np.ndarray = y - y.mean(dtype=np.float64)
    denominator: float = float(
        np.sqrt(
            np.sum(centered_x * centered_x, dtype=np.float64)
            * np.sum(centered_y * centered_y, dtype=np.float64)
        )
    )
    if denominator == 0.0:
        raise ValueError("Pearson r is undefined for a constant signal")
    correlation: float = float(
        np.sum(centered_x * centered_y, dtype=np.float64) / denominator
    )
    return min(1.0, max(0.0, correlation * correlation))


def _columnwise_pearson_r2(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError("control matrices must have equal two-dimensional shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("control matrices must be finite")
    mean_x: np.ndarray = np.mean(x, axis=0, dtype=np.float64)
    mean_y: np.ndarray = np.mean(y, axis=0, dtype=np.float64)
    centered_x: np.ndarray = x - mean_x
    centered_y: np.ndarray = y - mean_y
    numerator: np.ndarray = np.sum(centered_x * centered_y, axis=0, dtype=np.float64)
    denominator: np.ndarray = np.sqrt(
        np.sum(centered_x * centered_x, axis=0, dtype=np.float64)
        * np.sum(centered_y * centered_y, axis=0, dtype=np.float64)
    )
    if np.any(denominator == 0.0):
        raise ValueError("a control draw has a constant attribution signal")
    correlations: np.ndarray = numerator / denominator
    return np.clip(correlations * correlations, 0.0, 1.0)


def _mean_pair_r2(signals: Sequence[np.ndarray]) -> float:
    if len(signals) < 2:
        raise ValueError("at least two model signals are required")
    return float(
        np.mean(
            [
                _pearson_r2(signals[i], signals[j])
                for i, j in combinations(range(len(signals)), 2)
            ]
        )
    )


def _control_draw_pair_r2(signals: Sequence[np.ndarray]) -> np.ndarray:
    if len(signals) < 2:
        raise ValueError("at least two control matrices are required")
    num_draws: int = signals[0].shape[1]
    if any(signal.shape[1] != num_draws for signal in signals):
        raise ValueError("control matrices disagree on draw count")
    return np.stack(
        [
            _columnwise_pearson_r2(signals[first], signals[second])
            for first, second in combinations(range(len(signals)), 2)
        ],
        axis=0,
    )


def _observation_permutation_pair_r2(
    signals: Sequence[np.ndarray], permutations: np.ndarray
) -> np.ndarray:
    """Return pair scores after breaking shared observation coordinates."""
    if len(signals) < 2:
        raise ValueError("at least two model signals are required")
    if permutations.ndim != 2 or permutations.shape[1] != len(signals[0]):
        raise ValueError("observation permutations disagree with signal rows")
    if permutations.min() < 0 or permutations.max() >= permutations.shape[1]:
        raise ValueError("observation permutation contains an invalid row index")
    centered: list[np.ndarray] = [signal - signal.mean() for signal in signals]
    norms: list[float] = [float(np.linalg.norm(signal)) for signal in centered]
    if any(norm <= 0.0 or not np.isfinite(norm) for norm in norms):
        raise ValueError("observation permutation signal is constant or non-finite")
    output: np.ndarray = np.empty(
        (math.comb(len(signals), 2), len(permutations)), dtype=np.float64
    )
    for pair_index, (first, second) in enumerate(combinations(range(len(signals)), 2)):
        covariance: np.ndarray = centered[second][permutations] @ centered[first]
        output[pair_index] = np.square(covariance / (norms[first] * norms[second]))
    return np.clip(output, 0.0, 1.0)


def _control_draw_mean_pair_r2(signals: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(_control_draw_pair_r2(signals), axis=0)


def _permutation_seed(seed: int, assignment: int, model: str) -> int:
    payload: bytes = f"layer-control-shuffle-v1\0{seed}\0{assignment}\0{model}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _permutation_sha256(permutation: np.ndarray) -> str:
    canonical: np.ndarray = np.asarray(permutation, dtype="<i4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _shuffle_permutations(
    num_draws: int, assignments: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Create recorded independent cross-model draw permutations."""
    if assignments <= 0:
        raise ValueError("shuffle assignments must be positive")
    permutations: np.ndarray = np.empty(
        (assignments, len(OPEN_MODELS), num_draws), dtype=np.int32
    )
    identity: np.ndarray = np.arange(num_draws, dtype=np.int32)
    records: list[dict[str, Any]] = []
    for assignment in range(assignments):
        assignment_records: list[dict[str, Any]] = []
        for model_index, model in enumerate(OPEN_MODELS):
            if model_index == 0:
                model_seed: int | None = None
                permutation: np.ndarray = identity
            else:
                model_seed = _permutation_seed(seed, assignment, model)
                permutation = (
                    np.random.default_rng(model_seed)
                    .permutation(num_draws)
                    .astype(np.int32)
                )
            permutations[assignment, model_index] = permutation
            assignment_records.append(
                {
                    "model": model,
                    "seed": model_seed,
                    "permutation_sha256": _permutation_sha256(permutation),
                }
            )
        records.append({"assignment_idx": assignment, "models": assignment_records})
    return permutations, records


def _pairwise_r2_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return squared Pearson correlations for every cross-column pairing."""
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape or x.shape[0] < 2:
        raise ValueError("control matrices must have equal two-dimensional shapes")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("control matrices must be finite")
    centered_x: np.ndarray = x - np.mean(x, axis=0, dtype=np.float64)
    centered_y: np.ndarray = y - np.mean(y, axis=0, dtype=np.float64)
    norm_x: np.ndarray = np.sqrt(
        np.sum(centered_x * centered_x, axis=0, dtype=np.float64)
    )
    norm_y: np.ndarray = np.sqrt(
        np.sum(centered_y * centered_y, axis=0, dtype=np.float64)
    )
    if np.any(norm_x == 0.0) or np.any(norm_y == 0.0):
        raise ValueError("a control draw has a constant attribution signal")
    correlations: np.ndarray = (centered_x.T @ centered_y) / np.outer(norm_x, norm_y)
    return np.clip(correlations * correlations, 0.0, 1.0)


def _shared_and_shuffled_pair_r2(
    signals: Sequence[np.ndarray], permutations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pair-level matched-draw and shuffled-assignment scores."""
    num_draws: int = signals[0].shape[1]
    if permutations.ndim != 3 or permutations.shape[1:] != (
        len(signals),
        num_draws,
    ):
        raise ValueError("shuffle permutation shape disagrees with controls")
    model_pairs: list[tuple[int, int]] = list(combinations(range(len(signals)), 2))
    shared: np.ndarray = np.zeros((len(model_pairs), num_draws), dtype=np.float64)
    shuffled: np.ndarray = np.zeros(
        (len(model_pairs), permutations.shape[0]), dtype=np.float64
    )
    for pair_index, (first, second) in enumerate(model_pairs):
        matrix: np.ndarray = _pairwise_r2_matrix(signals[first], signals[second])
        shared[pair_index] = np.diag(matrix)
        for assignment in range(permutations.shape[0]):
            shuffled[pair_index, assignment] = float(
                np.mean(
                    matrix[
                        permutations[assignment, first],
                        permutations[assignment, second],
                    ]
                )
            )
    return shared, shuffled


def _shared_and_shuffled_mean_pair_r2(
    signals: Sequence[np.ndarray], permutations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute pair-averaged matched-draw and shuffled-assignment scores."""
    shared, shuffled = _shared_and_shuffled_pair_r2(signals, permutations)
    return np.mean(shared, axis=0), np.mean(shuffled, axis=0)


def _bootstrap_prompt_weights(
    prompt_ids: np.ndarray, resamples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique_prompts, prompt_codes = np.unique(prompt_ids, return_inverse=True)
    if len(unique_prompts) < 2:
        raise ValueError("prompt-cluster bootstrap needs at least two prompts")
    generator: np.random.Generator = np.random.default_rng(seed)
    weights: np.ndarray = np.empty((resamples, len(unique_prompts)), dtype=np.int32)
    for index in range(resamples):
        sampled: np.ndarray = generator.integers(
            0, len(unique_prompts), size=len(unique_prompts)
        )
        weights[index] = np.bincount(sampled, minlength=len(unique_prompts))
    return prompt_codes.astype(int), weights


def _bootstrap_pair_r2(
    x: np.ndarray,
    y: np.ndarray,
    prompt_codes: np.ndarray,
    prompt_weights: np.ndarray,
) -> np.ndarray:
    num_prompts: int = prompt_weights.shape[1]
    count: np.ndarray = np.bincount(prompt_codes, minlength=num_prompts).astype(float)
    sum_x: np.ndarray = np.bincount(prompt_codes, weights=x, minlength=num_prompts)
    sum_y: np.ndarray = np.bincount(prompt_codes, weights=y, minlength=num_prompts)
    sum_x2: np.ndarray = np.bincount(prompt_codes, weights=x * x, minlength=num_prompts)
    sum_y2: np.ndarray = np.bincount(prompt_codes, weights=y * y, minlength=num_prompts)
    sum_xy: np.ndarray = np.bincount(prompt_codes, weights=x * y, minlength=num_prompts)
    n: np.ndarray = prompt_weights @ count
    sx: np.ndarray = prompt_weights @ sum_x
    sy: np.ndarray = prompt_weights @ sum_y
    sxx: np.ndarray = prompt_weights @ sum_x2
    syy: np.ndarray = prompt_weights @ sum_y2
    sxy: np.ndarray = prompt_weights @ sum_xy
    covariance: np.ndarray = sxy - sx * sy / n
    variance_x: np.ndarray = sxx - sx * sx / n
    variance_y: np.ndarray = syy - sy * sy / n
    denominator: np.ndarray = np.sqrt(
        np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0)
    )
    result: np.ndarray = np.full(len(prompt_weights), np.nan, dtype=np.float64)
    valid: np.ndarray = denominator > 0.0
    result[valid] = np.square(covariance[valid] / denominator[valid])
    return np.clip(result, 0.0, 1.0)


def _target_statistics(
    signals: Sequence[np.ndarray],
    prompt_codes: np.ndarray,
    prompt_weights: np.ndarray,
    confidence_level: float,
) -> tuple[dict[str, float], np.ndarray]:
    point: float = _mean_pair_r2(signals)
    bootstrap: np.ndarray = np.zeros(len(prompt_weights), dtype=np.float64)
    pairs: int = 0
    for first, second in combinations(range(len(signals)), 2):
        pair_values: np.ndarray = _bootstrap_pair_r2(
            signals[first], signals[second], prompt_codes, prompt_weights
        )
        if not np.isfinite(pair_values).all():
            raise ValueError("target bootstrap produced an undefined correlation")
        bootstrap += pair_values
        pairs += 1
    bootstrap /= pairs
    alpha: float = (1.0 - confidence_level) / 2.0
    return (
        {
            "mean_pair_pearson_r2": point,
            "bootstrap_mean": float(np.mean(bootstrap)),
            "bootstrap_median": float(np.median(bootstrap)),
            "bootstrap_lower": float(np.quantile(bootstrap, alpha)),
            "bootstrap_upper": float(np.quantile(bootstrap, 1.0 - alpha)),
        },
        bootstrap,
    )


def _paired_difference_statistics(
    first_point: float,
    second_point: float,
    first_bootstrap: np.ndarray,
    second_bootstrap: np.ndarray,
    confidence_level: float,
) -> dict[str, float]:
    """Summarize a paired difference using shared bootstrap resamples."""
    if first_bootstrap.shape != second_bootstrap.shape:
        raise ValueError("paired bootstrap vectors must have the same shape")
    difference: np.ndarray = first_bootstrap - second_bootstrap
    if difference.ndim != 1 or not np.isfinite(difference).all():
        raise ValueError("paired bootstrap difference must be a finite vector")
    alpha: float = (1.0 - confidence_level) / 2.0
    return {
        "mean_pair_pearson_r2": first_point - second_point,
        "bootstrap_mean": float(np.mean(difference)),
        "bootstrap_median": float(np.median(difference)),
        "bootstrap_lower": float(np.quantile(difference, alpha)),
        "bootstrap_upper": float(np.quantile(difference, 1.0 - alpha)),
    }


def _randomization_statistics(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("randomization values must be a nonempty finite vector")
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _control_draw_rows(
    *,
    summary_kind: str,
    benchmark: str,
    pregrouper: str,
    scope: str,
    relative_depth_index: int | str,
    relative_depth: float | str,
    family: str,
    pair_values: np.ndarray,
    model_pairs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Return pair-level and pair-averaged rows for one control family."""
    if pair_values.ndim != 2 or pair_values.shape[0] != len(model_pairs):
        raise ValueError("control pair-value shape disagrees with model pairs")
    if family == OBSERVATION_PERMUTATION:
        randomization_unit: str = "observation_pair_permutation"
    elif family.startswith("shuffled_"):
        randomization_unit = "permutation_assignment"
    else:
        randomization_unit = "control_draw"
    rows: list[dict[str, Any]] = []
    mean_values: np.ndarray = np.mean(pair_values, axis=0)
    for randomization_idx, value in enumerate(mean_values):
        rows.append(
            {
                "summary_kind": summary_kind,
                "benchmark": benchmark,
                "pregrouper": pregrouper,
                "scope": scope,
                "relative_depth_index": relative_depth_index,
                "relative_depth": relative_depth,
                "control_family": family,
                "randomization_unit": randomization_unit,
                "randomization_idx": randomization_idx,
                "aggregation": "mean_over_model_pairs",
                "model_s": "",
                "model_t": "",
                "pearson_r2": float(value),
            }
        )
    for pair_index, (model_s, model_t) in enumerate(model_pairs):
        for randomization_idx, value in enumerate(pair_values[pair_index]):
            rows.append(
                {
                    "summary_kind": summary_kind,
                    "benchmark": benchmark,
                    "pregrouper": pregrouper,
                    "scope": scope,
                    "relative_depth_index": relative_depth_index,
                    "relative_depth": relative_depth,
                    "control_family": family,
                    "randomization_unit": randomization_unit,
                    "randomization_idx": randomization_idx,
                    "aggregation": "model_pair",
                    "model_s": model_s,
                    "model_t": model_t,
                    "pearson_r2": float(value),
                }
            )
    return rows


def _atomic_write_tsv(frame: pd.DataFrame, path: str) -> None:
    directory: str = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    suffix: str = ".tsv.gz" if path.endswith(".gz") else ".tsv"
    descriptor, temporary_path = tempfile.mkstemp(dir=directory, suffix=suffix)
    os.close(descriptor)
    try:
        if path.endswith(".gz"):
            with open(temporary_path, "wb") as raw_file:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw_file,
                    mtime=0,
                ) as compressed_file:
                    with io.TextIOWrapper(
                        compressed_file,
                        encoding="utf-8",
                        newline="",
                    ) as text_file:
                        frame.to_csv(
                            text_file,
                            sep="\t",
                            index=False,
                            lineterminator="\n",
                        )
        else:
            frame.to_csv(
                temporary_path,
                sep="\t",
                index=False,
                lineterminator="\n",
            )
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def analyze(
    source_results_dir: str,
    control_results_dir: str,
    control_spec_path: str,
    scope: str = "user",
    depth_grid_size: int = DEFAULT_DEPTH_GRID_SIZE,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    seed: int = DEFAULT_SEED,
    shuffle_assignments: int | None = None,
    include_embedding: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, str],
    int,
    list[dict[str, Any]],
]:
    """Validate inputs and return compact and optional draw-level results."""
    if depth_grid_size < 2:
        raise ValueError("depth_grid_size must be at least two")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie between zero and one")
    benchmark: str = "boolq"
    pregrouper: str = "sentence"
    manifest_path: str = _manifest_path(source_results_dir, benchmark, pregrouper)
    manifest: pd.DataFrame = _load_manifest(manifest_path)
    input_spec, input_spec_sha256 = _load_control_spec(control_spec_path)
    row_indices: np.ndarray = _scope_indices(manifest, scope)
    scoped_prompts: np.ndarray = manifest.iloc[row_indices]["prompt_idx"].to_numpy(
        dtype=int
    )
    prompt_codes, prompt_weights = _bootstrap_prompt_weights(
        scoped_prompts, bootstrap_resamples, seed
    )
    prediction_prompts: np.ndarray = (
        manifest["prompt_idx"].drop_duplicates().to_numpy(dtype=int)
    )
    prediction_prompt_codes, prediction_prompt_weights = _bootstrap_prompt_weights(
        prediction_prompts, bootstrap_resamples, seed
    )
    if not np.array_equal(
        np.unique(scoped_prompts), np.unique(prediction_prompts)
    ) or not np.array_equal(prompt_weights, prediction_prompt_weights):
        raise ValueError(
            "selected attribution scope does not cover the full prompt set"
        )

    inputs: dict[str, str] = {
        "boolq/sentence/segments": manifest_path,
        "boolq/sentence/control_spec": control_spec_path,
    }
    model_inputs: list[ModelInputs] = []
    for model in OPEN_MODELS:
        (
            grouped_prediction,
            grouped_logsumexp,
            grouped_projection,
            layer_path,
            layer_sidecar_path,
            layer_metadata,
        ) = _load_existing_targets(
            source_results_dir, benchmark, pregrouper, model, manifest
        )
        control_path, control_sidecar_path = _control_paths(
            control_results_dir, benchmark, pregrouper, model
        )
        control: ControlArtifact = _validate_control_artifact(
            control_path,
            control_sidecar_path,
            manifest_path,
            len(manifest),
            model,
            layer_metadata,
            layer_path,
            layer_sidecar_path,
            input_spec,
            control_spec_path,
            input_spec_sha256,
        )
        model_inputs.append(
            ModelInputs(
                control,
                grouped_prediction,
                grouped_logsumexp,
                grouped_projection,
                layer_path,
                layer_sidecar_path,
            )
        )
        prefix: str = f"boolq/sentence/{model}"
        inputs[f"{prefix}/controls"] = control_path
        inputs[f"{prefix}/controls_run"] = control_sidecar_path
        inputs[f"{prefix}/layers"] = layer_path
        inputs[f"{prefix}/layers_run"] = layer_sidecar_path
    _validate_cross_model_specs([item.control for item in model_inputs])
    num_draws: int = model_inputs[0].control.num_draws
    assignment_count: int = (
        num_draws if shuffle_assignments is None else shuffle_assignments
    )
    permutations, permutation_records = _shuffle_permutations(
        num_draws, assignment_count, seed
    )
    summary_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    target_points_by_depth: dict[str, list[float]] = {
        target: [] for target in TARGET_CURVES
    }
    target_bootstrap_by_depth: dict[str, list[np.ndarray]] = {
        target: [] for target in TARGET_CURVES
    }
    control_draws_by_depth: dict[str, list[np.ndarray]] = {
        family: [] for family in ANALYSIS_CONTROL_FAMILIES
    }
    control_pair_draws_by_depth: dict[str, list[np.ndarray]] = {
        family: [] for family in ANALYSIS_CONTROL_FAMILIES
    }
    model_pairs: list[tuple[str, str]] = list(combinations(OPEN_MODELS, 2))
    prediction_row_indices: np.ndarray = np.arange(len(prediction_prompts), dtype=int)
    observation_draws: int = min(num_draws, DEFAULT_OBSERVATION_PERMUTATION_DRAWS)
    observation_seed: int = seed ^ 0x4F425350
    observation_rng: np.random.Generator = np.random.default_rng(observation_seed)
    observation_permutations: np.ndarray = np.stack(
        [
            observation_rng.permutation(len(row_indices))
            for _ in range(observation_draws)
        ],
        axis=0,
    ).astype(np.int32)
    for depth_index, depth_value in enumerate(np.linspace(0.0, 1.0, depth_grid_size)):
        depth: float = float(depth_value)
        prediction_signals: list[np.ndarray] = [
            _interpolate_at_depth(
                item.grouped_logsumexp_prediction,
                prediction_row_indices,
                depth,
                include_embedding,
            )
            for item in model_inputs
        ]
        single_token_signals: list[np.ndarray] = [
            _interpolate_at_depth(
                item.control.values[:, :, 0], row_indices, depth, include_embedding
            )
            for item in model_inputs
        ]
        grouped_logsumexp_signals: list[np.ndarray] = [
            _interpolate_at_depth(
                item.grouped_logsumexp_attribution,
                row_indices,
                depth,
                include_embedding,
            )
            for item in model_inputs
        ]
        grouped_alias_signals: list[np.ndarray] = [
            _interpolate_at_depth(
                item.grouped_alias_projection, row_indices, depth, include_embedding
            )
            for item in model_inputs
        ]
        target_signals: dict[str, list[np.ndarray]] = {
            TARGET_CURVES[0]: prediction_signals,
            TARGET_CURVES[1]: grouped_logsumexp_signals,
            TARGET_CURVES[2]: single_token_signals,
            TARGET_CURVES[3]: grouped_alias_signals,
        }
        row: dict[str, Any] = {
            "summary_kind": "relative_depth",
            "interpretation_priority": (
                "primary_endpoint"
                if depth_index == depth_grid_size - 1
                else "trajectory"
            ),
            "benchmark": benchmark,
            "pregrouper": pregrouper,
            "scope": scope,
            "relative_depth_index": depth_index,
            "relative_depth": depth,
            "depth_grid_size": depth_grid_size,
            "depth_alignment": (
                "linear_interpolation_all_slots"
                if include_embedding
                else "linear_interpolation_block_outputs_only"
            ),
            "embedding_slot_included": include_embedding,
            "n_models": len(OPEN_MODELS),
            "n_model_pairs": math.comb(len(OPEN_MODELS), 2),
            "n_segments": len(row_indices),
            "n_prompts": len(np.unique(scoped_prompts)),
            "f_pred_n_observations": len(prediction_prompts),
            "f_attr_n_observations": len(row_indices),
            "num_control_draws": num_draws,
            "num_observation_permutations": observation_draws,
        }
        for target_name, signals in target_signals.items():
            target_prompt_codes: np.ndarray = (
                prediction_prompt_codes
                if target_name == TARGET_CURVES[0]
                else prompt_codes
            )
            statistics, bootstrap_values = _target_statistics(
                signals,
                target_prompt_codes,
                prompt_weights,
                confidence_level,
            )
            target_points_by_depth[target_name].append(
                statistics["mean_pair_pearson_r2"]
            )
            target_bootstrap_by_depth[target_name].append(bootstrap_values)
            for statistic, value in statistics.items():
                row[f"{target_name}_{statistic}"] = value

        gap_statistics: dict[str, float] = _paired_difference_statistics(
            target_points_by_depth[TARGET_CURVES[0]][-1],
            target_points_by_depth[TARGET_CURVES[1]][-1],
            target_bootstrap_by_depth[TARGET_CURVES[0]][-1],
            target_bootstrap_by_depth[TARGET_CURVES[1]][-1],
            confidence_level,
        )
        for statistic, value in gap_statistics.items():
            row[f"{GAP_CURVE}_{statistic}"] = value

        shared_signed_matrices: list[np.ndarray] = [
            _interpolate_at_depth(
                item.control.values[:, :, 1 : 1 + num_draws],
                row_indices,
                depth,
                include_embedding,
            )
            for item in model_inputs
        ]
        shared_pair_values, shuffled_shared_pair_values = _shared_and_shuffled_pair_r2(
            shared_signed_matrices, permutations
        )
        family_pair_draws: dict[str, np.ndarray] = {
            "shared_single_token_pair": shared_pair_values,
            "shuffled_shared_single_token_pair": shuffled_shared_pair_values,
        }
        del shared_signed_matrices
        grouped_signed_matrices: list[np.ndarray] = [
            _interpolate_at_depth(
                item.control.values[:, :, 1 + num_draws : 1 + 2 * num_draws],
                row_indices,
                depth,
                include_embedding,
            )
            for item in model_inputs
        ]
        grouped_pair_values, shuffled_grouped_pair_values = (
            _shared_and_shuffled_pair_r2(grouped_signed_matrices, permutations)
        )
        family_pair_draws["grouped_9v8_pseudo_label"] = grouped_pair_values
        family_pair_draws["shuffled_grouped_9v8_pseudo_label"] = (
            shuffled_grouped_pair_values
        )
        del grouped_signed_matrices
        isotropic_signed_matrices: list[np.ndarray] = [
            _interpolate_at_depth(
                item.control.values[:, :, 1 + 2 * num_draws :],
                row_indices,
                depth,
                include_embedding,
            )
            for item in model_inputs
        ]
        family_pair_draws["independent_isotropic"] = _control_draw_pair_r2(
            isotropic_signed_matrices
        )
        del isotropic_signed_matrices
        family_pair_draws[OBSERVATION_PERMUTATION] = _observation_permutation_pair_r2(
            grouped_logsumexp_signals, observation_permutations
        )
        for family in ANALYSIS_CONTROL_FAMILIES:
            pair_draws: np.ndarray = family_pair_draws[family]
            draws: np.ndarray = np.mean(pair_draws, axis=0)
            row[f"{family}_comparison_target"] = CONTROL_COMPARISON_TARGET[family]
            if family == OBSERVATION_PERMUTATION:
                row[f"{family}_randomization_unit"] = (
                    "observation_pair_permutation_mean_over_pairs"
                )
            elif family.startswith("shuffled_"):
                row[f"{family}_randomization_unit"] = (
                    "permutation_assignment_mean_over_draws_and_pairs"
                )
            else:
                row[f"{family}_randomization_unit"] = "control_draw_mean_over_pairs"
            control_draws_by_depth[family].append(draws)
            control_pair_draws_by_depth[family].append(pair_draws)
            for statistic, value in _randomization_statistics(draws).items():
                row[f"{family}_{statistic}"] = value
            draw_rows.extend(
                _control_draw_rows(
                    summary_kind="relative_depth",
                    benchmark=benchmark,
                    pregrouper=pregrouper,
                    scope=scope,
                    relative_depth_index=depth_index,
                    relative_depth=depth,
                    family=family,
                    pair_values=pair_draws,
                    model_pairs=model_pairs,
                )
            )
        summary_rows.append(row)
        logger.info("Analyzed relative depth %d/%d", depth_index + 1, depth_grid_size)

    depth_mean_row: dict[str, Any] = {
        "summary_kind": "equal_weight_depth_mean",
        "interpretation_priority": "secondary_depth_mean",
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "scope": scope,
        "relative_depth_index": "",
        "relative_depth": "",
        "depth_grid_size": depth_grid_size,
        "depth_alignment": (
            "linear_interpolation_all_slots"
            if include_embedding
            else "linear_interpolation_block_outputs_only"
        ),
        "embedding_slot_included": include_embedding,
        "n_models": len(OPEN_MODELS),
        "n_model_pairs": math.comb(len(OPEN_MODELS), 2),
        "n_segments": len(row_indices),
        "n_prompts": len(np.unique(scoped_prompts)),
        "f_pred_n_observations": len(prediction_prompts),
        "f_attr_n_observations": len(row_indices),
        "num_control_draws": num_draws,
        "num_observation_permutations": observation_draws,
    }
    alpha: float = (1.0 - confidence_level) / 2.0
    for target_name in TARGET_CURVES:
        point_depth_mean: float = float(np.mean(target_points_by_depth[target_name]))
        bootstrap_depth_mean: np.ndarray = np.mean(
            np.stack(target_bootstrap_by_depth[target_name], axis=0), axis=0
        )
        depth_mean_statistics: dict[str, float] = {
            "mean_pair_pearson_r2": point_depth_mean,
            "bootstrap_mean": float(np.mean(bootstrap_depth_mean)),
            "bootstrap_median": float(np.median(bootstrap_depth_mean)),
            "bootstrap_lower": float(np.quantile(bootstrap_depth_mean, alpha)),
            "bootstrap_upper": float(np.quantile(bootstrap_depth_mean, 1.0 - alpha)),
        }
        for statistic, value in depth_mean_statistics.items():
            depth_mean_row[f"{target_name}_{statistic}"] = value
    gap_depth_mean_statistics: dict[str, float] = _paired_difference_statistics(
        float(np.mean(target_points_by_depth[TARGET_CURVES[0]])),
        float(np.mean(target_points_by_depth[TARGET_CURVES[1]])),
        np.mean(np.stack(target_bootstrap_by_depth[TARGET_CURVES[0]], axis=0), axis=0),
        np.mean(np.stack(target_bootstrap_by_depth[TARGET_CURVES[1]], axis=0), axis=0),
        confidence_level,
    )
    for statistic, value in gap_depth_mean_statistics.items():
        depth_mean_row[f"{GAP_CURVE}_{statistic}"] = value
    for family in ANALYSIS_CONTROL_FAMILIES:
        draw_depth_mean: np.ndarray = np.mean(
            np.stack(control_draws_by_depth[family], axis=0), axis=0
        )
        pair_draw_depth_mean: np.ndarray = np.mean(
            np.stack(control_pair_draws_by_depth[family], axis=0), axis=0
        )
        for statistic, value in _randomization_statistics(draw_depth_mean).items():
            depth_mean_row[f"{family}_{statistic}"] = value
        depth_mean_row[f"{family}_comparison_target"] = CONTROL_COMPARISON_TARGET[
            family
        ]
        if family == OBSERVATION_PERMUTATION:
            depth_mean_row[f"{family}_randomization_unit"] = (
                "observation_pair_permutation_mean_over_pairs"
            )
        elif family.startswith("shuffled_"):
            depth_mean_row[f"{family}_randomization_unit"] = (
                "permutation_assignment_mean_over_draws_and_pairs"
            )
        else:
            depth_mean_row[f"{family}_randomization_unit"] = (
                "control_draw_mean_over_pairs"
            )
        draw_rows.extend(
            _control_draw_rows(
                summary_kind="equal_weight_depth_mean",
                benchmark=benchmark,
                pregrouper=pregrouper,
                scope=scope,
                relative_depth_index="",
                relative_depth="",
                family=family,
                pair_values=pair_draw_depth_mean,
                model_pairs=model_pairs,
            )
        )
    summary_rows.append(depth_mean_row)
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(draw_rows),
        inputs,
        num_draws,
        permutation_records,
    )


def _supporting_source_paths() -> dict[str, str]:
    root: str = os.path.dirname(os.path.dirname(__file__))
    names: tuple[str, ...] = (
        "benchmark_scripts/build_layer_control_spec.py",
        "benchmark_scripts/derived_provenance.py",
        "benchmark_scripts/f_table.py",
        "benchmark_scripts/layerwise_fidelity.py",
        "benchmark_scripts/provenance_sources.py",
        "benchmark_scripts/run_layer_controls.py",
        "surrogate/layer_control_scoring.py",
    )
    return {name: os.path.join(root, name) for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-results-dir", default="results")
    parser.add_argument("--control-results-dir", required=True)
    parser.add_argument("--control-spec", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--draw-output", default=None)
    parser.add_argument("--scope", choices=["all", "system", "user"], default="user")
    parser.add_argument("--depth-grid-size", type=int, default=DEFAULT_DEPTH_GRID_SIZE)
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument(
        "--confidence-level", type=float, default=DEFAULT_CONFIDENCE_LEVEL
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle-assignments", type=int, default=None)
    parser.add_argument(
        "--include-embedding",
        action="store_true",
        help="Use the paper-era all-slot depth axis, including the embedding slot.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output: str = args.output
    summary, draws, inputs, num_draws, permutation_records = analyze(
        args.source_results_dir,
        args.control_results_dir,
        args.control_spec,
        args.scope,
        args.depth_grid_size,
        args.bootstrap_resamples,
        args.confidence_level,
        args.seed,
        args.shuffle_assignments,
        args.include_embedding,
    )
    _atomic_write_tsv(summary, output)
    parameters: dict[str, Any] = {
        "benchmark": "boolq",
        "pregrouper": "sentence",
        "scope": args.scope,
        "models": list(OPEN_MODELS),
        "num_control_draws": num_draws,
        "shuffle_assignments": len(permutation_records),
        "shuffle_assignment_rng": "sha256_seeded_numpy_default_rng_per_model_v1",
        "shuffle_assignment_records": permutation_records,
        "control_families": list(ANALYSIS_CONTROL_FAMILIES),
        "observation_permutation": {
            "target": TARGET_CURVES[1],
            "unit": "flat_prompt_segment_coordinate",
            "draws": min(num_draws, DEFAULT_OBSERVATION_PERMUTATION_DRAWS),
            "draws_shared_across_depths_and_model_pairs": True,
            "seed": args.seed ^ 0x4F425350,
        },
        "target_curves": list(TARGET_CURVES),
        "control_comparison_target": CONTROL_COMPARISON_TARGET,
        "target_roles": {
            TARGET_CURVES[0]: "primary_prediction_fidelity_target",
            TARGET_CURVES[1]: "primary_attribution_fidelity_target",
            TARGET_CURVES[2]: "single_token_linear_attribution_diagnostic",
            TARGET_CURVES[3]: "grouped_alias_linear_attribution_diagnostic",
        },
        "derived_curves": {
            GAP_CURVE: {
                "definition": f"{TARGET_CURVES[0]}_minus_{TARGET_CURVES[1]}",
                "bootstrap": "paired_shared_prompt_resamples",
            }
        },
        "attribution_sign_convention": "interpolate_signed_then_correlate",
        "statistic": "mean_over_model_pairs_of_pearson_r_squared",
        "randomization_interval": [0.025, 0.975],
        "randomization_inference": "descriptive_only_no_p_values",
        "bootstrap_unit": "prompt_cluster",
        "bootstrap_resamples": args.bootstrap_resamples,
        "confidence_level": args.confidence_level,
        "bootstrap_seed": args.seed,
        "bootstrap_resamples_shared_across_depths_pairs_and_targets": True,
        "depth_grid_size": args.depth_grid_size,
        "depth_alignment": (
            "linear_interpolation_all_slots"
            if args.include_embedding
            else "linear_interpolation_block_outputs_only"
        ),
        "embedding_slot_included": args.include_embedding,
        "depth_mean_definition": (
            "arithmetic_mean_of_pointwise_pair_mean_r_squared_over_equally_"
            "spaced_relative_depth_grid"
        ),
        "r_squared_aggregation_interpretation": (
            "descriptive_pair_average_not_additive_explained_variance"
        ),
    }
    write_derived_provenance(
        output,
        generator_name="benchmark_scripts.analyze_layer_controls",
        generator_path=__file__,
        input_paths=inputs,
        parameters=parameters,
        root_dir=args.source_results_dir,
        supporting_source_paths=_supporting_source_paths(),
    )
    if args.draw_output is not None:
        _atomic_write_tsv(draws, args.draw_output)
        write_derived_provenance(
            args.draw_output,
            generator_name="benchmark_scripts.analyze_layer_controls",
            generator_path=__file__,
            input_paths=inputs,
            parameters={**parameters, "artifact_detail": "draw_level_diagnostic"},
            root_dir=args.source_results_dir,
            supporting_source_paths=_supporting_source_paths(),
        )
    logger.info("Wrote %d summary rows to %s", len(summary), output)


if __name__ == "__main__":
    main()
