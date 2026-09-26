# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Capture BoolQ layerwise random controls without rewriting main artifacts."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import logging
import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers

from benchmark_scripts.benchmark_config import (
    BENCHMARKS,
    BenchmarkSpec,
    load_benchmark_dataset,
    resolve_model_path,
)
from benchmark_scripts.provenance_sources import (
    canonical_file_hash_manifest_sha256,
    GOLD_OPEN_MODEL_REPOSITORIES,
)
from benchmark_scripts.run_layerwise import (
    _atomic_write_json,
    _character_length_permutation,
    _expected_manifest,
    _frame_sha256,
    _jsonable,
    _model_selection,
    _sha256_file,
    _tokenize_batch,
    _verified_model_identity,
)
from surrogate.layer_control_scoring import (
    grouped_logsumexp_contrasts,
    project_residual_deltas,
    seeded_isotropic_directions,
    summed_unembedding_group_contrast_norms,
    unit_unembedding_pair_directions,
)
from surrogate.layerwise_scoring import (
    capture_postnorm_residual_slots,
    ResidualSlotCapture,
)
from surrogate.model_types import Dialog, make_dialog
from surrogate.text_augmentation import (
    dialog_segments,
    PregrouperID,
    segment_and_ablate,
)
from surrogate.transformers_model import TransformersModel
from tqdm.auto import tqdm


logger: logging.Logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
PRODUCTION_DRAW_COUNT: int = 256
PRODUCTION_P9_POOL_SIZE: int = 278
PRODUCTION_P8_POOL_SIZE: int = 399
CANONICAL_BATCH_SIZE: int = 32
CANONICAL_POSITIVE_SURFACE: str = " true"
CANONICAL_NEGATIVE_SURFACE: str = " false"
LEGACY_BOOLQ_TOKEN_IDS_BY_MODEL: dict[str, tuple[int, int]] = {
    "qwen2.5-0.5b-instruct": (830, 895),
    "qwen2.5-3b-instruct": (830, 895),
    "qwen2.5-7b-instruct": (830, 895),
    "qwen2.5-14b-instruct": (830, 895),
    "llama-3.1-8b-instruct": (837, 905),
}
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
PRODUCTION_MODELS: frozenset[str] = frozenset(GOLD_OPEN_MODEL_REPOSITORIES)


@dataclass(frozen=True)
class BasisEntry:
    """One shared surface basis with IDs pinned for every declared model."""

    base_surface: str
    variants: tuple[str, ...]
    token_ids_by_model: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True)
class SelectedBasisPair:
    """One deterministic positive/negative basis pairing."""

    draw_idx: int
    positive_base_surface: str
    negative_base_surface: str


@dataclass(frozen=True)
class ControlSpec:
    """Validated random-control input specification."""

    namespace: str
    selection_algorithm: Any
    models: Mapping[str, Mapping[str, Any]]
    p9_basis_pool: tuple[BasisEntry, ...]
    p8_basis_pool: tuple[BasisEntry, ...]
    selected_paired_bases: tuple[SelectedBasisPair, ...]
    isotropic_seeds_by_model: Mapping[str, tuple[int, ...]]

    @property
    def num_draws(self) -> int:
        """Return the shared number of draws in every control family."""
        return len(self.selected_paired_bases)


@dataclass(frozen=True)
class ResolvedControls:
    """Model-specific tensors and JSON-safe direction metadata."""

    linear_unit_directions: torch.Tensor
    positive_group_ids: tuple[tuple[int, ...], ...]
    negative_group_ids: tuple[tuple[int, ...], ...]
    canonical_direction: Mapping[str, Any]
    shared_token_pairs: tuple[Mapping[str, Any], ...]
    grouped_pseudo_label_pairs: tuple[Mapping[str, Any], ...]
    isotropic: tuple[Mapping[str, Any], ...]


def _source_hashes() -> dict[str, str]:
    """Snapshot every public source dependency before control execution."""
    root: str = os.path.dirname(os.path.dirname(__file__))
    return {
        path: _sha256_file(os.path.join(root, path))
        for path in CONTROL_EXECUTION_SOURCE_FILES
    }


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_sha256(value: Any, field: str) -> str:
    text: str = _require_string(value, field)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload: bytes = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_selection_metadata(payload: Mapping[str, Any]) -> None:
    """Verify the builder's pool and self-hash selection contract."""
    selection: Any = payload["selection_algorithm"]
    if not isinstance(selection, dict):
        raise ValueError("selection_algorithm must be an object")
    required: set[str] = {
        "draw_count",
        "linear_surface_rule",
        "pool_sha256",
        "production_draw_count",
        "spec_hash_contract",
        "spec_sha256",
    }
    if not required.issubset(selection):
        raise ValueError("selection_algorithm metadata is incomplete")
    if (
        selection["draw_count"] != len(payload["selected_paired_bases"])
        or selection["production_draw_count"] != PRODUCTION_DRAW_COUNT
        or selection["linear_surface_rule"]
        != "single ASCII space + lowercase base_surface"
        or selection["spec_hash_contract"]
        != (
            "sha256_compact_sorted_utf8_json_with_selection_algorithm_"
            "spec_sha256_absent"
        )
    ):
        raise ValueError("selection_algorithm scalar contract disagrees")
    pool_hashes: Any = selection["pool_sha256"]
    if not isinstance(pool_hashes, dict) or pool_hashes != {
        "p8": _canonical_json_sha256(payload["p8_basis_pool"]),
        "p9": _canonical_json_sha256(payload["p9_basis_pool"]),
    }:
        raise ValueError("selection_algorithm pool hashes disagree")
    unhashed: dict[str, Any] = json.loads(
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
    )
    unhashed_selection: Any = unhashed["selection_algorithm"]
    if not isinstance(unhashed_selection, dict):
        raise ValueError("selection_algorithm must be an object")
    stored_hash: Any = unhashed_selection.pop("spec_sha256", None)
    if stored_hash != _canonical_json_sha256(unhashed):
        raise ValueError("selection_algorithm spec_sha256 disagrees")


def _parse_model_records(value: Any, canary: bool) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("models must be a nonempty object")
    names: set[str] = set(value)
    if not canary and names != PRODUCTION_MODELS:
        raise ValueError("production control specs require exactly five open models")
    if not names.issubset(PRODUCTION_MODELS):
        raise ValueError("control spec models must be canonical open models")
    expected_fields: set[str] = {
        "model_source",
        "model_revision",
        "model_artifact_manifest_sha256",
        "tokenizer_class",
        "tokenizer_files_sha256",
        "tokenizer_manifest_sha256",
        "vocabulary_size_including_added_tokens",
    }
    result: dict[str, dict[str, Any]] = {}
    for model_name, record_value in value.items():
        if not isinstance(record_value, dict) or set(record_value) != expected_fields:
            raise ValueError(f"model record fields disagree for {model_name!r}")
        record: dict[str, Any] = dict(record_value)
        _require_string(record["model_source"], "model_source")
        _require_string(record["model_revision"], "model_revision")
        _require_sha256(
            record["model_artifact_manifest_sha256"],
            "model_artifact_manifest_sha256",
        )
        _require_string(record["tokenizer_class"], "tokenizer_class")
        tokenizer_hashes: Any = record["tokenizer_files_sha256"]
        if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
            raise ValueError("tokenizer_files_sha256 must be a nonempty object")
        for filename, digest in tokenizer_hashes.items():
            _require_string(filename, "tokenizer filename")
            _require_sha256(digest, "tokenizer file SHA-256")
        expected_tokenizer_manifest: str = canonical_file_hash_manifest_sha256(
            tokenizer_hashes
        )
        if record["tokenizer_manifest_sha256"] != expected_tokenizer_manifest:
            raise ValueError(f"tokenizer manifest hash disagrees for {model_name!r}")
        vocabulary_size: int = _require_int(
            record["vocabulary_size_including_added_tokens"],
            "vocabulary_size_including_added_tokens",
        )
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size_including_added_tokens must be positive")
        result[model_name] = record
    return result


def _parse_basis_pool(
    value: Any,
    width: int,
    model_names: set[str],
    name: str,
) -> tuple[BasisEntry, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty list")
    result: list[BasisEntry] = []
    seen_bases: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {
            "base_surface",
            "variants",
            "token_ids_by_model",
        }:
            raise ValueError(f"{name}[{index}] fields disagree")
        base: str = _require_string(item["base_surface"], f"{name} base_surface")
        if base != base.strip() or base in seen_bases:
            raise ValueError(f"{name} base surfaces must be unique and unpadded")
        variants_value: Any = item["variants"]
        if not isinstance(variants_value, list) or len(variants_value) != width:
            raise ValueError(f"{name} variants must contain exactly {width} surfaces")
        variants: tuple[str, ...] = tuple(
            _require_string(surface, f"{name} variant") for surface in variants_value
        )
        if len(set(variants)) != width:
            raise ValueError(f"{name} variants must be unique")
        ids_value: Any = item["token_ids_by_model"]
        if not isinstance(ids_value, dict) or set(ids_value) != model_names:
            raise ValueError(f"{name} token-ID model keys disagree")
        ids_by_model: dict[str, tuple[int, ...]] = {}
        for model_name, ids_for_model in ids_value.items():
            if not isinstance(ids_for_model, list) or len(ids_for_model) != width:
                raise ValueError(f"{name} token IDs must have width {width}")
            token_ids: tuple[int, ...] = tuple(
                _require_int(token_id, f"{name} token ID") for token_id in ids_for_model
            )
            if any(token_id < 0 for token_id in token_ids):
                raise ValueError(f"{name} token IDs must be nonnegative")
            if len(set(token_ids)) != width:
                raise ValueError(f"{name} token IDs must be unique within a basis")
            ids_by_model[model_name] = token_ids
        seen_bases.add(base)
        result.append(
            BasisEntry(
                base_surface=base,
                variants=variants,
                token_ids_by_model=ids_by_model,
            )
        )
    return tuple(result)


def _parse_control_spec(payload: Any, canary: bool) -> ControlSpec:
    """Validate one complete shared random-control specification."""
    expected_fields: set[str] = {
        "schema_version",
        "namespace",
        "selection_algorithm",
        "models",
        "p9_basis_pool",
        "p8_basis_pool",
        "selected_paired_bases",
        "isotropic_seeds_by_model",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("control spec top-level fields disagree")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported control spec schema_version")
    namespace: str = _require_string(payload["namespace"], "namespace")
    _validate_selection_metadata(payload)
    selection_algorithm: Any = payload["selection_algorithm"]
    models: dict[str, dict[str, Any]] = _parse_model_records(payload["models"], canary)
    model_names: set[str] = set(models)
    p9_pool: tuple[BasisEntry, ...] = _parse_basis_pool(
        payload["p9_basis_pool"], 9, model_names, "p9_basis_pool"
    )
    p8_pool: tuple[BasisEntry, ...] = _parse_basis_pool(
        payload["p8_basis_pool"], 8, model_names, "p8_basis_pool"
    )
    if not canary and (
        len(p9_pool) != PRODUCTION_P9_POOL_SIZE
        or len(p8_pool) != PRODUCTION_P8_POOL_SIZE
    ):
        raise ValueError("production P9/P8 pools must contain 278/399 bases")
    selected_value: Any = payload["selected_paired_bases"]
    if not isinstance(selected_value, list) or not selected_value:
        raise ValueError("selected_paired_bases must be a nonempty list")
    if (not canary and len(selected_value) != PRODUCTION_DRAW_COUNT) or (
        canary and len(selected_value) > PRODUCTION_DRAW_COUNT
    ):
        raise ValueError("control draw count disagrees with canary/production mode")
    p9_by_base: dict[str, BasisEntry] = {item.base_surface: item for item in p9_pool}
    p8_by_base: dict[str, BasisEntry] = {item.base_surface: item for item in p8_pool}
    selected: list[SelectedBasisPair] = []
    seen_positive: set[str] = set()
    seen_negative: set[str] = set()
    for index, item in enumerate(selected_value):
        if not isinstance(item, dict) or set(item) != {
            "draw_idx",
            "positive_base_surface",
            "negative_base_surface",
        }:
            raise ValueError(f"selected_paired_bases[{index}] fields disagree")
        draw_idx: int = _require_int(item["draw_idx"], "draw_idx")
        positive: str = _require_string(
            item["positive_base_surface"], "positive_base_surface"
        )
        negative: str = _require_string(
            item["negative_base_surface"], "negative_base_surface"
        )
        if draw_idx != index:
            raise ValueError("selected draw_idx must equal list position")
        if positive not in p9_by_base or negative not in p8_by_base:
            raise ValueError("selected basis is absent from its full pool")
        if positive in seen_positive or negative in seen_negative:
            raise ValueError("selected bases must be sampled without replacement")
        positive_entry: BasisEntry = p9_by_base[positive]
        negative_entry: BasisEntry = p8_by_base[negative]
        if set(positive_entry.variants) & set(negative_entry.variants):
            raise ValueError("selected positive/negative surface groups overlap")
        for model_name in model_names:
            if set(positive_entry.token_ids_by_model[model_name]) & set(
                negative_entry.token_ids_by_model[model_name]
            ):
                raise ValueError("selected positive/negative token groups overlap")
        seen_positive.add(positive)
        seen_negative.add(negative)
        selected.append(SelectedBasisPair(draw_idx, positive, negative))
    seeds_value: Any = payload["isotropic_seeds_by_model"]
    if not isinstance(seeds_value, dict) or set(seeds_value) != model_names:
        raise ValueError("isotropic seed model keys disagree")
    seeds_by_model: dict[str, tuple[int, ...]] = {}
    all_seeds: set[int] = set()
    for model_name, model_seeds_value in seeds_value.items():
        if not isinstance(model_seeds_value, list) or len(model_seeds_value) != len(
            selected
        ):
            raise ValueError("isotropic seed count must equal control draw count")
        model_seeds: tuple[int, ...] = tuple(
            _require_int(seed, "isotropic seed") for seed in model_seeds_value
        )
        if any(seed < 0 or seed > (2**63 - 1) for seed in model_seeds):
            raise ValueError("isotropic seeds must be signed-64-bit nonnegative")
        if len(set(model_seeds)) != len(model_seeds):
            raise ValueError("isotropic seeds must be distinct within each model")
        if all_seeds & set(model_seeds):
            raise ValueError("isotropic seeds must be distinct across models")
        all_seeds.update(model_seeds)
        seeds_by_model[model_name] = model_seeds
    return ControlSpec(
        namespace=namespace,
        selection_algorithm=selection_algorithm,
        models=models,
        p9_basis_pool=p9_pool,
        p8_basis_pool=p8_pool,
        selected_paired_bases=tuple(selected),
        isotropic_seeds_by_model=seeds_by_model,
    )


def _load_control_spec(path: str, canary: bool) -> tuple[ControlSpec, str]:
    with open(path, encoding="utf-8") as source:
        payload: Any = json.load(source)
    return _parse_control_spec(payload, canary), _sha256_file(path)


def _encode_single_token(tokenizer: Any, surface: str) -> int:
    encoded_value: Any = tokenizer.encode(surface, add_special_tokens=False)
    encoded: list[int] = [int(value) for value in encoded_value]
    if len(encoded) != 1:
        raise ValueError(f"control surface {surface!r} is not exactly one token")
    return encoded[0]


def _validate_spec_model_identity(
    spec: ControlSpec,
    model_name: str,
    model_source: str,
    model_revision: str,
    artifact_hashes: Mapping[str, str],
    artifact_manifest_sha256: str,
    tokenizer: Any,
) -> None:
    """Bind the input spec's tokenizer and model identity to loaded bytes."""
    record: Mapping[str, Any] = spec.models[model_name]
    if (
        record["model_source"] != model_source
        or record["model_revision"] != model_revision
        or record["model_artifact_manifest_sha256"] != artifact_manifest_sha256
    ):
        raise ValueError(f"control spec model identity disagrees for {model_name}")
    tokenizer_hashes: Mapping[str, str] = record["tokenizer_files_sha256"]
    actual_tokenizer_hashes: dict[str, str] = {
        filename: artifact_hashes.get(filename, "") for filename in tokenizer_hashes
    }
    if actual_tokenizer_hashes != tokenizer_hashes:
        raise ValueError(f"control spec tokenizer hashes disagree for {model_name}")
    if type(tokenizer).__name__ != record["tokenizer_class"]:
        raise ValueError(f"control spec tokenizer class disagrees for {model_name}")
    if len(tokenizer) != record["vocabulary_size_including_added_tokens"]:
        raise ValueError(f"control spec vocabulary size disagrees for {model_name}")


def _resolve_basis_ids(
    tokenizer: Any,
    entries: Sequence[BasisEntry],
    model_name: str,
    name: str,
) -> dict[str, tuple[int, ...]]:
    """Verify every pre-resolved pool token ID against the loaded tokenizer."""
    result: dict[str, tuple[int, ...]] = {}
    for entry in entries:
        expected: tuple[int, ...] = entry.token_ids_by_model[model_name]
        observed: tuple[int, ...] = tuple(
            _encode_single_token(tokenizer, surface) for surface in entry.variants
        )
        if observed != expected:
            raise ValueError(
                f"{name} token IDs disagree for {entry.base_surface!r} in {model_name}"
            )
        result[entry.base_surface] = observed
    return result


def _tensor_row_sha256(value: torch.Tensor) -> str:
    array: np.ndarray = value.detach().cpu().numpy().astype(np.dtype("<f4"), copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _resolve_controls(
    spec: ControlSpec,
    model_name: str,
    tokenizer: Any,
    lm_head: Any,
) -> ResolvedControls:
    """Resolve and validate all model-specific control directions."""
    p9_by_base: dict[str, BasisEntry] = {
        item.base_surface: item for item in spec.p9_basis_pool
    }
    p8_by_base: dict[str, BasisEntry] = {
        item.base_surface: item for item in spec.p8_basis_pool
    }
    p9_ids: dict[str, tuple[int, ...]] = _resolve_basis_ids(
        tokenizer, spec.p9_basis_pool, model_name, "P9"
    )
    p8_ids: dict[str, tuple[int, ...]] = _resolve_basis_ids(
        tokenizer, spec.p8_basis_pool, model_name, "P8"
    )
    canonical_positive_id: int = _encode_single_token(
        tokenizer, CANONICAL_POSITIVE_SURFACE
    )
    canonical_negative_id: int = _encode_single_token(
        tokenizer, CANONICAL_NEGATIVE_SURFACE
    )
    if (canonical_positive_id, canonical_negative_id) != (
        LEGACY_BOOLQ_TOKEN_IDS_BY_MODEL[model_name]
    ):
        raise ValueError(
            f"legacy BoolQ token IDs disagree for {model_name}: "
            f"{(canonical_positive_id, canonical_negative_id)}"
        )
    shared_surfaces: list[tuple[str, str]] = []
    shared_ids: list[tuple[int, int]] = []
    positive_group_ids: list[tuple[int, ...]] = []
    negative_group_ids: list[tuple[int, ...]] = []
    for pair in spec.selected_paired_bases:
        positive_entry: BasisEntry = p9_by_base[pair.positive_base_surface]
        negative_entry: BasisEntry = p8_by_base[pair.negative_base_surface]
        positive_surface: str = f" {positive_entry.base_surface.lower()}"
        negative_surface: str = f" {negative_entry.base_surface.lower()}"
        if positive_surface not in positive_entry.variants:
            raise ValueError("P9 basis lacks its leading-space lowercase variant")
        if negative_surface not in negative_entry.variants:
            raise ValueError("P8 basis lacks its leading-space lowercase variant")
        positive_id: int = _encode_single_token(tokenizer, positive_surface)
        negative_id: int = _encode_single_token(tokenizer, negative_surface)
        expected_positive_id: int = p9_ids[positive_entry.base_surface][
            positive_entry.variants.index(positive_surface)
        ]
        expected_negative_id: int = p8_ids[negative_entry.base_surface][
            negative_entry.variants.index(negative_surface)
        ]
        if (positive_id, negative_id) != (
            expected_positive_id,
            expected_negative_id,
        ):
            raise ValueError("linear control IDs disagree with the resolved pools")
        if positive_id == negative_id:
            raise ValueError("linear control token IDs must differ")
        shared_surfaces.append((positive_surface, negative_surface))
        shared_ids.append((positive_id, negative_id))
        positive_group_ids.append(p9_ids[positive_entry.base_surface])
        negative_group_ids.append(p8_ids[negative_entry.base_surface])
    all_linear_ids: list[tuple[int, int]] = [
        (canonical_positive_id, canonical_negative_id),
        *shared_ids,
    ]
    if len({frozenset(pair) for pair in all_linear_ids}) != len(all_linear_ids):
        raise ValueError(
            "resolved linear control pairs contain target duplicates or reversals"
        )

    pair_directions, pair_norms = unit_unembedding_pair_directions(
        lm_head,
        all_linear_ids,
    )
    hidden_size: int = int(pair_directions.shape[1])
    isotropic_directions: torch.Tensor = seeded_isotropic_directions(
        hidden_size, spec.isotropic_seeds_by_model[model_name]
    )
    linear_directions: torch.Tensor = torch.cat(
        [
            pair_directions,
            isotropic_directions.to(device=pair_directions.device),
        ],
        dim=0,
    )
    group_norms: torch.Tensor = summed_unembedding_group_contrast_norms(
        lm_head, positive_group_ids, negative_group_ids
    )
    shared_metadata: list[dict[str, Any]] = []
    grouped_metadata: list[dict[str, Any]] = []
    isotropic_metadata: list[dict[str, Any]] = []
    for index, pair in enumerate(spec.selected_paired_bases):
        positive_entry = p9_by_base[pair.positive_base_surface]
        negative_entry = p8_by_base[pair.negative_base_surface]
        positive_surface, negative_surface = shared_surfaces[index]
        positive_id, negative_id = shared_ids[index]
        shared_metadata.append(
            {
                "draw_idx": index,
                "positive_surface": positive_surface,
                "negative_surface": negative_surface,
                "positive_token_id": positive_id,
                "negative_token_id": negative_id,
                "pre_normalization_norm": float(pair_norms[index + 1].item()),
            }
        )
        grouped_metadata.append(
            {
                "draw_idx": index,
                "positive_surfaces": list(positive_entry.variants),
                "negative_surfaces": list(negative_entry.variants),
                "positive_token_ids": list(positive_group_ids[index]),
                "negative_token_ids": list(negative_group_ids[index]),
                "pre_normalization_norm": float(group_norms[index].item()),
            }
        )
        isotropic_metadata.append(
            {
                "draw_idx": index,
                "seed": spec.isotropic_seeds_by_model[model_name][index],
                "vector_sha256": _tensor_row_sha256(isotropic_directions[index]),
            }
        )
    return ResolvedControls(
        linear_unit_directions=linear_directions,
        positive_group_ids=tuple(positive_group_ids),
        negative_group_ids=tuple(negative_group_ids),
        canonical_direction={
            "positive_label": "true",
            "negative_label": "false",
            "positive_surface": CANONICAL_POSITIVE_SURFACE,
            "negative_surface": CANONICAL_NEGATIVE_SURFACE,
            "positive_token_id": canonical_positive_id,
            "negative_token_id": canonical_negative_id,
            "pre_normalization_norm": float(pair_norms[0].item()),
        },
        shared_token_pairs=tuple(shared_metadata),
        grouped_pseudo_label_pairs=tuple(grouped_metadata),
        isotropic=tuple(isotropic_metadata),
    )


class ControlArtifactWriter:
    """Atomically stream projections into a manifest-ordered NPY artifact."""

    def __init__(
        self,
        output_path: str,
        manifest_keys: Sequence[tuple[int, int]],
        layer_slots: int,
        columns: int,
    ) -> None:
        if layer_slots <= 0 or columns <= 0:
            raise ValueError("layer_slots and columns must be positive")
        keys: list[tuple[int, int]] = list(manifest_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("manifest control keys must be nonempty and unique")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            dir=os.path.dirname(output_path),
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        self.output_path: str = output_path
        self.temporary_path: str = temporary_path
        self.shape: tuple[int, int, int] = (len(keys), layer_slots, columns)
        self._index_by_key: dict[tuple[int, int], int] = {
            key: index for index, key in enumerate(keys)
        }
        self._coverage: np.ndarray = np.zeros(len(keys), dtype=np.bool_)
        self._array: np.memmap | None = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.dtype("<f4"),
            shape=self.shape,
            fortran_order=False,
        )

    def write(
        self,
        prompt_idx: int,
        segment_indices: Sequence[int],
        projections: torch.Tensor,
    ) -> None:
        """Write one ablation batch, rejecting duplicates and shape drift."""
        if self._array is None:
            raise RuntimeError("control artifact writer is already closed")
        segment_values: list[int] = [int(value) for value in segment_indices]
        expected_shape: tuple[int, int, int] = (
            self.shape[1],
            len(segment_values),
            self.shape[2],
        )
        if tuple(projections.shape) != expected_shape:
            raise ValueError(
                f"projection shape {tuple(projections.shape)} differs from "
                f"expected {expected_shape}"
            )
        if projections.dtype != torch.float32 or not bool(
            torch.isfinite(projections).all()
        ):
            raise ValueError("control projections must be finite FP32 values")
        row_indices: list[int] = []
        for segment_idx in segment_values:
            key: tuple[int, int] = (prompt_idx, segment_idx)
            if key not in self._index_by_key:
                raise ValueError(f"control row is outside the manifest: {key}")
            row_index: int = self._index_by_key[key]
            if bool(self._coverage[row_index]):
                raise ValueError(f"control row was written twice: {key}")
            row_indices.append(row_index)
        if len(set(row_indices)) != len(row_indices):
            raise ValueError("control batch contains duplicate manifest rows")
        values: np.ndarray = (
            projections.detach().cpu().numpy().astype(np.dtype("<f4"), copy=False)
        ).transpose(1, 0, 2)
        self._array[row_indices, :, :] = values
        self._coverage[row_indices] = True

    def finalize(self) -> None:
        """Flush and atomically publish a complete exactly-once artifact."""
        if self._array is None:
            raise RuntimeError("control artifact writer is already closed")
        missing: int = int((~self._coverage).sum())
        if missing:
            raise RuntimeError(f"control artifact is missing {missing} manifest rows")
        self._array.flush()
        self._array = None
        with open(self.temporary_path, "rb") as artifact:
            os.fsync(artifact.fileno())
        os.replace(self.temporary_path, self.output_path)

    def abort(self) -> None:
        """Discard only the unpublished temporary artifact."""
        if self._array is not None:
            self._array.flush()
            self._array = None
        if os.path.exists(self.temporary_path):
            os.remove(self.temporary_path)


def _validate_existing_layer_binding(
    source_directory: str,
    benchmark: str,
    pregrouper: str,
    model_name: str,
    manifest_path: str,
    normalized_frame_sha256: str,
    model_revision: str,
    model_artifact_manifest_sha256: str,
    model_identity_hashes: Mapping[str, str],
) -> tuple[str, str, dict[str, Any]]:
    """Require exact main layer artifacts before a control-only run."""
    layer_path: str = os.path.join(source_directory, f"{model_name}_layers.tsv.gz")
    run_path: str = os.path.join(source_directory, f"{model_name}_layers_run.json")
    for path in (layer_path, run_path, manifest_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    with open(run_path, encoding="utf-8") as source:
        metadata: Any = json.load(source)
    if not isinstance(metadata, dict):
        raise ValueError(f"main layer sidecar is not an object: {run_path}")
    artifact: Any = metadata.get("artifact")
    if (
        not isinstance(artifact, dict)
        or artifact.get("filename") != os.path.basename(layer_path)
        or artifact.get("sha256") != _sha256_file(layer_path)
    ):
        raise ValueError(f"main layer artifact hash disagrees in {run_path}")
    if (
        metadata.get("schema_version") != 1
        or metadata.get("benchmark") != benchmark
        or metadata.get("pregrouper") != pregrouper
        or metadata.get("model") != model_name
        or metadata.get("manifest_sha256") != _sha256_file(manifest_path)
    ):
        raise ValueError(f"main layer identity disagrees in {run_path}")
    dataset: Any = metadata.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("normalized_frame_sha256") != normalized_frame_sha256
    ):
        raise ValueError(f"main layer dataset identity disagrees in {run_path}")
    if (
        metadata.get("model_revision") != model_revision
        or metadata.get("model_artifact_manifest_sha256")
        != model_artifact_manifest_sha256
        or metadata.get("model_identity_files_sha256") != model_identity_hashes
    ):
        raise ValueError(f"main layer model identity disagrees in {run_path}")
    layer_slots: Any = metadata.get("layer_slots")
    if (
        not isinstance(layer_slots, dict)
        or not isinstance(layer_slots.get("count"), int)
        or layer_slots["count"] <= 0
        or not isinstance(layer_slots.get("convention"), str)
    ):
        raise ValueError(f"main layer slot metadata is invalid in {run_path}")
    return layer_path, run_path, metadata


def _manifest_keys(path: str, expected: pd.DataFrame) -> list[tuple[int, int]]:
    existing: pd.DataFrame = pd.read_csv(
        path, sep="\t", keep_default_na=False, na_values=[""]
    )
    try:
        pd.testing.assert_frame_equal(
            existing.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(
            f"source manifest disagrees with control run: {path}"
        ) from error
    keys: list[tuple[int, int]] = [
        (int(row.prompt_idx), int(row.seg_idx))
        for row in existing.itertuples(index=False)
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("source manifest contains duplicate segment keys")
    return keys


def _capture(
    model: TransformersModel,
    texts: Sequence[str],
) -> ResidualSlotCapture:
    input_ids, attention_mask = _tokenize_batch(model, texts)
    return capture_postnorm_residual_slots(model._model, input_ids, attention_mask)


def _assemble_control_projections(
    linear: torch.Tensor,
    original_grouped: torch.Tensor,
    perturbed_grouped: torch.Tensor,
) -> torch.Tensor:
    """Assemble canonical, shared, grouped, and isotropic columns in order."""
    if original_grouped.ndim != 3 or perturbed_grouped.ndim != 3:
        raise ValueError("grouped control tensors must have three dimensions")
    if original_grouped.shape[1] != 1:
        raise ValueError("original grouped controls must have singleton batch")
    if (
        original_grouped.shape[0] != perturbed_grouped.shape[0]
        or original_grouped.shape[2] != perturbed_grouped.shape[2]
    ):
        raise ValueError("original and perturbed grouped control shapes disagree")
    draw_count: int = int(original_grouped.shape[2])
    expected_linear_shape: tuple[int, int, int] = (
        int(perturbed_grouped.shape[0]),
        int(perturbed_grouped.shape[1]),
        1 + 2 * draw_count,
    )
    if tuple(linear.shape) != expected_linear_shape:
        raise ValueError(
            f"linear control shape {tuple(linear.shape)} differs from "
            f"expected {expected_linear_shape}"
        )
    grouped_attribution: torch.Tensor = (original_grouped - perturbed_grouped).float()
    return torch.cat(
        [
            linear[..., : draw_count + 1],
            grouped_attribution,
            linear[..., draw_count + 1 :],
        ],
        dim=-1,
    )


async def _capture_controls(
    model: TransformersModel,
    dialogs: Sequence[Dialog],
    prompts_meta: Sequence[Mapping[str, Any]],
    pregrouper: PregrouperID,
    batch_size: int,
    controls: ResolvedControls,
    writer: ControlArtifactWriter,
) -> None:
    """Project every segment delta online and stream it to the writer."""
    for dialog, prompt_meta in tqdm(
        list(zip(dialogs, prompts_meta)), desc=f"Layer controls: {model.model_name}"
    ):
        prompt_idx: int = int(prompt_meta["prompt_idx"])
        original_text: str = model.dialog_to_text(dialog)
        original_capture: ResidualSlotCapture = _capture(model, [original_text])
        original_grouped: torch.Tensor = grouped_logsumexp_contrasts(
            original_capture,
            model._model.lm_head,
            controls.positive_group_ids,
            controls.negative_group_ids,
        )
        ablated_dialogs: list[Dialog] = await segment_and_ablate(dialog, pregrouper)
        segments = dialog_segments(dialog, pregrouper)
        if len(ablated_dialogs) != len(segments):
            raise RuntimeError(
                f"segment/ablation count mismatch for prompt {prompt_idx}"
            )
        ablated_texts: list[str] = [
            model.dialog_to_text(ablated) for ablated in ablated_dialogs
        ]
        permutation: list[int] = _character_length_permutation(ablated_texts)
        for start in range(0, len(permutation), batch_size):
            positions: list[int] = permutation[start : start + batch_size]
            texts: list[str] = [ablated_texts[index] for index in positions]
            perturbed_capture: ResidualSlotCapture = _capture(model, texts)
            linear: torch.Tensor = project_residual_deltas(
                original_capture,
                perturbed_capture,
                controls.linear_unit_directions,
            )
            perturbed_grouped: torch.Tensor = grouped_logsumexp_contrasts(
                perturbed_capture,
                model._model.lm_head,
                controls.positive_group_ids,
                controls.negative_group_ids,
            )
            projections: torch.Tensor = _assemble_control_projections(
                linear, original_grouped, perturbed_grouped
            )
            writer.write(
                prompt_idx,
                [segments[index].segment_idx for index in positions],
                projections,
            )
            del perturbed_capture, perturbed_grouped, projections, linear
        del original_capture, original_grouped


async def run_layer_controls(
    control_spec_path: str,
    source_results_dir: str,
    output_dir: str,
    benchmark_name: str = "boolq",
    pregrouper: PregrouperID = "sentence",
    batch_size: int = CANONICAL_BATCH_SIZE,
    max_samples: int | None = None,
    seed: int = 42,
    model_set: str = "Qwen2.5-Instruct",
    models: str | None = None,
    dataset_file: str | None = None,
    canary: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """Run the control-only BoolQ capture against existing main artifacts."""
    execution_source_sha256: dict[str, str] = _source_hashes()
    spec, control_spec_sha256 = _load_control_spec(control_spec_path, canary)
    if benchmark_name != "boolq" or pregrouper != "sentence":
        raise ValueError("layer random controls currently require BoolQ sentence")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not canary and batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"production controls require batch_size={CANONICAL_BATCH_SIZE}"
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if canary and max_samples is None:
        raise ValueError("canary controls require max_samples")
    if not canary and max_samples is not None:
        raise ValueError("production controls require the complete dataset")
    benchmark: BenchmarkSpec = BENCHMARKS[benchmark_name]
    frame: pd.DataFrame = load_benchmark_dataset(benchmark, dataset_file=dataset_file)
    if max_samples is not None and max_samples < len(frame):
        generator: np.random.Generator = np.random.default_rng(seed)
        positions: np.ndarray = generator.choice(
            len(frame), size=max_samples, replace=False
        )
        positions.sort()
        frame = frame.iloc[positions]
    if benchmark.eval_config is None:
        raise ValueError("BoolQ must define an evaluation configuration")
    system_prompt: str = (
        benchmark.system_prompt_override
        if benchmark.system_prompt_override is not None
        else benchmark.eval_config.system_prompt
    )
    dialogs: list[Dialog] = [
        make_dialog(system_prompt, benchmark.prompt_builder(row))
        for _, row in frame.iterrows()
    ]
    prompts_meta: list[dict[str, Any]] = [
        {
            "prompt_idx": int(source_idx),
            "answer": _jsonable(row[benchmark.answer_column]),
        }
        for source_idx, row in frame.iterrows()
    ]
    source_directory: str = os.path.join(source_results_dir, benchmark_name, pregrouper)
    manifest_path: str = os.path.join(source_directory, "segments.tsv.gz")
    expected_manifest: pd.DataFrame = _expected_manifest(
        dialogs, prompts_meta, pregrouper
    )
    manifest_keys: list[tuple[int, int]] = _manifest_keys(
        manifest_path, expected_manifest
    )
    normalized_frame_sha256: str = _frame_sha256(frame)
    selected_models: list[tuple[str, str]] = _model_selection(model_set, models)
    if not {name for name, _ in selected_models}.issubset(set(spec.models)):
        raise ValueError("selected models are absent from the control spec")
    destination_directory: str = os.path.join(output_dir, benchmark_name, pregrouper)
    os.makedirs(destination_directory, exist_ok=True)

    for model_name, model_source in selected_models:
        artifact_path: str = os.path.join(
            destination_directory, f"{model_name}_layer_controls.npy"
        )
        sidecar_path: str = os.path.join(
            destination_directory, f"{model_name}_layer_controls_run.json"
        )
        existing: list[str] = [
            path for path in (artifact_path, sidecar_path) if os.path.exists(path)
        ]
        if existing and not overwrite_existing:
            raise FileExistsError(
                f"refusing to overwrite controls without --overwrite-existing: {existing}"
            )
        model_path: str = resolve_model_path(model_source)
        (
            model_revision,
            model_artifact_hashes,
            model_artifact_manifest_sha256,
            model_identity_hashes,
        ) = _verified_model_identity(model_name, model_source, model_path)
        layer_path, layer_run_path, layer_metadata = _validate_existing_layer_binding(
            source_directory,
            benchmark_name,
            pregrouper,
            model_name,
            manifest_path,
            normalized_frame_sha256,
            model_revision,
            model_artifact_manifest_sha256,
            model_identity_hashes,
        )
        model: TransformersModel = TransformersModel(
            model_name=model_name,
            model_path=model_path,
            attn_implementation="sdpa",
        ).load()
        writer: ControlArtifactWriter | None = None
        try:
            _validate_spec_model_identity(
                spec,
                model_name,
                model_source,
                model_revision,
                model_artifact_hashes,
                model_artifact_manifest_sha256,
                model._tokenizer,
            )
            controls: ResolvedControls = _resolve_controls(
                spec, model_name, model._tokenizer, model._model.lm_head
            )
            layer_slots: Mapping[str, Any] = layer_metadata["layer_slots"]
            expected_slots: int = len(model._model.model.layers) + 1
            if layer_slots["count"] != expected_slots:
                raise ValueError("loaded model depth differs from main layer sidecar")
            columns: int = 1 + 3 * spec.num_draws
            writer = ControlArtifactWriter(
                artifact_path,
                manifest_keys,
                expected_slots,
                columns,
            )
            await _capture_controls(
                model,
                dialogs,
                prompts_meta,
                pregrouper,
                batch_size,
                controls,
                writer,
            )
            writer.finalize()
            writer = None
            artifact_shape: list[int] = [len(manifest_keys), expected_slots, columns]
            control_spec_metadata: dict[str, Any] = {
                "namespace": spec.namespace,
                "selection_algorithm": spec.selection_algorithm,
                "input_spec_filename": os.path.basename(control_spec_path),
                "input_spec_sha256": control_spec_sha256,
                "num_draws": spec.num_draws,
                "row_order": "segments_manifest_file_order",
                "column_layout": {
                    "legacy_true_false_linear": 0,
                    "shared_token_pair_start": 1,
                    "grouped_pseudo_label_start": 1 + spec.num_draws,
                    "isotropic_start": 1 + 2 * spec.num_draws,
                    "total": columns,
                },
                "projection_definition": (
                    "signed_per_segment_postnorm_delta_dot_unit_direction"
                ),
                "grouped_projection_definition": (
                    "signed_original_minus_ablated_grouped_logsumexp_contrast"
                ),
                "absolute_attribution": False,
                "canonical_direction": dict(controls.canonical_direction),
                "shared_token_pairs": [
                    dict(item) for item in controls.shared_token_pairs
                ],
                "grouped_pseudo_label_pairs": [
                    dict(item) for item in controls.grouped_pseudo_label_pairs
                ],
                "isotropic": [dict(item) for item in controls.isotropic],
            }
            metadata: dict[str, Any] = {
                "artifact_type": "layer_random_control_projections",
                "schema_version": SCHEMA_VERSION,
                "artifact": {
                    "filename": os.path.basename(artifact_path),
                    "shape": artifact_shape,
                    "dtype": "float32",
                    "byte_size": os.path.getsize(artifact_path),
                    "sha256": _sha256_file(artifact_path),
                },
                "benchmark": benchmark_name,
                "pregrouper": pregrouper,
                "segmentation_scope": "full_dialog_in_message_order",
                "model": model_name,
                "model_source": model_source,
                "model_revision": model_revision,
                "model_artifact_manifest_sha256": (model_artifact_manifest_sha256),
                "model_identity_files_sha256": model_identity_hashes,
                "manifest_sha256": _sha256_file(manifest_path),
                "source_layer_artifact": {
                    "filename": os.path.basename(layer_path),
                    "sha256": _sha256_file(layer_path),
                },
                "source_layer_run": {
                    "filename": os.path.basename(layer_run_path),
                    "sha256": _sha256_file(layer_run_path),
                },
                "layer_slots": dict(layer_slots),
                "control_spec": control_spec_metadata,
                "parameters": {
                    "attention_implementation": "sdpa",
                    "batch_size": batch_size,
                    "canary": canary,
                    "device_map": "auto",
                    "max_samples": max_samples,
                    "rendered_chat_add_special_tokens": False,
                    "seed": seed,
                    "torch_dtype": "bfloat16",
                },
                "software": {
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "torch": str(torch.__version__),
                    "transformers": transformers.__version__,
                    "cuda_runtime": torch.version.cuda,
                },
                "source_hash_timing": "run_start",
                "source_sha256": execution_source_sha256,
            }
            _atomic_write_json(sidecar_path, metadata)
            logger.info("Saved %s (%s)", artifact_path, artifact_shape)
        except Exception:
            if writer is not None:
                writer.abort()
            raise
        finally:
            model.unload()


def main() -> None:
    """Parse CLI arguments and run control-only capture."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--control-spec", required=True)
    parser.add_argument("--source-results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark", default="boolq", choices=["boolq"])
    parser.add_argument("--pregrouper", default="sentence", choices=["sentence"])
    parser.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-set", default="Qwen2.5-Instruct")
    parser.add_argument("--models")
    parser.add_argument("--dataset-file")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    asyncio.run(
        run_layer_controls(
            control_spec_path=args.control_spec,
            source_results_dir=args.source_results_dir,
            output_dir=args.output_dir,
            benchmark_name=args.benchmark,
            pregrouper=args.pregrouper,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            seed=args.seed,
            model_set=args.model_set,
            models=args.models,
            dataset_file=args.dataset_file,
            canary=args.canary,
            overwrite_existing=args.overwrite_existing,
        )
    )


if __name__ == "__main__":
    main()
