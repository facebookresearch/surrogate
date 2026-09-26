# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Generate matched-relative-depth fidelity statistics from layer artifacts.

Each input is ``results/<benchmark>/<pregrouper>/<model>_layers.tsv[.gz]``.
The canonical analysis linearly interpolates decoder-block outputs onto a
shared grid.  The embedding slot is validated and retained in the source
artifact, but deliberately excluded from interpolation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from benchmark_scripts.benchmark_config import MODEL_SETS
from benchmark_scripts.derived_provenance import sha256_file, write_derived_provenance
from benchmark_scripts.f_table import (
    OPEN_MODELS,
    _analysis_rng,
    _contrast_metadata,
    _emit_pair_corrs,
    _filter_segment_frame,
    _resolved_scope,
    _scope_keys,
)
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REVISIONS,
    LAYER_EXECUTION_SOURCE_FILES,
    canonical_file_hash_manifest_sha256,
)
from surrogate.eval_constants import label_column_alias

logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_CONFIGS: tuple[tuple[str, str], ...] = (
    ("boolq", "sentence"),
    ("anli_r1", "sentence"),
    ("anli_r2", "sentence"),
    ("anli_r3", "sentence"),
)
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "prompt_idx",
        "seg_idx",
        "kind",
        "answer",
        "layer_slot",
        "layer_kind",
        "block_idx",
        "delta_norm_postnorm",
    }
)
SHA256_PATTERN: re.Pattern[str] = re.compile(r"[0-9a-f]{64}")
EXPECTED_MODEL_SOURCES: dict[str, str] = {
    name: source for model_set in MODEL_SETS.values() for name, source in model_set
}


@dataclass(frozen=True)
class FinalLayerReadout:
    """Sidecar-validated final-block signals for one model and contrast."""

    alignment: pd.Series
    prediction: pd.Series
    attribution: pd.Series
    layer_path: str
    manifest_path: str
    sidecar_path: str


def _sanitize_label(label: str) -> str:
    """Return the column-safe label spelling used by layer artifacts."""
    return label_column_alias(label)


def _contrast_labels(benchmark: str, contrast: str) -> tuple[str, str]:
    """Resolve a public contrast name to its positive and negative labels."""
    if benchmark == "boolq" and contrast == "canonical":
        return "true", "false"
    if benchmark.startswith("anli_"):
        if contrast == "canonical" or contrast == "entailment_neutral":
            return "entailment", "neutral"
        if contrast == "entailment_contradiction":
            return "entailment", "contradiction"
    raise ValueError(f"Contrast {contrast!r} is not defined for {benchmark!r}")


def _coerce_integer_column(frame: pd.DataFrame, column: str, nullable: bool) -> None:
    """Validate and normalize an integer-valued identity column in place."""
    values: pd.Series = pd.to_numeric(frame[column], errors="coerce")
    if (not nullable and values.isna().any()) or ((values.dropna() % 1 != 0).any()):
        raise ValueError(f"{column} must contain integer values")
    frame[column] = values.astype("Int64")


def _validate_layer_frame(
    frame: pd.DataFrame,
    path: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    """Validate identities and complete layer grids in one model artifact."""
    score_columns: set[str] = {
        f"label_score_{_sanitize_label(positive_label)}",
        f"label_score_{_sanitize_label(negative_label)}",
    }
    missing: set[str] = (REQUIRED_COLUMNS | score_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns {sorted(missing)}")
    checked: pd.DataFrame = frame.copy()
    for column, nullable in (
        ("prompt_idx", False),
        ("seg_idx", True),
        ("layer_slot", False),
        ("block_idx", True),
    ):
        _coerce_integer_column(checked, column, nullable)
    for column in score_columns:
        checked[column] = pd.to_numeric(checked[column], errors="raise")
    if checked.empty:
        raise ValueError(f"{path} contains no layer rows")
    if not set(checked["kind"]).issubset({"orig", "ablated"}):
        raise ValueError(f"{path} kind must be 'orig' or 'ablated'")
    invalid_seg: pd.Series = (
        (checked["kind"] == "orig") & checked["seg_idx"].notna()
    ) | ((checked["kind"] == "ablated") & checked["seg_idx"].isna())
    if invalid_seg.any():
        raise ValueError(f"{path} has invalid seg_idx values for kind")
    identity: list[str] = ["prompt_idx", "seg_idx", "kind", "layer_slot"]
    if checked.duplicated(identity).any():
        raise ValueError(f"{path} has duplicate layer observation keys")

    layer_metadata: pd.DataFrame = checked[
        ["layer_slot", "layer_kind", "block_idx"]
    ].drop_duplicates()
    if layer_metadata["layer_slot"].duplicated().any():
        raise ValueError(f"{path} has conflicting metadata for a layer slot")
    embeddings: pd.DataFrame = layer_metadata[
        layer_metadata["layer_kind"] == "embedding"
    ]
    blocks: pd.DataFrame = layer_metadata[layer_metadata["layer_kind"] == "block"]
    if len(embeddings) != 1 or embeddings["block_idx"].notna().any():
        raise ValueError(f"{path} must contain exactly one non-block embedding slot")
    if not set(layer_metadata["layer_kind"]).issubset({"embedding", "block"}):
        raise ValueError(f"{path} has an unknown layer_kind")
    if len(blocks) < 2:
        raise ValueError(f"{path} needs at least two decoder blocks")
    if blocks["block_idx"].isna().any():
        raise ValueError(f"{path} has a block without block_idx")
    actual_blocks: list[int] = sorted(int(value) for value in blocks["block_idx"])
    if actual_blocks != list(range(len(blocks))):
        raise ValueError(f"{path} block_idx must be consecutive from zero")
    expected_layer_slots: list[int] = [0, *range(1, len(blocks) + 1)]
    actual_layer_slots: list[int] = sorted(
        int(value) for value in layer_metadata["layer_slot"]
    )
    if actual_layer_slots != expected_layer_slots:
        raise ValueError(f"{path} layer_slot must be consecutive from zero")
    block_slots: list[tuple[int, int]] = sorted(
        (int(row.layer_slot), int(row.block_idx))
        for row in blocks.itertuples(index=False)
    )
    if block_slots != [(block_idx + 1, block_idx) for block_idx in range(len(blocks))]:
        raise ValueError(f"{path} layer_slot and block_idx disagree")
    expected_slots: frozenset[int] = frozenset(
        int(value) for value in layer_metadata["layer_slot"]
    )
    group_columns: list[str] = ["prompt_idx", "kind", "seg_idx"]
    for observation, rows in checked.groupby(group_columns, dropna=False, sort=False):
        slots: frozenset[int] = frozenset(int(value) for value in rows["layer_slot"])
        if slots != expected_slots:
            raise ValueError(f"{path} has an incomplete layer grid for {observation}")
    original_prompts: set[int] = set(
        int(value) for value in checked.loc[checked["kind"] == "orig", "prompt_idx"]
    )
    ablated_prompts: set[int] = set(
        int(value) for value in checked.loc[checked["kind"] == "ablated", "prompt_idx"]
    )
    if not ablated_prompts.issubset(original_prompts):
        raise ValueError(f"{path} contains ablations without original rows")
    return checked


def _manifest_keys(
    results_dir: str, benchmark: str, pregrouper: str
) -> tuple[pd.DataFrame, pd.MultiIndex, str]:
    """Load a unique canonical manifest and return its segment identities."""
    base: str = os.path.join(results_dir, benchmark, pregrouper, "segments.tsv")
    path: str | None = next(
        (candidate for candidate in (base + ".gz", base) if os.path.isfile(candidate)),
        None,
    )
    if path is None:
        raise FileNotFoundError(base + "[.gz]")
    manifest: pd.DataFrame = pd.read_csv(path, sep="\t")
    required: set[str] = {"prompt_idx", "seg_idx", "message_role"}
    missing: set[str] = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing manifest columns {sorted(missing)}")
    _coerce_integer_column(manifest, "prompt_idx", nullable=False)
    _coerce_integer_column(manifest, "seg_idx", nullable=False)
    if manifest.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError(f"{path} has duplicate segment keys")
    keys: pd.MultiIndex = pd.MultiIndex.from_frame(manifest[["prompt_idx", "seg_idx"]])
    return manifest, keys, path


def _validate_manifest_coverage(
    frame: pd.DataFrame, manifest_keys: pd.MultiIndex, path: str
) -> None:
    """Require exact prompt and ablation-key coverage against the manifest."""
    ablated: pd.DataFrame = frame[frame["kind"] == "ablated"]
    actual_keys: pd.MultiIndex = pd.MultiIndex.from_frame(
        ablated[["prompt_idx", "seg_idx"]].drop_duplicates()
    )
    missing_keys: pd.MultiIndex = manifest_keys.difference(actual_keys)
    extra_keys: pd.MultiIndex = actual_keys.difference(manifest_keys)
    if len(missing_keys) or len(extra_keys):
        raise ValueError(
            f"{path} segment keys disagree with the manifest: "
            f"missing={len(missing_keys)}, extra={len(extra_keys)}"
        )
    expected_prompts: set[int] = set(
        int(value) for value in manifest_keys.get_level_values(0)
    )
    original_prompts: set[int] = set(
        int(value) for value in frame.loc[frame["kind"] == "orig", "prompt_idx"]
    )
    if original_prompts != expected_prompts:
        raise ValueError(f"{path} original prompt keys disagree with the manifest")


def _supporting_source_paths() -> dict[str, str]:
    """Return imported public sources that materially define the analysis."""
    root: str = os.path.dirname(os.path.dirname(__file__))
    relative_paths: tuple[str, ...] = (
        "benchmark_scripts/benchmark_config.py",
        "benchmark_scripts/derived_provenance.py",
        "benchmark_scripts/f_table.py",
        "surrogate/eval_constants.py",
    )
    return {path: os.path.join(root, path) for path in relative_paths}


def _layer_sidecar_path(layer_path: str) -> str:
    """Return the required execution-sidecar path for a layer table."""
    for suffix in (".tsv.gz", ".tsv"):
        if layer_path.endswith(suffix):
            return layer_path[: -len(suffix)] + "_run.json"
    raise ValueError(f"Layer artifact has an unsupported suffix: {layer_path}")


def _require_sha256(value: Any, field: str) -> str:
    """Validate one lowercase SHA-256 provenance value."""
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"Invalid SHA-256 in {field}")
    return value


def _validate_layer_sidecar(
    layer_path: str,
    manifest_path: str,
    frame: pd.DataFrame,
    benchmark: str,
    pregrouper: str,
    model: str,
) -> str:
    """Validate and return the sidecar cryptographically binding a layer run."""
    sidecar_path: str = _layer_sidecar_path(layer_path)
    if not os.path.isfile(sidecar_path):
        raise FileNotFoundError(sidecar_path)
    with open(sidecar_path, encoding="utf-8") as source:
        metadata: Any = json.load(source)
    if not isinstance(metadata, dict):
        raise ValueError(f"{sidecar_path} must contain a JSON object")
    expected_identity: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "model": model,
        "segmentation_scope": "full_dialog_in_message_order",
    }
    for field, expected in expected_identity.items():
        if metadata.get(field) != expected:
            raise ValueError(f"{sidecar_path} has invalid {field}")

    artifact: Any = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"{sidecar_path} is missing artifact provenance")
    if (
        artifact.get("filename") != os.path.basename(layer_path)
        or artifact.get("rows") != len(frame)
        or _require_sha256(artifact.get("sha256"), "artifact.sha256")
        != sha256_file(layer_path)
    ):
        raise ValueError(f"{sidecar_path} artifact identity disagrees")
    if _require_sha256(metadata.get("manifest_sha256"), "manifest_sha256") != (
        sha256_file(manifest_path)
    ):
        raise ValueError(f"{sidecar_path} manifest SHA disagrees")

    parameters: Any = metadata.get("parameters")
    if not isinstance(parameters, dict) or (
        parameters.get("rendered_chat_add_special_tokens") is not False
    ):
        raise ValueError(
            f"{sidecar_path} does not record safe rendered-chat tokenization"
        )
    if (
        parameters.get("attention_implementation") != "sdpa"
        or parameters.get("canary") is not False
        or parameters.get("device_map") != "auto"
        or parameters.get("max_samples") is not None
        or parameters.get("batch_size") != 32
        or parameters.get("torch_dtype") != "bfloat16"
    ):
        raise ValueError(f"{sidecar_path} is not a complete production layer run")

    expected_source: str | None = EXPECTED_MODEL_SOURCES.get(model)
    model_source: Any = metadata.get("model_source")
    if (
        not isinstance(model_source, str)
        or not model_source
        or (expected_source is not None and model_source != expected_source)
    ):
        raise ValueError(f"{sidecar_path} has invalid model_source")
    identity_hashes: Any = metadata.get("model_identity_files_sha256")
    if not isinstance(identity_hashes, dict) or not {
        "config.json",
        "tokenizer_config.json",
    }.issubset(identity_hashes):
        raise ValueError(f"{sidecar_path} lacks model identity provenance")
    for filename, digest in identity_hashes.items():
        if (
            not isinstance(filename, str)
            or os.path.basename(filename) != filename
            or _require_sha256(digest, f"model_identity_files_sha256.{filename}")
            != digest
        ):
            raise ValueError(f"{sidecar_path} has invalid model identity provenance")
    if model in GOLD_OPEN_MODEL_REVISIONS:
        artifact_hashes: Any = metadata.get("model_artifact_sha256")
        if (
            metadata.get("model_revision") != GOLD_OPEN_MODEL_REVISIONS[model]
            or metadata.get("model_artifact_hash_timing") != "pre_model_load"
            or metadata.get("source_hash_timing") != "run_start"
            or not isinstance(artifact_hashes, dict)
            or not artifact_hashes
        ):
            raise ValueError(f"{sidecar_path} has invalid model artifact provenance")
        validated_artifact_hashes: dict[str, str] = {}
        for filename, digest in artifact_hashes.items():
            if not isinstance(filename, str) or os.path.basename(filename) != filename:
                raise ValueError(
                    f"{sidecar_path} has invalid model artifact provenance"
                )
            validated_artifact_hashes[filename] = _require_sha256(
                digest, f"model_artifact_sha256.{filename}"
            )
        expected_manifest_digest: str = GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[model]
        if (
            metadata.get("model_artifact_manifest_sha256") != expected_manifest_digest
            or canonical_file_hash_manifest_sha256(validated_artifact_hashes)
            != expected_manifest_digest
            or any(
                validated_artifact_hashes.get(filename) != digest
                for filename, digest in identity_hashes.items()
            )
        ):
            raise ValueError(f"{sidecar_path} has invalid model artifact provenance")

    source_hashes: Any = metadata.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(
        LAYER_EXECUTION_SOURCE_FILES
    ):
        raise ValueError(f"{sidecar_path} lacks execution source provenance")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    for relative_path in LAYER_EXECUTION_SOURCE_FILES:
        recorded: str = _require_sha256(
            source_hashes.get(relative_path), f"source_sha256.{relative_path}"
        )
        if recorded != sha256_file(os.path.join(repository_root, relative_path)):
            raise ValueError(f"{sidecar_path} execution source SHA disagrees")

    slots: Any = metadata.get("layer_slots")
    if not isinstance(slots, dict) or slots.get("count") != int(
        frame["layer_slot"].nunique()
    ):
        raise ValueError(f"{sidecar_path} layer-slot metadata disagrees")
    dataset: Any = metadata.get("dataset")
    expected_prompts: int = int(
        frame.loc[frame["kind"] == "orig", "prompt_idx"].nunique()
    )
    if (
        not isinstance(dataset, dict)
        or dataset.get("prompts") != expected_prompts
        or not isinstance(dataset.get("snapshot_filename"), str)
        or _require_sha256(dataset.get("snapshot_sha256"), "dataset.snapshot_sha256")
        != dataset.get("snapshot_sha256")
        or _require_sha256(
            dataset.get("normalized_frame_sha256"),
            "dataset.normalized_frame_sha256",
        )
        != dataset.get("normalized_frame_sha256")
    ):
        raise ValueError(f"{sidecar_path} dataset provenance disagrees")
    return sidecar_path


def _native_signals(
    frame: pd.DataFrame,
    positive_label: str,
    negative_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return block-indexed prediction and attribution signals."""
    block_rows: pd.DataFrame = frame[frame["layer_kind"] == "block"].copy()
    score_column_positive: str = f"label_score_{_sanitize_label(positive_label)}"
    score_column_negative: str = f"label_score_{_sanitize_label(negative_label)}"
    block_rows["signal"] = (
        block_rows[score_column_positive] - block_rows[score_column_negative]
    )
    originals: pd.DataFrame = block_rows[block_rows["kind"] == "orig"][
        ["prompt_idx", "block_idx", "signal"]
    ]
    ablated: pd.DataFrame = block_rows[block_rows["kind"] == "ablated"][
        ["prompt_idx", "seg_idx", "block_idx", "signal"]
    ]
    attributes: pd.DataFrame = ablated.merge(
        originals.rename(columns={"signal": "original_signal"}),
        on=["prompt_idx", "block_idx"],
        validate="many_to_one",
    )
    attributes["signal"] = attributes["original_signal"] - attributes["signal"]
    return originals, attributes, int(block_rows["block_idx"].nunique())


def _interpolate_signal(
    frame: pd.DataFrame,
    identity_columns: list[str],
    num_blocks: int,
    depth: float,
) -> pd.Series:
    """Interpolate observation signals between decoder-block outputs."""
    native_depths: np.ndarray = np.linspace(0.0, 1.0, num_blocks)
    records: list[tuple[tuple[int, ...], float]] = []
    for identity, rows in frame.groupby(identity_columns, sort=False):
        identity_tuple: tuple[int, ...] = (
            tuple(int(value) for value in identity)
            if isinstance(identity, tuple)
            else (int(identity),)
        )
        ordered: pd.DataFrame = rows.sort_values("block_idx")
        records.append(
            (
                identity_tuple,
                float(
                    np.interp(
                        depth,
                        native_depths,
                        ordered["signal"].to_numpy(dtype=float),
                    )
                ),
            )
        )
    if len(identity_columns) == 1:
        return pd.Series(
            [value for _, value in records],
            index=pd.Index([identity[0] for identity, _ in records], name="prompt_idx"),
            dtype=float,
        )
    return pd.Series(
        [value for _, value in records],
        index=pd.MultiIndex.from_tuples(
            [identity for identity, _ in records], names=identity_columns
        ),
        dtype=float,
    )


def _nearest_signal(
    frame: pd.DataFrame,
    identity_columns: list[str],
    num_blocks: int,
    depth: float,
) -> pd.Series:
    """Select the nearest native decoder-block output at relative depth."""
    native_depths: np.ndarray = np.linspace(0.0, 1.0, num_blocks)
    # np.argmin resolves an exact midpoint toward the shallower block.
    block_index: int = int(np.argmin(np.abs(native_depths - depth)))
    selected: pd.DataFrame = frame[frame["block_idx"] == block_index]
    if len(identity_columns) == 1:
        return selected.set_index(identity_columns[0])["signal"].astype(float)
    return selected.set_index(identity_columns)["signal"].astype(float)


def _depth_signal(
    frame: pd.DataFrame,
    identity_columns: list[str],
    num_blocks: int,
    depth: float,
    depth_alignment: str,
) -> pd.Series:
    """Apply the requested cross-model relative-depth alignment."""
    if depth_alignment == "linear_interpolation":
        return _interpolate_signal(frame, identity_columns, num_blocks, depth)
    if depth_alignment == "nearest_native":
        return _nearest_signal(frame, identity_columns, num_blocks, depth)
    raise ValueError(f"unknown depth alignment {depth_alignment!r}")


def _model_path(results_dir: str, benchmark: str, pregrouper: str, model: str) -> str:
    base: str = os.path.join(results_dir, benchmark, pregrouper, f"{model}_layers.tsv")
    for path in (base + ".gz", base):
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(base + "[.gz]")


def _load_validated_layer_frame(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    positive_label: str,
    negative_label: str,
) -> tuple[pd.DataFrame, str, str, str]:
    """Load one complete layer artifact and validate all provenance bindings."""
    path: str = _model_path(results_dir, benchmark, pregrouper, model)
    frame: pd.DataFrame = _validate_layer_frame(
        pd.read_csv(path, sep="\t"), path, positive_label, negative_label
    )
    _manifest, manifest_keys, manifest_path = _manifest_keys(
        results_dir, benchmark, pregrouper
    )
    sidecar_path: str = _validate_layer_sidecar(
        path, manifest_path, frame, benchmark, pregrouper, model
    )
    _validate_manifest_coverage(frame, manifest_keys, path)
    return frame, path, manifest_path, sidecar_path


def load_final_layer_readout(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    scope: str,
    contrast: str,
) -> FinalLayerReadout:
    """Return final-block alignment, prediction, and attribution signals.

    The alignment is the cosine between the post-normalization representation
    delta and the requested sum-unembedding label direction. The returned
    signals are accepted only after validating the raw table, its full segment
    grid, and its execution sidecar. Zero-norm and non-finite alignment cells
    remain missing for pairwise-complete downstream analysis.
    """
    positive, negative = _contrast_labels(benchmark, contrast)
    frame, path, manifest_path, sidecar_path = _load_validated_layer_frame(
        results_dir,
        benchmark,
        pregrouper,
        model,
        positive,
        negative,
    )
    allowed_keys: pd.MultiIndex | None = _scope_keys(
        benchmark, pregrouper, results_dir, frame, scope
    )
    frame = _filter_segment_frame(frame, allowed_keys)
    prediction_frame, attribution_frame, _num_blocks = _native_signals(
        frame, positive, negative
    )
    final_block: int = int(frame.loc[frame["layer_kind"] == "block", "block_idx"].max())
    prediction_rows: pd.DataFrame = prediction_frame[
        prediction_frame["block_idx"] == final_block
    ]
    attribution_rows: pd.DataFrame = attribution_frame[
        attribution_frame["block_idx"] == final_block
    ]
    prediction: pd.Series = prediction_rows.set_index("prompt_idx")["signal"].astype(
        float
    )
    attribution: pd.Series = attribution_rows.set_index(["prompt_idx", "seg_idx"])[
        "signal"
    ].astype(float)

    pair_suffix: str = f"{_sanitize_label(positive)}_vs_{_sanitize_label(negative)}"
    numerator_column: str = f"w_dot_delta_z_postnorm_{pair_suffix}"
    norm_column: str = f"w_norm_{pair_suffix}"
    alignment_columns: set[str] = {
        numerator_column,
        norm_column,
        "delta_norm_postnorm",
    }
    missing_columns: set[str] = alignment_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{path} is missing requested alignment columns "
            f"{sorted(missing_columns)}"
        )
    alignment_rows: pd.DataFrame = frame[
        (frame["kind"] == "ablated")
        & (frame["layer_kind"] == "block")
        & (frame["block_idx"] == final_block)
    ].copy()
    for column in alignment_columns:
        alignment_rows[column] = pd.to_numeric(alignment_rows[column], errors="raise")
    denominator: pd.Series = (
        alignment_rows[norm_column] * alignment_rows["delta_norm_postnorm"]
    )
    alignment_values: pd.Series = alignment_rows[numerator_column] / denominator
    valid: pd.Series = denominator.ne(0.0) & np.isfinite(
        alignment_values.to_numpy(dtype=float)
    )
    alignment_values = alignment_values.where(valid, np.nan)
    alignment: pd.Series = pd.Series(
        alignment_values.to_numpy(dtype=float),
        index=pd.MultiIndex.from_frame(
            alignment_rows[["prompt_idx", "seg_idx"]].astype(int)
        ),
        dtype=float,
        name="alignment",
    )
    if not alignment.index.is_unique:
        raise ValueError(f"{path} has duplicate final-layer alignment keys")
    return FinalLayerReadout(
        alignment=alignment,
        prediction=prediction,
        attribution=attribution,
        layer_path=path,
        manifest_path=manifest_path,
        sidecar_path=sidecar_path,
    )


def _load_model_signals(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    scope: str,
    contrast: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, str, str, str]:
    """Load, validate, scope, and reduce one model's layer artifact."""
    positive, negative = _contrast_labels(benchmark, contrast)
    frame, path, manifest_path, sidecar_path = _load_validated_layer_frame(
        results_dir,
        benchmark,
        pregrouper,
        model,
        positive,
        negative,
    )
    allowed_keys: pd.MultiIndex | None = _scope_keys(
        benchmark, pregrouper, results_dir, frame, scope
    )
    frame = _filter_segment_frame(frame, allowed_keys)
    prediction, attribution, num_blocks = _native_signals(frame, positive, negative)
    return prediction, attribution, num_blocks, path, manifest_path, sidecar_path


def analyze_config(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    scope: str,
    contrast: str,
    models: tuple[str, ...],
    depth_grid_size: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
    depth_alignment: str = "linear_interpolation",
) -> tuple[list[dict[str, str | int | float | bool]], dict[str, str]]:
    """Compute layerwise fidelity for every unordered open-model pair."""
    if scope not in {"all", "system", "user"}:
        raise ValueError(f"unknown message scope {scope!r}")
    if depth_grid_size < 2:
        raise ValueError("depth_grid_size must be at least two")
    if depth_alignment not in {"linear_interpolation", "nearest_native"}:
        raise ValueError(f"unknown depth alignment {depth_alignment!r}")
    if len(models) < 2 or len(set(models)) != len(models):
        raise ValueError("models must contain at least two unique names")
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    native: dict[str, tuple[pd.DataFrame, pd.DataFrame, int]] = {}
    inputs: dict[str, str] = {}
    for model in models:
        (
            prediction,
            attribution,
            num_blocks,
            path,
            manifest_path,
            sidecar_path,
        ) = _load_model_signals(
            results_dir, benchmark, pregrouper, model, scope, contrast
        )
        native[model] = prediction, attribution, num_blocks
        inputs[f"{benchmark}/{pregrouper}/{model}/layers"] = path
        inputs[f"{benchmark}/{pregrouper}/{model}/layers_run"] = sidecar_path
        inputs[f"{benchmark}/{pregrouper}/segments"] = manifest_path
    rows_out: list[dict[str, str | int | float | bool]] = []
    for depth_index, depth_value in enumerate(np.linspace(0.0, 1.0, depth_grid_size)):
        depth: float = float(depth_value)
        signals_by_metric: dict[str, dict[str, pd.Series]] = {
            "F_pred": {
                model: _depth_signal(
                    values[0],
                    ["prompt_idx"],
                    values[2],
                    depth,
                    depth_alignment,
                )
                for model, values in native.items()
            },
            "F_attr": {
                model: _depth_signal(
                    values[1],
                    ["prompt_idx", "seg_idx"],
                    values[2],
                    depth,
                    depth_alignment,
                )
                for model, values in native.items()
            },
        }
        for metric, signals in signals_by_metric.items():
            source_contrast, target_contrast, readout = _contrast_metadata(
                benchmark, contrast, metric
            )
            metric_rows = _emit_pair_corrs(
                benchmark,
                metric,
                signals,
                bootstrap_resamples,
                confidence_level,
                _analysis_rng(seed, benchmark, pregrouper),
                bootstrap_seed=seed,
                rng_context=(
                    benchmark,
                    pregrouper,
                    _resolved_scope(metric, scope),
                    contrast,
                    depth_alignment,
                    str(depth_index),
                ),
            )
            for metric_row in metric_rows:
                source: str = str(metric_row["model_s"])
                target: str = str(metric_row["model_t"])
                point: float = float(metric_row["f_point"])
                enriched: dict[str, str | int | float | bool] = dict(metric_row)
                enriched.update(
                    {
                        "cohort": "open",
                        "pair_population": "open_open",
                        "pregrouper": pregrouper,
                        "scope": scope,
                        "requested_scope": scope,
                        "resolved_scope": _resolved_scope(metric, scope),
                        "contrast": contrast,
                        "requested_contrast": contrast,
                        "resolved_source_contrast": source_contrast,
                        "resolved_target_contrast": target_contrast,
                        "readout_contrast": readout,
                        "availability_status": (
                            "available" if np.isfinite(point) else "unavailable"
                        ),
                        "unavailable_reason": (
                            ""
                            if np.isfinite(point)
                            else "insufficient_joint_observations"
                        ),
                        "aggregation": "row_pooled",
                        "source_num_blocks": native[source][2],
                        "target_num_blocks": native[target][2],
                        "relative_depth_index": depth_index,
                        "relative_depth": depth,
                        "relative_depth_grid_size": depth_grid_size,
                        "depth_alignment": (
                            "linear_interpolation_block_outputs_only"
                            if depth_alignment == "linear_interpolation"
                            else "nearest_native_block_output_shallower_tie"
                        ),
                        "embedding_slot_included": False,
                    }
                )
                rows_out.append(enriched)
    return rows_out, inputs


def main() -> None:
    """Run the public layerwise fidelity command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default=None)
    parser.add_argument("--benchmarks", nargs="+", default=None)
    parser.add_argument("--pregrouper", default="sentence")
    parser.add_argument(
        "--scopes", nargs="+", choices=["all", "system", "user"], default=["all"]
    )
    parser.add_argument(
        "--anli-contrast",
        choices=["entailment_contradiction", "entailment_neutral"],
        default="entailment_contradiction",
    )
    parser.add_argument("--depth-grid-size", type=int, default=21)
    parser.add_argument(
        "--depth-alignment",
        choices=["linear_interpolation", "nearest_native"],
        default="linear_interpolation",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    output: str = args.output or os.path.join(
        args.results_dir, "layerwise_fidelity.tsv"
    )
    configs: list[tuple[str, str]] = (
        [(benchmark, args.pregrouper) for benchmark in args.benchmarks]
        if args.benchmarks is not None
        else list(DEFAULT_CONFIGS)
    )
    output_rows: list[dict[str, str | int | float | bool]] = []
    input_paths: dict[str, str] = {}
    for benchmark, pregrouper in configs:
        contrast: str = (
            args.anli_contrast if benchmark.startswith("anli_") else "canonical"
        )
        for scope in args.scopes:
            rows, inputs = analyze_config(
                args.results_dir,
                benchmark,
                pregrouper,
                scope,
                contrast,
                OPEN_MODELS,
                args.depth_grid_size,
                args.bootstrap_resamples,
                args.confidence_level,
                args.seed,
                args.depth_alignment,
            )
            output_rows.extend(rows)
            input_paths.update(inputs)
    pd.DataFrame(output_rows).to_csv(output, sep="\t", index=False)
    write_derived_provenance(
        output,
        generator_name="benchmark_scripts.layerwise_fidelity",
        generator_path=__file__,
        input_paths=input_paths,
        parameters={
            "benchmark_configs": [
                {"benchmark": benchmark, "pregrouper": pregrouper}
                for benchmark, pregrouper in configs
            ],
            "scopes": list(args.scopes),
            "anli_contrast": args.anli_contrast,
            "models": list(OPEN_MODELS),
            "depth_grid_size": args.depth_grid_size,
            "depth_alignment": args.depth_alignment,
            "depth_definition": "first_block_0_last_block_1_embedding_excluded",
            "bootstrap_resamples": args.bootstrap_resamples,
            "confidence_level": args.confidence_level,
            "seed": args.seed,
            "bootstrap_rng": "sha256_cell_key_v1",
            "missingness_policy": "pair_specific_complete_case",
        },
        root_dir=args.results_dir,
        supporting_source_paths=_supporting_source_paths(),
    )
    logger.info("Wrote %d rows to %s", len(output_rows), output)


if __name__ == "__main__":
    main()
