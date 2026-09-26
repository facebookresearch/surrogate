# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Capture compact layerwise scores for labeled benchmark ablations.

This runner intentionally writes label-level scalars rather than hidden
vectors.  Every configured label is retained, so contrasts such as ANLI
entailment--neutral and entailment--contradiction remain post-hoc choices.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
import gzip
import hashlib
import io
from itertools import combinations
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
    MODEL_SETS,
    resolve_model_path,
)
from benchmark_scripts.provenance_sources import (
    canonical_file_hash_manifest_sha256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    LAYER_EXECUTION_SOURCE_FILES,
    OPEN_MODEL_IDENTITY_FILENAMES,
)
from surrogate.eval_constants import label_column_alias, ReportToken
from surrogate.layerwise_scoring import (
    capture_postnorm_residual_slots,
    LayerwiseLabelScores,
    layerwise_delta_norms,
    ResidualSlotCapture,
    score_residual_slots,
)
from surrogate.model_types import Dialog, make_dialog
from surrogate.text_augmentation import (
    dialog_segments,
    PregrouperID,
    segment_and_ablate,
)
from surrogate.transformers_model import TransformersModel
from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

CANONICAL_BATCH_SIZE: int = 32

GZIP_COMPRESSION: dict[str, Any] = {
    "method": "gzip",
    "compresslevel": 9,
    "mtime": 0,
}
SCHEMA_VERSION: int = 1


def _sha256_file(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload: bytes = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_artifact_hashes(model_path: str) -> dict[str, str]:
    """Hash every local weight and identity artifact used by a model."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Pinned layerwise runs require a local model directory: {model_path}"
        )
    names: list[str] = sorted(
        name
        for name in os.listdir(model_path)
        if name.endswith((".safetensors", ".bin", ".json", ".model", ".txt"))
        and os.path.isfile(os.path.join(model_path, name))
    )
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"No model weight files found in {model_path}")
    return {name: _sha256_file(os.path.join(model_path, name)) for name in names}


def _verified_model_identity(
    model_name: str,
    model_source: str,
    model_path: str,
) -> tuple[str, dict[str, str], str, dict[str, str]]:
    """Verify a local model against the pinned public revision and file digest.

    Returns:
        The pinned revision, full artifact hashes, aggregate artifact-manifest
        hash, and the portable identity-file subset.
    """
    expected_source: str | None = GOLD_OPEN_MODEL_REPOSITORIES.get(model_name)
    revision: str | None = GOLD_OPEN_MODEL_REVISIONS.get(model_name)
    expected_manifest: str | None = GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256.get(
        model_name
    )
    if expected_source is None or revision is None or expected_manifest is None:
        raise ValueError(f"No pinned public model identity for {model_name!r}")
    if model_source != expected_source:
        raise ValueError(
            f"Model source for {model_name!r} is {model_source!r}, expected "
            f"{expected_source!r}"
        )
    logger.info("Hashing pinned local model artifacts for %s", model_name)
    artifact_hashes: dict[str, str] = _model_artifact_hashes(model_path)
    artifact_manifest: str = canonical_file_hash_manifest_sha256(artifact_hashes)
    if artifact_manifest != expected_manifest:
        raise ValueError(
            f"Model artifact manifest for {model_name!r} is {artifact_manifest}, "
            f"expected {expected_manifest}"
        )
    identity_hashes: dict[str, str] = {
        name: digest
        for name, digest in artifact_hashes.items()
        if name in OPEN_MODEL_IDENTITY_FILENAMES
    }
    if not identity_hashes:
        raise ValueError(f"No portable model identity files found for {model_name!r}")
    logger.info(
        "Verified pinned model artifact %s for %s", artifact_manifest, model_name
    )
    return revision, artifact_hashes, artifact_manifest, identity_hashes


def _jsonable(value: Any) -> Any:
    """Convert common NumPy scalar values to JSON-safe Python values."""
    return value.item() if isinstance(value, np.generic) else value


def _atomic_write_json(path: str, value: Mapping[str, Any]) -> None:
    """Write sorted JSON through a same-directory temporary file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _atomic_write_tsv(path: str, frame: pd.DataFrame) -> None:
    """Write a deterministic gzip-compressed TSV atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        dir=os.path.dirname(path), prefix=f".{os.path.basename(path)}.", suffix=".gz"
    )
    try:
        with os.fdopen(descriptor, "wb") as raw_output:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=int(GZIP_COMPRESSION["compresslevel"]),
                fileobj=raw_output,
                mtime=int(GZIP_COMPRESSION["mtime"]),
            ) as compressed_output:
                with io.TextIOWrapper(
                    compressed_output, encoding="utf-8", newline=""
                ) as text_output:
                    frame.to_csv(
                        text_output,
                        sep="\t",
                        index=False,
                        lineterminator="\n",
                    )
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary_path, path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise


def _pregrouper(value: str) -> PregrouperID:
    if value == "sentence":
        return "sentence"
    if value == "word":
        return "word"
    raise argparse.ArgumentTypeError("pregrouper must be 'sentence' or 'word'")


def _ordered_report_tokens(spec: BenchmarkSpec) -> dict[str, list[ReportToken]]:
    """Return deterministic token aliases for every configured label."""
    if spec.eval_config is None:
        raise ValueError(f"{spec.name!r} is not a labeled benchmark")
    result: dict[str, list[ReportToken]] = {}
    for label, configured_tokens in spec.eval_config.label_tokens.items():
        report_tokens: list[ReportToken] = spec.eval_config.report_tokens.get(label, [])
        if report_tokens:
            result[label] = report_tokens
        else:
            result[label] = [
                ReportToken(alias=surface, surface=surface)
                for surface in sorted(configured_tokens)
            ]
    return result


def _resolve_label_groups(
    tokenizer: Any,
    report_tokens: Mapping[str, Sequence[ReportToken]],
) -> tuple[dict[str, list[int]], dict[str, dict[str, Any]]]:
    """Resolve, deduplicate, and validate single-token label aliases.

    Multi-token aliases are recorded in metadata and excluded. A label with
    no surviving alias, or a token ID shared by two labels, is rejected.
    """
    groups: dict[str, list[int]] = {}
    metadata: dict[str, dict[str, Any]] = {}
    owner_by_id: dict[int, str] = {}
    for label, aliases in report_tokens.items():
        token_ids: list[int] = []
        accepted: list[dict[str, Any]] = []
        duplicate_aliases: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        retained_alias_by_id: dict[int, str] = {}
        for token in aliases:
            encoded_value: Any = tokenizer.encode(
                token.surface, add_special_tokens=False
            )
            encoded: list[int] = [int(value) for value in encoded_value]
            if len(encoded) != 1:
                rejected.append(
                    {
                        "alias": token.alias,
                        "surface": token.surface,
                        "token_ids": encoded,
                    }
                )
                continue
            token_id: int = encoded[0]
            if token_id in retained_alias_by_id:
                duplicate_aliases.append(
                    {
                        "alias": token.alias,
                        "surface": token.surface,
                        "token_id": token_id,
                        "duplicate_of_alias": retained_alias_by_id[token_id],
                    }
                )
                continue
            previous_owner: str | None = owner_by_id.get(token_id)
            if previous_owner is not None:
                raise ValueError(
                    f"token ID {token_id} overlaps labels "
                    f"{previous_owner!r} and {label!r}"
                )
            retained_alias_by_id[token_id] = token.alias
            owner_by_id[token_id] = label
            token_ids.append(token_id)
            accepted.append(
                {
                    "alias": token.alias,
                    "surface": token.surface,
                    "token_id": token_id,
                }
            )
        if not token_ids:
            raise ValueError(f"label {label!r} has no single-token aliases")
        groups[label] = token_ids
        metadata[label] = {
            "accepted_single_token_aliases": accepted,
            "deduplicated_single_token_aliases": duplicate_aliases,
            "rejected_multitoken_aliases": rejected,
        }
    if len(groups) < 2:
        raise ValueError("at least two nonempty label groups are required")
    return groups, metadata


def _column_names(
    labels: Sequence[str],
) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    """Build collision-free label and contrast column suffixes."""
    label_suffixes: dict[str, str] = {
        label: label_column_alias(label) for label in labels
    }
    if len(set(label_suffixes.values())) != len(label_suffixes):
        raise ValueError(f"label names collide after sanitization: {list(labels)}")
    contrast_suffixes: dict[tuple[str, str], str] = {
        (first, second): f"{label_suffixes[first]}_vs_{label_suffixes[second]}"
        for first, second in combinations(labels, 2)
    }
    return label_suffixes, contrast_suffixes


def _contrast_norms(
    lm_head: Any,
    label_groups: Mapping[str, Sequence[int]],
) -> dict[tuple[str, str], float]:
    """Compute norms of uniform-sum unembedding contrast directions."""
    weight: Any = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("lm_head.weight must be a rank-two tensor")
    sums: dict[str, torch.Tensor] = {
        label: weight[list(token_ids)].sum(dim=0)
        for label, token_ids in label_groups.items()
    }
    return {
        (first, second): float(
            torch.linalg.vector_norm(sums[first] - sums[second]).item()
        )
        for first, second in combinations(label_groups, 2)
    }


def _tokenize_batch(
    model: TransformersModel,
    texts: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad already-rendered chat text without adding special tokens."""
    tokenizer: Any = model._tokenizer
    causal_lm: Any = model._model
    if tokenizer is None or causal_lm is None:
        raise RuntimeError("model must be loaded before layerwise scoring")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenization_kwargs: dict[str, Any] = {
        "return_tensors": "pt",
        "padding": True,
    }
    encoded: Any = model._tokenize_rendered_text(list(texts), **tokenization_kwargs)
    input_ids: torch.Tensor = encoded["input_ids"].to(causal_lm.device)
    attention_mask: torch.Tensor = encoded["attention_mask"].to(causal_lm.device)
    return input_ids, attention_mask


def _capture_and_score(
    model: TransformersModel,
    texts: Sequence[str],
    label_groups: Mapping[str, Sequence[int]],
) -> tuple[ResidualSlotCapture, LayerwiseLabelScores]:
    input_ids, attention_mask = _tokenize_batch(model, texts)
    capture: ResidualSlotCapture = capture_postnorm_residual_slots(
        model._model, input_ids, attention_mask
    )
    scores: LayerwiseLabelScores = score_residual_slots(
        capture, model._model.lm_head, label_groups
    )
    return capture, scores


def _layer_metadata(layer_slot: int) -> tuple[str, int | None]:
    if layer_slot == 0:
        return "embedding", None
    return "block", layer_slot - 1


def _score_columns(
    scores: torch.Tensor,
    layer_slot: int,
    batch_index: int,
    labels: Sequence[str],
    label_suffixes: Mapping[str, str],
) -> dict[str, float]:
    return {
        f"label_score_{label_suffixes[label]}": float(
            scores[layer_slot, batch_index, label_index].item()
        )
        for label_index, label in enumerate(labels)
    }


def _original_rows(
    prompt_idx: int,
    answer: Any,
    scores: LayerwiseLabelScores,
    label_suffixes: Mapping[str, str],
    contrast_suffixes: Mapping[tuple[str, str], str],
    contrast_norms: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Build one compact original row per residual slot."""
    if scores.grouped_logsumexp.shape[1] != 1:
        raise ValueError("original scores must contain exactly one batch row")
    grouped: torch.Tensor = scores.grouped_logsumexp.float().cpu()
    rows: list[dict[str, Any]] = []
    for layer_slot in range(grouped.shape[0]):
        layer_kind, block_idx = _layer_metadata(layer_slot)
        row: dict[str, Any] = {
            "prompt_idx": prompt_idx,
            "seg_idx": None,
            "kind": "orig",
            "answer": answer,
            "layer_slot": layer_slot,
            "layer_kind": layer_kind,
            "block_idx": block_idx,
            **_score_columns(grouped, layer_slot, 0, scores.labels, label_suffixes),
            "delta_norm_postnorm": None,
        }
        for pair, suffix in contrast_suffixes.items():
            row[f"w_dot_delta_z_postnorm_{suffix}"] = None
            row[f"w_norm_{suffix}"] = contrast_norms[pair]
        rows.append(row)
    return rows


def _ablated_rows(
    prompt_idx: int,
    answer: Any,
    segment_indices: Sequence[int],
    original_scores: LayerwiseLabelScores,
    perturbed_scores: LayerwiseLabelScores,
    delta_norms: torch.Tensor,
    label_suffixes: Mapping[str, str],
    contrast_suffixes: Mapping[tuple[str, str], str],
    contrast_norms: Mapping[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Build compact rows for a batch of segment ablations."""
    if original_scores.labels != perturbed_scores.labels:
        raise ValueError("original and perturbed label orders disagree")
    expected_shape: tuple[int, int] = (
        int(perturbed_scores.grouped_logsumexp.shape[0]),
        len(segment_indices),
    )
    if tuple(delta_norms.shape) != expected_shape:
        raise ValueError(
            f"delta norm shape {tuple(delta_norms.shape)} differs from "
            f"expected {expected_shape}"
        )
    if int(perturbed_scores.grouped_logsumexp.shape[1]) != len(segment_indices):
        raise ValueError("segment index count differs from perturbed score batch")
    grouped: torch.Tensor = perturbed_scores.grouped_logsumexp.float().cpu()
    original_projection: torch.Tensor = (
        original_scores.summed_unembedding_projection.float().cpu()
    )
    perturbed_projection: torch.Tensor = (
        perturbed_scores.summed_unembedding_projection.float().cpu()
    )
    norms: torch.Tensor = delta_norms.float().cpu()
    label_index: dict[str, int] = {
        label: index for index, label in enumerate(perturbed_scores.labels)
    }
    rows: list[dict[str, Any]] = []
    for batch_index, seg_idx in enumerate(segment_indices):
        for layer_slot in range(grouped.shape[0]):
            layer_kind, block_idx = _layer_metadata(layer_slot)
            row: dict[str, Any] = {
                "prompt_idx": prompt_idx,
                "seg_idx": seg_idx,
                "kind": "ablated",
                "answer": answer,
                "layer_slot": layer_slot,
                "layer_kind": layer_kind,
                "block_idx": block_idx,
                **_score_columns(
                    grouped,
                    layer_slot,
                    batch_index,
                    perturbed_scores.labels,
                    label_suffixes,
                ),
                "delta_norm_postnorm": float(norms[layer_slot, batch_index].item()),
            }
            for (first, second), suffix in contrast_suffixes.items():
                first_index: int = label_index[first]
                second_index: int = label_index[second]
                original_dot: torch.Tensor = (
                    original_projection[layer_slot, 0, first_index]
                    - original_projection[layer_slot, 0, second_index]
                )
                perturbed_dot: torch.Tensor = (
                    perturbed_projection[layer_slot, batch_index, first_index]
                    - perturbed_projection[layer_slot, batch_index, second_index]
                )
                row[f"w_dot_delta_z_postnorm_{suffix}"] = float(
                    (original_dot - perturbed_dot).item()
                )
                row[f"w_norm_{suffix}"] = contrast_norms[(first, second)]
            rows.append(row)
    return rows


def _expected_manifest(
    dialogs: Sequence[Dialog],
    prompts_meta: Sequence[Mapping[str, Any]],
    pregrouper: PregrouperID,
) -> pd.DataFrame:
    """Construct the exact full-dialog segment manifest for this run."""
    rows: list[dict[str, Any]] = []
    for dialog, prompt_meta in zip(dialogs, prompts_meta):
        segments = dialog_segments(dialog, pregrouper)
        for segment in segments:
            rows.append(
                {
                    "prompt_idx": int(prompt_meta["prompt_idx"]),
                    "answer": prompt_meta["answer"],
                    "seg_idx": segment.segment_idx,
                    "message_idx": segment.message_idx,
                    "message_role": segment.message_role,
                    "message_seg_idx": segment.message_segment_idx,
                    "segment_text": segment.text,
                    "n_segments": len(segments),
                }
            )
    return pd.DataFrame(rows)


def _validate_or_write_manifest(
    path: str,
    expected: pd.DataFrame,
    canary: bool,
) -> None:
    """Require exact manifest identity, except when creating a canary."""
    if not os.path.exists(path):
        if not canary:
            raise FileNotFoundError(
                f"Required segment manifest does not exist: {path}. "
                "Use --canary with --max-samples only for an isolated canary run."
            )
        _atomic_write_tsv(path, expected)
        logger.info("Wrote canary segment manifest %s", path)
        return
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
            f"Existing segment manifest disagrees with run: {path}"
        ) from error


def _model_selection(
    model_set: str,
    model_filter: str | None,
) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = MODEL_SETS[model_set]
    if model_filter is None:
        model_filter = os.environ.get("BENCHMARK_MODELS")
    if not model_filter:
        return candidates
    requested: list[str] = [
        name.strip() for name in model_filter.split(",") if name.strip()
    ]
    available: set[str] = {name for name, _ in candidates}
    unknown: set[str] = set(requested) - available
    if unknown:
        raise ValueError(
            f"models {sorted(unknown)} are not in {model_set}; available: {sorted(available)}"
        )
    requested_set: set[str] = set(requested)
    return [(name, source) for name, source in candidates if name in requested_set]


async def _score_model(
    model: TransformersModel,
    dialogs: Sequence[Dialog],
    prompts_meta: Sequence[Mapping[str, Any]],
    pregrouper: PregrouperID,
    label_groups: Mapping[str, Sequence[int]],
    batch_size: int,
) -> pd.DataFrame:
    """Score every original and full-dialog segment ablation for one model."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    labels: tuple[str, ...] = tuple(label_groups)
    label_suffixes, contrast_suffixes = _column_names(labels)
    contrast_norms: dict[tuple[str, str], float] = _contrast_norms(
        model._model.lm_head, label_groups
    )
    rows: list[dict[str, Any]] = []
    for dialog, prompt_meta in tqdm(
        list(zip(dialogs, prompts_meta)), desc=f"Layerwise: {model.model_name}"
    ):
        prompt_idx: int = int(prompt_meta["prompt_idx"])
        answer: Any = prompt_meta["answer"]
        original_text: str = model.dialog_to_text(dialog)
        original_capture, original_scores = _capture_and_score(
            model, [original_text], label_groups
        )
        rows.extend(
            _original_rows(
                prompt_idx,
                answer,
                original_scores,
                label_suffixes,
                contrast_suffixes,
                contrast_norms,
            )
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
        # Keep the batching topology identical to the ordinary public scorer.
        # Its canonical length sort uses rendered character length rather than
        # tokenizer length. This matters numerically for BF16 inference because
        # changing a row's padding/batch context can slightly change its result.
        permutation: list[int] = _character_length_permutation(ablated_texts)
        for start in range(0, len(permutation), batch_size):
            segment_indices: list[int] = permutation[start : start + batch_size]
            texts: list[str] = [ablated_texts[index] for index in segment_indices]
            perturbed_capture, perturbed_scores = _capture_and_score(
                model, texts, label_groups
            )
            delta_norms: torch.Tensor = layerwise_delta_norms(
                original_capture, perturbed_capture
            )
            rows.extend(
                _ablated_rows(
                    prompt_idx,
                    answer,
                    segment_indices,
                    original_scores,
                    perturbed_scores,
                    delta_norms,
                    label_suffixes,
                    contrast_suffixes,
                    contrast_norms,
                )
            )
        del original_capture, original_scores

    frame: pd.DataFrame = pd.DataFrame(rows)
    kind_order: pd.Series = frame["kind"].map({"orig": 0, "ablated": 1})
    frame = (
        frame.assign(_kind_order=kind_order, _seg_order=frame["seg_idx"].fillna(-1))
        .sort_values(
            ["prompt_idx", "_kind_order", "_seg_order", "layer_slot"],
            kind="stable",
        )
        .drop(columns=["_kind_order", "_seg_order"])
        .reset_index(drop=True)
    )
    return frame


def _source_hashes() -> dict[str, str]:
    root: str = os.path.dirname(os.path.dirname(__file__))
    return {
        path: _sha256_file(os.path.join(root, path))
        for path in LAYER_EXECUTION_SOURCE_FILES
    }


def _character_length_permutation(texts: Sequence[str]) -> list[int]:
    """Return the stable rendered-character ordering used by ordinary scoring."""
    return sorted(range(len(texts)), key=lambda index: (len(texts[index]), index))


async def run_layerwise(
    benchmark_name: str,
    pregrouper: PregrouperID = "sentence",
    batch_size: int = CANONICAL_BATCH_SIZE,
    max_samples: int | None = None,
    seed: int = 42,
    model_set: str = "Qwen2.5-Instruct",
    models: str | None = None,
    results_dir: str = "results",
    dataset_file: str | None = None,
    canary: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """Run compact layerwise scoring for a labeled benchmark.

    Args:
        benchmark_name: Key in ``BENCHMARKS``.
        pregrouper: Full-dialog segmentation granularity.
        batch_size: Number of character-length-sorted ablations per forward
            pass. Canonical runs require 32 to match ordinary public scoring.
        max_samples: Optional deterministic prompt subsample.
        seed: Subsampling seed.
        model_set: Key in ``MODEL_SETS``.
        models: Optional comma-separated model-name filter.
        results_dir: Root containing benchmark/pregrouper result directories.
        dataset_file: Optional frozen dataset TSV.
        canary: Permit creating an isolated manifest; requires ``max_samples``.
        overwrite_existing: Replace existing layerwise model outputs.
    """
    execution_source_sha256: dict[str, str] = _source_hashes()
    if benchmark_name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {benchmark_name!r}")
    if model_set not in MODEL_SETS:
        raise ValueError(f"unknown model set {model_set!r}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not canary and batch_size != CANONICAL_BATCH_SIZE:
        raise ValueError(
            f"canonical layerwise runs require batch_size={CANONICAL_BATCH_SIZE}; "
            "use --canary for diagnostic batch sizes"
        )
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if canary and max_samples is None:
        raise ValueError("--canary requires --max-samples")
    spec: BenchmarkSpec = BENCHMARKS[benchmark_name]
    report_tokens = _ordered_report_tokens(spec)
    frame: pd.DataFrame = load_benchmark_dataset(spec, dataset_file=dataset_file)
    if max_samples is not None and max_samples < len(frame):
        generator: np.random.Generator = np.random.default_rng(seed)
        positions: np.ndarray = generator.choice(
            len(frame), size=max_samples, replace=False
        )
        positions.sort()
        frame = frame.iloc[positions]

    if spec.eval_config is None:
        raise ValueError(f"{benchmark_name!r} is not a labeled benchmark")
    system_prompt: str = (
        spec.system_prompt_override
        if spec.system_prompt_override is not None
        else spec.eval_config.system_prompt
    )
    dialogs: list[Dialog] = [
        make_dialog(system_prompt, spec.prompt_builder(row))
        for _, row in frame.iterrows()
    ]
    prompts_meta: list[dict[str, Any]] = [
        {
            "prompt_idx": int(source_idx),
            "answer": _jsonable(row[spec.answer_column]),
        }
        for source_idx, row in frame.iterrows()
    ]
    output_dir: str = os.path.join(results_dir, benchmark_name, pregrouper)
    os.makedirs(output_dir, exist_ok=True)
    manifest_path: str = os.path.join(output_dir, "segments.tsv.gz")
    expected_manifest: pd.DataFrame = _expected_manifest(
        dialogs, prompts_meta, pregrouper
    )
    _validate_or_write_manifest(manifest_path, expected_manifest, canary)

    selected_models: list[tuple[str, str]] = _model_selection(model_set, models)
    for model_name, model_source in selected_models:
        layer_path: str = os.path.join(output_dir, f"{model_name}_layers.tsv.gz")
        run_path: str = os.path.join(output_dir, f"{model_name}_layers_run.json")
        existing: list[str] = [
            path for path in (layer_path, run_path) if os.path.exists(path)
        ]
        if existing and not overwrite_existing:
            raise FileExistsError(
                f"refusing to overwrite layerwise outputs without "
                f"--overwrite-existing: {existing}"
            )
        model_path: str = resolve_model_path(model_source)
        (
            model_revision,
            model_artifact_hashes,
            model_artifact_manifest_sha256,
            model_identity_hashes,
        ) = _verified_model_identity(model_name, model_source, model_path)
        model: TransformersModel = TransformersModel(
            model_name=model_name,
            model_path=model_path,
            attn_implementation="sdpa",
        ).load()
        try:
            label_groups, label_metadata = _resolve_label_groups(
                model._tokenizer, report_tokens
            )
            result: pd.DataFrame = await _score_model(
                model,
                dialogs,
                prompts_meta,
                pregrouper,
                label_groups,
                batch_size,
            )
            _atomic_write_tsv(layer_path, result)
            metadata: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "artifact": {
                    "filename": os.path.basename(layer_path),
                    "rows": len(result),
                    "sha256": _sha256_file(layer_path),
                },
                "benchmark": benchmark_name,
                "pregrouper": pregrouper,
                "segmentation_scope": "full_dialog_in_message_order",
                "layer_slots": {
                    "count": int(result["layer_slot"].max()) + 1,
                    "convention": (
                        "slot 0 is the embedding output; slot k+1 is decoder "
                        "block k output; final norm is applied before all scores"
                    ),
                },
                "model": model_name,
                "model_source": model_source,
                "model_revision": model_revision,
                "model_artifact_hash_timing": "pre_model_load",
                "model_artifact_sha256": model_artifact_hashes,
                "model_artifact_manifest_sha256": (model_artifact_manifest_sha256),
                "model_identity_files_sha256": model_identity_hashes,
                "dataset": {
                    "hf_path": spec.hf_dataset_path,
                    "hf_name": spec.hf_dataset_name,
                    "hf_split": spec.hf_split,
                    "snapshot_filename": (
                        os.path.basename(dataset_file)
                        if dataset_file is not None
                        else None
                    ),
                    "snapshot_sha256": (
                        _sha256_file(dataset_file) if dataset_file is not None else None
                    ),
                    "normalized_frame_sha256": _frame_sha256(frame),
                    "prompts": len(frame),
                },
                "manifest_sha256": _sha256_file(manifest_path),
                "labels": label_metadata,
                "alignment": {
                    "direction": (
                        "uniform sum of accepted label unembedding rows; each "
                        "unordered contrast follows configured label order"
                    ),
                    "multi_alias_status": (
                        "diagnostic approximation to grouped-logsumexp attribution"
                    ),
                },
                "label_score_definition": (
                    "intermediate slots store logsumexp of accepted alias logits; "
                    "the final slot stores logsumexp of the model's native-dtype "
                    "full-head log-probabilities to match ordinary outputs; "
                    "pairwise differences are grouped-label log-probability "
                    "contrasts"
                ),
                "parameters": {
                    "attention_implementation": "sdpa",
                    "rendered_chat_add_special_tokens": False,
                    "batch_size": batch_size,
                    "canary": canary,
                    "device_map": "auto",
                    "max_samples": max_samples,
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
            _atomic_write_json(run_path, metadata)
            logger.info("Saved %s (%d rows)", layer_path, len(result))
        finally:
            model.unload()


def main() -> None:
    """Parse CLI arguments and execute the layerwise runner."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--pregrouper",
        type=_pregrouper,
        default="sentence",
        choices=["sentence", "word"],
    )
    parser.add_argument("--batch-size", type=int, default=CANONICAL_BATCH_SIZE)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-set", default="Qwen2.5-Instruct", choices=sorted(MODEL_SETS)
    )
    parser.add_argument(
        "--models", help="Comma-separated model names selected from --model-set"
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--dataset-file")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    asyncio.run(
        run_layerwise(
            benchmark_name=args.benchmark,
            pregrouper=args.pregrouper,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            seed=args.seed,
            model_set=args.model_set,
            models=args.models,
            results_dir=args.results_dir,
            dataset_file=args.dataset_file,
            canary=args.canary,
            overwrite_existing=args.overwrite_existing,
        )
    )


if __name__ == "__main__":
    main()
