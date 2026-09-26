# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Validation helpers for portable hosted completion-score payloads."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

CANARY_SAMPLE_SIZE: int = 10
CANARY_SEED: int = 42
CANARY_SELECTION: str = (
    "numpy_default_rng_choice_without_replacement_sorted_from_5153_frozen_rows"
)
# Audited against the frozen 5,153-row LAMBADA source and full-dialog word
# segmentation. These prompts need not occur in the separately subsampled
# 10,000-ablation result manifest.
LAMBADA_GOLD_CANARY: tuple[tuple[int, int], ...] = (
    (442, 83),
    (459, 117),
    (485, 86),
    (1037, 73),
    (2229, 73),
    (2258, 84),
    (3368, 81),
    (3592, 75),
    (3982, 72),
    (4420, 79),
)
LAMBADA_GOLD_CANARY_ANSWERS: dict[int, str] = {
    442: "wendy",
    459: "crib",
    485: "bare",
    1037: "owl",
    2229: "bree",
    2258: "then",
    3368: "winnie",
    3592: "jared",
    3982: "tiger",
    4420: "angie",
}


def expected_canary_prompt_indices(
    seed: int,
    sample_size: int = CANARY_SAMPLE_SIZE,
) -> list[int]:
    """Return the audited canary sample from the frozen LAMBADA population."""
    if seed != CANARY_SEED or sample_size != CANARY_SAMPLE_SIZE:
        raise ValueError(
            f"Gold LAMBADA canary requires seed={CANARY_SEED} and "
            f"sample_size={CANARY_SAMPLE_SIZE}"
        )
    return [prompt_idx for prompt_idx, _ in LAMBADA_GOLD_CANARY]


def _finite_score(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def _completion_values_for_manifest(
    result: dict[str, Any],
    prompt_segments: pd.DataFrame,
) -> list[Any]:
    """Align one prompt-level ablation list to its canonical manifest rows.

    A non-empty payload must either contain the full ``n_segments``-long list,
    indexed by global ``seg_idx``, or carry explicit ``ablation_indices``. An
    empty list is a supported representation of a wholly failed prompt and is
    expanded to canonical missing values so it cannot shrink the denominator.
    """
    if "n_segments" not in prompt_segments.columns:
        raise ValueError("Segment manifest lacks n_segments")
    declared_counts: set[int] = set(prompt_segments["n_segments"].astype(int))
    if len(declared_counts) != 1:
        raise ValueError("Manifest has inconsistent n_segments within a prompt")
    expected_total: int = next(iter(declared_counts))
    canonical_indices: list[int] = prompt_segments["seg_idx"].astype(int).tolist()
    if len(set(canonical_indices)) != len(canonical_indices):
        raise ValueError("Manifest has duplicate segment indices within a prompt")
    if any(index < 0 or index >= expected_total for index in canonical_indices):
        raise ValueError("Manifest segment index is outside the declared segment count")

    raw_values: Any = result.get("ablated_logprob")
    if raw_values is None:
        values: list[Any] = []
    elif isinstance(raw_values, list):
        values = raw_values
    else:
        raise ValueError("Prompt-level ablated_logprob must be a list or null")

    declared_result_count: int = int(result.get("n_segments", expected_total))
    explicit_indices_value: Any = result.get("ablation_indices")
    explicit_indices: list[int] | None = (
        [int(value) for value in explicit_indices_value]
        if explicit_indices_value is not None
        else None
    )

    if not values:
        if declared_result_count not in {0, expected_total}:
            raise ValueError(
                f"Empty completion result declares n_segments={declared_result_count}; "
                f"expected 0 or {expected_total}"
            )
        if explicit_indices not in (None, []):
            raise ValueError("Empty completion result has non-empty ablation_indices")
        return [None] * len(canonical_indices)

    if declared_result_count != expected_total:
        raise ValueError(
            f"Completion result declares n_segments={declared_result_count}; "
            f"manifest declares {expected_total}"
        )
    if explicit_indices is None:
        if len(values) != expected_total:
            raise ValueError(
                f"Completion result has {len(values)} ablations without indices; "
                f"expected the full {expected_total}"
            )
        indexed_values: dict[int, Any] = dict(enumerate(values))
    else:
        if len(explicit_indices) != len(values) or len(set(explicit_indices)) != len(
            explicit_indices
        ):
            raise ValueError("Completion ablation_indices are duplicated or mis-sized")
        if any(index < 0 or index >= expected_total for index in explicit_indices):
            raise ValueError("Completion ablation index is outside n_segments")
        indexed_values = dict(zip(explicit_indices, values))
        missing_indices: set[int] = set(canonical_indices) - set(indexed_values)
        if missing_indices:
            raise ValueError(
                "Completion result omits canonical ablation indices "
                f"{sorted(missing_indices)}"
            )
    return [indexed_values[index] for index in canonical_indices]


def _completion_statuses_for_manifest(
    result: dict[str, Any],
    prompt_segments: pd.DataFrame,
    values: list[Any],
) -> list[str]:
    """Align optional prompt-level reason codes to canonical manifest rows."""
    raw_plural: Any = result.get("ablated_statuses")
    raw_singular: Any = result.get("ablated_status")
    if raw_plural is not None and raw_singular is not None:
        raise ValueError(
            "Prompt-level completion result has both ablated_status and "
            "ablated_statuses"
        )
    raw_statuses: Any = raw_plural if raw_plural is not None else raw_singular
    if raw_statuses is None:
        return ["ok" if _finite_score(value) else "unavailable" for value in values]
    if not isinstance(raw_statuses, list):
        raise ValueError("Prompt-level ablation statuses must be a list")
    status_result: dict[str, Any] = dict(result)
    status_result["ablated_logprob"] = raw_statuses
    aligned: list[Any] = _completion_values_for_manifest(status_result, prompt_segments)
    if any(not isinstance(status, str) or not status for status in aligned):
        raise ValueError("Prompt-level ablation statuses must be non-empty strings")
    return [str(status) for status in aligned]


def expand_prompt_completion_payload(
    payload: list[dict[str, Any]],
    manifest: pd.DataFrame,
    expected_prompt_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Expand prompt-level completion JSON to canonical per-segment records."""
    manifest_by_prompt: dict[int, pd.DataFrame] = {
        int(prompt_idx): rows.sort_values("seg_idx")
        for prompt_idx, rows in manifest.groupby("prompt_idx", sort=False)
    }
    expected: list[int] = (
        sorted(manifest_by_prompt)
        if expected_prompt_indices is None
        else list(expected_prompt_indices)
    )
    observed: list[int] = [int(row["prompt_idx"]) for row in payload]
    if len(set(observed)) != len(observed):
        raise ValueError("Prompt-level completion payload has duplicate prompt IDs")
    if expected_prompt_indices is not None and observed != expected:
        raise ValueError(
            "Prompt-level completion IDs disagree with the canonical expected "
            f"order: expected {expected}, observed {observed}"
        )
    if expected_prompt_indices is None and not set(expected).issubset(observed):
        raise ValueError(
            "Prompt-level completion coverage differs from the manifest: "
            f"{len(set(expected) - set(observed))} canonical prompts missing"
        )

    rows: list[dict[str, Any]] = []
    results_by_prompt: dict[int, dict[str, Any]] = dict(zip(observed, payload))
    for prompt_idx in expected:
        result: dict[str, Any] = results_by_prompt[prompt_idx]
        if prompt_idx not in manifest_by_prompt:
            raise ValueError(f"Prompt {prompt_idx} is absent from segment manifest")
        prompt_segments: pd.DataFrame = manifest_by_prompt[prompt_idx]
        manifest_answers: set[str] = set(prompt_segments["answer"].map(str))
        if "answer" not in result or manifest_answers != {str(result["answer"])}:
            raise ValueError(
                f"Prompt {prompt_idx}: answer disagrees with manifest "
                f"answer(s) {sorted(manifest_answers)}"
            )
        values: list[Any] = _completion_values_for_manifest(result, prompt_segments)
        statuses: list[str] = _completion_statuses_for_manifest(
            result, prompt_segments, values
        )
        original_status: Any = result.get(
            "orig_status",
            "ok" if _finite_score(result.get("orig_logprob")) else "unavailable",
        )
        if not isinstance(original_status, str) or not original_status:
            raise ValueError("Prompt-level orig_status must be a non-empty string")
        for segment, value, status in zip(
            prompt_segments.itertuples(index=False), values, statuses
        ):
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "answer": result["answer"],
                    "ablation_idx": int(segment.seg_idx),
                    "n_segments": int(segment.n_segments),
                    "orig_logprob": result.get("orig_logprob"),
                    "orig_status": original_status,
                    "ablated_logprob": value,
                    "ablated_status": status,
                }
            )
    return rows


def summarize_completion_canary(
    payload: list[dict[str, Any]],
    seed: int,
    sample_size: int = CANARY_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Validate and summarize the audited frozen-dataset completion canary."""
    expected_indices: list[int] = expected_canary_prompt_indices(seed, sample_size)
    observed: list[int] = [int(row["prompt_idx"]) for row in payload]
    if observed != expected_indices:
        raise ValueError(
            "Canary prompt IDs disagree with the audited frozen LAMBADA sample: "
            f"expected {expected_indices}, observed {observed}"
        )
    expected_counts: dict[int, int] = dict(LAMBADA_GOLD_CANARY)
    expanded: list[dict[str, Any]] = []
    for result in payload:
        prompt_idx: int = int(result["prompt_idx"])
        expected_count: int = expected_counts[prompt_idx]
        expected_answer: str = LAMBADA_GOLD_CANARY_ANSWERS[prompt_idx]
        if str(result.get("answer", "")) != expected_answer:
            raise ValueError(
                f"Canary prompt {prompt_idx} answer disagrees with the frozen dataset"
            )
        if int(result.get("n_segments", -1)) != expected_count:
            raise ValueError(
                f"Canary prompt {prompt_idx} must declare "
                f"n_segments={expected_count}"
            )
        raw_values: Any = result.get("ablated_logprob")
        if not isinstance(raw_values, list) or len(raw_values) != expected_count:
            raise ValueError(
                f"Canary prompt {prompt_idx} must contain exactly "
                f"{expected_count} ablated scores"
            )
        explicit_indices: Any = result.get("ablation_indices")
        if explicit_indices is not None and [
            int(value) for value in explicit_indices
        ] != list(range(expected_count)):
            raise ValueError(
                f"Canary prompt {prompt_idx} ablation indices are not canonical"
            )
        prompt_segments: pd.DataFrame = pd.DataFrame(
            {
                "seg_idx": range(expected_count),
                "n_segments": [expected_count] * expected_count,
            }
        )
        values: list[Any] = _completion_values_for_manifest(result, prompt_segments)
        expanded.extend(
            {
                "prompt_idx": prompt_idx,
                "orig_logprob": result.get("orig_logprob"),
                "ablated_logprob": value,
            }
            for value in values
        )
    original_available_by_prompt: dict[int, bool] = {}
    ablated_available: list[bool] = []
    paired_available: list[bool] = []
    for row in expanded:
        prompt_idx: int = int(row["prompt_idx"])
        original_ok: bool = _finite_score(row.get("orig_logprob"))
        ablated_ok: bool = _finite_score(row.get("ablated_logprob"))
        original_available_by_prompt[prompt_idx] = original_ok
        ablated_available.append(ablated_ok)
        paired_available.append(original_ok and ablated_ok)
    return {
        "seed": seed,
        "selection": CANARY_SELECTION,
        "sample_size": sample_size,
        "prompt_indices": expected_indices,
        "expected_segments": len(expanded),
        "original_coverage": sum(original_available_by_prompt.values()) / sample_size,
        "ablated_coverage": sum(ablated_available) / len(expanded),
        "paired_attribution_coverage": sum(paired_available) / len(expanded),
    }
