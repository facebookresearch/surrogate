# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Consolidated benchmark runner for attention and ablation scoring across any
supported benchmark (BoolQ, ANLI R1-R3, WinoGrande, LAMBADA).

For each model and prompt:
  1. Score attention (mean/max/rollout) via single forward pass (skippable)
  2. Score ablation via leave-one-out (sentence- or word-level), recording
     per-segment representation metrics + per-(prompt, seg, kind, label,
     token) raw logprobs

Results are saved per-model under ``results/{benchmark}/{pregrouper}/`` (override
with ``--results-dir``):
  {model}_segment.tsv  one row per (prompt, seg)        — attention + rep metrics
  {model}_tokens.tsv   one row per (prompt, seg, kind,
                       label, token)                    — raw token logprobs

The benchmark script writes raw token logprobs only — log-odds and label-level
log-probabilities are computed downstream by ``compute_logodds`` so that
aggregation choices (logsumexp vs first-token, pos-vs-neg vs higher-dim
log-odds) can be revisited without rerunning the model.

Usage:
    # Default: BoolQ, sentence-level, all 4 Qwen2.5 instruct models
    python -m benchmark_scripts.run_benchmark --benchmark boolq

    # Word-level ablation requires --max-forward-passes (full pool too large)
    python -m benchmark_scripts.run_benchmark --benchmark boolq --pregrouper word --max-forward-passes 10000

    # Score only ablation into a fresh output directory
    python -m benchmark_scripts.run_benchmark --benchmark boolq --model-set Qwen2.5-Base --phases ablation --results-dir results/ablation-only

    # Filter to specific models within a set (env var, comma-separated)
    BENCHMARK_MODELS=qwen2.5-32b-base python -m benchmark_scripts.run_benchmark --benchmark boolq --model-set Qwen2.5-Base

Models and datasets are loaded from HuggingFace Hub.
"""

import argparse
import asyncio
from collections.abc import Mapping
import hashlib
import json
import logging
import os
from types import MappingProxyType
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from surrogate.attention_scoring import (
    attention_segment_scores,
    AttentionConfig,
    PregrouperID,
)
from benchmark_scripts.benchmark_config import (
    BENCHMARKS,
    BenchmarkSpec,
    load_benchmark_dataset,
    MODEL_SETS,
    resolve_model_path,
)
from surrogate.eval_constants import ReportToken
from surrogate.representation_scoring import (
    extract_repr_for_dialogs,
    representation_metrics_batch,
    w_norm,
)
from surrogate.text_augmentation import dialog_segments, segment_and_ablate
from surrogate.transformers_model import TransformersModel
from surrogate.transformers_scoring import (
    transformers_completion_logprob_batch,
    transformers_label_logprob_breakdown,
    transformers_label_logprob_breakdown_batch,
)
from surrogate.model_types import Dialog, LogprobBreakdown, make_dialog
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)

ATTENTION_CONFIGS: list[tuple[str, AttentionConfig]] = [
    ("mean", AttentionConfig(aggregation="mean")),
    ("max", AttentionConfig(aggregation="max")),
    ("rollout", AttentionConfig(aggregation="rollout")),
]
GZIP_COMPRESSION: dict[str, Any] = {
    "method": "gzip",
    "compresslevel": 9,
    "mtime": 0,
}
EXECUTION_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/run_benchmark.py",
    "surrogate/attention_scoring.py",
    "surrogate/eval_constants.py",
    "surrogate/model_types.py",
    "surrogate/representation_scoring.py",
    "surrogate/text_augmentation.py",
    "surrogate/transformers_model.py",
    "surrogate/transformers_scoring.py",
    "surrogate/utils.py",
)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    payload: bytes = frame.to_csv(index=True, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _snapshot_execution_source_hashes() -> Mapping[str, str]:
    """Capture the complete scoring dependency set before a run begins."""
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    return MappingProxyType(
        {
            relative_path: _sha256_file(os.path.join(repository_root, relative_path))
            for relative_path in EXECUTION_SOURCE_FILES
        }
    )


def _model_file_hashes(model_path: str) -> dict[str, str]:
    """Hash small identity/config files without traversing model weights."""
    if not os.path.isdir(model_path):
        return {}
    result: dict[str, str] = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ):
        path: str = os.path.join(model_path, name)
        if os.path.isfile(path):
            result[name] = _sha256_file(path)
    return result


def _write_run_metadata(
    output_dir: str,
    model_name: str,
    model_source: str,
    model_path: str,
    spec: BenchmarkSpec,
    pregrouper: PregrouperID,
    phases: set[str],
    batch_size: int,
    max_samples: int | None,
    max_forward_passes: int | None,
    seed: int,
    dataset_file: str | None,
    dataset: pd.DataFrame,
    execution_source_sha256: Mapping[str, str],
) -> None:
    """Write portable provenance for one public runner invocation."""
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "benchmark": spec.name,
        "pregrouper": pregrouper,
        "segmentation_scope": "full_dialog_in_message_order",
        "model": model_name,
        "model_source": model_source,
        "model_identity_files_sha256": _model_file_hashes(model_path),
        "dataset": {
            "hf_path": spec.hf_dataset_path,
            "hf_name": spec.hf_dataset_name,
            "hf_split": spec.hf_split,
            "snapshot_filename": (
                os.path.basename(dataset_file) if dataset_file is not None else None
            ),
            "snapshot_sha256": (
                _sha256_file(dataset_file) if dataset_file is not None else None
            ),
            "normalized_frame_sha256": _dataframe_sha256(dataset),
            "rows": len(dataset),
        },
        "parameters": {
            "phases": sorted(phases),
            "phase_attention_implementation": {
                phase: _attention_implementation_for_phase(phase)
                for phase in sorted(phases)
            },
            "batch_size": batch_size,
            "max_samples": max_samples,
            "max_forward_passes": max_forward_passes,
            "seed": seed,
        },
        "software": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": str(torch.__version__),
        },
        "source_sha256": dict(execution_source_sha256),
    }
    path: str = os.path.join(output_dir, f"{model_name}_run.json")
    with open(path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def _segment_metadata(
    dialog: Dialog,
    pregrouper_id: PregrouperID,
) -> list[dict[str, str | int]]:
    """Return auditable source metadata for every global dialog segment."""
    return [
        {
            "seg_idx": segment.segment_idx,
            "message_idx": segment.message_idx,
            "message_role": segment.message_role,
            "message_seg_idx": segment.message_segment_idx,
            "segment_text": segment.text,
        }
        for segment in dialog_segments(dialog, pregrouper_id)
    ]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _build_dialog(prompt_text: str, system_prompt: str) -> Dialog:
    return make_dialog(system_prompt, prompt_text)


def _resolve_label_token_ids(
    model: TransformersModel,
    tokens: set[str] | list[str] | str,
) -> list[int]:
    """Map label-token strings to single-token IDs in the model's vocab.

    Multi-token strings are silently dropped (matches the behavior of the
    logit-difference scorers, which can only score single-token labels).
    """
    if isinstance(tokens, str):
        tokens_list: list[str] = [tokens]
    else:
        tokens_list = list(tokens)
    ids: list[int] = []
    for t in tokens_list:
        tid: int | None = model.encode_single_token(t)
        if tid is not None:
            ids.append(tid)
    return ids


def _as_token_set(tokens: set[str] | list[str] | str) -> set[str]:
    return {tokens} if isinstance(tokens, str) else set(tokens)


def _contrast_token_sets(
    eval_config: Any,
    scoring_mode: str,
    answer: Any,
) -> tuple[set[str], set[str]]:
    """Resolve the positive and negative token sets for one prompt."""
    label_tokens: dict[str, set[str] | list[str] | str] = eval_config.label_tokens
    if scoring_mode != "per_prompt_logit_difference":
        first, second = list(label_tokens.values())[:2]
        return _as_token_set(first), _as_token_set(second)

    answer_key: str = str(answer)
    if answer_key not in label_tokens:
        raise ValueError(
            f"Answer {answer_key!r} is not a configured label: " f"{list(label_tokens)}"
        )
    positive: set[str] = _as_token_set(label_tokens[answer_key])
    negative: set[str] = set().union(
        *(
            _as_token_set(tokens)
            for label, tokens in label_tokens.items()
            if label != answer_key
        )
    )
    return positive, negative


def _length_sort_perm(
    model: TransformersModel,
    dialogs: list[Dialog],
) -> tuple[list[int], list[int]]:
    """Return ``(perm, inv_perm)`` sorting ``dialogs`` by chat-templated text length.

    Sorting first eliminates >99% of intra-batch padding waste; critical
    for fitting the larger models (Llama-8B, Qwen-14B) at batch_size=32.
    """
    texts: list[str] = [model.dialog_to_text(d) for d in dialogs]
    lengths: list[int] = [len(t) for t in texts]
    perm: list[int] = sorted(range(len(dialogs)), key=lambda i: lengths[i])
    inv_perm: list[int] = [0] * len(perm)
    for sorted_i, orig_i in enumerate(perm):
        inv_perm[orig_i] = sorted_i
    return perm, inv_perm


def _completion_logprob_length_sorted(
    model: TransformersModel,
    dialogs: list[Dialog],
    target_texts: list[str],
    batch_size: int,
) -> list[float | None]:
    if len(dialogs) <= 1:
        return transformers_completion_logprob_batch(
            model, dialogs, target_texts, batch_size=batch_size
        )
    perm, inv_perm = _length_sort_perm(model, dialogs)
    sorted_dialogs: list[Dialog] = [dialogs[i] for i in perm]
    sorted_targets: list[str] = [target_texts[i] for i in perm]
    sorted_scores: list[float | None] = transformers_completion_logprob_batch(
        model, sorted_dialogs, sorted_targets, batch_size=batch_size
    )
    return [sorted_scores[inv_perm[i]] for i in range(len(dialogs))]


def _label_logprob_breakdown_length_sorted(
    model: TransformersModel,
    dialogs: list[Dialog],
    label_tokens: dict[str, set[str] | list[str] | str],
    report_tokens: dict[str, list[ReportToken]],
    batch_size: int,
) -> list[LogprobBreakdown | None]:
    if len(dialogs) <= 1:
        return transformers_label_logprob_breakdown_batch(
            model,
            dialogs,
            label_tokens=label_tokens,
            report_tokens=report_tokens,
            batch_size=batch_size,
        )
    perm, inv_perm = _length_sort_perm(model, dialogs)
    sorted_dialogs: list[Dialog] = [dialogs[i] for i in perm]
    sorted_breakdowns: list[LogprobBreakdown | None] = (
        transformers_label_logprob_breakdown_batch(
            model,
            sorted_dialogs,
            label_tokens=label_tokens,
            report_tokens=report_tokens,
            batch_size=batch_size,
        )
    )
    return [sorted_breakdowns[inv_perm[i]] for i in range(len(dialogs))]


def _extract_repr_length_sorted(
    model: TransformersModel,
    dialogs: list[Dialog],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(dialogs) <= 1:
        return extract_repr_for_dialogs(model, dialogs, batch_size=batch_size)
    perm, inv_perm = _length_sort_perm(model, dialogs)
    sorted_dialogs: list[Dialog] = [dialogs[i] for i in perm]
    pre, post = extract_repr_for_dialogs(model, sorted_dialogs, batch_size=batch_size)
    inv_idx: torch.Tensor = torch.tensor(inv_perm, device=pre.device)
    return pre[inv_idx], post[inv_idx]


# ---------------------------------------------------------------------------
# Token-row builder: turn LogprobBreakdown into long-format rows
# ---------------------------------------------------------------------------


def _token_rows_from_breakdown(
    breakdown: LogprobBreakdown | None,
    prompt_idx: int,
    seg_idx: int | None,
    kind: str,
    answer: Any,
) -> list[dict[str, Any]]:
    """Flatten a LogprobBreakdown into one row per (label, token).

    ``seg_idx`` is the segment index for ``kind="ablated"`` rows, or
    ``None`` for ``kind="orig"`` rows (the original prompt has no segment
    grain — the same orig logprobs apply to all segment ablations).
    """
    if breakdown is None:
        return []
    rows: list[dict[str, Any]] = []
    for label, tokens in breakdown.token_logprobs.items():
        for token_alias, logprob in tokens.items():
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "seg_idx": seg_idx,
                    "kind": kind,
                    "answer": answer,
                    "label": label,
                    "token": token_alias,
                    "logprob": logprob,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Attention scoring: returns segment_rows
# ---------------------------------------------------------------------------


async def _score_attention(
    model: TransformersModel,
    dialogs: list[Dialog],
    prompts_meta: list[dict[str, Any]],
    pregrouper_id: PregrouperID = "sentence",
    continue_on_error: bool = False,
) -> list[dict[str, Any]]:
    """Compute attention scores. Returns list of (prompt_idx, seg_idx) rows
    with attention_{mean,max,rollout} columns."""
    seg_rows: list[dict[str, Any]] = []
    for i, dialog in enumerate(tqdm(dialogs, desc=f"Attention: {model.model_name}")):
        segment_metadata: list[dict[str, str | int]] = _segment_metadata(
            dialog, pregrouper_id
        )
        per_config: dict[str, list[float] | None] = {}
        for config_name, config in ATTENTION_CONFIGS:
            try:
                scores: list[float] = await attention_segment_scores(
                    model=model,
                    dialog=dialog,
                    pregrouper_id=pregrouper_id,
                    config=config,
                )
                per_config[config_name] = scores
            except Exception as error:
                prompt_idx: int = int(prompts_meta[i]["prompt_idx"])
                if not continue_on_error:
                    raise RuntimeError(
                        f"Attention ({config_name}) failed for prompt " f"{prompt_idx}"
                    ) from error
                logger.warning(
                    f"Attention ({config_name}) failed for prompt "
                    f"{prompt_idx}: {error}"
                )
                per_config[config_name] = None
        # Width: every config returns N segments (same N); pick from any
        # successful one. If all failed, skip the prompt.
        n_seg: int = max(
            (len(v) for v in per_config.values() if v is not None), default=0
        )
        if n_seg != len(segment_metadata):
            raise ValueError(
                f"Attention returned {n_seg} scores for "
                f"{len(segment_metadata)} segments on prompt "
                f"{prompts_meta[i]['prompt_idx']}"
            )
        for seg_idx in range(n_seg):
            row: dict[str, Any] = {
                **prompts_meta[i],
                **segment_metadata[seg_idx],
                "n_segments": n_seg,
            }
            for config_name in (n for n, _ in ATTENTION_CONFIGS):
                v: list[float] | None = per_config[config_name]
                row[f"attention_{config_name}"] = (
                    v[seg_idx] if v is not None and seg_idx < len(v) else None
                )
            seg_rows.append(row)
    return seg_rows


# ---------------------------------------------------------------------------
# Ablation scoring: returns (segment_rows, token_rows)
# ---------------------------------------------------------------------------


async def _score_ablation_full(
    model: TransformersModel,
    dialogs: list[Dialog],
    prompts_meta: list[dict[str, Any]],
    eval_config: Any,
    scoring_mode: str,
    pregrouper_id: PregrouperID = "sentence",
    batch_size: int = 32,
    continue_on_error: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Full-mode ablation: every prompt × every segment. Returns
    (segment_rows, token_rows).

    Drops all log-odds / label-aggregated logprobs from output — only raw
    per-token logprobs and per-segment representation metrics are written.
    """
    label_tokens: dict[str, set[str] | list[str] | str] = eval_config.label_tokens
    report_tokens: dict[str, list[ReportToken]] = eval_config.report_tokens
    direction_cache: dict[str, tuple[Any, Any, float]] = {}

    seg_rows: list[dict[str, Any]] = []
    tok_rows: list[dict[str, Any]] = []

    for i, dialog in enumerate(tqdm(dialogs, desc=f"Ablation: {model.model_name}")):
        prompt_idx: int = int(prompts_meta[i]["prompt_idx"])
        try:
            direction_key: str = (
                str(prompts_meta[i]["answer"])
                if scoring_mode == "per_prompt_logit_difference"
                else "fixed"
            )
            if direction_key not in direction_cache:
                pos_tokens, neg_tokens = _contrast_token_sets(
                    eval_config,
                    scoring_mode,
                    prompts_meta[i]["answer"],
                )
                pos_ids: list[int] = _resolve_label_token_ids(model, pos_tokens)
                neg_ids: list[int] = _resolve_label_token_ids(model, neg_tokens)
                w_vec: Any = model.get_unembedding_direction(pos_ids, neg_ids)
                w_vec_pre: Any = model.get_unembedding_direction_prenorm(
                    pos_ids, neg_ids
                )
                direction_cache[direction_key] = (
                    w_vec,
                    w_vec_pre,
                    w_norm(w_vec),
                )
            w_vec, w_vec_pre, model_w_norm = direction_cache[direction_key]
            orig_breakdown: LogprobBreakdown | None = (
                await transformers_label_logprob_breakdown(
                    model,
                    dialog,
                    label_tokens=label_tokens,
                    report_tokens=report_tokens,
                )
            )
            ablated_dialogs: list[Dialog] = await segment_and_ablate(
                dialog,
                pregrouper_id=cast(Any, pregrouper_id),
            )
            ablated_breakdowns: list[LogprobBreakdown | None] = (
                _label_logprob_breakdown_length_sorted(
                    model,
                    ablated_dialogs,
                    label_tokens=label_tokens,
                    report_tokens=report_tokens,
                    batch_size=batch_size,
                )
            )
            orig_pre, orig_post = extract_repr_for_dialogs(
                model, [dialog], batch_size=batch_size
            )
            pert_pre, pert_post = extract_repr_for_dialogs(
                model, ablated_dialogs, batch_size=batch_size
            )
            rep_rows: list[dict[str, float]] = representation_metrics_batch(
                z_orig_prenorm=orig_pre,
                z_orig_postnorm=orig_post,
                z_pert_prenorm=pert_pre,
                z_pert_postnorm=pert_post,
                w_postnorm=w_vec,
                w_prenorm=w_vec_pre,
            )
        except Exception as error:
            if not continue_on_error:
                raise RuntimeError(
                    f"Ablation failed for prompt {prompt_idx}"
                ) from error
            logger.warning(f"Ablation failed for prompt {prompt_idx}: {error}")
            continue

        n_seg: int = len(ablated_dialogs)
        segment_metadata: list[dict[str, str | int]] = _segment_metadata(
            dialog, pregrouper_id
        )
        if n_seg != len(segment_metadata):
            raise ValueError(
                f"Ablation generated {n_seg} dialogs for "
                f"{len(segment_metadata)} segments on prompt {prompt_idx}"
            )
        # One orig row per prompt — appears once with seg_idx=None.
        tok_rows.extend(
            _token_rows_from_breakdown(
                orig_breakdown,
                prompt_idx,
                None,
                "orig",
                prompts_meta[i]["answer"],
            )
        )
        for seg_idx in range(n_seg):
            seg_rows.append(
                {
                    **prompts_meta[i],
                    **segment_metadata[seg_idx],
                    "n_segments": n_seg,
                    "w_norm": model_w_norm,
                    **(rep_rows[seg_idx] if seg_idx < len(rep_rows) else {}),
                }
            )
            tok_rows.extend(
                _token_rows_from_breakdown(
                    (
                        ablated_breakdowns[seg_idx]
                        if seg_idx < len(ablated_breakdowns)
                        else None
                    ),
                    prompt_idx,
                    seg_idx,
                    "ablated",
                    prompts_meta[i]["answer"],
                )
            )
    return seg_rows, tok_rows


def _score_ablation_subsampled(
    model: TransformersModel,
    dialogs: list[Dialog],
    prompts_meta: list[dict[str, Any]],
    all_ablated: list[list[Dialog]],
    selected_pairs: list[tuple[int, int]],
    eval_config: Any,
    scoring_mode: str,
    pregrouper_id: PregrouperID,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Subsampled-mode ablation: only selected (prompt_idx, seg_idx) pairs."""
    label_tokens: dict[str, set[str] | list[str] | str] = eval_config.label_tokens
    report_tokens: dict[str, list[ReportToken]] = eval_config.report_tokens

    unique_prompt_indices: list[int] = sorted({pi for pi, _ in selected_pairs})
    orig_dialogs: list[Dialog] = [dialogs[pi] for pi in unique_prompt_indices]
    logger.info(
        f"Scoring {len(orig_dialogs)} originals + "
        f"{len(selected_pairs)} ablations for {model.model_name}"
    )
    orig_breakdowns_list: list[LogprobBreakdown | None] = (
        _label_logprob_breakdown_length_sorted(
            model,
            orig_dialogs,
            label_tokens=label_tokens,
            report_tokens=report_tokens,
            batch_size=batch_size,
        )
    )
    orig_breakdowns: dict[int, LogprobBreakdown | None] = dict(
        zip(unique_prompt_indices, orig_breakdowns_list)
    )

    torch.cuda.empty_cache()
    orig_pre_stack, orig_post_stack = _extract_repr_length_sorted(
        model, orig_dialogs, batch_size=batch_size
    )
    orig_pre: dict[int, Any] = {
        pi: orig_pre_stack[k] for k, pi in enumerate(unique_prompt_indices)
    }
    orig_post: dict[int, Any] = {
        pi: orig_post_stack[k] for k, pi in enumerate(unique_prompt_indices)
    }

    directions: dict[int, tuple[Any, Any, float]] = {}
    direction_cache: dict[str, tuple[Any, Any, float]] = {}
    for pi in unique_prompt_indices:
        direction_key: str = (
            str(prompts_meta[pi]["answer"])
            if scoring_mode == "per_prompt_logit_difference"
            else "fixed"
        )
        if direction_key not in direction_cache:
            pos_tokens, neg_tokens = _contrast_token_sets(
                eval_config,
                scoring_mode,
                prompts_meta[pi]["answer"],
            )
            pos_ids: list[int] = _resolve_label_token_ids(model, pos_tokens)
            neg_ids: list[int] = _resolve_label_token_ids(model, neg_tokens)
            w_vec: Any = model.get_unembedding_direction(pos_ids, neg_ids)
            w_vec_pre: Any = model.get_unembedding_direction_prenorm(pos_ids, neg_ids)
            direction_cache[direction_key] = (
                w_vec,
                w_vec_pre,
                w_norm(w_vec),
            )
        directions[pi] = direction_cache[direction_key]

    selected_dialogs: list[Dialog] = [all_ablated[pi][ai] for pi, ai in selected_pairs]
    ablated_breakdowns: list[LogprobBreakdown | None] = (
        _label_logprob_breakdown_length_sorted(
            model,
            selected_dialogs,
            label_tokens=label_tokens,
            report_tokens=report_tokens,
            batch_size=batch_size,
        )
    )

    torch.cuda.empty_cache()
    pert_pre_stack, pert_post_stack = _extract_repr_length_sorted(
        model, selected_dialogs, batch_size=batch_size
    )

    seg_rows: list[dict[str, Any]] = []
    tok_rows: list[dict[str, Any]] = []
    segment_metadata_by_prompt: dict[int, list[dict[str, str | int]]] = {}
    for pi in unique_prompt_indices:
        prompt_idx: int = int(prompts_meta[pi]["prompt_idx"])
        tok_rows.extend(
            _token_rows_from_breakdown(
                orig_breakdowns[pi],
                prompt_idx,
                None,
                "orig",
                prompts_meta[pi]["answer"],
            )
        )

    for idx, (pi, ai) in enumerate(selected_pairs):
        prompt_idx = int(prompts_meta[pi]["prompt_idx"])
        if pi not in segment_metadata_by_prompt:
            segment_metadata_by_prompt[pi] = _segment_metadata(
                dialogs[pi], pregrouper_id
            )
        segment_metadata = segment_metadata_by_prompt[pi][ai]
        w_vec, w_vec_pre, model_w_norm = directions[pi]
        rep_row: dict[str, float] = representation_metrics_batch(
            z_orig_prenorm=orig_pre[pi].unsqueeze(0),
            z_orig_postnorm=orig_post[pi].unsqueeze(0),
            z_pert_prenorm=pert_pre_stack[idx].unsqueeze(0),
            z_pert_postnorm=pert_post_stack[idx].unsqueeze(0),
            w_postnorm=w_vec,
            w_prenorm=w_vec_pre,
        )[0]
        seg_rows.append(
            {
                **prompts_meta[pi],
                **segment_metadata,
                "n_segments": len(all_ablated[pi]),
                "w_norm": model_w_norm,
                **rep_row,
            }
        )
        tok_rows.extend(
            _token_rows_from_breakdown(
                ablated_breakdowns[idx],
                prompt_idx,
                ai,
                "ablated",
                prompts_meta[pi]["answer"],
            )
        )

    return seg_rows, tok_rows


def _score_ablation_subsampled_completion(
    model: TransformersModel,
    dialogs: list[Dialog],
    prompts_meta: list[dict[str, Any]],
    all_ablated: list[list[Dialog]],
    selected_pairs: list[tuple[int, int]],
    target_texts: list[str],
    pregrouper_id: PregrouperID,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Completion-logprob ablation (LAMBADA). Returns segment_rows with
    ``orig_completion_logprob`` / ``ablated_completion_logprob`` columns
    + per-segment rep metrics. tokens TSV is empty for this path because
    the per-target-token logprobs aren't currently exposed by the
    underlying scorer."""
    unique_prompt_indices: list[int] = sorted({pi for pi, _ in selected_pairs})
    orig_dialogs: list[Dialog] = [dialogs[pi] for pi in unique_prompt_indices]
    orig_targets: list[str] = [target_texts[pi] for pi in unique_prompt_indices]
    logger.info(
        f"Scoring {len(orig_dialogs)} originals + "
        f"{len(selected_pairs)} ablations for {model.model_name}"
    )
    orig_scores_list: list[float | None] = _completion_logprob_length_sorted(
        model, orig_dialogs, orig_targets, batch_size=batch_size
    )
    orig_scores: dict[int, float | None] = dict(
        zip(unique_prompt_indices, orig_scores_list)
    )

    torch.cuda.empty_cache()
    orig_pre_stack, orig_post_stack = _extract_repr_length_sorted(
        model, orig_dialogs, batch_size=batch_size
    )
    orig_pre: dict[int, Any] = {
        pi: orig_pre_stack[k] for k, pi in enumerate(unique_prompt_indices)
    }
    orig_post: dict[int, Any] = {
        pi: orig_post_stack[k] for k, pi in enumerate(unique_prompt_indices)
    }

    # Per-prompt w (sum of lm_head rows for the target's tokens). For
    # LAMBADA there's no negative class — w is just the positive-target sum.
    w_post_per_prompt: dict[int, Any] = {}
    w_pre_per_prompt: dict[int, Any] = {}
    for pi in unique_prompt_indices:
        target_tokens: list[int] = model._tokenizer.encode(  # type: ignore[union-attr]
            target_texts[pi], add_special_tokens=False
        )
        if not target_tokens:
            continue
        w_post_per_prompt[pi] = model.get_unembedding_direction(target_tokens, [])
        w_pre_per_prompt[pi] = model.get_unembedding_direction_prenorm(
            target_tokens, []
        )

    selected_dialogs: list[Dialog] = [all_ablated[pi][ai] for pi, ai in selected_pairs]
    selected_targets: list[str] = [target_texts[pi] for pi, _ in selected_pairs]
    ablated_scores: list[float | None] = _completion_logprob_length_sorted(
        model, selected_dialogs, selected_targets, batch_size=batch_size
    )

    torch.cuda.empty_cache()
    pert_pre_stack, pert_post_stack = _extract_repr_length_sorted(
        model, selected_dialogs, batch_size=batch_size
    )

    seg_rows: list[dict[str, Any]] = []
    segment_metadata_by_prompt: dict[int, list[dict[str, str | int]]] = {}
    for idx, (pi, ai) in enumerate(selected_pairs):
        if pi not in segment_metadata_by_prompt:
            segment_metadata_by_prompt[pi] = _segment_metadata(
                dialogs[pi], pregrouper_id
            )
        segment_metadata: dict[str, str | int] = segment_metadata_by_prompt[pi][ai]
        if pi not in w_post_per_prompt:
            continue
        rep_row: dict[str, float] = representation_metrics_batch(
            z_orig_prenorm=orig_pre[pi].unsqueeze(0),
            z_orig_postnorm=orig_post[pi].unsqueeze(0),
            z_pert_prenorm=pert_pre_stack[idx].unsqueeze(0),
            z_pert_postnorm=pert_post_stack[idx].unsqueeze(0),
            w_postnorm=w_post_per_prompt[pi],
            w_prenorm=w_pre_per_prompt[pi],
        )[0]
        seg_rows.append(
            {
                **prompts_meta[pi],
                **segment_metadata,
                "n_segments": len(all_ablated[pi]),
                "w_norm": float(w_post_per_prompt[pi].norm().item()),
                "orig_completion_logprob": orig_scores[pi],
                "ablated_completion_logprob": ablated_scores[idx],
                **rep_row,
            }
        )
    return seg_rows, []


async def _generate_all_ablations(
    dialogs: list[Dialog],
    pregrouper_id: str,
) -> list[list[Dialog]]:
    """Generate all ablated dialogs up front (text-only, no GPU)."""
    all_ablated: list[list[Dialog]] = []
    for dialog in tqdm(dialogs, desc=f"Generating {pregrouper_id} ablations"):
        ablated: list[Dialog] = await segment_and_ablate(
            dialog, pregrouper_id=cast(Any, pregrouper_id)
        )
        all_ablated.append(ablated)
    return all_ablated


def _subsample_ablation_pairs(
    all_ablated: list[list[Dialog]],
    max_forward_passes: int,
    seed: int,
) -> list[tuple[int, int]]:
    """Select a deterministic random subset of (prompt_idx, ablation_idx) pairs."""
    pool: list[tuple[int, int]] = [
        (pi, ai) for pi, ablated in enumerate(all_ablated) for ai in range(len(ablated))
    ]
    n_select: int = min(max_forward_passes, len(pool))
    rng: np.random.Generator = np.random.default_rng(seed)
    selected_indices: np.ndarray = rng.choice(len(pool), size=n_select, replace=False)
    selected_indices.sort()
    selected: list[tuple[int, int]] = [pool[i] for i in selected_indices]
    n_prompts: int = len({pi for pi, _ in selected})
    logger.info(
        f"Subsampled {n_select} ablations from {len(pool)} total "
        f"(across {n_prompts} unique prompts, seed={seed}). "
        f"Originals add {n_prompts} forward passes -> "
        f"{n_select + n_prompts} total."
    )
    return selected


# ---------------------------------------------------------------------------
# TSV writers
# ---------------------------------------------------------------------------


def _write_segment_manifest(
    output_dir: str,
    dialogs: list[Dialog],
    prompts_meta: list[dict[str, Any]],
    pregrouper_id: PregrouperID,
    selected_pairs: list[tuple[int, int]] | None,
) -> None:
    """Write the canonical segment identities used by every model and phase."""
    selected: set[tuple[int, int]] | None = (
        set(selected_pairs) if selected_pairs is not None else None
    )
    rows: list[dict[str, Any]] = []
    for prompt_position, dialog in enumerate(dialogs):
        metadata: list[dict[str, str | int]] = _segment_metadata(dialog, pregrouper_id)
        for segment in metadata:
            segment_idx: int = int(segment["seg_idx"])
            if selected is not None and (prompt_position, segment_idx) not in selected:
                continue
            rows.append(
                {
                    **prompts_meta[prompt_position],
                    **segment,
                    "n_segments": len(metadata),
                }
            )
    manifest_path: str = os.path.join(output_dir, "segments.tsv.gz")
    manifest: pd.DataFrame = pd.DataFrame(rows)
    if os.path.exists(manifest_path):
        existing: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
        try:
            pd.testing.assert_frame_equal(
                existing,
                manifest,
                check_dtype=False,
                check_like=False,
            )
        except AssertionError as error:
            raise ValueError(
                f"Existing segment manifest disagrees with this run: "
                f"{manifest_path}"
            ) from error
        logger.info(f"Validated existing {manifest_path} ({len(rows)} rows)")
        return
    manifest.to_csv(
        manifest_path,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )
    logger.info(f"Saved {manifest_path} ({len(rows)} rows)")


def _merge_segment_rows(
    attn_rows: list[dict[str, Any]],
    abl_rows: list[dict[str, Any]],
    restrict_to_ablation: bool = False,
) -> pd.DataFrame:
    """Join attention and ablation rows after validating segment alignment.

    Args:
        attn_rows: Full attention results.
        abl_rows: Full or deterministically subsampled ablation results.
        restrict_to_ablation: Whether to retain only ablated keys. This is
            required for forward-pass-budgeted runs so every model is compared
            on the same sampled segment set.
    """
    if not attn_rows and not abl_rows:
        return pd.DataFrame()
    df_attn: pd.DataFrame = (
        pd.DataFrame(attn_rows)
        if attn_rows
        else pd.DataFrame(columns=["prompt_idx", "seg_idx"])
    )
    df_abl: pd.DataFrame = (
        pd.DataFrame(abl_rows)
        if abl_rows
        else pd.DataFrame(columns=["prompt_idx", "seg_idx"])
    )
    key_cols: list[str] = ["prompt_idx", "seg_idx"]
    for phase, frame in (("attention", df_attn), ("ablation", df_abl)):
        if not frame.empty and frame.duplicated(key_cols).any():
            raise ValueError(f"Duplicate (prompt_idx, seg_idx) keys in {phase} rows")

    if attn_rows and abl_rows:
        attn_keys: set[tuple[int, int]] = set(
            df_attn[key_cols].itertuples(index=False, name=None)
        )
        abl_keys: set[tuple[int, int]] = set(
            df_abl[key_cols].itertuples(index=False, name=None)
        )
        expected_attn_keys: set[tuple[int, int]] = (
            abl_keys if restrict_to_ablation else attn_keys
        )
        if not abl_keys.issubset(attn_keys) or expected_attn_keys != abl_keys:
            raise ValueError(
                "Attention and ablation segments are misaligned: "
                f"{len(attn_keys - abl_keys)} attention-only keys and "
                f"{len(abl_keys - attn_keys)} ablation-only keys"
            )
        if restrict_to_ablation:
            key_index: pd.MultiIndex = pd.MultiIndex.from_frame(df_attn[key_cols])
            df_attn = df_attn[key_index.isin(abl_keys)]

    merged: pd.DataFrame = df_attn.merge(
        df_abl,
        on=["prompt_idx", "seg_idx"],
        how="outer",
        suffixes=("", "_abl"),
    )
    # Validate duplicate metadata before dropping the `_abl` copies.
    drop_cols: list[str] = [c for c in merged.columns if c.endswith("_abl")]
    for right_col in drop_cols:
        left_col: str = right_col[: -len("_abl")]
        if left_col not in merged.columns:
            continue
        left: pd.Series = merged[left_col]
        right: pd.Series = merged[right_col]
        mismatch: pd.Series = ~(left.eq(right) | (left.isna() & right.isna()))
        if mismatch.any():
            raise ValueError(f"Attention and ablation metadata differ for {left_col!r}")
    merged = merged.drop(columns=drop_cols)
    return merged.sort_values(["prompt_idx", "seg_idx"]).reset_index(drop=True)


def _write_per_model_outputs(
    output_dir: str,
    model_name: str,
    attn_rows: list[dict[str, Any]],
    abl_rows: list[dict[str, Any]],
    tok_rows: list[dict[str, Any]],
    restrict_to_ablation: bool = False,
    overwrite_existing: bool = False,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    seg_df: pd.DataFrame = _merge_segment_rows(
        attn_rows,
        abl_rows,
        restrict_to_ablation=restrict_to_ablation,
    )
    seg_path: str = os.path.join(output_dir, f"{model_name}_segment.tsv.gz")
    seg_df.to_csv(
        seg_path,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )
    logger.info(f"Saved {seg_path} ({len(seg_df)} rows)")

    tok_path: str = os.path.join(output_dir, f"{model_name}_tokens.tsv.gz")
    if tok_rows:
        tok_df: pd.DataFrame = pd.DataFrame(tok_rows)
        tok_df.to_csv(
            tok_path,
            sep="\t",
            index=False,
            compression=GZIP_COMPRESSION,
        )
        logger.info(f"Saved {tok_path} ({len(tok_df)} rows)")
    elif overwrite_existing and os.path.exists(tok_path):
        os.remove(tok_path)
        logger.info(f"Removed stale {tok_path}")

    if overwrite_existing:
        for legacy_suffix in ("_segment.tsv", "_tokens.tsv"):
            legacy_path: str = os.path.join(output_dir, f"{model_name}{legacy_suffix}")
            if os.path.exists(legacy_path):
                os.remove(legacy_path)
                logger.info(f"Removed stale {legacy_path}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


VALID_PHASES: set[str] = {"attention", "ablation"}


def _ensure_model_outputs_writable(
    output_dir: str,
    model_name: str,
    overwrite_existing: bool,
) -> None:
    """Reject an implicit overwrite before loading or scoring a model."""
    output_paths: list[str] = [
        os.path.join(output_dir, f"{model_name}{suffix}")
        for suffix in (
            "_segment.tsv",
            "_segment.tsv.gz",
            "_tokens.tsv",
            "_tokens.tsv.gz",
            "_run.json",
        )
    ]
    existing: list[str] = [path for path in output_paths if os.path.exists(path)]
    if existing and not overwrite_existing:
        raise FileExistsError(
            "Refusing to overwrite existing model outputs without "
            f"--overwrite-existing: {existing}"
        )


def _attention_implementation_for_phase(phase: str) -> str:
    """Return the transformer attention backend requested for one phase."""
    if phase not in VALID_PHASES:
        raise ValueError(f"Unknown phase {phase!r}. Valid: {sorted(VALID_PHASES)}")
    return "eager" if phase == "attention" else "sdpa"


def _load_phase_model(
    model_name: str,
    model_path: str,
    phase: str,
) -> TransformersModel:
    """Load a model with the backend required by one scoring phase."""
    return TransformersModel(
        model_name=model_name,
        model_path=model_path,
        attn_implementation=_attention_implementation_for_phase(phase),
    ).load()


async def run_benchmark(
    benchmark_name: str,
    pregrouper: PregrouperID = "sentence",
    phases: set[str] | None = None,
    batch_size: int = 32,
    max_samples: int | None = None,
    max_forward_passes: int | None = None,
    seed: int = 42,
    model_set: str = "Qwen2.5-Instruct",
    results_dir: str = "results",
    continue_on_error: bool = False,
    dataset_file: str | None = None,
    overwrite_existing: bool = False,
) -> None:
    execution_source_sha256: Mapping[str, str] = _snapshot_execution_source_hashes()
    if phases is None:
        phases = VALID_PHASES.copy()
    invalid: set[str] = phases - VALID_PHASES
    if invalid:
        raise ValueError(f"Unknown phases {invalid}. Valid: {sorted(VALID_PHASES)}")
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark '{benchmark_name}'. "
            f"Available: {list(BENCHMARKS.keys())}"
        )
    if model_set not in MODEL_SETS:
        raise ValueError(
            f"Unknown model set '{model_set}'. Available: {list(MODEL_SETS.keys())}"
        )

    run_attention: bool = "attention" in phases
    run_ablation: bool = "ablation" in phases

    spec: BenchmarkSpec = BENCHMARKS[benchmark_name]

    if spec.scoring_mode == "completion_logprob" and max_forward_passes is None:
        raise ValueError(
            f"benchmark {spec.name!r} uses scoring_mode=completion_logprob "
            f"which only supports subsampled ablation. Pass --max-forward-passes."
        )

    df: pd.DataFrame = load_benchmark_dataset(spec, dataset_file=dataset_file)

    if max_samples is not None and max_samples < len(df):
        rng: np.random.Generator = np.random.default_rng(seed)
        indices: np.ndarray = rng.choice(len(df), size=max_samples, replace=False)
        indices.sort()
        df = df.iloc[indices]
        logger.info(f"Subsampled to {len(df)} prompts (seed={seed})")

    prompt_texts: list[str] = [spec.prompt_builder(row) for _, row in df.iterrows()]
    system_prompt: str = (
        spec.system_prompt_override
        if spec.system_prompt_override is not None
        else spec.eval_config.system_prompt if spec.eval_config is not None else ""
    )
    dialogs: list[Dialog] = [_build_dialog(t, system_prompt) for t in prompt_texts]
    prompts_meta: list[dict[str, Any]] = [
        {"prompt_idx": int(source_idx), "answer": row[spec.answer_column]}
        for source_idx, row in df.iterrows()
    ]

    target_texts: list[str] | None = None
    if spec.target_column is not None:
        target_texts = [str(row[spec.target_column]) for _, row in df.iterrows()]
        if not any(t.strip() for t in target_texts):
            raise ValueError(
                f"target_column {spec.target_column!r} for benchmark "
                f"{spec.name!r} has no non-empty values; cannot score "
                f"completion_logprob with empty targets"
            )

    all_ablated: list[list[Dialog]] | None = None
    selected_pairs: list[tuple[int, int]] | None = None
    if max_forward_passes is not None:
        all_ablated = await _generate_all_ablations(dialogs, pregrouper)
        selected_pairs = _subsample_ablation_pairs(
            all_ablated, max_forward_passes, seed
        )

    output_dir: str = os.path.join(results_dir, spec.name, pregrouper)
    os.makedirs(output_dir, exist_ok=True)
    _write_segment_manifest(
        output_dir,
        dialogs,
        prompts_meta,
        pregrouper,
        selected_pairs,
    )

    models_to_run: list[tuple[str, str]] = MODEL_SETS[model_set]
    if os.environ.get("BENCHMARK_MODELS"):
        allowed: set[str] = set(os.environ["BENCHMARK_MODELS"].split(","))
        models_to_run = [(n, p) for n, p in models_to_run if n in allowed]
    logger.info(f"Model set: {model_set} -> {[n for n, _ in models_to_run]}")

    for model_name, model_subpath in models_to_run:
        _ensure_model_outputs_writable(
            output_dir,
            model_name,
            overwrite_existing=overwrite_existing,
        )
        model_path: str = resolve_model_path(model_subpath)
        logger.info(f"\n{'=' * 60}\n{spec.name} / {model_name}\n{'=' * 60}")

        attn_rows: list[dict[str, Any]] = []
        abl_rows: list[dict[str, Any]] = []
        tok_rows: list[dict[str, Any]] = []

        if run_attention:
            attention_model: TransformersModel = _load_phase_model(
                model_name, model_path, "attention"
            )
            try:
                attn_rows = await _score_attention(
                    attention_model,
                    dialogs,
                    prompts_meta,
                    pregrouper_id=pregrouper,
                    continue_on_error=continue_on_error,
                )
            finally:
                attention_model.unload()

        if run_ablation:
            ablation_model: TransformersModel = _load_phase_model(
                model_name, model_path, "ablation"
            )
            try:
                if selected_pairs is not None and all_ablated is not None:
                    if (
                        spec.scoring_mode == "completion_logprob"
                        and target_texts is not None
                    ):
                        abl_rows, tok_rows = _score_ablation_subsampled_completion(
                            ablation_model,
                            dialogs,
                            prompts_meta,
                            all_ablated,
                            selected_pairs,
                            target_texts,
                            pregrouper_id=pregrouper,
                            batch_size=batch_size,
                        )
                    else:
                        abl_rows, tok_rows = _score_ablation_subsampled(
                            ablation_model,
                            dialogs,
                            prompts_meta,
                            all_ablated,
                            selected_pairs,
                            spec.eval_config,
                            spec.scoring_mode,
                            pregrouper_id=pregrouper,
                            batch_size=batch_size,
                        )
                else:
                    abl_rows, tok_rows = await _score_ablation_full(
                        ablation_model,
                        dialogs,
                        prompts_meta,
                        spec.eval_config,
                        spec.scoring_mode,
                        pregrouper_id=pregrouper,
                        batch_size=batch_size,
                        continue_on_error=continue_on_error,
                    )
            finally:
                ablation_model.unload()

        _write_per_model_outputs(
            output_dir,
            model_name,
            attn_rows,
            abl_rows,
            tok_rows,
            restrict_to_ablation=selected_pairs is not None,
            overwrite_existing=overwrite_existing,
        )
        _write_run_metadata(
            output_dir=output_dir,
            model_name=model_name,
            model_source=model_subpath,
            model_path=model_path,
            spec=spec,
            pregrouper=pregrouper,
            phases=phases,
            batch_size=batch_size,
            max_samples=max_samples,
            max_forward_passes=max_forward_passes,
            seed=seed,
            dataset_file=dataset_file,
            dataset=df,
            execution_source_sha256=execution_source_sha256,
        )
    logger.info(f"Done with {spec.name}!")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=list(BENCHMARKS.keys()),
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--pregrouper",
        type=str,
        default="sentence",
        choices=["word", "sentence"],
        help="Segmentation granularity (default: sentence)",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="attention,ablation",
        help="Comma-separated phases to run (default: attention,ablation)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for ablation scoring (default: 32)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max prompts to include (deterministic subset, same across models)",
    )
    parser.add_argument(
        "--max-forward-passes",
        type=int,
        default=None,
        help="Max ablation forward passes (subsample across all prompts, same across models)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic subsampling (default: 42)",
    )
    parser.add_argument(
        "--model-set",
        type=str,
        default="Qwen2.5-Instruct",
        choices=list(MODEL_SETS.keys()),
        help="Which set of HuggingFace models to score (default: Qwen2.5-Instruct)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Output directory for per-model TSVs (default: results)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help=(
            "Keep partial results after a prompt fails. The default is to "
            "fail fast so incomplete outputs cannot look complete."
        ),
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help=(
            "Deliberately replace every existing output for the selected "
            "model/configuration. Without this flag, any existing segment, "
            "token, or run file makes the command fail before model loading."
        ),
    )
    parser.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help=(
            "Optional frozen TSV snapshot. Its first column is used as the "
            "stable prompt index; otherwise the HuggingFace dataset is loaded."
        ),
    )
    args: argparse.Namespace = parser.parse_args()
    asyncio.run(
        run_benchmark(
            args.benchmark,
            pregrouper=cast(PregrouperID, args.pregrouper),
            phases=set(args.phases.split(",")),
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_forward_passes=args.max_forward_passes,
            seed=args.seed,
            model_set=args.model_set,
            results_dir=args.results_dir,
            continue_on_error=args.continue_on_error,
            dataset_file=args.dataset_file,
            overwrite_existing=args.overwrite_existing,
        )
    )


if __name__ == "__main__":
    main()
