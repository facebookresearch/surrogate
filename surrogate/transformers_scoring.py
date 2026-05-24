# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from typing import Any

import torch
from surrogate.eval_constants import BOOLQ_CONFIG, ReportToken
from surrogate.transformers_model import TransformersModel
from surrogate.model_types import Dialog, LogprobBreakdown, SingleScore

logger: logging.Logger = logging.getLogger(__name__)


def _resolve_token_ids(
    model: TransformersModel,
    tokens: set[str] | list[str] | str,
) -> list[int]:
    """
    Map token variation strings to vocabulary IDs.

    Only includes tokens that encode as single tokens in this model's
    vocabulary. Multi-token strings (e.g., "contradiction") are silently
    skipped — top-k logprob APIs also only find single-token matches.
    """
    if isinstance(tokens, str):
        tokens = [tokens]
    elif isinstance(tokens, set):
        tokens = list(tokens)

    ids: list[int] = []
    for token in tokens:
        token_id: int | None = model.encode_single_token(token)
        if token_id is not None:
            ids.append(token_id)
    return ids


def _aggregate_log_probs(
    log_probs: torch.Tensor,
    token_ids: list[int],
) -> float:
    """
    Aggregate log-probabilities for a set of token IDs using log-sum-exp.

    This is equivalent to computing log( sum_i P(token_i) ), i.e. the
    log-probability of any of the target tokens being generated.

    Returns -inf if no token IDs are provided.
    """
    if not token_ids:
        return -float("inf")
    selected: torch.Tensor = log_probs[token_ids]
    return torch.logsumexp(selected, dim=0).item()


def _aggregate_log_probs_or_none(
    log_probs: torch.Tensor,
    token_ids: list[int],
) -> float | None:
    if not token_ids:
        return None
    return _aggregate_log_probs(log_probs, token_ids)


def _resolve_label_token_ids_by_label(
    model: TransformersModel,
    label_tokens: dict[str, set[str] | list[str] | str],
) -> dict[str, list[int]]:
    return {
        label: _resolve_token_ids(model, tokens)
        for label, tokens in label_tokens.items()
    }


def _resolve_report_token_ids_by_label(
    model: TransformersModel,
    report_tokens: dict[str, list[ReportToken]],
) -> dict[str, dict[str, int | None]]:
    return {
        label: {
            token.alias: model.encode_single_token(token.surface) for token in tokens
        }
        for label, tokens in report_tokens.items()
    }


def _build_logprob_breakdown(
    log_probs: torch.Tensor,
    resolved_label_token_ids: dict[str, list[int]],
    resolved_report_token_ids: dict[str, dict[str, int | None]],
) -> LogprobBreakdown:
    label_logprobs: dict[str, float | None] = {
        label: _aggregate_log_probs_or_none(log_probs, token_ids)
        for label, token_ids in resolved_label_token_ids.items()
    }
    token_logprobs: dict[str, dict[str, float | None]] = {}
    for label, token_ids_by_alias in resolved_report_token_ids.items():
        token_logprobs[label] = {
            alias: (None if token_id is None else float(log_probs[token_id].item()))
            for alias, token_id in token_ids_by_alias.items()
        }
    return LogprobBreakdown(
        label_logprobs=label_logprobs,
        token_logprobs=token_logprobs,
    )


def _has_any_resolved_label_or_report_tokens(
    resolved_label_token_ids: dict[str, list[int]],
    resolved_report_token_ids: dict[str, dict[str, int | None]],
) -> bool:
    return any(token_ids for token_ids in resolved_label_token_ids.values()) or any(
        token_id is not None
        for token_ids_by_alias in resolved_report_token_ids.values()
        for token_id in token_ids_by_alias.values()
    )


async def transformers_label_logprob_breakdown(
    model: TransformersModel,
    prompt: Dialog,
    label_tokens: dict[str, set[str] | list[str] | str],
    report_tokens: dict[str, list[ReportToken]] | None = None,
) -> LogprobBreakdown | None:
    """Return per-label and per-token next-token log-probabilities.

    ``label_tokens`` defines the aggregated label groups scored via
    log-sum-exp; ``report_tokens`` optionally defines an ordered subset of
    single-token surfaces to expose individually in JSON/TSV outputs.
    """
    resolved_label_token_ids: dict[str, list[int]] = _resolve_label_token_ids_by_label(
        model, label_tokens
    )
    resolved_report_token_ids: dict[str, dict[str, int | None]] = (
        _resolve_report_token_ids_by_label(model, report_tokens or {})
    )
    if not _has_any_resolved_label_or_report_tokens(
        resolved_label_token_ids, resolved_report_token_ids
    ):
        logger.warning(
            f"No matching single-token IDs found for {model.model_name} "
            f"across labels {sorted(label_tokens.keys())}"
        )
        return None

    try:
        text: str = model.dialog_to_text(prompt)
        log_probs: torch.Tensor = model.get_next_token_log_probs(text)
    except Exception as e:
        logger.warning(f"Error during inference for {model.model_name}: {e}")
        return None

    return _build_logprob_breakdown(
        log_probs,
        resolved_label_token_ids=resolved_label_token_ids,
        resolved_report_token_ids=resolved_report_token_ids,
    )


def transformers_label_logprob_breakdown_batch(
    model: TransformersModel,
    dialogs: list[Dialog],
    label_tokens: dict[str, set[str] | list[str] | str],
    report_tokens: dict[str, list[ReportToken]] | None = None,
    batch_size: int = 32,
) -> list[LogprobBreakdown | None]:
    """Batched sibling of ``transformers_label_logprob_breakdown``."""
    resolved_label_token_ids: dict[str, list[int]] = _resolve_label_token_ids_by_label(
        model, label_tokens
    )
    resolved_report_token_ids: dict[str, dict[str, int | None]] = (
        _resolve_report_token_ids_by_label(model, report_tokens or {})
    )
    if not _has_any_resolved_label_or_report_tokens(
        resolved_label_token_ids, resolved_report_token_ids
    ):
        logger.warning(
            f"No matching single-token IDs found for {model.model_name} "
            f"across labels {sorted(label_tokens.keys())}"
        )
        return [None] * len(dialogs)

    texts: list[str] = [model.dialog_to_text(d) for d in dialogs]
    results: list[LogprobBreakdown | None] = []
    for start in range(0, len(texts), batch_size):
        chunk: list[str] = texts[start : start + batch_size]
        try:
            log_probs_list: list[torch.Tensor] = model.get_next_token_log_probs_batch(
                chunk
            )
            for log_probs in log_probs_list:
                results.append(
                    _build_logprob_breakdown(
                        log_probs,
                        resolved_label_token_ids=resolved_label_token_ids,
                        resolved_report_token_ids=resolved_report_token_ids,
                    )
                )
        except Exception as e:
            logger.warning(f"Batch logprob breakdown failed: {e}")
            results.extend([None] * len(chunk))
    return results


def transformers_completion_logprob_batch(
    model: Any,  # TransformersModel
    dialogs: list[Any],  # list[Dialog]
    target_texts: list[str],
    batch_size: int = 32,
) -> list[float | None]:
    """
    Batched completion log-probability scoring.

    Each dialog is scored against its corresponding target text.
    Processes in chunks of batch_size for efficient GPU utilization.

    Args:
        model: A loaded TransformersModel.
        dialogs: List of Dialogs to condition on.
        target_texts: List of target completions, one per dialog.
        batch_size: Number of dialogs per batch.

    Returns:
        List of summed log-probabilities, one per dialog.
    """
    texts: list[str] = [model.dialog_to_text(d) for d in dialogs]

    results: list[float | None] = []
    for start in range(0, len(texts), batch_size):
        chunk_texts: list[str] = texts[start : start + batch_size]
        chunk_targets: list[str] = target_texts[start : start + batch_size]
        try:
            scores: list[float] = model.get_sequence_log_probs_batch(
                chunk_texts, chunk_targets
            )
            results.extend(scores)
        except Exception as e:
            logger.warning(f"Batch completion logprob failed: {e}")
            results.extend([None] * len(chunk_texts))

    return results


def transformers_logit_difference_batch(
    model: Any,  # TransformersModel
    dialogs: list[Any],  # list[Dialog]
    pos_tokens: set[str] | list[str] | str = BOOLQ_CONFIG.label_tokens["true"],
    neg_tokens: set[str] | list[str] | str = BOOLQ_CONFIG.label_tokens["false"],
    batch_size: int = 32,
) -> list[float | None]:
    """
    Batched logit difference scoring. Processes dialogs in chunks of
    batch_size for ~10-50x speedup over sequential scoring.

    Args:
        model: A loaded TransformersModel.
        dialogs: List of Dialogs to score.
        pos_tokens: Positive label token variations.
        neg_tokens: Negative label token variations.
        batch_size: Number of dialogs per batch.

    Returns:
        List of logit differences, one per dialog.
    """
    pos_ids: list[int] = _resolve_token_ids(model, pos_tokens)
    neg_ids: list[int] = _resolve_token_ids(model, neg_tokens)

    if not pos_ids or not neg_ids:
        logger.warning("No matching token IDs for batch scoring")
        return [None] * len(dialogs)

    # Convert all dialogs to text
    texts: list[str] = [model.dialog_to_text(d) for d in dialogs]

    results: list[float | None] = []
    for start in range(0, len(texts), batch_size):
        chunk: list[str] = texts[start : start + batch_size]
        try:
            log_probs_list: list[Any] = model.get_next_token_log_probs_batch(chunk)
            for log_probs in log_probs_list:
                pos_lp: float = _aggregate_log_probs(log_probs, pos_ids)
                neg_lp: float = _aggregate_log_probs(log_probs, neg_ids)
                results.append(pos_lp - neg_lp)
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            results.extend([None] * len(chunk))

    return results
