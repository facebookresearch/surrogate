# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""CPU tests for compact per-layer capture and scoring."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest import TestCase

import torch

from surrogate.layerwise_scoring import (
    ResidualSlotCapture,
    capture_postnorm_residual_slots,
    layerwise_delta_norms,
    score_residual_slots,
)


class _FakeBlock(torch.nn.Module):
    """Deterministic residual block used to distinguish capture slots."""

    def __init__(self, hidden_size: int, offset: float) -> None:
        super().__init__()
        self.register_buffer(
            "offset", torch.arange(hidden_size, dtype=torch.float32) * offset
        )

    def forward(self, hidden: torch.Tensor, **_kwargs: Any) -> tuple[torch.Tensor]:
        """Add a fixed nonuniform residual and mimic tuple-style HF output."""
        return (hidden + self.offset,)


class _FakeBackbone(torch.nn.Module):
    """Small Llama/Qwen2-shaped backbone with two fake decoder blocks."""

    def __init__(self, vocab_size: int = 9, hidden_size: int = 4) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList(
            [_FakeBlock(hidden_size, 1.0), _FakeBlock(hidden_size, 2.0)]
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        """Run embedding, decoder blocks, and the final normalization."""
        del attention_mask, use_cache
        hidden: torch.Tensor = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return self.norm(hidden)


class _FakeCausalLM(torch.nn.Module):
    """Backbone plus biased language-model head for scoring tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeBackbone()
        self.lm_head = torch.nn.Linear(4, 9, bias=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> SimpleNamespace:
        """Return full-head logits while exercising all backbone hooks."""
        hidden: torch.Tensor = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        return SimpleNamespace(logits=self.lm_head(hidden))


class CaptureResidualSlotsTest(TestCase):
    """Tests for residual slot ordering, indexing, and hook completeness."""

    def test_embedding_and_block_slots_use_last_unmasked_token(self) -> None:
        torch.manual_seed(4)
        model: _FakeCausalLM = _FakeCausalLM()
        input_ids: torch.Tensor = torch.tensor([[1, 2, 0], [3, 4, 5]])
        attention_mask: torch.Tensor = torch.tensor([[1, 1, 0], [1, 1, 1]])

        capture: ResidualSlotCapture = capture_postnorm_residual_slots(
            model, input_ids, attention_mask
        )

        embedded: torch.Tensor = model.model.embed_tokens(input_ids)
        expected_raw: list[torch.Tensor] = [
            embedded[torch.arange(2), torch.tensor([1, 2])]
        ]
        expected_raw.append(expected_raw[-1] + model.model.layers[0].offset)
        expected_raw.append(expected_raw[-1] + model.model.layers[1].offset)
        expected: torch.Tensor = torch.stack(
            [model.model.norm(value) for value in expected_raw]
        )
        self.assertEqual(capture.postnorm.shape, (3, 2, 4))
        self.assertEqual(capture.final_logits.shape, (2, 9))
        self.assertEqual(capture.num_layers, 2)
        self.assertTrue(torch.allclose(capture.postnorm, expected))

    def test_missing_hook_fails_instead_of_returning_partial_capture(self) -> None:
        model: _FakeCausalLM = _FakeCausalLM()

        def skip_last_layer(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            use_cache: bool = False,
        ) -> torch.Tensor:
            del attention_mask, use_cache
            hidden: torch.Tensor = model.model.embed_tokens(input_ids)
            hidden = model.model.layers[0](hidden)[0]
            return model.model.norm(hidden)

        model.model.forward = skip_last_layer
        with self.assertRaisesRegex(RuntimeError, r"slots \[2\]"):
            capture_postnorm_residual_slots(model, torch.tensor([[1, 2]]))

    def test_empty_mask_row_is_rejected(self) -> None:
        model: _FakeCausalLM = _FakeCausalLM()
        with self.assertRaisesRegex(ValueError, "at least one unmasked"):
            capture_postnorm_residual_slots(
                model,
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[1, 1], [0, 0]]),
            )


class ScoreResidualSlotsTest(TestCase):
    """Tests for compact grouped-label scores and input validation."""

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model: _FakeCausalLM = _FakeCausalLM()
        self.capture: ResidualSlotCapture = capture_postnorm_residual_slots(
            self.model, torch.tensor([[1, 2], [3, 4]])
        )

    def test_scores_match_selected_logits_at_every_slot(self) -> None:
        labels: dict[str, list[int]] = {
            "entailment": [1, 2],
            "neutral": [4],
            "contradiction": [6, 7],
        }
        scores = score_residual_slots(self.capture, self.model.lm_head, labels)
        logits: torch.Tensor = self.model.lm_head(self.capture.postnorm)

        expected_entailment: torch.Tensor = torch.logsumexp(logits[..., [1, 2]], dim=-1)
        expected_entailment[-1] = torch.logsumexp(
            torch.log_softmax(self.capture.final_logits, dim=-1)[:, [1, 2]], dim=-1
        )
        expected_projection: torch.Tensor = self.capture.postnorm @ (
            self.model.lm_head.weight[1] + self.model.lm_head.weight[2]
        )
        self.assertEqual(scores.labels, ("entailment", "neutral", "contradiction"))
        self.assertTrue(
            torch.allclose(scores.grouped_logsumexp[..., 0], expected_entailment)
        )
        self.assertTrue(
            torch.allclose(
                scores.summed_unembedding_projection[..., 0], expected_projection
            )
        )
        expected_ec: torch.Tensor = expected_entailment - torch.logsumexp(
            logits[..., [6, 7]], dim=-1
        )
        expected_ec[-1] = expected_entailment[-1] - torch.logsumexp(
            torch.log_softmax(self.capture.final_logits, dim=-1)[:, [6, 7]], dim=-1
        )
        self.assertTrue(
            torch.allclose(scores.contrast("entailment", "contradiction"), expected_ec)
        )

    def test_final_slot_uses_captured_full_head_logits(self) -> None:
        capture = ResidualSlotCapture(
            postnorm=self.capture.postnorm,
            final_logits=torch.arange(18, dtype=torch.float32).reshape(2, 9),
        )
        scores = score_residual_slots(
            capture,
            self.model.lm_head,
            {"positive": [1, 2], "negative": [6, 7]},
        )
        expected = torch.logsumexp(
            torch.log_softmax(capture.final_logits, dim=-1)[:, [1, 2]], dim=-1
        )
        self.assertTrue(torch.equal(scores.grouped_logsumexp[-1, :, 0], expected))

    def test_rejects_empty_overlapping_duplicate_and_out_of_range_groups(self) -> None:
        invalid_groups: list[dict[str, list[int]]] = [
            {"a": [], "b": [1]},
            {"a": [1], "b": [1]},
            {"a": [1, 1], "b": [2]},
            {"a": [1], "b": [9]},
        ]
        for groups in invalid_groups:
            with (
                self.subTest(groups=groups),
                self.assertRaises((TypeError, ValueError)),
            ):
                score_residual_slots(self.capture, self.model.lm_head, groups)

    def test_unknown_or_self_contrast_is_rejected(self) -> None:
        scores = score_residual_slots(
            self.capture, self.model.lm_head, {"a": [1], "b": [2]}
        )
        with self.assertRaises(ValueError):
            scores.contrast("a", "missing")
        with self.assertRaises(ValueError):
            scores.projection_contrast("a", "a")


class LayerwiseDeltaNormsTest(TestCase):
    """Tests for post-normalization residual delta norms."""

    def test_single_original_broadcasts_across_ablations(self) -> None:
        original: ResidualSlotCapture = ResidualSlotCapture(
            postnorm=torch.zeros(3, 1, 4)
        )
        perturbed: ResidualSlotCapture = ResidualSlotCapture(
            postnorm=torch.ones(3, 5, 4)
        )
        result: torch.Tensor = layerwise_delta_norms(original, perturbed)
        self.assertEqual(result.shape, (3, 5))
        self.assertTrue(torch.equal(result, torch.full((3, 5), 2.0)))

    def test_incompatible_batch_sizes_fail(self) -> None:
        original: ResidualSlotCapture = ResidualSlotCapture(
            postnorm=torch.zeros(3, 2, 4)
        )
        perturbed: ResidualSlotCapture = ResidualSlotCapture(
            postnorm=torch.zeros(3, 5, 4)
        )
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            layerwise_delta_norms(original, perturbed)
