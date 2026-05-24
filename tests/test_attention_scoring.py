# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from surrogate.attention_scoring import (
    _aggregate_max,
    _aggregate_mean,
    _attention_rollout,
    _map_segments_to_token_indices,
    AttentionConfig,
)
from unittest import TestCase


class TestAggregationStrategies(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # 2 layers, 2 heads, 4 tokens
        self.attentions: torch.Tensor = torch.zeros(2, 2, 4, 4)
        # Layer 0, head 0: uniform attention
        self.attentions[0, 0] = torch.tensor(
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
                [0.25, 0.25, 0.25, 0.25],
            ]
        )
        # Layer 0, head 1: focused on token 0
        self.attentions[0, 1] = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        # Layer 1: identity-like (each token attends to itself)
        self.attentions[1, 0] = torch.eye(4)
        self.attentions[1, 1] = torch.eye(4)

    def test_mean_shape(self) -> None:
        result: torch.Tensor = _aggregate_mean(self.attentions)
        self.assertEqual(result.shape, (4, 4))

    def test_mean_values(self) -> None:
        """Mean over 2 layers x 2 heads = 4 matrices."""
        result: torch.Tensor = _aggregate_mean(self.attentions)
        # All entries should be finite
        self.assertTrue(torch.isfinite(result).all())
        # Rows should sum to ~1 (since each input matrix has rows summing to 1)
        row_sums: torch.Tensor = result.sum(dim=-1)
        for s in row_sums.tolist():
            self.assertAlmostEqual(s, 1.0, places=5)

    def test_max_shape(self) -> None:
        result: torch.Tensor = _aggregate_max(self.attentions)
        self.assertEqual(result.shape, (4, 4))

    def test_max_picks_largest(self) -> None:
        """Max should pick the highest attention value across heads/layers."""
        result: torch.Tensor = _aggregate_max(self.attentions)
        # Token 3 attending to token 0: max across heads/layers includes
        # head 1 layer 0 which has value 1.0
        self.assertAlmostEqual(result[3, 0].item(), 1.0, places=5)

    def test_rollout_shape(self) -> None:
        result: torch.Tensor = _attention_rollout(self.attentions)
        self.assertEqual(result.shape, (4, 4))

    def test_rollout_rows_sum_to_one(self) -> None:
        """Rollout matrix rows should sum to ~1 (it's a stochastic matrix)."""
        result: torch.Tensor = _attention_rollout(self.attentions)
        row_sums: torch.Tensor = result.sum(dim=-1)
        for s in row_sums.tolist():
            self.assertAlmostEqual(s, 1.0, places=4)

    def test_rollout_identity_input(self) -> None:
        """If all layers have identity attention, rollout should be identity."""
        eye_attn: torch.Tensor = (
            torch.eye(4).unsqueeze(0).unsqueeze(0).expand(3, 2, 4, 4)
        )
        result: torch.Tensor = _attention_rollout(eye_attn)
        # With residual (0.5*I + 0.5*I = I), rollout of identity is identity
        for i in range(4):
            self.assertAlmostEqual(result[i, i].item(), 1.0, places=4)


class TestSegmentToTokenMapping(TestCase):
    def test_simple_mapping(self) -> None:
        """Each word maps to its token."""
        full_text: str = "Hello world foo"
        segments: list[str] = ["Hello", "world", "foo"]
        # Token offsets: "Hello"=(0,5), " world"=(5,11), " foo"=(11,15)
        offset_mapping: list[tuple[int, int]] = [(0, 5), (5, 11), (11, 15)]

        result: list[list[int]] = _map_segments_to_token_indices(
            full_text, segments, offset_mapping
        )
        self.assertEqual(result, [[0], [1], [2]])

    def test_multi_token_segment(self) -> None:
        """A segment that spans multiple tokens."""
        full_text: str = "Hello beautiful world"
        segments: list[str] = ["Hello beautiful", "world"]
        offset_mapping: list[tuple[int, int]] = [(0, 5), (5, 15), (15, 21)]

        result: list[list[int]] = _map_segments_to_token_indices(
            full_text, segments, offset_mapping
        )
        self.assertEqual(result[0], [0, 1])  # "Hello beautiful" spans 2 tokens
        self.assertEqual(result[1], [2])

    def test_special_tokens_skipped(self) -> None:
        """Special tokens (0,0 offsets) should be skipped."""
        full_text: str = "Hello world"
        segments: list[str] = ["Hello", "world"]
        # BOS token has (0,0), then real tokens
        offset_mapping: list[tuple[int, int]] = [(0, 0), (0, 5), (5, 11)]

        result: list[list[int]] = _map_segments_to_token_indices(
            full_text, segments, offset_mapping
        )
        self.assertEqual(result, [[1], [2]])

    def test_segment_not_found(self) -> None:
        """Missing segment should produce empty list."""
        full_text: str = "Hello world"
        segments: list[str] = ["missing"]
        offset_mapping: list[tuple[int, int]] = [(0, 5), (5, 11)]

        result: list[list[int]] = _map_segments_to_token_indices(
            full_text, segments, offset_mapping
        )
        self.assertEqual(result, [[]])


class TestAttentionConfig(TestCase):
    def test_defaults(self) -> None:
        config: AttentionConfig = AttentionConfig()
        self.assertEqual(config.aggregation, "mean")
