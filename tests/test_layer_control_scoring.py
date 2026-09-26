# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Focused CPU tests for layerwise random-control scoring."""

from __future__ import annotations

from unittest import TestCase

import torch

from surrogate.layer_control_scoring import (
    grouped_logsumexp_contrasts,
    project_residual_deltas,
    seeded_isotropic_directions,
    summed_unembedding_group_contrast_norms,
    unit_unembedding_pair_directions,
)
from surrogate.layerwise_scoring import ResidualSlotCapture


class DirectionConstructionTest(TestCase):
    """Tests for deterministic normalized control directions."""

    def test_unembedding_pairs_are_unit_normalized_and_ordered(self) -> None:
        head: torch.nn.Linear = torch.nn.Linear(3, 5, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 2.0, 0.0],
                        [0.0, 0.0, 3.0],
                        [1.0, 2.0, 0.0],
                        [0.0, 2.0, 3.0],
                    ]
                )
            )

        directions, norms = unit_unembedding_pair_directions(head, [(3, 0), (4, 1)])

        self.assertTrue(torch.allclose(norms, torch.tensor([2.0, 3.0])))
        self.assertTrue(
            torch.allclose(
                directions,
                torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            )
        )

    def test_isotropic_draws_are_independent_deterministic_and_unit(self) -> None:
        first: torch.Tensor = seeded_isotropic_directions(7, [11, 29, 47])
        second: torch.Tensor = seeded_isotropic_directions(7, [11, 29, 47])

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(
            torch.allclose(
                torch.linalg.vector_norm(first, dim=1), torch.ones(3), atol=1e-6
            )
        )
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_invalid_pair_and_seed_inputs_fail_closed(self) -> None:
        head: torch.nn.Linear = torch.nn.Linear(3, 5, bias=False)
        with self.assertRaisesRegex(ValueError, "must differ"):
            unit_unembedding_pair_directions(head, [(1, 1)])
        with self.assertRaisesRegex(ValueError, "unique"):
            unit_unembedding_pair_directions(head, [(1, 2), (1, 2)])
        with self.assertRaisesRegex(ValueError, "distinct"):
            seeded_isotropic_directions(3, [7, 7])


class GroupedControlTest(TestCase):
    """Tests for 9-vs-8-style grouped logsumexp control scoring."""

    def setUp(self) -> None:
        self.head: torch.nn.Linear = torch.nn.Linear(3, 8, bias=True)
        with torch.no_grad():
            self.head.weight.copy_(
                torch.arange(24, dtype=torch.float32).reshape(8, 3) / 7
            )
            self.head.bias.copy_(torch.arange(8, dtype=torch.float32) / 11)
        self.capture = ResidualSlotCapture(
            postnorm=torch.tensor(
                [
                    [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]],
                    [[2.0, 1.0, 0.0], [2.5, 1.5, 0.5]],
                ]
            ),
            final_logits=torch.arange(16, dtype=torch.float32).reshape(2, 8) / 5,
        )

    def test_grouped_scores_match_selected_head_and_final_logprob_paths(self) -> None:
        positive: list[list[int]] = [[0, 1, 2], [1, 3, 5]]
        negative: list[list[int]] = [[3, 4], [2, 6]]

        actual: torch.Tensor = grouped_logsumexp_contrasts(
            self.capture, self.head, positive, negative
        )

        logits: torch.Tensor = self.head(self.capture.postnorm)
        expected_first: torch.Tensor = torch.logsumexp(
            logits[..., positive[0]], dim=-1
        ) - torch.logsumexp(logits[..., negative[0]], dim=-1)
        final_logits: torch.Tensor | None = self.capture.final_logits
        self.assertIsNotNone(final_logits)
        assert final_logits is not None
        final_log_probs: torch.Tensor = torch.log_softmax(final_logits, dim=-1)
        expected_first[-1] = torch.logsumexp(
            final_log_probs[:, positive[0]], dim=-1
        ) - torch.logsumexp(final_log_probs[:, negative[0]], dim=-1)
        self.assertEqual(actual.shape, (2, 2, 2))
        self.assertTrue(torch.allclose(actual[..., 0], expected_first))

    def test_group_norms_match_summed_unembedding_difference(self) -> None:
        positive: list[list[int]] = [[0, 1, 2], [1, 3, 5]]
        negative: list[list[int]] = [[3, 4], [2, 6]]

        actual: torch.Tensor = summed_unembedding_group_contrast_norms(
            self.head, positive, negative
        )
        expected: torch.Tensor = torch.stack(
            [
                torch.linalg.vector_norm(
                    self.head.weight[p].sum(dim=0) - self.head.weight[n].sum(dim=0)
                )
                for p, n in zip(positive, negative)
            ]
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_overlap_and_nonrectangular_groups_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            grouped_logsumexp_contrasts(self.capture, self.head, [[0, 1]], [[1, 2]])
        with self.assertRaisesRegex(ValueError, "rectangular"):
            grouped_logsumexp_contrasts(
                self.capture, self.head, [[0], [1, 2]], [[3], [4]]
            )


class ProjectionTest(TestCase):
    """Tests for signed per-segment delta projection semantics."""

    def test_projects_original_minus_ablated_in_fp32(self) -> None:
        original = ResidualSlotCapture(
            postnorm=torch.tensor([[[3.0, 4.0]], [[5.0, 7.0]]])
        )
        perturbed = ResidualSlotCapture(
            postnorm=torch.tensor(
                [
                    [[2.0, 2.0], [1.0, 4.0]],
                    [[2.0, 3.0], [4.0, 5.0]],
                ]
            )
        )
        directions: torch.Tensor = torch.eye(2, dtype=torch.float32)

        actual: torch.Tensor = project_residual_deltas(original, perturbed, directions)

        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                actual,
                torch.tensor(
                    [
                        [[1.0, 2.0], [2.0, 0.0]],
                        [[3.0, 4.0], [1.0, 2.0]],
                    ]
                ),
            )
        )

    def test_nonunit_directions_are_rejected(self) -> None:
        capture = ResidualSlotCapture(postnorm=torch.zeros(2, 1, 3))
        with self.assertRaisesRegex(ValueError, "unit normalized"):
            project_residual_deltas(capture, capture, torch.tensor([[2.0, 0.0, 0.0]]))
