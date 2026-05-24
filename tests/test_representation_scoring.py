# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Linear-algebra sanity checks for representation_scoring.py.

Pure tensor math — no real LM is loaded. The "exact attribution under
linear readout" claim is exercised by constructing a synthetic
``W ∈ R^{vocab × hidden}`` and a ``z ∈ R^{hidden}``, computing
``log p[pos] - log p[neg]`` directly, perturbing z, and confirming that
``w · (z - z')`` reproduces the change in log-odds when there is a
single token per side.
"""

from __future__ import annotations

import torch
from surrogate.representation_scoring import (
    cosine_similarity,
    delta_norm,
    representation_metrics_batch,
    w_dot_delta,
    w_norm,
)
from unittest import TestCase


HIDDEN: int = 8
VOCAB: int = 20


def _make_W(seed: int = 0) -> torch.Tensor:
    """Random unembedding matrix (vocab, hidden)."""
    g: torch.Generator = torch.Generator().manual_seed(seed)
    return torch.randn(VOCAB, HIDDEN, generator=g)


def _logit_diff(W: torch.Tensor, z: torch.Tensor, pos: int, neg: int) -> float:
    """log p[pos] - log p[neg] from logits = W @ z. Single-token case."""
    logits: torch.Tensor = W @ z  # (vocab,)
    log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
    return float(log_probs[pos] - log_probs[neg])


class DeltaNormTest(TestCase):
    def test_zero_when_identical(self) -> None:
        z: torch.Tensor = torch.randn(3, HIDDEN)
        result: torch.Tensor = delta_norm(z, z.clone())
        self.assertTrue(torch.allclose(result, torch.zeros(3), atol=1e-6))

    def test_matches_manual_formula(self) -> None:
        a: torch.Tensor = torch.tensor([[3.0, 4.0]])
        b: torch.Tensor = torch.tensor([[0.0, 0.0]])
        # ‖[3,4]‖₂ = 5
        self.assertAlmostEqual(float(delta_norm(a, b)[0]), 5.0, places=6)

    def test_nonnegative(self) -> None:
        a: torch.Tensor = torch.randn(10, HIDDEN)
        b: torch.Tensor = torch.randn(10, HIDDEN)
        self.assertTrue((delta_norm(a, b) >= 0).all().item())


class CosineSimilarityTest(TestCase):
    def test_one_for_identical(self) -> None:
        z: torch.Tensor = torch.randn(5, HIDDEN)
        result: torch.Tensor = cosine_similarity(z, z.clone())
        self.assertTrue(torch.allclose(result, torch.ones(5), atol=1e-5))

    def test_minus_one_for_opposite(self) -> None:
        z: torch.Tensor = torch.randn(5, HIDDEN)
        result: torch.Tensor = cosine_similarity(z, -z)
        self.assertTrue(torch.allclose(result, -torch.ones(5), atol=1e-5))

    def test_zero_for_orthogonal(self) -> None:
        a: torch.Tensor = torch.tensor([[1.0, 0.0]])
        b: torch.Tensor = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(float(cosine_similarity(a, b)[0]), 0.0, places=6)

    def test_within_unit_interval(self) -> None:
        a: torch.Tensor = torch.randn(20, HIDDEN)
        b: torch.Tensor = torch.randn(20, HIDDEN)
        cs: torch.Tensor = cosine_similarity(a, b)
        self.assertTrue(((cs >= -1.0 - 1e-5) & (cs <= 1.0 + 1e-5)).all().item())


class WDotDeltaTest(TestCase):
    def test_zero_when_identical(self) -> None:
        w: torch.Tensor = torch.randn(HIDDEN)
        z: torch.Tensor = torch.randn(7, HIDDEN)
        self.assertTrue(torch.allclose(w_dot_delta(w, z, z.clone()), torch.zeros(7)))

    def test_matches_manual_inner_product(self) -> None:
        w: torch.Tensor = torch.tensor([1.0, 2.0, 3.0])
        z_orig: torch.Tensor = torch.tensor([[5.0, 5.0, 5.0]])
        z_pert: torch.Tensor = torch.tensor([[1.0, 2.0, 3.0]])
        # Δz = [4, 3, 2]; w · Δz = 4*1 + 3*2 + 2*3 = 16
        self.assertAlmostEqual(float(w_dot_delta(w, z_orig, z_pert)[0]), 16.0)


class CauchySchwarzBoundTest(TestCase):
    """The bound used in the user's motivation: |w·Δz| ≤ ‖w‖·‖Δz‖."""

    def test_bound_holds(self) -> None:
        torch.manual_seed(7)
        w: torch.Tensor = torch.randn(HIDDEN)
        z_orig: torch.Tensor = torch.randn(50, HIDDEN)
        z_pert: torch.Tensor = torch.randn(50, HIDDEN)
        lhs: torch.Tensor = w_dot_delta(w, z_orig, z_pert).abs()
        rhs: torch.Tensor = w_norm(w) * delta_norm(z_orig, z_pert)
        # Strict inequality up to floating-point slack
        self.assertTrue((lhs <= rhs + 1e-5).all().item())

    def test_bound_tight_when_aligned(self) -> None:
        """If Δz is parallel to w, equality holds in Cauchy-Schwarz."""
        w: torch.Tensor = torch.randn(HIDDEN)
        # Δz = α·w
        z_orig: torch.Tensor = torch.randn(HIDDEN)
        delta: torch.Tensor = 2.5 * w
        z_pert: torch.Tensor = z_orig - delta
        lhs: float = float(w_dot_delta(w, z_orig.unsqueeze(0), z_pert.unsqueeze(0))[0])
        rhs: float = w_norm(w) * float(
            delta_norm(z_orig.unsqueeze(0), z_pert.unsqueeze(0))[0]
        )
        self.assertAlmostEqual(abs(lhs), rhs, places=4)


class LinearReadoutEquivalenceTest(TestCase):
    """The headline math: w · Δz_postnorm == change in log-odds.

    Holds when both pos and neg are single tokens (so logsumexp collapses
    to a single logit) and we read from the post-norm hidden state (so
    the logit really is linear in z). With a synthetic W and arbitrary z,
    the partition function cancels in (logit_pos - logit_neg) → log-odds.
    """

    def test_single_token_exactness(self) -> None:
        torch.manual_seed(1)
        W: torch.Tensor = _make_W()
        pos: int = 3
        neg: int = 11

        # w as the lm_head difference (post-norm linear direction)
        w: torch.Tensor = W[pos] - W[neg]

        # z and a perturbation
        z_orig: torch.Tensor = torch.randn(HIDDEN)
        z_pert: torch.Tensor = z_orig + 0.3 * torch.randn(HIDDEN)

        # Direct: change in log-odds via softmax math
        lo_orig: float = _logit_diff(W, z_orig, pos, neg)
        lo_pert: float = _logit_diff(W, z_pert, pos, neg)
        delta_logodds: float = lo_orig - lo_pert

        # Predicted by the linear-readout formula
        predicted: float = float(
            w_dot_delta(w, z_orig.unsqueeze(0), z_pert.unsqueeze(0))[0]
        )
        self.assertAlmostEqual(predicted, delta_logodds, places=4)

    def test_multi_token_is_only_approximate(self) -> None:
        """With multiple pos/neg tokens, w = ΣW[pos] - ΣW[neg] is
        not the exact linearization — log-odds uses logsumexp, whose
        gradient is a softmax-weighted combination of W rows. This test
        simply documents that the equality fails in general; it's not a
        soundness test of our code, just a guardrail against future
        readers expecting more than the math gives."""
        torch.manual_seed(2)
        W: torch.Tensor = _make_W()
        pos_ids: list[int] = [3, 4]
        neg_ids: list[int] = [11, 12]
        w: torch.Tensor = W[pos_ids].sum(dim=0) - W[neg_ids].sum(dim=0)

        z_orig: torch.Tensor = torch.randn(HIDDEN)
        z_pert: torch.Tensor = z_orig + 0.3 * torch.randn(HIDDEN)

        def logsumexp_diff(z: torch.Tensor) -> float:
            logits: torch.Tensor = W @ z
            log_probs: torch.Tensor = torch.log_softmax(logits, dim=-1)
            pos_lp: float = float(torch.logsumexp(log_probs[pos_ids], dim=0))
            neg_lp: float = float(torch.logsumexp(log_probs[neg_ids], dim=0))
            return pos_lp - neg_lp

        delta_true: float = logsumexp_diff(z_orig) - logsumexp_diff(z_pert)
        predicted: float = float(
            w_dot_delta(w, z_orig.unsqueeze(0), z_pert.unsqueeze(0))[0]
        )
        # Document that they generally differ (we don't claim equality).
        # Use a loose bound: just check Cauchy-Schwarz still holds for the
        # naive w against the true delta.
        self.assertNotAlmostEqual(predicted, delta_true, places=2)


class RepresentationMetricsBatchTest(TestCase):
    def test_keys_present(self) -> None:
        z_orig: torch.Tensor = torch.randn(1, HIDDEN)
        z_pert: torch.Tensor = torch.randn(3, HIDDEN)
        w: torch.Tensor = torch.randn(HIDDEN)
        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig.expand(3, HIDDEN),
            z_orig_postnorm=z_orig.expand(3, HIDDEN),
            z_pert_prenorm=z_pert,
            z_pert_postnorm=z_pert,
            w_postnorm=w,
            w_prenorm=w,
        )
        self.assertEqual(len(rows), 3)
        expected: set[str] = {
            "delta_norm_prenorm",
            "delta_norm_postnorm",
            "cossim_prenorm",
            "cossim_postnorm",
            "w_dot_delta_z_postnorm",
            "w_dot_delta_z_prenorm",
            "z_orig_norm_prenorm",
            "z_pert_norm_prenorm",
            "z_orig_norm_postnorm",
            "z_pert_norm_postnorm",
            "w_dot_z_orig_postnorm",
            "w_dot_z_pert_postnorm",
            "w_dot_z_orig_prenorm",
            "w_dot_z_pert_prenorm",
        }
        for row in rows:
            self.assertEqual(set(row.keys()), expected)

    def test_omits_prenorm_w_when_none(self) -> None:
        z_orig: torch.Tensor = torch.randn(1, HIDDEN).expand(2, HIDDEN)
        z_pert: torch.Tensor = torch.randn(2, HIDDEN)
        w: torch.Tensor = torch.randn(HIDDEN)
        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig,
            z_orig_postnorm=z_orig,
            z_pert_prenorm=z_pert,
            z_pert_postnorm=z_pert,
            w_postnorm=w,
            w_prenorm=None,
        )
        for row in rows:
            self.assertNotIn("w_dot_delta_z_prenorm", row)
            self.assertNotIn("w_dot_z_orig_prenorm", row)
            self.assertNotIn("w_dot_z_pert_prenorm", row)
            self.assertIn("w_dot_delta_z_postnorm", row)
            self.assertIn("w_dot_z_orig_postnorm", row)
            self.assertIn("w_dot_z_pert_postnorm", row)
            # Norms are unconditional.
            self.assertIn("z_orig_norm_postnorm", row)
            self.assertIn("z_pert_norm_prenorm", row)

    def test_additivity_w_dot_delta_equals_diff_of_projections(self) -> None:
        """w·(z - z') == w·z - w·z' for both pre- and post-norm columns."""
        torch.manual_seed(7)
        n: int = 4
        z_orig: torch.Tensor = torch.randn(1, HIDDEN).expand(n, HIDDEN).contiguous()
        z_pert: torch.Tensor = torch.randn(n, HIDDEN)
        z_orig_pre: torch.Tensor = torch.randn(1, HIDDEN).expand(n, HIDDEN).contiguous()
        z_pert_pre: torch.Tensor = torch.randn(n, HIDDEN)
        w: torch.Tensor = torch.randn(HIDDEN)
        w_pre: torch.Tensor = torch.randn(HIDDEN)
        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig_pre,
            z_orig_postnorm=z_orig,
            z_pert_prenorm=z_pert_pre,
            z_pert_postnorm=z_pert,
            w_postnorm=w,
            w_prenorm=w_pre,
        )
        for row in rows:
            self.assertAlmostEqual(
                row["w_dot_delta_z_postnorm"],
                row["w_dot_z_orig_postnorm"] - row["w_dot_z_pert_postnorm"],
                places=4,
            )
            self.assertAlmostEqual(
                row["w_dot_delta_z_prenorm"],
                row["w_dot_z_orig_prenorm"] - row["w_dot_z_pert_prenorm"],
                places=4,
            )

    def test_orig_scalars_broadcast_across_rows(self) -> None:
        """When z_orig is (1, H), per-row orig norms/projections all equal."""
        torch.manual_seed(13)
        n: int = 5
        z_orig: torch.Tensor = torch.randn(1, HIDDEN).expand(n, HIDDEN).contiguous()
        z_pert: torch.Tensor = torch.randn(n, HIDDEN)
        w: torch.Tensor = torch.randn(HIDDEN)
        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig,
            z_orig_postnorm=z_orig,
            z_pert_prenorm=z_pert,
            z_pert_postnorm=z_pert,
            w_postnorm=w,
            w_prenorm=w,
        )
        for key in (
            "z_orig_norm_prenorm",
            "z_orig_norm_postnorm",
            "w_dot_z_orig_postnorm",
            "w_dot_z_orig_prenorm",
        ):
            values: list[float] = [row[key] for row in rows]
            for v in values[1:]:
                self.assertAlmostEqual(v, values[0], places=5)

    def test_ln_decomposition_holds_under_synthetic_rmsnorm(self) -> None:
        """For RMSNorm with v_eff = gamma * w, the LN-vs-linear decomposition
        reconstructs the post-norm attribution exactly:

            a = sqrt(d) * [(z_pre · v_eff)/||z_pre|| - (z'_pre · v_eff)/||z'_pre||]
        """
        torch.manual_seed(101)
        d: int = HIDDEN
        gamma: torch.Tensor = torch.randn(d).abs() + 0.5
        w: torch.Tensor = torch.randn(d)
        v_eff: torch.Tensor = gamma * w

        z_pre_orig: torch.Tensor = torch.randn(1, d)
        z_pre_pert: torch.Tensor = torch.randn(3, d)

        def _rmsnorm(x: torch.Tensor) -> torch.Tensor:
            rms: torch.Tensor = torch.linalg.vector_norm(x, dim=-1, keepdim=True) / (
                d**0.5
            )
            return gamma * x / rms

        z_post_orig: torch.Tensor = _rmsnorm(z_pre_orig)
        z_post_pert: torch.Tensor = _rmsnorm(z_pre_pert)

        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_pre_orig.expand(3, d),
            z_orig_postnorm=z_post_orig.expand(3, d),
            z_pert_prenorm=z_pre_pert,
            z_pert_postnorm=z_post_pert,
            w_postnorm=w,
            w_prenorm=v_eff,
        )
        sqrt_d: float = float(d**0.5)
        for row in rows:
            n0: float = row["z_orig_norm_prenorm"]
            n1: float = row["z_pert_norm_prenorm"]
            p0: float = row["w_dot_z_orig_prenorm"]
            p1: float = row["w_dot_z_pert_prenorm"]
            reconstructed: float = sqrt_d * (p0 / n0 - p1 / n1)
            self.assertAlmostEqual(
                reconstructed, row["w_dot_delta_z_postnorm"], places=4
            )

    def test_per_row_matches_individual_calls(self) -> None:
        """Batched output[i] equals the same metric computed on row i alone."""
        torch.manual_seed(11)
        n: int = 5
        z_orig: torch.Tensor = torch.randn(1, HIDDEN).expand(n, HIDDEN).contiguous()
        z_pert: torch.Tensor = torch.randn(n, HIDDEN)
        w: torch.Tensor = torch.randn(HIDDEN)
        batched: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig,
            z_orig_postnorm=z_orig,
            z_pert_prenorm=z_pert,
            z_pert_postnorm=z_pert,
            w_postnorm=w,
        )
        for i in range(n):
            single_dn: float = float(
                delta_norm(z_orig[i : i + 1], z_pert[i : i + 1])[0]
            )
            single_cs: float = float(
                cosine_similarity(z_orig[i : i + 1], z_pert[i : i + 1])[0]
            )
            single_wd: float = float(
                w_dot_delta(w, z_orig[i : i + 1], z_pert[i : i + 1])[0]
            )
            self.assertAlmostEqual(
                batched[i]["delta_norm_postnorm"], single_dn, places=5
            )
            self.assertAlmostEqual(batched[i]["cossim_postnorm"], single_cs, places=5)
            self.assertAlmostEqual(
                batched[i]["w_dot_delta_z_postnorm"], single_wd, places=5
            )


class EndToEndAblationSanityTest(TestCase):
    """Integration-style: simulate the full pipeline for a single ablation
    using a synthetic linear readout and verify w_dot_delta_z_postnorm
    equals the change in log-odds (the exact-attribution claim from the
    representation_scoring module docstring).
    """

    def test_pipeline_matches_logodds_change(self) -> None:
        torch.manual_seed(42)
        W: torch.Tensor = _make_W(seed=42)
        pos: int = 5
        neg: int = 9

        # Three "ablated" hidden states (perturbations of z_orig)
        z_orig: torch.Tensor = torch.randn(HIDDEN)
        z_perts: torch.Tensor = z_orig.unsqueeze(0) + 0.5 * torch.randn(3, HIDDEN)

        # True log-odds change for each ablation
        true_deltas: list[float] = [
            _logit_diff(W, z_orig, pos, neg) - _logit_diff(W, z_perts[i], pos, neg)
            for i in range(3)
        ]

        # The pipeline's predicted delta from rep metrics
        w: torch.Tensor = W[pos] - W[neg]
        rows: list[dict[str, float]] = representation_metrics_batch(
            z_orig_prenorm=z_orig.unsqueeze(0).expand(3, HIDDEN),
            z_orig_postnorm=z_orig.unsqueeze(0).expand(3, HIDDEN),
            z_pert_prenorm=z_perts,
            z_pert_postnorm=z_perts,
            w_postnorm=w,
        )
        for i in range(3):
            self.assertAlmostEqual(
                rows[i]["w_dot_delta_z_postnorm"], true_deltas[i], places=4
            )
            # Cauchy-Schwarz: |attr| ≤ ‖w‖·‖Δz‖
            self.assertLessEqual(
                abs(rows[i]["w_dot_delta_z_postnorm"]),
                w_norm(w) * rows[i]["delta_norm_postnorm"] + 1e-5,
            )
