# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Hookpoint correctness tests for ``TransformersModel``.

These tests pin down the mathematical/code properties that the bf16-on-GPU
smoke test (``benchmark_scripts/smoke_test_repr.py``) checks against a real
Qwen2.5 model, but verifies them on a tiny CPU-only randomly-initialized
``Qwen2ForCausalLM``. That keeps the tests fast (<1s), deterministic, and
free of external dependencies (no large model download, no GPU), while still
exercising the actual HuggingFace forward path that the smoke test relies
on (``model.model.norm`` as the hookpoint, ``model.lm_head`` as the linear
readout).

Each test docstring states the property being verified.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import patch

import torch
from surrogate.transformers_model import TransformersModel
from unittest import TestCase
from transformers import Qwen2Config, Qwen2ForCausalLM


def _make_tiny_qwen2(seed: int = 0) -> Any:
    """Build a small randomly-initialized Qwen2 on CPU. ~50k params.

    Returns ``Any`` rather than ``Qwen2ForCausalLM`` because the HF stubs
    don't expose all the dynamically-defined attributes (``eval``,
    ``model``, ``lm_head``) that we touch at runtime; pyre would flag
    every access otherwise.
    """
    torch.manual_seed(seed)
    config: Qwen2Config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    model: Any = Qwen2ForCausalLM(config)
    model.eval()
    return model


def _make_wrapper(model: Any) -> TransformersModel:
    """Wrap a pre-built model in TransformersModel, bypassing .load()."""
    tm: TransformersModel = TransformersModel(
        model_name="tiny-qwen2", model_path="(in-memory)"
    )
    tm._model = model
    tm._loaded = True
    return tm


class HookpointCapturesLmHeadInputTest(TestCase):
    """Property: ``lm_head(post_captured)`` reproduces ``model(input_ids).logits``
    exactly (modulo fp32 numerics).

    The forward hook on ``model.model.norm`` captures the post-norm
    activation, which is exactly what ``lm_head`` reads. Re-applying
    ``lm_head`` to the captured tensor must yield the same logits the
    full-model forward returns. If this property breaks, the linear-readout
    identity (``w · z_post = logit_pos − logit_neg``) and every
    representation-level metric built on it are wrong by an unknown factor.
    """

    def test_post_norm_capture_reproduces_logits(self) -> None:
        model: Any = _make_tiny_qwen2()
        tm: TransformersModel = _make_wrapper(model)

        torch.manual_seed(1)
        input_ids: torch.Tensor = torch.randint(0, 128, (3, 16))
        attention_mask: torch.Tensor = torch.ones_like(input_ids)

        with torch.no_grad():
            ref_logits: torch.Tensor = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits

        # Replicate the hookpoint capture used by get_last_layer_repr_batch.
        norm_module: Any = tm._locate_final_norm()
        self.assertIsNotNone(norm_module)
        captured: dict[str, torch.Tensor] = {}

        def post_hook(
            _mod: Any, _args: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            captured["post"] = output.detach()

        h: Any = norm_module.register_forward_hook(post_hook)
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            h.remove()

        with torch.no_grad():
            recon_logits: torch.Tensor = model.lm_head(captured["post"])

        self.assertEqual(ref_logits.shape, recon_logits.shape)
        self.assertTrue(torch.allclose(ref_logits, recon_logits, atol=1e-5))


class SingleTokenLinearReadoutExactTest(TestCase):
    """Property: for any single-token (pos, neg) and any input,

        (W[pos] − W[neg]) · z_post[batch, last]  ==  logits[batch, last, pos] − logits[batch, last, neg]

    where ``W = lm_head.weight`` and ``z_post`` is the post-final-norm
    hidden state at the prediction position (last index after left-padding).

    This is the single-token half of the smoke test, made deterministic on
    a tiny CPU model. The benchmark's multi-token aggregation gap (slope
    ≈ 1/|set|) is intentionally excluded — it's an asymptotic claim that
    needs many real-model prompts to verify, and is exercised by
    smoke_test_repr.py instead.
    """

    def test_w_dot_post_equals_logit_diff(self) -> None:
        model: Any = _make_tiny_qwen2()
        tm: TransformersModel = _make_wrapper(model)

        torch.manual_seed(2)
        input_ids: torch.Tensor = torch.randint(0, 128, (4, 12))
        attention_mask: torch.Tensor = torch.ones_like(input_ids)

        # Pick arbitrary single-token pos/neg ids.
        pos_id: int = 17
        neg_id: int = 91

        # Reference: actual logit diff at the last token.
        with torch.no_grad():
            ref_logits: torch.Tensor = model(
                input_ids=input_ids, attention_mask=attention_mask
            ).logits
        ref_diff: torch.Tensor = ref_logits[:, -1, pos_id] - ref_logits[:, -1, neg_id]

        # Reconstruct via captured post-norm and the unembedding direction.
        w: torch.Tensor = tm.get_unembedding_direction([pos_id], [neg_id])
        norm_module: Any = tm._locate_final_norm()
        captured_post: dict[str, torch.Tensor] = {}

        def post_hook(
            _mod: Any, _args: tuple[torch.Tensor, ...], output: torch.Tensor
        ) -> None:
            captured_post["x"] = output.detach()

        h: Any = norm_module.register_forward_hook(post_hook)
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            h.remove()
        z_last: torch.Tensor = captured_post["x"][:, -1, :]
        recon_diff: torch.Tensor = z_last @ w

        self.assertEqual(ref_diff.shape, recon_diff.shape)
        self.assertTrue(torch.allclose(ref_diff, recon_diff, atol=1e-4))


class UnembeddingDirectionConstructionTest(TestCase):
    """Properties of ``get_unembedding_direction``:

    1. Returns ``Σ_i W[pos_i] − Σ_j W[neg_j]``, where ``W = lm_head.weight``.
    2. Output shape is ``(hidden_size,)``.
    3. Output has ``requires_grad=False`` (the function is decorated with
       ``@torch.no_grad()`` so downstream uses don't accidentally retain a
       graph through ``lm_head.weight``).
    """

    def test_construction_matches_manual_sum(self) -> None:
        model: Any = _make_tiny_qwen2()
        tm: TransformersModel = _make_wrapper(model)

        pos_ids: list[int] = [3, 17, 42]
        neg_ids: list[int] = [11, 91]
        w: torch.Tensor = tm.get_unembedding_direction(pos_ids, neg_ids)

        with torch.no_grad():
            expected: torch.Tensor = model.lm_head.weight[pos_ids].sum(
                dim=0
            ) - model.lm_head.weight[neg_ids].sum(dim=0)
        self.assertTrue(torch.allclose(w, expected))
        self.assertEqual(w.shape, (model.config.hidden_size,))

    def test_no_grad_propagation(self) -> None:
        model: Any = _make_tiny_qwen2()
        tm: TransformersModel = _make_wrapper(model)
        w: torch.Tensor = tm.get_unembedding_direction([3], [11])
        self.assertFalse(
            w.requires_grad,
            "get_unembedding_direction must run under @torch.no_grad() so "
            "downstream arithmetic with w doesn't carry an unintended graph "
            "through lm_head.weight.",
        )


class LocateFinalNormFallbackTest(TestCase):
    """Property: ``_locate_final_norm()`` returns the final RMSNorm module
    when present (Llama/Qwen2 expose it as ``model.model.norm``), and
    returns ``None`` with a warning otherwise. The ``None`` fallback drives
    ``get_last_layer_repr_batch`` to a degenerate path where pre-norm and
    post-norm hidden states are identical — preferable to silently producing
    wrong values from a missing/misnamed norm.
    """

    def test_returns_norm_when_present(self) -> None:
        model: Any = _make_tiny_qwen2()
        tm: TransformersModel = _make_wrapper(model)
        norm: Any = tm._locate_final_norm()
        self.assertIs(norm, model.model.norm)

    def test_returns_none_and_warns_when_missing(self) -> None:
        # A model without `.model.norm` — use a bare nn.Module stub.
        bare: torch.nn.Module = torch.nn.Linear(4, 4)
        tm: TransformersModel = TransformersModel(
            model_name="bare", model_path="(in-memory)"
        )
        tm._model = bare
        tm._loaded = True
        with patch.object(
            logging.getLogger("surrogate.transformers_model"),
            "warning",
        ) as mock_warn:
            result: Any = tm._locate_final_norm()
        self.assertIsNone(result)
        mock_warn.assert_called_once()
