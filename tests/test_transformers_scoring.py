# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import torch
from surrogate.eval_constants import ReportToken
from surrogate.transformers_model import TransformersModel
from surrogate.transformers_scoring import (
    transformers_label_logprob_breakdown,
    transformers_label_logprob_breakdown_batch,
)
from surrogate.model_types import make_dialog
from unittest import IsolatedAsyncioTestCase as TestCase


def _mock_transformers_model() -> TransformersModel:
    model: TransformersModel = MagicMock(spec=TransformersModel)
    model.model_name = "mock-transformer"
    model.dialog_to_text.return_value = "prompt"
    return model


class TestTransformersScoring(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.dialog = make_dialog("system", "hello")
        self.dialog2 = make_dialog("system", "goodbye")
        self.label_tokens: dict[str, set[str] | list[str] | str] = {
            "true": {"true", "True"},
            "false": {"false"},
        }
        self.report_tokens: dict[str, list[ReportToken]] = {
            "true": [
                ReportToken(alias="true", surface="true"),
                ReportToken(alias="True", surface="True"),
                ReportToken(alias="us_TRUE", surface="_TRUE"),
            ],
            "false": [ReportToken(alias="false", surface="false")],
        }
        self.model = _mock_transformers_model()
        token_to_id: dict[str, int] = {
            "true": 0,
            "True": 1,
            "false": 2,
        }
        self.model.encode_single_token.side_effect = lambda token: token_to_id.get(
            token
        )
        self.log_probs1: torch.Tensor = torch.tensor([-1.0, -0.2, -2.0])
        self.log_probs2: torch.Tensor = torch.tensor([-3.0, -2.4, -0.1])

    async def test_label_breakdown_reports_group_and_token_logprobs(self) -> None:
        self.model.get_next_token_log_probs.return_value = self.log_probs1

        result = await transformers_label_logprob_breakdown(
            self.model,
            self.dialog,
            label_tokens=self.label_tokens,
            report_tokens=self.report_tokens,
        )

        self.assertIsNotNone(result)
        assert result is not None
        true_group: float | None = result.label_logprobs["true"]
        false_group: float | None = result.label_logprobs["false"]
        true_tok: float | None = result.token_logprobs["true"]["true"]
        title_true_tok: float | None = result.token_logprobs["true"]["True"]
        false_tok: float | None = result.token_logprobs["false"]["false"]
        assert true_group is not None
        assert false_group is not None
        assert true_tok is not None
        assert title_true_tok is not None
        assert false_tok is not None
        self.assertAlmostEqual(
            true_group,
            float(torch.logsumexp(self.log_probs1[[0, 1]], dim=0).item()),
            places=5,
        )
        self.assertAlmostEqual(false_group, -2.0, places=5)
        self.assertAlmostEqual(true_tok, -1.0, places=5)
        self.assertAlmostEqual(title_true_tok, -0.2, places=5)
        self.assertIsNone(result.token_logprobs["true"]["us_TRUE"])
        self.assertAlmostEqual(false_tok, -2.0, places=5)

    def test_batch_breakdown_preserves_input_order(self) -> None:
        self.model.get_next_token_log_probs_batch.return_value = [
            self.log_probs1,
            self.log_probs2,
        ]

        results = transformers_label_logprob_breakdown_batch(
            self.model,
            [self.dialog, self.dialog2],
            label_tokens=self.label_tokens,
            report_tokens=self.report_tokens,
            batch_size=2,
        )

        self.assertEqual(len(results), 2)
        self.assertIsNotNone(results[0])
        self.assertIsNotNone(results[1])
        assert results[0] is not None
        assert results[1] is not None
        first_true: float | None = results[0].label_logprobs["true"]
        first_false: float | None = results[0].label_logprobs["false"]
        second_true: float | None = results[1].label_logprobs["true"]
        second_false: float | None = results[1].label_logprobs["false"]
        assert first_true is not None
        assert first_false is not None
        assert second_true is not None
        assert second_false is not None
        self.assertGreater(
            first_true,
            first_false,
        )
        self.assertLess(
            second_true,
            second_false,
        )

    def test_batch_breakdown_returns_none_for_failed_chunk(self) -> None:
        self.model.get_next_token_log_probs_batch.side_effect = RuntimeError("boom")

        results = transformers_label_logprob_breakdown_batch(
            self.model,
            [self.dialog, self.dialog2],
            label_tokens=self.label_tokens,
            report_tokens=self.report_tokens,
            batch_size=2,
        )

        self.assertEqual(results, [None, None])
