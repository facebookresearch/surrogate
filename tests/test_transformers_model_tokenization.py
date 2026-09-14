# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for tokenizing already-rendered chat prompts exactly once."""

from __future__ import annotations

from typing import Any
from unittest import TestCase

from surrogate.transformers_model import TransformersModel


class _RecordingTokenizer:
    """Minimal tokenizer that records its latest invocation."""

    def __init__(self) -> None:
        self.call_kwargs: dict[str, Any] = {}

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        self.call_kwargs = {"text": text, **kwargs}
        return {"input_ids": [[1]]}


class RenderedTextTokenizationTest(TestCase):
    """Ensure chat-template control tokens are never added a second time."""

    def test_disables_tokenizer_special_tokens(self) -> None:
        model: TransformersModel = TransformersModel("test", "unused")
        tokenizer = _RecordingTokenizer()
        model._tokenizer = tokenizer
        model._loaded = True

        output: dict[str, Any] = model._tokenize_rendered_text(
            "<bos>rendered prompt", return_tensors="pt"
        )

        self.assertEqual(output, {"input_ids": [[1]]})
        self.assertEqual(tokenizer.call_kwargs["add_special_tokens"], False)

    def test_rejects_caller_override(self) -> None:
        model: TransformersModel = TransformersModel("test", "unused")
        model._tokenizer = _RecordingTokenizer()
        model._loaded = True

        with self.assertRaisesRegex(ValueError, "owns add_special_tokens"):
            model._tokenize_rendered_text("rendered prompt", add_special_tokens=True)
