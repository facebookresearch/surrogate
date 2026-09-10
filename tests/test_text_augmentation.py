# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal
from unittest import IsolatedAsyncioTestCase

from surrogate.text_augmentation import segment_and_ablate
from surrogate.model_types import Dialog, Message


class TestTextAugmentation(IsolatedAsyncioTestCase):
    async def test_segment_and_ablate(self) -> None:
        cases: list[tuple[str, Literal["word", "sentence", "line"], list[str]]] = [
            ("1 2 3", "word", [" 2 3", "1 3", "1 2"]),
            ("123", "word", [""]),
            ("1 2 3", "sentence", [""]),
        ]
        for input_prompt, pregrouper_type, expected_outputs in cases:
            with self.subTest(
                input_prompt=input_prompt, pregrouper_type=pregrouper_type
            ):
                prompt = Dialog(messages=[Message(role="system", content=input_prompt)])
                actual_outputs: list[Dialog] = await segment_and_ablate(
                    prompt, pregrouper_id=pregrouper_type
                )

                actual_outputs_text = [
                    actual_output.messages[0].content
                    for actual_output in actual_outputs
                ]
                self.assertEqual(actual_outputs_text, expected_outputs)

                expected_outputs_dialog = [
                    Dialog(messages=[Message(role="system", content=expected_output)])
                    for expected_output in expected_outputs
                ]
                self.assertEqual(actual_outputs, expected_outputs_dialog)

    async def test_segments_all_messages_in_dialog_order(self) -> None:
        prompt = Dialog(
            messages=[
                Message(role="system", content="System one. System two."),
                Message(role="user", content="User one. User two."),
            ]
        )

        actual_outputs: list[Dialog] = await segment_and_ablate(
            prompt, pregrouper_id="sentence"
        )

        self.assertEqual(
            actual_outputs,
            [
                Dialog(
                    messages=[
                        Message(role="system", content=" System two."),
                        Message(role="user", content="User one. User two."),
                    ]
                ),
                Dialog(
                    messages=[
                        Message(role="system", content="System one."),
                        Message(role="user", content="User one. User two."),
                    ]
                ),
                Dialog(
                    messages=[
                        Message(role="system", content="System one. System two."),
                        Message(role="user", content=" User two."),
                    ]
                ),
                Dialog(
                    messages=[
                        Message(role="system", content="System one. System two."),
                        Message(role="user", content="User one."),
                    ]
                ),
            ],
        )

    async def test_sentence_segmentation_does_not_split_line_initials(self) -> None:
        prompt = Dialog(
            messages=[Message(role="user", content="Hypothesis:\nF.T. Island")]
        )

        actual_outputs: list[Dialog] = await segment_and_ablate(
            prompt, pregrouper_id="sentence"
        )

        self.assertEqual(
            actual_outputs,
            [
                Dialog(messages=[Message(role="user", content=" Island")]),
                Dialog(messages=[Message(role="user", content="Hypothesis:\nF.T.")]),
            ],
        )
