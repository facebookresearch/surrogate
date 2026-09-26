# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from surrogate.model_types import Dialog, Message
from surrogate.utils import segment_text

PregrouperID = Literal["word", "sentence"]
SegmentedMessage = tuple[int, list[str], str]


@dataclass(frozen=True)
class DialogSegment:
    """One addressable text segment in a dialog."""

    segment_idx: int
    message_idx: int
    message_role: str
    message_segment_idx: int
    text: str


def segment_dialog(
    prompt: Dialog,
    pregrouper_id: PregrouperID = "word",
) -> list[SegmentedMessage]:
    """Segment every message in a dialog, preserving message order.

    Returns:
        Tuples containing the message index, its ordered segments, and the
        template used to reconstruct that message.
    """
    return [
        (message_idx, *segment_text(message.content, level=pregrouper_id))
        for message_idx, message in enumerate(prompt.messages)
    ]


def dialog_segments(
    prompt: Dialog,
    pregrouper_id: PregrouperID = "word",
) -> list[DialogSegment]:
    """Return all dialog segments with stable global and message-local indices."""
    result: list[DialogSegment] = []
    for message_idx, segments, _ in segment_dialog(prompt, pregrouper_id):
        for message_segment_idx, text in enumerate(segments):
            result.append(
                DialogSegment(
                    segment_idx=len(result),
                    message_idx=message_idx,
                    message_role=prompt.messages[message_idx].role,
                    message_segment_idx=message_segment_idx,
                    text=text,
                )
            )
    return result


async def segment_and_ablate(
    prompt: Dialog,
    pregrouper_id: PregrouperID = "word",
) -> list[Dialog]:
    """
    Given a prompt, segment with a regex-based segmenter and ablate each segment.
    Returns a list of Dialog objects, each of which has one segment ablated out.

    This is a text augmentation in the sense that it turns a single prompt into a list
    of modified prompts. This is useful for the surrogate model scoring project because
    we are interested in measuring how different models respond to perturbations like
    this one.

    Args:
        prompt: The prompt to segment and ablate.
        pregrouper_id: The segmentation level to use ("word" or "sentence").
            Defaults to "word".

    Returns:
        A list of Dialog objects, each of which has one segment ablated out.
    """
    ablated_dialogs: list[Dialog] = []
    for message_idx, segments, template in segment_dialog(prompt, pregrouper_id):
        for segment_idx in range(len(segments)):
            new_segments: list[str] = list(segments)
            new_segments[segment_idx] = ""
            new_text: str = template.format(*new_segments)
            new_dialog = Dialog(
                messages=[
                    Message(
                        role=message.role,
                        content=new_text if index == message_idx else message.content,
                    )
                    for index, message in enumerate(prompt.messages)
                ]
            )
            ablated_dialogs.append(new_dialog)
    return ablated_dialogs
