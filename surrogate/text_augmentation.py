# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Literal

from surrogate.model_types import Dialog, Message
from surrogate.utils import segment_text


async def segment_and_ablate(
    prompt: Dialog,
    pregrouper_id: Literal["word", "sentence"] = "word",
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
    text = prompt.messages[0].content
    segments, template = segment_text(text, level=pregrouper_id)

    ablated_dialogs: list[Dialog] = []
    for i in range(len(segments)):
        new_segments = list(segments)
        new_segments[i] = ""
        new_text = template.format(*new_segments)
        new_dialog = Dialog(
            messages=[
                Message(
                    role=msg.role,
                    content=new_text if j == 0 else msg.content,
                )
                for j, msg in enumerate(prompt.messages)
            ]
        )
        ablated_dialogs.append(new_dialog)
    return ablated_dialogs
