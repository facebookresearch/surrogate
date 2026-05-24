# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import TypeAlias


@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class Dialog:
    messages: list[Message]


@dataclass(frozen=True)
class LogprobBreakdown:
    label_logprobs: dict[str, float | None]
    token_logprobs: dict[str, dict[str, float | None]]


def make_dialog(system_prompt: str, user_prompt: str) -> Dialog:
    return Dialog(
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_prompt),
        ]
    )


SingleScore: TypeAlias = list[float | None]
