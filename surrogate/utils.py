# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re


# Currency symbols for all currencies.
CURRENCY_SYMBOLS: str = (
    "$¢£¤¥֏؋৲৳৻૱௹฿៛\u20a0-\u20bd\ua838\ufdfc\ufe69\uff04\uffe0\uffe1\uffe5\uffe6"
)
# Word regex that matches a word together with up to two leading spaces so
# word-level ablations preserve the historical pregrouper behavior.
WORD_RX: re.Pattern[str] = re.compile(
    r"((?: {,2})?'?(?<!\w)["
    + CURRENCY_SYMBOLS
    + r"]?(?:[\w\d\-]|[\.][^\.\s]|,\d)+["
    + CURRENCY_SYMBOLS
    + r"]?(?!\w))",
    flags=re.UNICODE,
)
# Sentence regex
SENTENCE_RX: re.Pattern[str] = re.compile(
    r"((?:^\s*(?:-|(?:\d+|\w)[\.\)])|\.(?=$|\s+)|;|[!?]+)\n*)",
    flags=re.MULTILINE,
)


def _escape_braces(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


def segment_text(text: str, level: str = "word") -> tuple[list[str], str]:
    if level == "word":
        return _segment_words(text)
    elif level == "sentence":
        return _segment_sentences(text)
    else:
        raise ValueError(f"Unknown segmentation level: {level!r}")


def _segment_words(text: str) -> tuple[list[str], str]:
    words_and_delimiters = WORD_RX.split(text)
    delimiters = words_and_delimiters[::2]
    words = words_and_delimiters[1::2]
    assert len(words) + 1 == len(delimiters), f"{len(words)=} {len(delimiters)=}"

    template = "{}".join(_escape_braces(delimiter) for delimiter in delimiters)
    return words, template


def _segment_sentences(text: str) -> tuple[list[str], str]:
    sentences_and_delimiters = SENTENCE_RX.split(text)
    sentences_raw = sentences_and_delimiters[::2]
    delimiters = sentences_and_delimiters[1::2]
    assert len(sentences_raw) == len(delimiters) + 1, (
        f"{len(sentences_raw)=} {len(delimiters)=}"
    )

    segments: list[str] = []
    pending_prefix = sentences_raw[0]
    if pending_prefix:
        segments.append(pending_prefix)
        pending_prefix = ""

    for sentence, delimiter in zip(sentences_raw[1:], delimiters):
        if segments:
            segments[-1] += delimiter
        else:
            pending_prefix += delimiter
        if sentence:
            segments.append(pending_prefix + sentence)
            pending_prefix = ""

    if pending_prefix:
        segments.append(pending_prefix)

    template = "{}" * len(segments)
    return segments, template


