# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from dataclasses import dataclass, field

_NON_ALNUM_RE: re.Pattern[str] = re.compile(r"[^0-9A-Za-z_]+")


@dataclass(frozen=True)
class ReportToken:
    alias: str
    surface: str


def _sanitize_column_component(value: str) -> str:
    sanitized: str = _NON_ALNUM_RE.sub("_", value).strip("_")
    return sanitized or "token"


def label_column_alias(label: str) -> str:
    return _sanitize_column_component(label.lower())


def report_token_alias(token: str) -> str:
    prefix: str = ""
    surface: str = token
    if token.startswith(" "):
        prefix = "sp_"
        surface = token[1:]
    elif token.startswith("_"):
        prefix = "us_"
        surface = token[1:]
    return f"{prefix}{_sanitize_column_component(surface)}"


def ordered_token_variations(token: str) -> list[ReportToken]:
    """Generate ordered case and spacing variations of a token.

    Order is stable and preserves the common plain/title/upper progression
    within each prefix family:
    - Prefix: none, space, underscore
    - Case: original, lower, Title, UPPER

    Duplicate surfaces are removed while preserving first occurrence, so
    lower-case inputs like ``"true"`` still produce the intuitive order:
    ``true``, ``True``, ``TRUE``, `` true``, ...
    """
    cases: list[str] = [token, token.lower(), token.capitalize(), token.upper()]
    prefixes: list[str] = ["", " ", "_"]
    seen: set[str] = set()
    out: list[ReportToken] = []
    for prefix in prefixes:
        for case in cases:
            surface: str = f"{prefix}{case}"
            if surface in seen:
                continue
            seen.add(surface)
            out.append(ReportToken(alias=report_token_alias(surface), surface=surface))
    return out


def token_variations(token: str) -> set[str]:
    """Generate case and spacing variations of a token."""
    return {item.surface for item in ordered_token_variations(token)}


@dataclass
class EvalConfig:
    """
    Configuration for an evaluation benchmark.
    """

    system_prompt: str
    label_tokens: dict[str, set[str]]  # label -> set of matching tokens
    report_tokens: dict[str, list[ReportToken]] = field(default_factory=dict)


BOOLQ_CONFIG = EvalConfig(
    system_prompt="You are a general-purpose binary question-answering machine. You will be shown a passage (marked 'Context'), followed by a true/false question (marked 'Question') about that passage. Answer the question on the basis of the context, and respond only with 'true' or 'false'.",
    label_tokens={
        "true": token_variations("true"),
        "false": token_variations("false"),
    },
    report_tokens={
        "true": ordered_token_variations("true"),
        "false": ordered_token_variations("false"),
    },
)

ANLI_CONFIG = EvalConfig(
    system_prompt="You are a general-purpose logic machine. You will be shown a passage (marked 'Premise'), followed by a hypothesis (marked 'Hypothesis'). Your task is to determine whether the hypothesis is implied by the premise. Please respond ONLY with one of the following: {'entailment', 'neutral', 'contradiction'.}",
    label_tokens={
        "entailment": token_variations("ent"),
        "neutral": token_variations("neutral"),
        "contradiction": token_variations("contr"),
    },
    report_tokens={
        "entailment": ordered_token_variations("ent"),
        "neutral": ordered_token_variations("neutral"),
        "contradiction": ordered_token_variations("contr"),
    },
)

WINOGRANDE_CONFIG = EvalConfig(
    system_prompt="You are a general-purpose logic machine. You will be shown a passage (marked 'Sentence'), followed by two options (marked 'Option 1' and 'Option 2'). The sentence will have a blank in it; your task is to determine which option is the most likely to fill in the blank in the passage. Please respond ONLY with one of the following: {'1', '2'.}",
    label_tokens={
        "option 1": {"1"},
        "option 2": {"2"},
    },
    report_tokens={
        "option 1": [ReportToken(alias="1", surface="1")],
        "option 2": [ReportToken(alias="2", surface="2")],
    },
)
