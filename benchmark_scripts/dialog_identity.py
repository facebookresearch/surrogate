# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Canonical prompt and ablated-dialog identity digests for public artifacts."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from surrogate.model_types import Dialog, make_dialog
from surrogate.text_augmentation import dialog_segments, segment_and_ablate


IDENTITY_FORMAT: str = "canonical_jsonl_prompt_and_full_dialog_v1"


def _messages(dialog: Dialog) -> list[list[str]]:
    return [[message.role, message.content] for message in dialog.messages]


def _canonical_line(value: list[object]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


async def compute_dialog_identity(
    benchmark: str,
    pregrouper: str,
    dataset_path: str,
    manifest_path: str,
) -> tuple[str, str]:
    """Hash every represented original prompt and selected ablated dialog.

    The manifest is also checked against a fresh segmentation of each prompt,
    so the returned ablation digest cannot silently bless a different segment
    coordinate system.
    """

    spec = BENCHMARKS[benchmark]
    if spec.eval_config is None:
        raise ValueError("Dialog identity is defined here only for classification")
    frame: pd.DataFrame = pd.read_csv(dataset_path, sep="\t")
    if spec.dataset_preprocessor is not None:
        frame = spec.dataset_preprocessor(frame.copy())
    manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
    selected: dict[int, pd.DataFrame] = {
        int(prompt_idx): rows.sort_values("seg_idx")
        for prompt_idx, rows in manifest.groupby("prompt_idx", sort=False)
    }
    prompt_hash = hashlib.sha256()
    ablated_hash = hashlib.sha256()
    for prompt_idx in sorted(selected):
        if prompt_idx < 0 or prompt_idx >= len(frame):
            raise ValueError(f"Manifest prompt {prompt_idx} is outside the dataset")
        row: pd.Series = frame.iloc[prompt_idx]
        dialog: Dialog = make_dialog(
            spec.eval_config.system_prompt,
            spec.prompt_builder(row),
        )
        prompt_hash.update(_canonical_line([prompt_idx, _messages(dialog)]))
        ablated: list[Dialog] = await segment_and_ablate(
            dialog, pregrouper_id=pregrouper
        )
        segments = dialog_segments(dialog, pregrouper_id=pregrouper)
        declared: int = int(selected[prompt_idx]["n_segments"].iloc[0])
        if len(ablated) != len(segments) or len(segments) != declared:
            raise ValueError(f"Ablation count mismatch at {benchmark}/{prompt_idx}")
        for manifest_row in selected[prompt_idx].itertuples(index=False):
            seg_idx: int = int(manifest_row.seg_idx)
            segment = segments[seg_idx]
            if (
                segment.message_idx != int(manifest_row.message_idx)
                or segment.message_role != str(manifest_row.message_role)
                or segment.message_segment_idx != int(manifest_row.message_seg_idx)
                or segment.text != str(manifest_row.segment_text)
            ):
                raise ValueError(
                    f"Manifest segment mismatch at {benchmark}/{prompt_idx}/{seg_idx}"
                )
            ablated_hash.update(
                _canonical_line([prompt_idx, seg_idx, _messages(ablated[seg_idx])])
            )
    return prompt_hash.hexdigest(), ablated_hash.hexdigest()
