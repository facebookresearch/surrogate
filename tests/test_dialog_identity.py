# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import asyncio
import os
import tempfile
from unittest import TestCase

import pandas as pd

from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.dialog_identity import compute_dialog_identity
from surrogate.model_types import make_dialog
from surrogate.text_augmentation import dialog_segments


class TestDialogIdentity(TestCase):
    def test_digest_is_stable_and_manifest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path: str = os.path.join(directory, "boolq.tsv")
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            frame: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "passage": "A short passage",
                        "question": "is this a test",
                        "answer": True,
                    }
                ]
            )
            frame.to_csv(dataset_path, sep="\t", index=False)
            spec = BENCHMARKS["boolq"]
            dialog = make_dialog(
                spec.eval_config.system_prompt,
                spec.prompt_builder(frame.iloc[0]),
            )
            segments = dialog_segments(dialog, pregrouper_id="sentence")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "answer": True,
                        "seg_idx": segment.segment_idx,
                        "message_idx": segment.message_idx,
                        "message_role": segment.message_role,
                        "message_seg_idx": segment.message_segment_idx,
                        "segment_text": segment.text,
                        "n_segments": len(segments),
                    }
                    for segment in segments
                ]
            )
            manifest.to_csv(manifest_path, sep="\t", index=False)

            first: tuple[str, str] = asyncio.run(
                compute_dialog_identity(
                    "boolq", "sentence", dataset_path, manifest_path
                )
            )
            second: tuple[str, str] = asyncio.run(
                compute_dialog_identity(
                    "boolq", "sentence", dataset_path, manifest_path
                )
            )
            self.assertEqual(first, second)
            self.assertTrue(all(len(value) == 64 for value in first))

            manifest.loc[0, "segment_text"] = "tampered"
            manifest.to_csv(manifest_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "Manifest segment mismatch"):
                asyncio.run(
                    compute_dialog_identity(
                        "boolq", "sentence", dataset_path, manifest_path
                    )
                )
