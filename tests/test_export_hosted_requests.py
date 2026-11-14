# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from benchmark_scripts.export_hosted_requests import build_requests
from surrogate.model_types import make_dialog
from surrogate.text_augmentation import dialog_segments


class TestExportHostedRequests(TestCase):
    def test_requests_match_canonical_dialog_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset_path: str = os.path.join(directory, "boolq.tsv")
            manifest_path: str = os.path.join(directory, "segments.tsv.gz")
            frame: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "passage": "First sentence here. Second one follows.",
                        "question": "is this a test",
                        "answer": True,
                    }
                ]
            )
            frame.to_csv(dataset_path, sep="\t", index=False)
            spec = BENCHMARKS["boolq"]
            dialog = make_dialog(
                spec.eval_config.system_prompt, spec.prompt_builder(frame.iloc[0])
            )
            segments = dialog_segments(dialog, pregrouper_id="sentence")
            pd.DataFrame(
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
            ).to_csv(manifest_path, sep="\t", index=False)

            requests, prompt_sha256, ablated_sha256 = asyncio.run(
                build_requests("boolq", "sentence", dataset_path, manifest_path)
            )
            expected = asyncio.run(
                compute_dialog_identity(
                    "boolq", "sentence", dataset_path, manifest_path
                )
            )

        self.assertEqual((prompt_sha256, ablated_sha256), expected)
        self.assertEqual(len(requests), 1 + len(segments))
        original = requests[0]
        self.assertIsNone(original["seg_idx"])
        self.assertIs(original["answer"], True)
        self.assertEqual(original["n_segments"], len(segments))
        self.assertEqual(
            original["messages"],
            [[message.role, message.content] for message in dialog.messages],
        )
        self.assertEqual(
            [request["seg_idx"] for request in requests[1:]],
            [segment.segment_idx for segment in segments],
        )
        user_segment = next(s for s in segments if s.message_role == "user")
        ablated_user = requests[1 + user_segment.segment_idx]["messages"][1][1]
        self.assertNotIn(user_segment.text, ablated_user)
