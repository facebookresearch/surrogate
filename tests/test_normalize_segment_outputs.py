# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import hashlib
import tempfile
from unittest import TestCase

import pandas as pd

from benchmark_scripts.normalize_segment_outputs import normalize_file


class TestNormalizeSegmentOutputs(TestCase):
    def test_fills_missing_keys_and_records_availability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "model_segment.tsv.gz")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": index,
                        "answer": False,
                        "message_idx": index,
                        "message_role": "system" if index == 0 else "user",
                        "message_seg_idx": 0,
                        "segment_text": f"segment {index}",
                        "n_segments": 2,
                    }
                    for index in range(2)
                ]
            )
            manifest.iloc[:1][["prompt_idx", "seg_idx", "answer", "n_segments"]].to_csv(
                path, sep="\t", index=False
            )

            with self.assertRaisesRegex(ValueError, "missing 1"):
                normalize_file(path, manifest)

            with self.assertRaisesRegex(ValueError, "lacks identity columns"):
                normalize_file(path, manifest, allow_missing=True)

            normalize_file(
                path,
                manifest,
                allow_missing=True,
                legacy_identity_attestation="Legacy keys independently verified.",
            )

            result: pd.DataFrame = pd.read_csv(path, sep="\t")
            self.assertEqual(len(result), 2)
            self.assertEqual(result["segment_result_available"].tolist(), [True, False])
            self.assertEqual(result["original_result_available"].tolist(), [True, True])
            self.assertEqual(result["message_role"].tolist(), ["system", "user"])

            with open(path, "rb") as source:
                before: str = hashlib.sha256(source.read()).hexdigest()
            normalize_file(path, manifest)
            with open(path, "rb") as source:
                after: str = hashlib.sha256(source.read()).hexdigest()
            self.assertEqual(before, after)

    def test_completion_availability_is_derived_from_finite_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "model_segment.tsv.gz")
            manifest: pd.DataFrame = pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": index,
                        "answer": "target",
                        "message_idx": 1,
                        "message_role": "user",
                        "message_seg_idx": index,
                        "segment_text": f"word {index}",
                        "n_segments": 2,
                    }
                    for index in range(2)
                ]
            )
            frame: pd.DataFrame = manifest.copy()
            frame["orig_completion_logprob"] = [-1.0, -1.0]
            frame["ablated_completion_logprob"] = [-2.0, float("nan")]
            frame.to_csv(path, sep="\t", index=False)

            normalize_file(path, manifest)

            result: pd.DataFrame = pd.read_csv(path, sep="\t")
            self.assertEqual(result["segment_result_available"].tolist(), [True, False])
            self.assertEqual(result["original_result_available"].tolist(), [True, True])
