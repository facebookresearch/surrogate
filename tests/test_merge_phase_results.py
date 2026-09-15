# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for merging independently executed benchmark phases."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import pandas as pd

from benchmark_scripts.merge_phase_results import _validate_receipts, merge


def _receipt(phase: str) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "benchmark": "boolq",
        "pregrouper": "sentence",
        "segmentation_scope": "full_dialog_in_message_order",
        "model": "model",
        "model_source": "source",
        "model_revision": "revision",
        "model_revision_resolution": "content_hashes",
        "model_identity_files_sha256": {"config.json": "digest"},
        "model_protocol": {"prefix": "direct"},
        "dataset": {"rows": 10},
        "software": {"torch": "version"},
        "source_sha256": {"runner.py": "digest"},
        "parameters": {
            "phases": [phase],
            "phase_attention_implementation": {
                phase: "eager" if phase == "attention" else "sdpa"
            },
            "batch_size": 2,
            "seed": 42,
        },
    }


class MergePhaseResultsTest(TestCase):
    """Validate the phase-receipt compatibility checks."""

    def test_compatible_receipts(self) -> None:
        _validate_receipts(_receipt("attention"), _receipt("ablation"))

    def test_execution_source_disagreement_is_rejected(self) -> None:
        attention: dict[str, Any] = _receipt("attention")
        ablation: dict[str, Any] = deepcopy(_receipt("ablation"))
        ablation["source_sha256"]["runner.py"] = "other"
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            _validate_receipts(attention, ablation)

    def test_wrong_phase_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only attention"):
            _validate_receipts(_receipt("ablation"), _receipt("ablation"))

    def test_merge_normalizes_and_binds_outputs(self) -> None:
        with TemporaryDirectory() as directory:
            attention_root: str = os.path.join(directory, "attention")
            ablation_root: str = os.path.join(directory, "ablation")
            output_root: str = os.path.join(directory, "output")
            for root in (attention_root, ablation_root):
                config_dir: str = os.path.join(root, "boolq", "sentence")
                os.makedirs(config_dir)
                manifest: pd.DataFrame = pd.DataFrame(
                    {
                        "prompt_idx": [0],
                        "answer": [True],
                        "seg_idx": [0],
                        "message_idx": [0],
                        "message_role": ["system"],
                        "message_seg_idx": [0],
                        "segment_text": ["system"],
                        "n_segments": [1],
                    }
                )
                manifest.to_csv(
                    os.path.join(config_dir, "segments.tsv.gz"),
                    sep="\t",
                    index=False,
                )
            attention_dir: str = os.path.join(attention_root, "boolq", "sentence")
            ablation_dir: str = os.path.join(ablation_root, "boolq", "sentence")
            manifest.assign(attention_mean=0.25).to_csv(
                os.path.join(attention_dir, "model_segment.tsv.gz"),
                sep="\t",
                index=False,
            )
            manifest.assign(delta_norm_postnorm=0.5).to_csv(
                os.path.join(ablation_dir, "model_segment.tsv.gz"),
                sep="\t",
                index=False,
            )
            pd.DataFrame(
                {
                    "prompt_idx": [0],
                    "seg_idx": [0],
                    "kind": ["ablated"],
                    "answer": [True],
                    "label": ["True"],
                    "token": ["True"],
                    "logprob": [-0.1],
                }
            ).to_csv(
                os.path.join(ablation_dir, "model_tokens.tsv.gz"),
                sep="\t",
                index=False,
            )
            for config_dir, phase in (
                (attention_dir, "attention"),
                (ablation_dir, "ablation"),
            ):
                with open(
                    os.path.join(config_dir, "model_run.json"),
                    "w",
                    encoding="utf-8",
                ) as output:
                    json.dump(_receipt(phase), output)

            merge(
                attention_root,
                ablation_root,
                output_root,
                "boolq",
                "sentence",
                "model",
            )

            output_dir: str = os.path.join(output_root, "boolq", "sentence")
            merged: pd.DataFrame = pd.read_csv(
                os.path.join(output_dir, "model_segment.tsv.gz"), sep="\t"
            )
            self.assertEqual(len(merged), 1)
            self.assertTrue(merged["segment_result_available"].all())
            self.assertTrue(merged["original_result_available"].all())
            with open(
                os.path.join(output_dir, "model_run.json"), encoding="utf-8"
            ) as source:
                receipt: dict[str, Any] = json.load(source)
            self.assertEqual(receipt["assembly"]["kind"], "parallel_independent_phases")
            self.assertEqual(receipt["parameters"]["phases"], ["ablation", "attention"])
            self.assertEqual(set(receipt["output_sha256"]), {"segment", "tokens"})
