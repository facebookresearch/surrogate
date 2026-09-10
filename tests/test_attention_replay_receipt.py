# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for separate BoolQ-word attention replay receipts."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmark_scripts import attention_replay_receipt
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
)


class AttentionReplayReceiptTest(unittest.TestCase):
    """Validate coordinate, identity, source, and numerical replay checks."""

    def _write_fixture(self, root: str, model: str, offset: float = 0.0) -> None:
        directory: str = os.path.join(root, "boolq", "word")
        os.makedirs(directory)
        values: np.ndarray = np.arange(10_000, dtype=float) / 10_000.0
        frame: pd.DataFrame = pd.DataFrame(
            {
                "prompt_idx": np.arange(10_000),
                "seg_idx": np.zeros(10_000, dtype=int),
                "message_idx": np.ones(10_000, dtype=int),
                "message_role": ["user"] * 10_000,
                "message_seg_idx": np.zeros(10_000, dtype=int),
                "segment_text": [f"word-{index}" for index in range(10_000)],
                "n_segments": np.ones(10_000, dtype=int),
                "attention_mean": values + offset,
                "attention_max": values * 2.0 + offset,
                "attention_rollout": values * 0.5 + offset,
            }
        )
        frame.to_csv(
            os.path.join(directory, f"{model}_segment.tsv.gz"),
            sep="\t",
            index=False,
        )
        frame[list(attention_replay_receipt.IDENTITY_COLUMNS)].to_csv(
            os.path.join(directory, "segments.tsv.gz"),
            sep="\t",
            index=False,
        )
        is_replay: bool = os.path.basename(root) == "replay"
        with open(
            os.path.join(directory, f"{model}_run.json"),
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                {
                    "benchmark": "boolq",
                    "dataset": attention_replay_receipt.EXPECTED_DATASET,
                    "model": model,
                    "model_identity_files_sha256": {"config.json": "1" * 64},
                    "parameters": {
                        "phase_attention_implementation": (
                            {"attention": "eager"}
                            if is_replay
                            else {"ablation": "sdpa", "attention": "eager"}
                        ),
                        "max_forward_passes": None,
                        "max_samples": None,
                        "phases": (
                            ["attention"] if is_replay else ["ablation", "attention"]
                        ),
                    },
                    "pregrouper": "word",
                    "segmentation_scope": "full_dialog_in_message_order",
                    "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
                },
                output,
            )

    def _write_execution_evidence(self, temporary: str) -> tuple[str, str]:
        observation_path: str = os.path.join(temporary, "observation.json")
        completion_path: str = os.path.join(temporary, "completion.json")
        with open(observation_path, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "source_files": {
                        path: {"mtime_ns": 1, "sha256": digest}
                        for path, digest in (
                            GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256.items()
                        )
                    }
                },
                output,
            )
        with open(completion_path, "w", encoding="utf-8") as output:
            json.dump(
                {
                    "schema_version": 1,
                    "started_epoch": 10,
                    "completed_epoch": 20,
                },
                output,
            )
        return observation_path, completion_path

    def test_build_receipt_checks_exact_replay(self) -> None:
        model: str = "qwen2.5-0.5b-instruct"
        with tempfile.TemporaryDirectory() as temporary:
            results_dir: str = os.path.join(temporary, "results")
            replay_dir: str = os.path.join(temporary, "replay")
            self._write_fixture(results_dir, model)
            self._write_fixture(replay_dir, model)
            observation_path, completion_path = self._write_execution_evidence(
                temporary
            )
            replay_manifest_sha256: str = attention_replay_receipt._sha256(
                os.path.join(replay_dir, "boolq", "word", "segments.tsv.gz")
            )
            with (
                patch.object(attention_replay_receipt, "QWEN_MODELS", (model,)),
                patch.object(attention_replay_receipt, "EXPECTED_REPLAY_ROWS", 10_000),
                patch.object(
                    attention_replay_receipt, "EXPECTED_REPLAY_PROMPTS", 10_000
                ),
                patch.object(
                    attention_replay_receipt,
                    "EXPECTED_REPLAY_MANIFEST_SHA256",
                    replay_manifest_sha256,
                ),
            ):
                receipt = attention_replay_receipt.build_receipt(
                    results_dir,
                    replay_dir,
                    observation_path,
                    completion_path,
                )
                attention_replay_receipt.validate_receipt(receipt, results_dir)
                receipt["purpose"] = "unreviewed text"
                with self.assertRaisesRegex(ValueError, "metadata disagrees"):
                    attention_replay_receipt.validate_receipt(receipt, results_dir)
                receipt["purpose"] = attention_replay_receipt.PURPOSE
                receipt["models"][model]["replay_segment_sha256"] = "invalid"
                with self.assertRaisesRegex(ValueError, "provenance disagrees"):
                    attention_replay_receipt.validate_receipt(receipt, results_dir)
            self.assertEqual(10_000, receipt["models"][model]["matched_rows"])
            self.assertEqual(
                0.0,
                receipt["models"][model]["metrics"]["attention_mean"]["max_abs_error"],
            )

    def test_build_receipt_rejects_numerical_mismatch(self) -> None:
        model: str = "qwen2.5-0.5b-instruct"
        with tempfile.TemporaryDirectory() as temporary:
            results_dir: str = os.path.join(temporary, "results")
            replay_dir: str = os.path.join(temporary, "replay")
            self._write_fixture(results_dir, model)
            self._write_fixture(replay_dir, model, offset=0.01)
            observation_path, completion_path = self._write_execution_evidence(
                temporary
            )
            replay_manifest_sha256: str = attention_replay_receipt._sha256(
                os.path.join(replay_dir, "boolq", "word", "segments.tsv.gz")
            )
            with (
                patch.object(attention_replay_receipt, "QWEN_MODELS", (model,)),
                patch.object(attention_replay_receipt, "EXPECTED_REPLAY_ROWS", 10_000),
                patch.object(
                    attention_replay_receipt, "EXPECTED_REPLAY_PROMPTS", 10_000
                ),
                patch.object(
                    attention_replay_receipt,
                    "EXPECTED_REPLAY_MANIFEST_SHA256",
                    replay_manifest_sha256,
                ),
                self.assertRaisesRegex(ValueError, "Attention replay mismatch"),
            ):
                attention_replay_receipt.build_receipt(
                    results_dir,
                    replay_dir,
                    observation_path,
                    completion_path,
                )

    def test_build_receipt_rejects_unpinned_full_manifest(self) -> None:
        model: str = "qwen2.5-0.5b-instruct"
        with tempfile.TemporaryDirectory() as temporary:
            results_dir: str = os.path.join(temporary, "results")
            replay_dir: str = os.path.join(temporary, "replay")
            self._write_fixture(results_dir, model)
            self._write_fixture(replay_dir, model)
            observation_path, completion_path = self._write_execution_evidence(
                temporary
            )
            with (
                patch.object(attention_replay_receipt, "QWEN_MODELS", (model,)),
                patch.object(attention_replay_receipt, "EXPECTED_REPLAY_ROWS", 10_000),
                patch.object(
                    attention_replay_receipt, "EXPECTED_REPLAY_PROMPTS", 10_000
                ),
                patch.object(
                    attention_replay_receipt,
                    "EXPECTED_REPLAY_MANIFEST_SHA256",
                    "0" * 64,
                ),
                self.assertRaisesRegex(ValueError, "provenance mismatch"),
            ):
                attention_replay_receipt.build_receipt(
                    results_dir,
                    replay_dir,
                    observation_path,
                    completion_path,
                )


if __name__ == "__main__":
    unittest.main()
