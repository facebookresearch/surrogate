# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for the isolated Muse Glimmer robustness-table driver."""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from benchmark_scripts.glimmer_robustness import (
    _pair_population,
    _validate_extension_outputs,
)
from benchmark_scripts.validate_results import OPEN_SEGMENT_COLUMNS
from surrogate.eval_constants import BOOLQ_CONFIG


class GlimmerRobustnessTest(TestCase):
    """Validate explicit reference-population labels."""

    def test_open_reference_population(self) -> None:
        row: pd.Series = pd.Series({"model_s": "qwen2.5-7b-instruct"})
        self.assertEqual(_pair_population(row), "glimmer_open")

    def test_hosted_reference_population(self) -> None:
        row: pd.Series = pd.Series({"model_s": "gpt-4o"})
        self.assertEqual(_pair_population(row), "glimmer_hosted")

    def test_unknown_reference_population_is_rejected(self) -> None:
        row: pd.Series = pd.Series({"model_s": "unknown"})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            _pair_population(row)

    def _raw_fixture(self, directory: str) -> tuple[str, str, pd.DataFrame]:
        manifest: pd.DataFrame = pd.DataFrame(
            {
                "prompt_idx": [0],
                "answer": [True],
                "seg_idx": [0],
                "message_idx": [1],
                "message_role": ["user"],
                "message_seg_idx": [0],
                "segment_text": ["question"],
                "n_segments": [1],
            }
        )
        segments: pd.DataFrame = manifest.assign(
            original_result_available=True,
            segment_result_available=True,
            **{metric: 0.5 for metric in OPEN_SEGMENT_COLUMNS},
        )
        aliases: list[tuple[str, str]] = [
            (label, token.alias)
            for label, report_tokens in BOOLQ_CONFIG.report_tokens.items()
            for token in report_tokens
        ]
        token_rows: list[dict[str, object]] = []
        for kind, seg_idx in (("orig", float("nan")), ("ablated", 0)):
            for label, token in aliases:
                token_rows.append(
                    {
                        "prompt_idx": 0,
                        "seg_idx": seg_idx,
                        "kind": kind,
                        "answer": True,
                        "label": label,
                        "token": token,
                        "logprob": -1.0,
                    }
                )
        segment_path: str = os.path.join(directory, "segment.tsv.gz")
        token_path: str = os.path.join(directory, "tokens.tsv.gz")
        segments.to_csv(segment_path, sep="\t", index=False)
        pd.DataFrame(token_rows).to_csv(token_path, sep="\t", index=False)
        return segment_path, token_path, manifest

    def test_complete_raw_grid_is_accepted(self) -> None:
        with TemporaryDirectory() as directory:
            segment_path, token_path, manifest = self._raw_fixture(directory)
            segments, tokens = _validate_extension_outputs(
                segment_path, token_path, manifest
            )
            self.assertEqual(len(segments), 1)
            self.assertEqual(len(tokens), 36)
            self.assertNotIn("seg_idx_key", tokens)

    def test_missing_alias_row_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            segment_path, token_path, manifest = self._raw_fixture(directory)
            tokens: pd.DataFrame = pd.read_csv(token_path, sep="\t").iloc[:-1]
            tokens.to_csv(token_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "grid is incomplete"):
                _validate_extension_outputs(segment_path, token_path, manifest)
