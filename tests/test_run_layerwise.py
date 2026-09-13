# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Focused tests for the compact public layerwise runner."""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import torch

from benchmark_scripts.run_layerwise import (
    CANONICAL_BATCH_SIZE,
    _ablated_rows,
    _atomic_write_tsv,
    _character_length_permutation,
    _column_names,
    _model_artifact_hashes,
    _model_selection,
    _resolve_label_groups,
    _sha256_file,
    _source_hashes,
    _tokenize_batch,
    _validate_or_write_manifest,
    _verified_model_identity,
    run_layerwise,
)
from benchmark_scripts.provenance_sources import canonical_file_hash_manifest_sha256
from surrogate.eval_constants import ReportToken
from surrogate.layerwise_scoring import LayerwiseLabelScores


class _FakeTokenizer:
    """Map test surfaces to fixed token sequences."""

    def __init__(self, token_ids: dict[str, list[int]]) -> None:
        self._token_ids: dict[str, list[int]] = token_ids

    def encode(self, surface: str, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("label aliases must never add special tokens")
        return self._token_ids[surface]


class _RenderedTextTokenizer:
    """Minimal tokenizer state needed by ``_tokenize_batch``."""

    def __init__(self) -> None:
        self.padding_side: str = "right"
        self.pad_token: str | None = None
        self.eos_token: str = "<eos>"


class _RenderedTextModel:
    """Record calls to the fail-closed rendered-text tokenizer helper."""

    def __init__(self) -> None:
        self._tokenizer: _RenderedTextTokenizer = _RenderedTextTokenizer()
        self._model: Any = type("CausalLM", (), {"device": torch.device("cpu")})()
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def _tokenize_rendered_text(
        self, texts: list[str], **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        self.calls.append((texts, kwargs))
        return {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.ones((2, 2), dtype=torch.long),
        }


class RenderedTextTokenizationTest(TestCase):
    """Tests that layerwise scoring cannot request tokenizer special tokens."""

    def test_uses_fail_closed_rendered_text_helper(self) -> None:
        model: Any = _RenderedTextModel()

        input_ids, attention_mask = _tokenize_batch(model, ["first", "second"])

        self.assertEqual(input_ids.tolist(), [[1, 2], [3, 4]])
        self.assertEqual(attention_mask.tolist(), [[1, 1], [1, 1]])
        self.assertEqual(
            model.calls,
            [
                (
                    ["first", "second"],
                    {"return_tensors": "pt", "padding": True},
                )
            ],
        )
        self.assertEqual(model._tokenizer.padding_side, "left")
        self.assertEqual(model._tokenizer.pad_token, "<eos>")


class ResolveLabelGroupsTest(TestCase):
    """Tests for strict alias resolution and token ownership."""

    def test_deduplicates_ids_and_records_multitoken_aliases(self) -> None:
        tokenizer = _FakeTokenizer(
            {"ent": [1], " ent": [1], "ENT": [2, 3], "neutral": [4]}
        )
        groups, metadata = _resolve_label_groups(
            tokenizer,
            {
                "entailment": [
                    ReportToken(alias="ent", surface="ent"),
                    ReportToken(alias="sp_ent", surface=" ent"),
                    ReportToken(alias="ENT", surface="ENT"),
                ],
                "neutral": [ReportToken(alias="neutral", surface="neutral")],
            },
        )

        self.assertEqual(groups, {"entailment": [1], "neutral": [4]})
        self.assertEqual(
            metadata["entailment"]["deduplicated_single_token_aliases"][0][
                "duplicate_of_alias"
            ],
            "ent",
        )
        self.assertEqual(
            metadata["entailment"]["rejected_multitoken_aliases"][0]["token_ids"],
            [2, 3],
        )

    def test_rejects_empty_and_overlapping_groups(self) -> None:
        with self.assertRaisesRegex(ValueError, "no single-token aliases"):
            _resolve_label_groups(
                _FakeTokenizer({"many": [1, 2], "other": [3]}),
                {
                    "empty": [ReportToken(alias="many", surface="many")],
                    "other": [ReportToken(alias="other", surface="other")],
                },
            )
        with self.assertRaisesRegex(ValueError, "overlaps labels"):
            _resolve_label_groups(
                _FakeTokenizer({"a": [5], "b": [5]}),
                {
                    "first": [ReportToken(alias="a", surface="a")],
                    "second": [ReportToken(alias="b", surface="b")],
                },
            )


class LayerRowTest(TestCase):
    """Tests for exact grouped scores and historical alignment scalars."""

    def test_ablated_rows_retain_all_label_contrasts(self) -> None:
        labels: tuple[str, ...] = ("entailment", "neutral", "contradiction")
        original = LayerwiseLabelScores(
            labels=labels,
            grouped_logsumexp=torch.tensor([[[10.0, 7.0, 2.0]], [[11.0, 8.0, 4.0]]]),
            summed_unembedding_projection=torch.tensor(
                [[[9.0, 6.0, 1.0]], [[10.0, 7.0, 3.0]]]
            ),
        )
        perturbed = LayerwiseLabelScores(
            labels=labels,
            grouped_logsumexp=torch.tensor(
                [
                    [[8.0, 6.0, 3.0], [7.0, 5.0, 4.0]],
                    [[9.0, 6.0, 5.0], [8.0, 5.0, 6.0]],
                ]
            ),
            summed_unembedding_projection=torch.tensor(
                [
                    [[7.0, 5.0, 2.0], [6.0, 4.0, 3.0]],
                    [[8.0, 5.0, 4.0], [7.0, 4.0, 5.0]],
                ]
            ),
        )
        label_suffixes, contrast_suffixes = _column_names(labels)
        contrast_norms: dict[tuple[str, str], float] = {
            pair: float(index + 1) for index, pair in enumerate(contrast_suffixes)
        }

        rows: list[dict[str, Any]] = _ablated_rows(
            prompt_idx=12,
            answer=0,
            segment_indices=[4, 1],
            original_scores=original,
            perturbed_scores=perturbed,
            delta_norms=torch.tensor([[0.5, 0.7], [1.5, 1.7]]),
            label_suffixes=label_suffixes,
            contrast_suffixes=contrast_suffixes,
            contrast_norms=contrast_norms,
        )

        self.assertEqual(len(rows), 4)
        first: dict[str, Any] = rows[0]
        self.assertEqual(first["seg_idx"], 4)
        self.assertEqual(first["layer_kind"], "embedding")
        self.assertIsNone(first["block_idx"])
        self.assertEqual(first["label_score_contradiction"], 3.0)
        # Original E-C projection is 9-1=8; ablated is 7-2=5.
        self.assertEqual(
            first["w_dot_delta_z_postnorm_entailment_vs_contradiction"],
            3.0,
        )
        self.assertIn("w_dot_delta_z_postnorm_entailment_vs_neutral", first)
        self.assertIn("w_dot_delta_z_postnorm_neutral_vs_contradiction", first)


class OutputSafetyTest(TestCase):
    """Tests for deterministic atomic output and manifest binding."""

    def test_gzip_payload_is_filename_independent_and_deterministic(self) -> None:
        frame: pd.DataFrame = pd.DataFrame([{"a": 1, "b": "text"}])
        with tempfile.TemporaryDirectory() as directory:
            first: str = os.path.join(directory, "first.tsv.gz")
            second: str = os.path.join(directory, "second.tsv.gz")
            _atomic_write_tsv(first, frame)
            _atomic_write_tsv(second, frame)
            self.assertEqual(_sha256_file(first), _sha256_file(second))
            pd.testing.assert_frame_equal(pd.read_csv(first, sep="\t"), frame)

    def test_manifest_must_match_exactly_but_canary_can_create_one(self) -> None:
        expected: pd.DataFrame = pd.DataFrame(
            [
                {
                    "prompt_idx": 0,
                    "answer": True,
                    "seg_idx": 0,
                    "message_idx": 0,
                    "message_role": "system",
                    "message_seg_idx": 0,
                    "segment_text": "instruction",
                    "n_segments": 1,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "segments.tsv.gz")
            with self.assertRaises(FileNotFoundError):
                _validate_or_write_manifest(path, expected, canary=False)
            _validate_or_write_manifest(path, expected, canary=True)
            _validate_or_write_manifest(path, expected, canary=False)

            changed: pd.DataFrame = expected.copy()
            changed.loc[0, "segment_text"] = "different"
            with self.assertRaisesRegex(ValueError, "disagrees"):
                _validate_or_write_manifest(path, changed, canary=False)

    def test_model_filter_preserves_model_set_order_and_rejects_unknowns(self) -> None:
        selected: list[tuple[str, str]] = _model_selection(
            "Qwen2.5-Instruct",
            "qwen2.5-7b-instruct,qwen2.5-0.5b-instruct",
        )
        self.assertEqual(
            [name for name, _ in selected],
            ["qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct"],
        )
        with self.assertRaisesRegex(ValueError, "not in Qwen2.5-Instruct"):
            _model_selection("Qwen2.5-Instruct", "missing-model")

    def test_character_length_order_matches_ordinary_stable_sort(self) -> None:
        texts: list[str] = ["four", "a", "also", "bb"]

        self.assertEqual(_character_length_permutation(texts), [1, 3, 0, 2])

    def test_noncanary_requires_canonical_batch_size(self) -> None:
        self.assertEqual(CANONICAL_BATCH_SIZE, 32)
        with self.assertRaisesRegex(
            ValueError, "canonical layerwise runs require batch_size=32"
        ):
            asyncio.run(run_layerwise("boolq", batch_size=8))

    def test_canary_permits_diagnostic_batch_size(self) -> None:
        def reject_dataset(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
            raise RuntimeError("diagnostic batch accepted")

        with (
            patch(
                "benchmark_scripts.run_layerwise.load_benchmark_dataset",
                side_effect=reject_dataset,
            ),
            self.assertRaisesRegex(RuntimeError, "diagnostic batch accepted"),
        ):
            asyncio.run(
                run_layerwise(
                    "boolq",
                    batch_size=8,
                    max_samples=1,
                    canary=True,
                )
            )


class ProvenanceTest(TestCase):
    """Tests for entry-time source and pinned full-model identity binding."""

    def test_source_snapshot_precedes_dataset_loading(self) -> None:
        events: list[str] = []

        def snapshot() -> dict[str, str]:
            events.append("source")
            return {}

        def reject_dataset(*_args: Any, **_kwargs: Any) -> pd.DataFrame:
            events.append("dataset")
            raise RuntimeError("stop after ordering check")

        with (
            patch(
                "benchmark_scripts.run_layerwise._source_hashes",
                side_effect=snapshot,
            ),
            patch(
                "benchmark_scripts.run_layerwise.load_benchmark_dataset",
                side_effect=reject_dataset,
            ),
            self.assertRaisesRegex(RuntimeError, "ordering check"),
        ):
            asyncio.run(run_layerwise("boolq"))
        self.assertEqual(events, ["source", "dataset"])

    def test_source_inventory_covers_transitive_public_dependencies(self) -> None:
        self.assertEqual(
            set(_source_hashes()),
            {
                "benchmark_scripts/benchmark_config.py",
                "benchmark_scripts/provenance_sources.py",
                "benchmark_scripts/run_layerwise.py",
                "surrogate/eval_constants.py",
                "surrogate/layerwise_scoring.py",
                "surrogate/model_types.py",
                "surrogate/text_augmentation.py",
                "surrogate/transformers_model.py",
                "surrogate/utils.py",
            },
        )

    def test_verifies_full_local_model_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths: dict[str, bytes] = {
                "config.json": b"config",
                "tokenizer.json": b"tokenizer",
                "model-00001-of-00001.safetensors": b"weights",
            }
            for name, payload in paths.items():
                with open(os.path.join(directory, name), "wb") as output:
                    output.write(payload)
            with open(os.path.join(directory, "README.md"), "wb") as output:
                output.write(b"not a runtime model artifact")
            hashes: dict[str, str] = _model_artifact_hashes(directory)
            aggregate: str = canonical_file_hash_manifest_sha256(hashes)

            with (
                patch.dict(
                    "benchmark_scripts.run_layerwise.GOLD_OPEN_MODEL_REPOSITORIES",
                    {"test-model": "test/repository"},
                ),
                patch.dict(
                    "benchmark_scripts.run_layerwise.GOLD_OPEN_MODEL_REVISIONS",
                    {"test-model": "revision"},
                ),
                patch.dict(
                    "benchmark_scripts.run_layerwise.GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {"test-model": aggregate},
                ),
            ):
                revision, verified, verified_aggregate, identity = (
                    _verified_model_identity("test-model", "test/repository", directory)
                )

            self.assertEqual(revision, "revision")
            self.assertEqual(verified, hashes)
            self.assertEqual(verified_aggregate, aggregate)
            self.assertEqual(set(identity), {"config.json", "tokenizer.json"})
            self.assertNotIn("README.md", verified)

    def test_rejects_unpinned_or_mismatched_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No pinned"):
                _verified_model_identity("missing", "missing/repository", directory)
