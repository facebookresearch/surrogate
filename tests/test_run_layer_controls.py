# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Focused tests for the control-only layer capture runner."""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest import TestCase

import numpy as np
import torch

from benchmark_scripts.provenance_sources import canonical_file_hash_manifest_sha256
from benchmark_scripts.run_layer_controls import (
    _assemble_control_projections,
    _canonical_json_sha256,
    ControlArtifactWriter,
    _parse_control_spec,
    _resolve_controls,
    _source_hashes,
)


_MODEL: str = "qwen2.5-0.5b-instruct"


def _model_record() -> dict[str, Any]:
    tokenizer_hashes: dict[str, str] = {"tokenizer.json": "1" * 64}
    return {
        "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
        "model_revision": "revision",
        "model_artifact_manifest_sha256": "2" * 64,
        "tokenizer_class": "FakeTokenizer",
        "tokenizer_files_sha256": tokenizer_hashes,
        "tokenizer_manifest_sha256": canonical_file_hash_manifest_sha256(
            tokenizer_hashes
        ),
        "vocabulary_size_including_added_tokens": 64,
    }


def _pool_entry(
    base: str,
    width: int,
    start_id: int,
) -> dict[str, Any]:
    linear: str = f" {base.lower()}"
    variants: list[str] = [linear] + [f"{base}_{index}" for index in range(width - 1)]
    return {
        "base_surface": base,
        "variants": variants,
        "token_ids_by_model": {_MODEL: list(range(start_id, start_id + width))},
    }


def _canary_payload(draws: int = 2) -> dict[str, Any]:
    p9_pool: list[dict[str, Any]] = [
        _pool_entry(f"Positive{index}", 9, 20 * index) for index in range(draws)
    ]
    p8_pool: list[dict[str, Any]] = [
        _pool_entry(f"Negative{index}", 8, 200 + 20 * index) for index in range(draws)
    ]
    selection: dict[str, Any] = {
        "draw_count": draws,
        "linear_surface_rule": "single ASCII space + lowercase base_surface",
        "pool_sha256": {
            "p8": _canonical_json_sha256(p8_pool),
            "p9": _canonical_json_sha256(p9_pool),
        },
        "production_draw_count": 256,
        "spec_hash_contract": (
            "sha256_compact_sorted_utf8_json_with_selection_algorithm_"
            "spec_sha256_absent"
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "namespace": "unit-test-controls",
        "selection_algorithm": selection,
        "models": {_MODEL: _model_record()},
        "p9_basis_pool": p9_pool,
        "p8_basis_pool": p8_pool,
        "selected_paired_bases": [
            {
                "draw_idx": index,
                "positive_base_surface": f"Positive{index}",
                "negative_base_surface": f"Negative{index}",
            }
            for index in range(draws)
        ],
        "isotropic_seeds_by_model": {_MODEL: [1000 + index for index in range(draws)]},
    }
    selection["spec_sha256"] = _canonical_json_sha256(payload)
    return payload


def _refresh_spec_hash(payload: dict[str, Any]) -> None:
    selection: dict[str, Any] = payload["selection_algorithm"]
    selection.pop("spec_sha256", None)
    selection["spec_sha256"] = _canonical_json_sha256(payload)


class _FakeTokenizer:
    """Exact one-token mapping used for resolution tests."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping: dict[str, int] = mapping

    def encode(self, surface: str, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("control surfaces must not add special tokens")
        return [self.mapping[surface]]


class ControlSpecTest(TestCase):
    """Tests for strict shared-pool and independent-seed validation."""

    def test_canary_accepts_fewer_draws_and_declared_model_subset(self) -> None:
        spec = _parse_control_spec(_canary_payload(), canary=True)

        self.assertEqual(spec.num_draws, 2)
        self.assertEqual(spec.selected_paired_bases[1].draw_idx, 1)
        self.assertEqual(len(spec.p9_basis_pool[0].variants), 9)
        self.assertEqual(len(spec.p8_basis_pool[0].variants), 8)

    def test_production_rejects_canary_draw_count_and_model_subset(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly five"):
            _parse_control_spec(_canary_payload(), canary=False)

    def test_rejects_reused_bases_and_seed_collisions(self) -> None:
        reused: dict[str, Any] = _canary_payload()
        reused["selected_paired_bases"][1]["positive_base_surface"] = "Positive0"
        _refresh_spec_hash(reused)
        with self.assertRaisesRegex(ValueError, "without replacement"):
            _parse_control_spec(reused, canary=True)

        duplicated_seed: dict[str, Any] = _canary_payload()
        duplicated_seed["isotropic_seeds_by_model"][_MODEL] = [7, 7]
        _refresh_spec_hash(duplicated_seed)
        with self.assertRaisesRegex(ValueError, "distinct"):
            _parse_control_spec(duplicated_seed, canary=True)

    def test_rejects_misaligned_pool_ids_and_variants(self) -> None:
        payload: dict[str, Any] = _canary_payload()
        payload["p9_basis_pool"][0]["token_ids_by_model"][_MODEL] = [1, 2]
        payload["selection_algorithm"]["pool_sha256"]["p9"] = _canonical_json_sha256(
            payload["p9_basis_pool"]
        )
        _refresh_spec_hash(payload)
        with self.assertRaisesRegex(ValueError, "width 9"):
            _parse_control_spec(payload, canary=True)


class ControlResolutionTest(TestCase):
    """Tests model-specific ID verification and direction metadata."""

    def test_resolves_target_shared_grouped_and_isotropic_controls(self) -> None:
        payload: dict[str, Any] = _canary_payload(draws=1)
        spec = _parse_control_spec(payload, canary=True)
        mapping: dict[str, int] = {" true": 830, " false": 895}
        positive: dict[str, Any] = payload["p9_basis_pool"][0]
        negative: dict[str, Any] = payload["p8_basis_pool"][0]
        mapping.update(
            zip(
                positive["variants"],
                positive["token_ids_by_model"][_MODEL],
            )
        )
        mapping.update(
            zip(
                negative["variants"],
                negative["token_ids_by_model"][_MODEL],
            )
        )
        tokenizer: _FakeTokenizer = _FakeTokenizer(mapping)
        head: torch.nn.Linear = torch.nn.Linear(4, 1000, bias=False)
        torch.manual_seed(3)
        with torch.no_grad():
            head.weight.copy_(torch.randn_like(head.weight))

        controls = _resolve_controls(spec, _MODEL, tokenizer, head)

        self.assertEqual(controls.linear_unit_directions.shape, (3, 4))
        self.assertEqual(controls.positive_group_ids[0], tuple(range(9)))
        self.assertEqual(controls.negative_group_ids[0], tuple(range(200, 208)))
        self.assertEqual(controls.canonical_direction["positive_surface"], " true")
        self.assertEqual(controls.shared_token_pairs[0]["draw_idx"], 0)
        self.assertEqual(len(controls.isotropic[0]["vector_sha256"]), 64)

    def test_live_tokenizer_disagreement_is_rejected(self) -> None:
        payload: dict[str, Any] = _canary_payload(draws=1)
        spec = _parse_control_spec(payload, canary=True)
        mapping: dict[str, int] = {" true": 830, " false": 895}
        for entry in (
            payload["p9_basis_pool"][0],
            payload["p8_basis_pool"][0],
        ):
            mapping.update(zip(entry["variants"], entry["token_ids_by_model"][_MODEL]))
        mapping[payload["p9_basis_pool"][0]["variants"][0]] = 63
        head: torch.nn.Linear = torch.nn.Linear(4, 1000, bias=False)

        with self.assertRaisesRegex(ValueError, "token IDs disagree"):
            _resolve_controls(spec, _MODEL, _FakeTokenizer(mapping), head)


class ControlArtifactWriterTest(TestCase):
    """Tests atomic manifest-order streaming without hidden vectors."""

    def test_out_of_order_batches_land_in_exact_manifest_order(self) -> None:
        keys: list[tuple[int, int]] = [(4, 0), (4, 1), (9, 0)]
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "controls.npy")
            writer: ControlArtifactWriter = ControlArtifactWriter(path, keys, 2, 3)
            writer.write(
                4,
                [1, 0],
                torch.tensor(
                    [
                        [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
                        [[13.0, 14.0, 15.0], [23.0, 24.0, 25.0]],
                    ],
                    dtype=torch.float32,
                ),
            )
            writer.write(
                9,
                [0],
                torch.tensor(
                    [[[30.0, 31.0, 32.0]], [[33.0, 34.0, 35.0]]],
                    dtype=torch.float32,
                ),
            )
            writer.finalize()

            result: np.ndarray = np.load(path, mmap_mode="r")
            self.assertEqual(result.shape, (3, 2, 3))
            np.testing.assert_array_equal(result[0, 0], [20.0, 21.0, 22.0])
            np.testing.assert_array_equal(result[1, 0], [10.0, 11.0, 12.0])
            np.testing.assert_array_equal(result[2, 1], [33.0, 34.0, 35.0])

    def test_incomplete_or_duplicate_rows_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "controls.npy")
            writer: ControlArtifactWriter = ControlArtifactWriter(
                path, [(0, 0), (0, 1)], 1, 1
            )
            value: torch.Tensor = torch.ones((1, 1, 1), dtype=torch.float32)
            writer.write(0, [0], value)
            with self.assertRaisesRegex(ValueError, "twice"):
                writer.write(0, [0], value)
            with self.assertRaisesRegex(RuntimeError, "missing 1"):
                writer.finalize()
            writer.abort()
            self.assertFalse(os.path.exists(path))

    def test_duplicate_rows_within_one_batch_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer: ControlArtifactWriter = ControlArtifactWriter(
                os.path.join(directory, "controls.npy"), [(0, 0)], 1, 1
            )
            with self.assertRaisesRegex(ValueError, "duplicate manifest"):
                writer.write(
                    0,
                    [0, 0],
                    torch.ones((1, 2, 1), dtype=torch.float32),
                )
            writer.abort()


class ControlColumnLayoutTest(TestCase):
    """Tests the exact 1 + 3K binary control column contract."""

    def test_assembles_signed_control_families_in_contract_order(self) -> None:
        # Linear layout before assembly is target, two shared, two isotropic.
        linear: torch.Tensor = torch.tensor(
            [[[10.0, 11.0, 12.0, 13.0, 14.0]]], dtype=torch.float32
        )
        original_grouped: torch.Tensor = torch.tensor(
            [[[30.0, 40.0]]], dtype=torch.float32
        )
        perturbed_grouped: torch.Tensor = torch.tensor(
            [[[3.0, 4.0]]], dtype=torch.float32
        )

        result: torch.Tensor = _assemble_control_projections(
            linear, original_grouped, perturbed_grouped
        )

        self.assertEqual(result.shape, (1, 1, 7))
        self.assertEqual(
            result.tolist(), [[[10.0, 11.0, 12.0, 27.0, 36.0, 13.0, 14.0]]]
        )


class ControlProvenanceTest(TestCase):
    """Tests the complete control-only source dependency inventory."""

    def test_source_inventory_includes_new_and_frozen_capture_modules(self) -> None:
        self.assertEqual(
            set(_source_hashes()),
            {
                "benchmark_scripts/benchmark_config.py",
                "benchmark_scripts/provenance_sources.py",
                "benchmark_scripts/run_layer_controls.py",
                "benchmark_scripts/run_layerwise.py",
                "surrogate/eval_constants.py",
                "surrogate/layer_control_scoring.py",
                "surrogate/layerwise_scoring.py",
                "surrogate/model_types.py",
                "surrogate/text_augmentation.py",
                "surrogate/transformers_model.py",
                "surrogate/utils.py",
            },
        )
