# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Focused tests for the BoolQ layer-control analyzer."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from benchmark_scripts import analyze_layer_controls as analyzer
from benchmark_scripts.derived_provenance import sha256_file
from benchmark_scripts.f_table import OPEN_MODELS


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _layer_metadata(model: str, slots: int = 3) -> dict[str, object]:
    identity: dict[str, str] = {
        "config.json": _digest(model + ":config"),
        "tokenizer_config.json": _digest(model + ":tokenizer"),
    }
    return {
        "model": model,
        "model_source": f"example/{model}",
        "model_revision": _digest(model + ":revision"),
        "model_artifact_manifest_sha256": _digest(model + ":artifact"),
        "model_artifact_sha256": identity,
        "model_identity_files_sha256": identity,
        "layer_slots": {"count": slots, "convention": "synthetic"},
        "labels": {
            "true": {
                "accepted_single_token_aliases": [
                    {"alias": "sp_true", "surface": " true", "token_id": 1}
                ]
            },
            "false": {
                "accepted_single_token_aliases": [
                    {"alias": "sp_false", "surface": " false", "token_id": 2}
                ]
            },
        },
    }


def _basis_pool(width: int, size: int) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for basis_idx in range(size):
        base: str = f"basis{width}_{basis_idx}"
        variants: list[str] = [f" {base}"] + [
            f" {base}_{variant_idx}" for variant_idx in range(1, width)
        ]
        ids_by_model: dict[str, list[int]] = {}
        for model_idx, model in enumerate(OPEN_MODELS):
            start: int = (
                10_000_000 * (model_idx + 1) + 1_000_000 * width + 100 * basis_idx
            )
            ids_by_model[model] = list(range(start, start + width))
        result.append(
            {
                "base_surface": base,
                "variants": variants,
                "token_ids_by_model": ids_by_model,
            }
        )
    return result


def _control_spec(model: str, num_draws: int = 5) -> dict[str, object]:
    p9: list[dict[str, object]] = _basis_pool(9, 278)
    p8: list[dict[str, object]] = _basis_pool(8, 399)
    selected: list[dict[str, object]] = [
        {
            "draw_idx": draw_idx,
            "positive_base_surface": p9[draw_idx]["base_surface"],
            "negative_base_surface": p8[draw_idx]["base_surface"],
        }
        for draw_idx in range(num_draws)
    ]
    shared: list[dict[str, object]] = []
    grouped: list[dict[str, object]] = []
    for draw_idx in range(num_draws):
        positive: dict[str, object] = p9[draw_idx]
        negative: dict[str, object] = p8[draw_idx]
        positive_by_model: object = positive["token_ids_by_model"]
        negative_by_model: object = negative["token_ids_by_model"]
        self_positive_ids: object = (
            positive_by_model[model] if isinstance(positive_by_model, dict) else None
        )
        self_negative_ids: object = (
            negative_by_model[model] if isinstance(negative_by_model, dict) else None
        )
        if not isinstance(self_positive_ids, list) or not isinstance(
            self_negative_ids, list
        ):
            raise AssertionError("test basis fixture has invalid token IDs")
        positive_ids: list[int] = [int(value) for value in self_positive_ids]
        negative_ids: list[int] = [int(value) for value in self_negative_ids]
        shared.append(
            {
                "draw_idx": draw_idx,
                "positive_surface": " " + str(positive["base_surface"]).lower(),
                "negative_surface": " " + str(negative["base_surface"]).lower(),
                "positive_token_id": positive_ids[0],
                "negative_token_id": negative_ids[0],
                "pre_normalization_norm": 1.0,
            }
        )
        grouped.append(
            {
                "draw_idx": draw_idx,
                "positive_surfaces": positive["variants"],
                "negative_surfaces": negative["variants"],
                "positive_token_ids": positive_ids,
                "negative_token_ids": negative_ids,
                "pre_normalization_norm": 2.0,
            }
        )
    return {
        "namespace": "synthetic-control-v1",
        "selection_algorithm": {"algorithm": "synthetic"},
        "input_spec_filename": "control_spec.json",
        "input_spec_sha256": "0" * 64,
        "num_draws": num_draws,
        "column_layout": {
            "legacy_true_false_linear": 0,
            "shared_token_pair_start": 1,
            "grouped_pseudo_label_start": 1 + num_draws,
            "isotropic_start": 1 + 2 * num_draws,
            "total": 1 + 3 * num_draws,
        },
        "projection_definition": analyzer.PROJECTION_DEFINITION,
        "grouped_projection_definition": (
            "signed_original_minus_ablated_grouped_logsumexp_contrast"
        ),
        "absolute_attribution": False,
        "row_order": analyzer.ROW_ORDER,
        "canonical_direction": {
            "positive_label": "true",
            "negative_label": "false",
            "positive_surface": " true",
            "negative_surface": " false",
            "positive_token_id": 1,
            "negative_token_id": 2,
            "pre_normalization_norm": 3.0,
        },
        "p9_basis_pool": p9,
        "p8_basis_pool": p8,
        "selected_paired_bases": selected,
        "shared_token_pairs": shared,
        "grouped_pseudo_label_pairs": grouped,
        "isotropic": [
            {
                "draw_idx": draw_idx,
                "seed": 1000 + draw_idx,
                "vector_sha256": _digest(f"{model}:vector:{draw_idx}"),
            }
            for draw_idx in range(num_draws)
        ],
    }


def _input_spec(resolved_spec: dict[str, object]) -> dict[str, object]:
    num_draws: int = int(cast(Any, resolved_spec["num_draws"]))
    models: dict[str, dict[str, object]] = {}
    for model in OPEN_MODELS:
        metadata: dict[str, object] = _layer_metadata(model)
        identity_hashes: dict[str, str] = cast(
            dict[str, str], metadata["model_identity_files_sha256"]
        )
        tokenizer_hash: str = identity_hashes["tokenizer_config.json"]
        models[model] = {
            "model_source": metadata["model_source"],
            "model_revision": metadata["model_revision"],
            "model_artifact_manifest_sha256": metadata[
                "model_artifact_manifest_sha256"
            ],
            "tokenizer_class": "SyntheticTokenizer",
            "tokenizer_files_sha256": {"tokenizer_config.json": tokenizer_hash},
            "tokenizer_manifest_sha256": _digest(
                json.dumps(
                    {"tokenizer_config.json": tokenizer_hash},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "vocabulary_size_including_added_tokens": 100_000_000,
        }
    return {
        "namespace": resolved_spec["namespace"],
        "selection_algorithm": resolved_spec["selection_algorithm"],
        "models": models,
        "p9_basis_pool": resolved_spec["p9_basis_pool"],
        "p8_basis_pool": resolved_spec["p8_basis_pool"],
        "selected_paired_bases": resolved_spec["selected_paired_bases"],
        "isotropic_seeds_by_model": {
            model: [1000 + draw_idx for draw_idx in range(num_draws)]
            for model in OPEN_MODELS
        },
    }


def _artifact(
    model: str, values: np.ndarray, spec: dict[str, object]
) -> analyzer.ControlArtifact:
    metadata: dict[str, object] = {"control_spec": spec}
    return analyzer.ControlArtifact(
        model=model,
        path=f"/{model}.npy",
        sidecar_path=f"/{model}.json",
        values=values,
        metadata=metadata,
        num_draws=int(cast(Any, spec["num_draws"])),
    )


class AtomicOutputTest(unittest.TestCase):
    def test_gzip_output_is_deterministic(self) -> None:
        frame: pd.DataFrame = pd.DataFrame(
            {"model": ["qwen", "llama"], "pearson_r2": [0.25, 0.5]}
        )
        with tempfile.TemporaryDirectory() as directory:
            first_path: str = os.path.join(directory, "first.tsv.gz")
            second_path: str = os.path.join(directory, "second.tsv.gz")
            analyzer._atomic_write_tsv(frame, first_path)
            analyzer._atomic_write_tsv(frame, second_path)
            with open(first_path, "rb") as first_file:
                first_bytes: bytes = first_file.read()
            with open(second_path, "rb") as second_file:
                second_bytes: bytes = second_file.read()

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_bytes[4:8], b"\x00\x00\x00\x00")
        self.assertEqual(first_bytes[3] & 0x08, 0)


class NumericalControlTest(unittest.TestCase):
    def test_interpolation_excludes_embedding_and_retains_sign(self) -> None:
        values: np.ndarray = np.array(
            [
                [[999.0], [-2.0], [2.0]],
                [[-999.0], [4.0], [8.0]],
            ],
            dtype=np.float32,
        )
        first_block: np.ndarray = analyzer._interpolate_at_depth(
            values, np.array([0, 1]), 0.0
        )
        midpoint: np.ndarray = analyzer._interpolate_at_depth(
            values, np.array([0, 1]), 0.5
        )
        embedding: np.ndarray = analyzer._interpolate_at_depth(
            values, np.array([0, 1]), 0.0, include_embedding=True
        )
        np.testing.assert_allclose(first_block[:, 0], [-2.0, 4.0])
        np.testing.assert_allclose(midpoint[:, 0], [0.0, 6.0])
        np.testing.assert_allclose(embedding[:, 0], [999.0, -999.0])

    def test_permutations_are_deterministic_and_recorded(self) -> None:
        first, first_records = analyzer._shuffle_permutations(7, 4, 91)
        second, second_records = analyzer._shuffle_permutations(7, 4, 91)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first_records, second_records)
        for assignment in range(4):
            np.testing.assert_array_equal(first[assignment, 0], np.arange(7))
            for model_index in range(1, len(OPEN_MODELS)):
                self.assertEqual(set(first[assignment, model_index]), set(range(7)))
                self.assertEqual(
                    first_records[assignment]["models"][model_index][
                        "permutation_sha256"
                    ],
                    analyzer._permutation_sha256(first[assignment, model_index]),
                )

    def test_matched_and_shuffled_statistics_match_direct_calculation(self) -> None:
        base: np.ndarray = np.array(
            [
                [1.0, 2.0, 4.0],
                [2.0, 1.0, 3.0],
                [4.0, 3.0, 2.0],
                [8.0, 5.0, 1.0],
            ]
        )
        signals: list[np.ndarray] = [base + index for index in range(3)]
        permutations: np.ndarray = np.array(
            [[[0, 1, 2], [1, 2, 0], [2, 0, 1]]], dtype=np.int32
        )
        matched, shuffled = analyzer._shared_and_shuffled_mean_pair_r2(
            signals, permutations
        )
        np.testing.assert_allclose(matched, np.ones(3))
        expected: list[float] = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            pair: list[float] = []
            for draw_idx in range(3):
                pair.append(
                    analyzer._pearson_r2(
                        signals[first][:, permutations[0, first, draw_idx]],
                        signals[second][:, permutations[0, second, draw_idx]],
                    )
                )
            expected.append(float(np.mean(pair)))
        self.assertAlmostEqual(shuffled[0], float(np.mean(expected)))

    def test_observation_permutation_breaks_row_pairing(self) -> None:
        signals: list[np.ndarray] = [
            np.array([1.0, 2.0, 4.0, 8.0]),
            np.array([2.0, 3.0, 7.0, 9.0]),
            np.array([-1.0, 5.0, 6.0, 10.0]),
        ]
        permutations: np.ndarray = np.array(
            [[3, 2, 1, 0], [1, 3, 0, 2]], dtype=np.int32
        )
        observed: np.ndarray = analyzer._observation_permutation_pair_r2(
            signals, permutations
        )
        expected: np.ndarray = np.stack(
            [
                [
                    analyzer._pearson_r2(first, second[permutation])
                    for permutation in permutations
                ]
                for first, second in itertools.combinations(signals, 2)
            ]
        )
        np.testing.assert_allclose(observed, expected)

    def test_prompt_bootstrap_is_deterministic_and_clustered(self) -> None:
        prompts: np.ndarray = np.array([0, 0, 1, 1, 2, 2])
        codes_a, weights_a = analyzer._bootstrap_prompt_weights(prompts, 20, 17)
        codes_b, weights_b = analyzer._bootstrap_prompt_weights(prompts, 20, 17)
        np.testing.assert_array_equal(codes_a, codes_b)
        np.testing.assert_array_equal(weights_a, weights_b)
        self.assertTrue(np.all(weights_a.sum(axis=1) == 3))
        signals: list[np.ndarray] = [
            np.arange(1.0, 7.0),
            np.arange(1.0, 7.0) * 2.0,
            np.arange(1.0, 7.0) * -3.0,
        ]
        statistics, bootstrap = analyzer._target_statistics(
            signals, codes_a, weights_a, 0.95
        )
        self.assertAlmostEqual(statistics["mean_pair_pearson_r2"], 1.0)
        np.testing.assert_allclose(bootstrap, np.ones(20))

    def test_gap_interval_preserves_paired_bootstrap_resamples(self) -> None:
        prediction: np.ndarray = np.array([0.8, 0.6, 0.7, 0.9])
        attribution: np.ndarray = np.array([0.5, 0.4, 0.6, 0.3])
        statistics: dict[str, float] = analyzer._paired_difference_statistics(
            0.75,
            0.45,
            prediction,
            attribution,
            0.5,
        )
        difference: np.ndarray = prediction - attribution
        self.assertAlmostEqual(statistics["mean_pair_pearson_r2"], 0.30)
        self.assertAlmostEqual(statistics["bootstrap_mean"], float(difference.mean()))
        self.assertAlmostEqual(
            statistics["bootstrap_lower"], float(np.quantile(difference, 0.25))
        )
        self.assertAlmostEqual(
            statistics["bootstrap_upper"], float(np.quantile(difference, 0.75))
        )


class SidecarValidationTest(unittest.TestCase):
    def _write_fixture(self, directory: str, model: str = OPEN_MODELS[0]) -> tuple[
        str,
        str,
        str,
        dict[str, object],
        str,
        dict[str, object],
        str,
        str,
    ]:
        manifest_path: str = os.path.join(directory, "segments.tsv")
        pd.DataFrame(
            {
                "prompt_idx": [0, 0, 1, 1],
                "seg_idx": [0, 1, 0, 1],
                "message_role": ["system", "user", "system", "user"],
            }
        ).to_csv(manifest_path, sep="\t", index=False)
        values: np.ndarray = np.arange(4 * 3 * 16, dtype="<f4").reshape(4, 3, 16)
        path: str = os.path.join(directory, f"{model}_layer_controls.npy")
        np.save(path, values, allow_pickle=False)
        layer_metadata: dict[str, object] = _layer_metadata(model)
        layer_path: str = os.path.join(directory, f"{model}_layers.tsv.gz")
        layer_sidecar_path: str = os.path.join(directory, f"{model}_layers_run.json")
        with open(layer_path, "wb") as output:
            output.write(b"synthetic layer artifact")
        with open(layer_sidecar_path, "w", encoding="utf-8") as output:
            json.dump(layer_metadata, output)
        resolved_spec: dict[str, object] = _control_spec(model)
        input_spec: dict[str, object] = _input_spec(resolved_spec)
        input_spec_path: str = os.path.join(directory, "control_spec.json")
        with open(input_spec_path, "w", encoding="utf-8") as output:
            json.dump(input_spec, output)
        resolved_spec["input_spec_sha256"] = sha256_file(input_spec_path)
        metadata: dict[str, object] = {
            "schema_version": analyzer.SCHEMA_VERSION,
            "artifact_type": analyzer.ARTIFACT_TYPE,
            "benchmark": "boolq",
            "pregrouper": "sentence",
            "model": model,
            "segmentation_scope": "full_dialog_in_message_order",
            "manifest_sha256": sha256_file(manifest_path),
            "model_source": layer_metadata["model_source"],
            "model_revision": layer_metadata["model_revision"],
            "model_artifact_manifest_sha256": layer_metadata[
                "model_artifact_manifest_sha256"
            ],
            "model_identity_files_sha256": layer_metadata[
                "model_identity_files_sha256"
            ],
            "layer_slots": layer_metadata["layer_slots"],
            "source_layer_artifact": {
                "filename": os.path.basename(layer_path),
                "sha256": sha256_file(layer_path),
            },
            "source_layer_run": {
                "filename": os.path.basename(layer_sidecar_path),
                "sha256": sha256_file(layer_sidecar_path),
            },
            "parameters": {
                "attention_implementation": "sdpa",
                "batch_size": 32,
                "canary": False,
                "device_map": "auto",
                "max_samples": None,
                "rendered_chat_add_special_tokens": False,
                "seed": 42,
                "torch_dtype": "bfloat16",
            },
            "source_hash_timing": "run_start",
            "source_sha256": {},
            "control_spec": resolved_spec,
            "artifact": {
                "filename": os.path.basename(path),
                "shape": [4, 3, 16],
                "dtype": "float32",
                "byte_size": os.path.getsize(path),
                "sha256": sha256_file(path),
            },
        }
        sidecar_path: str = os.path.join(directory, f"{model}_layer_controls_run.json")
        with open(sidecar_path, "w", encoding="utf-8") as output:
            json.dump(metadata, output)
        return (
            path,
            sidecar_path,
            manifest_path,
            layer_metadata,
            input_spec_path,
            input_spec,
            layer_path,
            layer_sidecar_path,
        )

    def test_strict_sidecar_accepts_complete_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                path,
                sidecar,
                manifest,
                layer_metadata,
                spec_path,
                input_spec,
                layer_path,
                layer_sidecar_path,
            ) = self._write_fixture(directory)
            with patch.object(analyzer, "CONTROL_EXECUTION_SOURCE_FILES", ()):
                artifact: analyzer.ControlArtifact = (
                    analyzer._validate_control_artifact(
                        path,
                        sidecar,
                        manifest,
                        4,
                        OPEN_MODELS[0],
                        layer_metadata,
                        layer_path,
                        layer_sidecar_path,
                        input_spec,
                        spec_path,
                        sha256_file(spec_path),
                    )
                )
            self.assertEqual(artifact.num_draws, 5)
            self.assertEqual(artifact.values.shape, (4, 3, 16))

    def test_tampered_contract_fields_are_rejected(self) -> None:
        mutations = {
            "row order": lambda value: value["control_spec"].__setitem__(
                "row_order", "sorted"
            ),
            "shape": lambda value: value["artifact"].__setitem__("shape", [4, 3, 15]),
            "artifact digest": lambda value: value["artifact"].__setitem__(
                "sha256", "f" * 64
            ),
            "input spec digest": lambda value: value["control_spec"].__setitem__(
                "input_spec_sha256", "f" * 64
            ),
            "resolved ID": lambda value: value["control_spec"][
                "grouped_pseudo_label_pairs"
            ][0]["positive_token_ids"].__setitem__(0, -1),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                (
                    path,
                    sidecar,
                    manifest,
                    layer_metadata,
                    spec_path,
                    input_spec,
                    layer_path,
                    layer_sidecar_path,
                ) = self._write_fixture(directory)
                with open(sidecar, encoding="utf-8") as source:
                    metadata: dict[str, object] = json.load(source)
                mutate(metadata)
                with open(sidecar, "w", encoding="utf-8") as output:
                    json.dump(metadata, output)
                with (
                    patch.object(analyzer, "CONTROL_EXECUTION_SOURCE_FILES", ()),
                    self.assertRaises(ValueError),
                ):
                    analyzer._validate_control_artifact(
                        path,
                        sidecar,
                        manifest,
                        4,
                        OPEN_MODELS[0],
                        layer_metadata,
                        layer_path,
                        layer_sidecar_path,
                        input_spec,
                        spec_path,
                        sha256_file(spec_path),
                    )

    def test_reused_basis_in_input_spec_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (
                path,
                sidecar,
                manifest,
                layer_metadata,
                spec_path,
                input_spec,
                layer_path,
                layer_sidecar_path,
            ) = self._write_fixture(directory)
            tampered: dict[str, object] = copy.deepcopy(input_spec)
            paired_bases: list[dict[str, object]] = cast(
                list[dict[str, object]], tampered["selected_paired_bases"]
            )
            paired_bases[1]["positive_base_surface"] = paired_bases[0][
                "positive_base_surface"
            ]
            with (
                patch.object(analyzer, "CONTROL_EXECUTION_SOURCE_FILES", ()),
                self.assertRaisesRegex(ValueError, "no-replacement"),
            ):
                analyzer._validate_control_artifact(
                    path,
                    sidecar,
                    manifest,
                    4,
                    OPEN_MODELS[0],
                    layer_metadata,
                    layer_path,
                    layer_sidecar_path,
                    tampered,
                    spec_path,
                    sha256_file(spec_path),
                )

    def test_cross_model_specs_require_shared_surfaces_and_independent_vectors(
        self,
    ) -> None:
        values: np.ndarray = np.ones((4, 3, 16), dtype=np.float32)
        artifacts: list[analyzer.ControlArtifact] = [
            _artifact(model, values, _control_spec(model)) for model in OPEN_MODELS
        ]
        analyzer._validate_cross_model_specs(artifacts)

        altered: list[analyzer.ControlArtifact] = list(artifacts)
        altered_metadata: dict[str, object] = copy.deepcopy(altered[1].metadata)
        altered_spec: dict[str, Any] = cast(
            dict[str, Any], altered_metadata["control_spec"]
        )
        altered_pairs: list[dict[str, Any]] = cast(
            list[dict[str, Any]], altered_spec["shared_token_pairs"]
        )
        altered_pairs[0]["positive_surface"] = " altered"
        altered[1] = analyzer.ControlArtifact(
            altered[1].model,
            altered[1].path,
            altered[1].sidecar_path,
            altered[1].values,
            altered_metadata,
            altered[1].num_draws,
        )
        with self.assertRaisesRegex(ValueError, "shared control surfaces"):
            analyzer._validate_cross_model_specs(altered)


class EndToEndAnalysisTest(unittest.TestCase):
    def test_compact_curve_draw_output_and_equal_weight_depth_mean(self) -> None:
        num_draws: int = 5
        slots: int = 3
        rows: int = 8
        manifest: pd.DataFrame = pd.DataFrame(
            {
                "prompt_idx": np.repeat(np.arange(4), 2),
                "seg_idx": np.tile([0, 1], 4),
                "message_role": ["user"] * rows,
            }
        )
        controls: dict[str, analyzer.ControlArtifact] = {}
        predictions: dict[str, np.ndarray] = {}
        grouped: dict[str, np.ndarray] = {}
        delta: dict[str, np.ndarray] = {}
        for model_idx, model in enumerate(OPEN_MODELS):
            row_signal: np.ndarray = np.arange(1.0, rows + 1.0)[:, None, None]
            layer_signal: np.ndarray = np.arange(1.0, slots + 1.0)[None, :, None]
            column_signal: np.ndarray = np.arange(1.0, 2.0 + 3 * num_draws)[
                None, None, :
            ]
            values: np.ndarray = (
                row_signal * (1.0 + 0.03 * model_idx)
                + layer_signal * column_signal
                + 0.01 * row_signal * column_signal * (model_idx + 1)
            ).astype(np.float32)
            controls[model] = _artifact(model, values, _control_spec(model, num_draws))
            base: np.ndarray = np.arange(rows * slots, dtype=float).reshape(rows, slots)
            prediction_base: np.ndarray = np.arange(
                (rows // 2) * slots, dtype=float
            ).reshape(rows // 2, slots)
            predictions[model] = (prediction_base - (2.0 + 0.25 * model_idx)) * (
                1.0 + model_idx * 0.08
            )
            grouped[model] = (base - (4.0 + 0.5 * model_idx)) * (1.0 + model_idx * 0.1)
            delta[model] = np.sqrt(base + 1.0 + model_idx)

        with tempfile.TemporaryDirectory() as directory:
            manifest_path: str = os.path.join(
                directory, "boolq", "sentence", "segments.tsv"
            )
            os.makedirs(os.path.dirname(manifest_path))
            manifest.to_csv(manifest_path, sep="\t", index=False)

            def load_targets(
                _results_dir: str,
                _benchmark: str,
                _pregrouper: str,
                model: str,
                _manifest: pd.DataFrame,
            ) -> tuple[
                np.ndarray,
                np.ndarray,
                np.ndarray,
                str,
                str,
                dict[str, object],
            ]:
                return (
                    predictions[model],
                    grouped[model],
                    delta[model],
                    f"/{model}_layers.tsv.gz",
                    f"/{model}_layers_run.json",
                    _layer_metadata(model, slots),
                )

            def load_control(
                _path: str,
                _sidecar: str,
                _manifest_path: str,
                _manifest_rows: int,
                model: str,
                _layer_metadata_value: dict[str, object],
                _layer_path: str,
                _layer_sidecar_path: str,
                _input_spec_value: dict[str, object],
                _input_spec_path: str,
                _input_spec_sha256: str,
            ) -> analyzer.ControlArtifact:
                return controls[model]

            with (
                patch.object(analyzer, "_load_existing_targets", load_targets),
                patch.object(analyzer, "_validate_control_artifact", load_control),
                patch.object(
                    analyzer,
                    "_load_control_spec",
                    return_value=(_input_spec(_control_spec(OPEN_MODELS[0])), "0" * 64),
                ),
            ):
                summary, draws, _inputs, reported_draws, records = analyzer.analyze(
                    directory,
                    "/controls",
                    "/controls/control_spec.json",
                    scope="user",
                    depth_grid_size=3,
                    bootstrap_resamples=20,
                    seed=9,
                )

        self.assertEqual(reported_draws, num_draws)
        self.assertEqual(len(records), num_draws)
        self.assertEqual(len(summary), 4)
        self.assertEqual(
            len(draws),
            4
            * len(analyzer.ANALYSIS_CONTROL_FAMILIES)
            * num_draws
            * (1 + len(list(itertools.combinations(OPEN_MODELS, 2)))),
        )
        self.assertEqual(
            {"mean_over_model_pairs", "model_pair"}, set(draws["aggregation"])
        )
        depth_rows: pd.DataFrame = summary[summary["summary_kind"] == "relative_depth"]
        depth_mean: pd.Series = summary[
            summary["summary_kind"] == "equal_weight_depth_mean"
        ].iloc[0]
        for target in analyzer.TARGET_CURVES:
            column: str = f"{target}_mean_pair_pearson_r2"
            self.assertAlmostEqual(
                float(depth_mean[column]), float(depth_rows[column].mean())
            )
        gap_column: str = f"{analyzer.GAP_CURVE}_mean_pair_pearson_r2"
        prediction_column: str = "grouped_logsumexp_prediction_mean_pair_pearson_r2"
        attribution_column: str = "grouped_logsumexp_attribution_mean_pair_pearson_r2"
        self.assertTrue(
            np.allclose(
                depth_rows[gap_column],
                depth_rows[prediction_column] - depth_rows[attribution_column],
            )
        )
        self.assertAlmostEqual(
            float(depth_mean[gap_column]), float(depth_rows[gap_column].mean())
        )
        endpoint: pd.Series = depth_rows[depth_rows["relative_depth"] == 1.0].iloc[0]
        prediction_endpoint: list[np.ndarray] = [
            predictions[model][:, -1] for model in OPEN_MODELS
        ]
        self.assertAlmostEqual(
            float(endpoint["grouped_logsumexp_prediction_mean_pair_pearson_r2"]),
            analyzer._mean_pair_r2(prediction_endpoint),
        )
        signed_grouped: list[np.ndarray] = [
            grouped[model][:, -1] for model in OPEN_MODELS
        ]
        self.assertAlmostEqual(
            float(endpoint["grouped_logsumexp_attribution_mean_pair_pearson_r2"]),
            analyzer._mean_pair_r2(signed_grouped),
        )
        self.assertEqual(
            depth_rows.iloc[-1]["interpretation_priority"], "primary_endpoint"
        )


if __name__ == "__main__":
    unittest.main()
