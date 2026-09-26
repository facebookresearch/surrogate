# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Tests for matched-relative-depth fidelity analysis."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest import TestCase

import numpy as np
import pandas as pd

from benchmark_scripts.derived_provenance import sha256_file
from benchmark_scripts.layerwise_fidelity import (
    LAYER_EXECUTION_SOURCE_FILES,
    _contrast_labels,
    _interpolate_signal,
    _load_model_signals,
    _nearest_signal,
    _native_signals,
    _validate_layer_sidecar,
    _validate_layer_frame,
    _validate_manifest_coverage,
    analyze_config,
    load_final_layer_readout,
)


def _layer_rows(model_sign: float = 1.0) -> pd.DataFrame:
    rows: list[dict[str, str | int | float | None]] = []
    layer_metadata: list[tuple[int, str, int | None]] = [
        (0, "embedding", None),
        (1, "block", 0),
        (2, "block", 1),
        (3, "block", 2),
    ]
    for prompt_idx in range(4):
        original_signal: float = model_sign * float(prompt_idx + 1)
        for layer_slot, layer_kind, block_idx in layer_metadata:
            if layer_kind == "embedding":
                multiplier: float = 100.0
            else:
                assert block_idx is not None
                multiplier = float(block_idx + 1)
            rows.append(
                {
                    "prompt_idx": prompt_idx,
                    "seg_idx": None,
                    "kind": "orig",
                    "answer": "entailment",
                    "layer_slot": layer_slot,
                    "layer_kind": layer_kind,
                    "block_idx": block_idx,
                    "label_score_entailment": original_signal * multiplier,
                    "label_score_neutral": 0.0,
                    "label_score_contradiction": -original_signal * multiplier,
                    "delta_norm_postnorm": 0.0,
                    "w_dot_delta_z_postnorm_entailment_vs_contradiction": 0.0,
                    "w_norm_entailment_vs_contradiction": 2.0,
                }
            )
            for seg_idx in range(2):
                attribution: float = model_sign * float(prompt_idx + seg_idx + 1)
                rows.append(
                    {
                        "prompt_idx": prompt_idx,
                        "seg_idx": seg_idx,
                        "kind": "ablated",
                        "answer": "entailment",
                        "layer_slot": layer_slot,
                        "layer_kind": layer_kind,
                        "block_idx": block_idx,
                        "label_score_entailment": (original_signal - attribution)
                        * multiplier,
                        "label_score_neutral": 0.0,
                        "label_score_contradiction": -(original_signal - attribution)
                        * multiplier,
                        "delta_norm_postnorm": abs(attribution * multiplier),
                        "w_dot_delta_z_postnorm_entailment_vs_contradiction": (
                            (-1.0 if seg_idx == 0 else 1.0)
                            * 2.0
                            * abs(attribution * multiplier)
                        ),
                        "w_norm_entailment_vs_contradiction": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def _manifest_rows() -> pd.DataFrame:
    """Return the complete two-segment manifest for the synthetic layer rows."""
    return pd.DataFrame(
        [
            {
                "prompt_idx": prompt_idx,
                "seg_idx": seg_idx,
                "message_role": "system" if seg_idx == 0 else "user",
            }
            for prompt_idx in range(4)
            for seg_idx in range(2)
        ]
    )


def _write_sidecar(
    config_dir: str,
    model: str,
    frame: pd.DataFrame,
    *,
    artifact_sha256: str | None = None,
) -> str:
    """Write a valid synthetic execution sidecar beside a layer table."""
    layer_path: str = os.path.join(config_dir, f"{model}_layers.tsv.gz")
    manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "artifact": {
            "filename": os.path.basename(layer_path),
            "rows": len(frame),
            "sha256": artifact_sha256 or sha256_file(layer_path),
        },
        "benchmark": "anli_r1",
        "pregrouper": "sentence",
        "segmentation_scope": "full_dialog_in_message_order",
        "layer_slots": {"count": 4, "convention": "synthetic test slots"},
        "model": model,
        "model_source": f"example/{model}",
        "model_identity_files_sha256": {
            "config.json": "a" * 64,
            "tokenizer_config.json": "b" * 64,
        },
        "dataset": {
            "snapshot_filename": "facebook_anli_test_r1.tsv",
            "snapshot_sha256": "c" * 64,
            "normalized_frame_sha256": "d" * 64,
            "prompts": 4,
        },
        "manifest_sha256": sha256_file(manifest_path),
        "parameters": {
            "attention_implementation": "sdpa",
            "rendered_chat_add_special_tokens": False,
            "batch_size": 32,
            "canary": False,
            "device_map": "auto",
            "max_samples": None,
            "seed": 42,
            "torch_dtype": "bfloat16",
        },
        "source_sha256": {
            path: sha256_file(os.path.join(repository_root, path))
            for path in LAYER_EXECUTION_SOURCE_FILES
        },
    }
    sidecar_path: str = os.path.join(config_dir, f"{model}_layers_run.json")
    with open(sidecar_path, "w", encoding="utf-8") as output:
        json.dump(metadata, output)
    return sidecar_path


class LayerwiseValidationTest(TestCase):
    """Validate contrast and layer-grid invariants."""

    def test_resolves_boolq_and_both_anli_contrasts(self) -> None:
        self.assertEqual(_contrast_labels("boolq", "canonical"), ("true", "false"))
        self.assertEqual(
            _contrast_labels("anli_r1", "entailment_contradiction"),
            ("entailment", "contradiction"),
        )
        self.assertEqual(
            _contrast_labels("anli_r3", "entailment_neutral"),
            ("entailment", "neutral"),
        )

    def test_rejects_duplicate_and_incomplete_observation_keys(self) -> None:
        frame: pd.DataFrame = _layer_rows()
        duplicate: pd.DataFrame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _validate_layer_frame(
                duplicate, "duplicate.tsv", "entailment", "contradiction"
            )

        incomplete: pd.DataFrame = frame.drop(index=frame.index[-1])
        with self.assertRaisesRegex(ValueError, "incomplete layer grid"):
            _validate_layer_frame(
                incomplete, "incomplete.tsv", "entailment", "contradiction"
            )

    def test_interpolation_uses_blocks_and_not_embedding(self) -> None:
        frame: pd.DataFrame = _validate_layer_frame(
            _layer_rows(), "layers.tsv", "entailment", "contradiction"
        )
        predictions, _attributes, num_blocks = _native_signals(
            frame, "entailment", "neutral"
        )

        interpolated: pd.Series = _interpolate_signal(
            predictions, ["prompt_idx"], num_blocks, 0.25
        )

        # Prompt zero has block signals 1, 2, 3 and an embedding signal 100.
        self.assertAlmostEqual(float(interpolated.loc[0]), 1.5)

    def test_nearest_alignment_uses_native_block_and_shallower_tie(self) -> None:
        frame: pd.DataFrame = _validate_layer_frame(
            _layer_rows(), "layers.tsv", "entailment", "contradiction"
        )
        predictions, _attributes, num_blocks = _native_signals(
            frame, "entailment", "neutral"
        )

        selected: pd.Series = _nearest_signal(
            predictions, ["prompt_idx"], num_blocks, 0.25
        )

        # Three native blocks lie at 0, 0.5, 1; the midpoint tie selects 0.
        self.assertAlmostEqual(float(selected.loc[0]), 1.0)

    def test_manifest_coverage_rejects_a_missing_segment_key(self) -> None:
        frame: pd.DataFrame = _validate_layer_frame(
            _layer_rows(), "layers.tsv", "entailment", "contradiction"
        )
        incomplete_manifest: pd.MultiIndex = pd.MultiIndex.from_tuples(
            [(0, 0), (0, 1), (4, 0)], names=["prompt_idx", "seg_idx"]
        )

        with self.assertRaisesRegex(ValueError, "segment keys disagree"):
            _validate_manifest_coverage(frame, incomplete_manifest, "layers.tsv")

    def test_missing_execution_sidecar_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "anli_r1", "sentence")
            os.makedirs(config_dir)
            _layer_rows().to_csv(
                os.path.join(config_dir, "source_layers.tsv.gz"),
                sep="\t",
                index=False,
            )
            _manifest_rows().to_csv(
                os.path.join(config_dir, "segments.tsv.gz"), sep="\t", index=False
            )

            with self.assertRaisesRegex(FileNotFoundError, "layers_run.json"):
                _load_model_signals(
                    results_dir,
                    "anli_r1",
                    "sentence",
                    "source",
                    "user",
                    "entailment_contradiction",
                )

    def test_tampered_sidecar_and_single_bos_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "anli_r1", "sentence")
            os.makedirs(config_dir)
            frame: pd.DataFrame = _layer_rows()
            layer_path: str = os.path.join(config_dir, "source_layers.tsv.gz")
            manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
            frame.to_csv(layer_path, sep="\t", index=False)
            _manifest_rows().to_csv(manifest_path, sep="\t", index=False)
            sidecar_path: str = _write_sidecar(config_dir, "source", frame)

            with open(sidecar_path, encoding="utf-8") as source:
                valid: dict[str, Any] = json.load(source)
            tampered_hash: dict[str, Any] = json.loads(json.dumps(valid))
            tampered_hash["artifact"]["sha256"] = "0" * 64
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(tampered_hash, output)
            with self.assertRaisesRegex(ValueError, "artifact identity"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )

            invalid_bos: dict[str, Any] = json.loads(json.dumps(valid))
            invalid_bos["parameters"]["rendered_chat_add_special_tokens"] = True
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(invalid_bos, output)
            with self.assertRaisesRegex(ValueError, "rendered-chat tokenization"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )

            invalid_batch: dict[str, Any] = json.loads(json.dumps(valid))
            invalid_batch["parameters"]["batch_size"] = 8
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(invalid_batch, output)
            with self.assertRaisesRegex(ValueError, "complete production"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )

            invalid_manifest: dict[str, Any] = json.loads(json.dumps(valid))
            invalid_manifest["manifest_sha256"] = "0" * 64
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(invalid_manifest, output)
            with self.assertRaisesRegex(ValueError, "manifest SHA"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )

            invalid_model: dict[str, Any] = json.loads(json.dumps(valid))
            invalid_model["model_identity_files_sha256"] = {}
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(invalid_model, output)
            with self.assertRaisesRegex(ValueError, "model identity provenance"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )

            invalid_source: dict[str, Any] = json.loads(json.dumps(valid))
            invalid_source["source_sha256"][LAYER_EXECUTION_SOURCE_FILES[0]] = "0" * 64
            with open(sidecar_path, "w", encoding="utf-8") as output:
                json.dump(invalid_source, output)
            with self.assertRaisesRegex(ValueError, "execution source SHA"):
                _validate_layer_sidecar(
                    layer_path,
                    manifest_path,
                    frame,
                    "anli_r1",
                    "sentence",
                    "source",
                )


class LayerwiseAnalysisTest(TestCase):
    """Exercise scoping, correlation signs, coverage, and depth endpoints."""

    def test_final_layer_readout_uses_requested_alignment_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "anli_r1", "sentence")
            os.makedirs(config_dir)
            frame: pd.DataFrame = _layer_rows()
            final_user: pd.Series = (
                (frame["kind"] == "ablated")
                & (frame["seg_idx"] == 1)
                & (frame["block_idx"] == 2)
            )
            frame.loc[
                final_user & (frame["prompt_idx"] == 3),
                "w_norm_entailment_vs_contradiction",
            ] = 0.0
            frame.to_csv(
                os.path.join(config_dir, "source_layers.tsv.gz"),
                sep="\t",
                index=False,
            )
            _manifest_rows().to_csv(
                os.path.join(config_dir, "segments.tsv.gz"), sep="\t", index=False
            )
            _write_sidecar(config_dir, "source", frame)

            result = load_final_layer_readout(
                results_dir,
                "anli_r1",
                "sentence",
                "source",
                "user",
                "entailment_contradiction",
            )

        self.assertEqual(
            result.alignment.index.get_level_values("seg_idx").tolist(), [1] * 4
        )
        self.assertEqual(result.alignment.iloc[:3].tolist(), [1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(result.alignment.iloc[3]))
        self.assertEqual(float(result.prediction.loc[0]), 6.0)
        self.assertEqual(float(result.attribution.loc[(0, 1)]), 12.0)

    def test_analysis_emits_pairwise_complete_cluster_bootstraps(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "anli_r1", "sentence")
            os.makedirs(config_dir)
            models: tuple[str, ...] = ("source", "target")
            _layer_rows(1.0).to_csv(
                os.path.join(config_dir, "source_layers.tsv.gz"),
                sep="\t",
                index=False,
            )
            target: pd.DataFrame = _layer_rows(-1.0)
            # One target value is unavailable: complete-case counts must be local
            # to this pair and depth rather than imputed.
            target.loc[
                (target["kind"] == "ablated")
                & (target["prompt_idx"] == 3)
                & (target["seg_idx"] == 1)
                & (target["block_idx"] == 2),
                "label_score_entailment",
            ] = np.nan
            target.to_csv(
                os.path.join(config_dir, "target_layers.tsv.gz"),
                sep="\t",
                index=False,
            )
            _manifest_rows().to_csv(
                os.path.join(config_dir, "segments.tsv.gz"), sep="\t", index=False
            )
            _write_sidecar(config_dir, "source", _layer_rows(1.0))
            _write_sidecar(config_dir, "target", target)

            rows, inputs = analyze_config(
                results_dir=results_dir,
                benchmark="anli_r1",
                pregrouper="sentence",
                scope="user",
                contrast="entailment_contradiction",
                models=models,
                depth_grid_size=3,
                bootstrap_resamples=20,
                confidence_level=0.95,
                seed=42,
            )

        self.assertEqual(len(rows), 18)
        self.assertEqual(
            set(inputs),
            {
                "anli_r1/sentence/segments",
                "anli_r1/sentence/source/layers",
                "anli_r1/sentence/source/layers_run",
                "anli_r1/sentence/target/layers",
                "anli_r1/sentence/target/layers_run",
            },
        )
        final_attr_pearson = next(
            row
            for row in rows
            if row["relative_depth"] == 1.0
            and row["metric"] == "F_attr"
            and row["statistic"] == "pearson_r"
        )
        self.assertAlmostEqual(float(final_attr_pearson["f_point"]), -1.0)
        self.assertEqual(final_attr_pearson["n_observations"], 3)
        self.assertEqual(final_attr_pearson["expected_observations"], 4)
        self.assertEqual(final_attr_pearson["n_prompts"], 3)
        self.assertEqual(
            final_attr_pearson["resolved_scope"],
            "user_segment_coordinates_from_full_dialog",
        )
        final_pred_r2 = next(
            row
            for row in rows
            if row["relative_depth"] == 1.0
            and row["metric"] == "F_pred"
            and row["statistic"] == "pearson_r2"
        )
        self.assertAlmostEqual(float(final_pred_r2["f_point"]), 1.0)
        self.assertEqual(final_pred_r2["resolved_scope"], "prompt_level_full_dialog")
