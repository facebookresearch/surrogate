# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from benchmark_scripts.derived_provenance import (
    DERIVED_SUPPORTING_SOURCE_FILES,
    PROVENANCE_SCHEMA_VERSION,
    sha256_file,
    write_derived_provenance,
)
from benchmark_scripts import f_table, race_multiclass, race_rv


class TestDerivedProvenance(TestCase):
    def test_binds_output_inputs_parameters_and_software(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            input_path: str = os.path.join(root_dir, "inputs", "source.tsv")
            output_path: str = os.path.join(root_dir, "derived.tsv")
            os.makedirs(os.path.dirname(input_path))
            with open(input_path, "w", encoding="utf-8") as output:
                output.write("value\n1\n")
            with open(output_path, "w", encoding="utf-8") as output:
                output.write("metric\tvalue\nx\t1.0\n")

            parameters = {
                "seed": 42,
                "bootstrap_resamples": 500,
                "confidence_level": 0.95,
                "cohort": "open",
                "scopes": ["all", "system", "user"],
                "contrasts": ["canonical"],
                "missingness_policy": "strict_complete",
            }
            sidecar_path: str = write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.example",
                generator_path=__file__,
                input_paths={"source": input_path},
                parameters=parameters,
                root_dir=root_dir,
            )
            with open(sidecar_path, encoding="utf-8") as source:
                payload = json.load(source)

            self.assertEqual(payload["schema_version"], PROVENANCE_SCHEMA_VERSION)
            self.assertEqual(payload["artifact_type"], "derived_table")
            self.assertEqual(
                payload["generator"]["module"], "benchmark_scripts.example"
            )
            self.assertEqual(payload["output"]["path"], "derived.tsv")
            self.assertEqual(payload["output"]["sha256"], sha256_file(output_path))
            self.assertEqual(payload["inputs"]["source"]["path"], "inputs/source.tsv")
            self.assertEqual(
                payload["inputs"]["source"]["sha256"], sha256_file(input_path)
            )
            self.assertEqual(payload["parameters"], parameters)
            self.assertEqual(
                set(payload["software"]),
                {
                    "python",
                    "python_implementation",
                    "numpy",
                    "pandas",
                    "scipy",
                    "tqdm",
                },
            )
            serialized: str = json.dumps(payload, sort_keys=True)
            self.assertNotIn(root_dir, serialized)

            with open(sidecar_path, encoding="utf-8") as source:
                first_bytes: str = source.read()
            write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.example",
                generator_path=__file__,
                input_paths={"source": input_path},
                parameters=parameters,
                root_dir=root_dir,
            )
            with open(sidecar_path, encoding="utf-8") as source:
                self.assertEqual(source.read(), first_bytes)

    def test_external_input_records_only_its_basename(self) -> None:
        with (
            tempfile.TemporaryDirectory() as root_dir,
            tempfile.TemporaryDirectory() as external,
        ):
            input_path: str = os.path.join(external, "external.tsv")
            output_path: str = os.path.join(root_dir, "derived.tsv")
            for path in (input_path, output_path):
                with open(path, "w", encoding="utf-8") as output:
                    output.write("data\n")
            sidecar_path: str = write_derived_provenance(
                output_path,
                generator_name="benchmark_scripts.example",
                generator_path=__file__,
                input_paths={"external": input_path},
                parameters={},
                root_dir=root_dir,
            )
            with open(sidecar_path, encoding="utf-8") as source:
                payload = json.load(source)
            self.assertEqual(payload["inputs"]["external"]["path"], "external.tsv")
            self.assertNotIn(external, json.dumps(payload))

    def test_rejects_missing_declared_input(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            output_path: str = os.path.join(root_dir, "derived.tsv")
            with open(output_path, "w", encoding="utf-8") as output:
                output.write("data\n")
            with self.assertRaises(FileNotFoundError):
                write_derived_provenance(
                    output_path,
                    generator_name="benchmark_scripts.example",
                    generator_path=__file__,
                    input_paths={"missing": os.path.join(root_dir, "missing.tsv")},
                    parameters={},
                    root_dir=root_dir,
                )

    def test_all_three_generators_write_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            config_dir: str = os.path.join(results_dir, "race", "sentence")
            os.makedirs(config_dir)
            logodds_path: str = os.path.join(results_dir, "race_sentence_logodds.tsv")
            manifest_path: str = os.path.join(config_dir, "segments.tsv.gz")
            for path in (logodds_path, manifest_path):
                with open(path, "w", encoding="utf-8") as output:
                    output.write("fixture\n")
            for name in (
                "model-a_segment.tsv.gz",
                "model-a_tokens.tsv.gz",
                "model-a_run.json",
            ):
                with open(
                    os.path.join(config_dir, name), "w", encoding="utf-8"
                ) as output:
                    output.write("fixture\n")

            race_frame: pd.DataFrame = pd.DataFrame(
                [{"model_s": "model-a", "model_t": "model-b", "f_point": 1.0}]
            )
            rv_output: str = os.path.join(results_dir, "race_rv.tsv")
            with (
                patch.object(race_rv, "compute_race_rv", return_value=race_frame),
                patch.object(
                    sys,
                    "argv",
                    [
                        "race_rv",
                        "--results-dir",
                        results_dir,
                        "--output",
                        rv_output,
                        "--cohort",
                        "open",
                    ],
                ),
            ):
                race_rv.main()
            self._assert_sidecar(rv_output, "benchmark_scripts.race_rv")

            multiclass_output: str = os.path.join(results_dir, "race_multiclass.tsv")
            with (
                patch.object(
                    race_multiclass,
                    "compute_race_multiclass",
                    return_value=race_frame,
                ),
                patch.object(
                    sys,
                    "argv",
                    [
                        "race_multiclass",
                        "--results-dir",
                        results_dir,
                        "--output",
                        multiclass_output,
                        "--cohort",
                        "open",
                    ],
                ),
            ):
                race_multiclass.main()
            self._assert_sidecar(multiclass_output, "benchmark_scripts.race_multiclass")

            scalar_input: str = os.path.join(results_dir, "boolq_sentence_segments.tsv")
            with open(scalar_input, "w", encoding="utf-8") as output:
                output.write("fixture\n")
            scalar_config_dir: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(scalar_config_dir)
            for name in (
                "segments.tsv.gz",
                "model-a_segment.tsv.gz",
                "model-a_run.json",
            ):
                with open(
                    os.path.join(scalar_config_dir, name), "w", encoding="utf-8"
                ) as output:
                    output.write("fixture\n")
            scalar_output: str = os.path.join(results_dir, "f_table_test.tsv")
            scalar_row: dict[str, str | int | float] = {
                "benchmark": "boolq",
                "pregrouper": "sentence",
                "scope": "all",
                "requested_scope": "all",
                "resolved_scope": "prompt_level_full_dialog",
                "contrast": "canonical",
                "requested_contrast": "true_minus_false",
                "resolved_source_contrast": "true_minus_false",
                "resolved_target_contrast": "true_minus_false",
                "readout_contrast": "not_applicable",
                "availability_status": "available",
                "unavailable_reason": "",
                "api_infinity_policy": "drop",
                "aggregation": "row_pooled",
                "model_s": "model-a",
                "model_t": "model-b",
                "metric": "F_pred",
                "statistic": "pearson_r2",
                "n_observations": 3,
                "n_prompts": 3,
                "f_point": 1.0,
                "f_lo": 1.0,
                "f_hi": 1.0,
            }
            with (
                patch.object(
                    f_table,
                    "DEFAULT_BENCHMARK_CONFIGS",
                    [("boolq", "sentence")],
                ),
                patch.object(f_table, "_process_benchmark", return_value=[scalar_row]),
                patch.object(
                    sys,
                    "argv",
                    [
                        "f_table",
                        "--results-dir",
                        results_dir,
                        "--output",
                        scalar_output,
                        "--cohort",
                        "open",
                        "--bootstrap-resamples",
                        "17",
                    ],
                ),
            ):
                f_table.main()
            self._assert_sidecar(scalar_output, "benchmark_scripts.f_table")
            with open(scalar_output + ".provenance.json", encoding="utf-8") as source:
                scalar_provenance = json.load(source)
            self.assertEqual(scalar_provenance["parameters"]["bootstrap_resamples"], 17)

    def _assert_sidecar(self, output_path: str, generator: str) -> None:
        """Assert that a generator sidecar binds its output bytes."""
        with open(output_path + ".provenance.json", encoding="utf-8") as source:
            payload = json.load(source)
        self.assertEqual(payload["generator"]["module"], generator)
        self.assertEqual(payload["output"]["sha256"], sha256_file(output_path))
        self.assertTrue(any("/raw/" in identifier for identifier in payload["inputs"]))
        self.assertEqual(
            set(payload["supporting_sources"]),
            set(DERIVED_SUPPORTING_SOURCE_FILES),
        )
