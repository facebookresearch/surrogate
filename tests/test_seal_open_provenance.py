# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
    canonical_file_hash_manifest_sha256,
)
from benchmark_scripts.seal_open_provenance import (
    _model_artifact_hashes,
    _release_source_corrections,
    seal,
)


class TestSealOpenProvenance(TestCase):
    def test_release_source_correction_is_exact_and_reruns_need_none(self) -> None:
        execution: dict[str, str] = {"source.py": "1" * 64}
        release: dict[str, str] = {"source.py": "2" * 64}
        correction: dict[str, dict[str, str]] = {
            "source.py": {
                "execution_sha256": "1" * 64,
                "release_sha256": "2" * 64,
                "reason": "Canonicalize public Llama repository locators only.",
                "scope": "model_locator_only_no_numerical_change",
            }
        }
        with (
            patch.dict(
                "benchmark_scripts.seal_open_provenance."
                "GOLD_OPEN_EXECUTION_SOURCE_SHA256",
                execution,
                clear=True,
            ),
            patch.dict(
                "benchmark_scripts.seal_open_provenance."
                "GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS",
                correction,
                clear=True,
            ),
            patch(
                "benchmark_scripts.seal_open_provenance." "OPEN_EXECUTION_SOURCE_FILES",
                ("source.py",),
            ),
        ):
            self.assertEqual(
                correction,
                _release_source_corrections(execution, release),
            )
            self.assertEqual({}, _release_source_corrections(release, release))
            with self.assertRaisesRegex(ValueError, "Unapproved"):
                _release_source_corrections(execution, {"source.py": "3" * 64})

    def test_download_script_pins_every_public_model_revision(self) -> None:
        repository_root: str = os.path.dirname(os.path.dirname(__file__))
        with open(
            os.path.join(repository_root, "download_all.sh"), encoding="utf-8"
        ) as source:
            script: str = source.read()
        for model, repository in GOLD_OPEN_MODEL_REPOSITORIES.items():
            local_name: str = (
                "Meta-Llama-3.1-8B-Instruct"
                if model == "llama-3.1-8b-instruct"
                else repository.rsplit("/", 1)[1]
            )
            self.assertIn(
                f"{repository} {local_name} {GOLD_OPEN_MODEL_REVISIONS[model]}",
                script,
            )
        self.assertIn('hf download "$repository" --revision "$revision"', script)

    def test_seals_model_and_output_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir: str = os.path.join(directory, "results")
            config_dir: str = os.path.join(results_dir, "boolq", "sentence")
            model_dir: str = os.path.join(directory, "model")
            os.makedirs(config_dir)
            os.makedirs(model_dir)
            model: str = "qwen2.5-0.5b-instruct"
            with open(os.path.join(model_dir, "model.safetensors"), "wb") as output:
                output.write(b"weights")
            with open(
                os.path.join(model_dir, "config.json"), "w", encoding="utf-8"
            ) as output:
                output.write("{}\n")
            for name in ("segments.tsv.gz", f"{model}_segment.tsv.gz"):
                pd.DataFrame([{"prompt_idx": 0, "seg_idx": 0}]).to_csv(
                    os.path.join(config_dir, name), sep="\t", index=False
                )
            metadata_path: str = os.path.join(config_dir, f"{model}_run.json")
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "model": model,
                        "model_source": GOLD_OPEN_MODEL_REPOSITORIES[model],
                        "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
                        "model_identity_files_sha256": {
                            "config.json": _model_artifact_hashes(model_dir)[
                                "config.json"
                            ]
                        },
                        "software": {},
                        "complete_source_sha256": {"stale": "metadata"},
                    },
                    output,
                )

            model_manifest_sha256: str = canonical_file_hash_manifest_sha256(
                _model_artifact_hashes(model_dir)
            )
            with patch.dict(
                "benchmark_scripts.seal_open_provenance."
                "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                {model: model_manifest_sha256},
            ):
                self.assertEqual(seal(results_dir, model, model_dir), 1)

            with open(metadata_path, encoding="utf-8") as source:
                metadata = json.load(source)
            self.assertIn("model.safetensors", metadata["model_artifact_sha256"])
            self.assertIn("segment", metadata["artifact_sha256"])
            self.assertIn("manifest_sha256", metadata)
            self.assertNotIn("complete_source_sha256", metadata)
            self.assertIn("release_source_sha256", metadata)
            self.assertEqual(
                GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
                metadata["release_source_corrections"],
            )
            self.assertEqual(3, metadata["schema_version"])
            self.assertEqual(
                GOLD_OPEN_MODEL_REPOSITORIES[model],
                metadata["execution_model_source"],
            )
            self.assertEqual("run_completion", metadata["execution_source_hash_timing"])
            self.assertEqual("post_run_seal", metadata["model_artifact_hash_timing"])
            self.assertIn("transformers", metadata["software"])
            self.assertEqual(
                metadata["model_revision"],
                "7ae557604adf67be50417f59c2c2f167def9a775",
            )
            self.assertEqual(
                metadata["model_artifact_manifest_sha256"], model_manifest_sha256
            )

    def test_rejects_execution_identity_mismatch_before_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir: str = os.path.join(directory, "results")
            config_dir: str = os.path.join(results_dir, "boolq", "sentence")
            model_dir: str = os.path.join(directory, "model")
            os.makedirs(config_dir)
            os.makedirs(model_dir)
            model: str = "qwen2.5-0.5b-instruct"
            for filename, value in (
                ("model.safetensors", b"weights"),
                ("config.json", b"{}\n"),
            ):
                with open(os.path.join(model_dir, filename), "wb") as output:
                    output.write(value)
            for name in ("segments.tsv.gz", f"{model}_segment.tsv.gz"):
                pd.DataFrame([{"prompt_idx": 0, "seg_idx": 0}]).to_csv(
                    os.path.join(config_dir, name), sep="\t", index=False
                )
            metadata_path: str = os.path.join(config_dir, f"{model}_run.json")
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "model": model,
                        "model_source": GOLD_OPEN_MODEL_REPOSITORIES[model],
                        "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
                        "model_identity_files_sha256": {"config.json": "0" * 64},
                        "software": {},
                    },
                    output,
                )
            model_manifest_sha256: str = canonical_file_hash_manifest_sha256(
                _model_artifact_hashes(model_dir)
            )
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                self.assertRaisesRegex(ValueError, "identities disagree"),
            ):
                seal(results_dir, model, model_dir)
            with open(metadata_path, encoding="utf-8") as source:
                metadata = json.load(source)
            self.assertNotIn("model_revision", metadata)

    def test_preserves_legacy_execution_locator_and_rejects_unknown_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            results_dir: str = os.path.join(directory, "results")
            config_dir: str = os.path.join(results_dir, "boolq", "sentence")
            model_dir: str = os.path.join(directory, "model")
            os.makedirs(config_dir)
            os.makedirs(model_dir)
            model: str = "llama-3.1-8b-instruct"
            for filename, value in (
                ("model.safetensors", b"weights"),
                ("config.json", b"{}\n"),
            ):
                with open(os.path.join(model_dir, filename), "wb") as output:
                    output.write(value)
            for name in ("segments.tsv.gz", f"{model}_segment.tsv.gz"):
                pd.DataFrame([{"prompt_idx": 0, "seg_idx": 0}]).to_csv(
                    os.path.join(config_dir, name), sep="\t", index=False
                )
            metadata_path: str = os.path.join(config_dir, f"{model}_run.json")
            identity: dict[str, str] = {
                "config.json": _model_artifact_hashes(model_dir)["config.json"]
            }
            metadata: dict[str, object] = {
                "model": model,
                "model_identity_files_sha256": identity,
                "model_source": GOLD_OPEN_EXECUTION_MODEL_SOURCES[model],
                "software": {},
                "source_sha256": GOLD_OPEN_EXECUTION_SOURCE_SHA256,
            }
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(metadata, output)
            model_manifest_sha256: str = canonical_file_hash_manifest_sha256(
                _model_artifact_hashes(model_dir)
            )
            with patch.dict(
                "benchmark_scripts.seal_open_provenance."
                "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                {model: model_manifest_sha256},
            ):
                self.assertEqual(1, seal(results_dir, model, model_dir))
            with open(metadata_path, encoding="utf-8") as source:
                sealed = json.load(source)
            self.assertEqual(
                GOLD_OPEN_EXECUTION_MODEL_SOURCES[model],
                sealed["execution_model_source"],
            )
            self.assertEqual(
                GOLD_OPEN_MODEL_REPOSITORIES[model], sealed["model_source"]
            )

            sealed["execution_model_source"] = GOLD_OPEN_MODEL_REPOSITORIES[model]
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(sealed, output)
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                self.assertRaisesRegex(ValueError, "Execution model source disagrees"),
            ):
                seal(results_dir, model, model_dir)

            sealed["execution_model_source"] = "unknown/repository"
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(sealed, output)
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                self.assertRaisesRegex(ValueError, "Execution model source disagrees"),
            ):
                seal(results_dir, model, model_dir)
