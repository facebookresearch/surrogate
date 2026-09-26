# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import os
import tempfile
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import pandas as pd

from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
    GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS,
    OPEN_EXECUTION_SOURCE_FILES,
    build_release_source_corrections,
    canonical_file_hash_manifest_sha256,
)
from benchmark_scripts.seal_open_provenance import (
    _model_artifact_hashes,
    _release_source_corrections,
    _sha256,
    _verify_rendered_chat_tokenization,
    seal,
)


class TestSealOpenProvenance(TestCase):
    def test_verifies_model_specific_rendered_chat_tokenization(self) -> None:
        class FakeTokenizer:
            def __init__(self, bos_token_id: int | None, adds_bos: bool) -> None:
                self.bos_token_id: int | None = bos_token_id
                self._adds_bos: bool = adds_bos

            def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> str:
                return "rendered"

            def encode(self, _text: str, add_special_tokens: bool) -> list[int]:
                base: list[int] = (
                    [self.bos_token_id, 2] if self.bos_token_id is not None else [2]
                )
                if add_special_tokens and self._adds_bos:
                    if self.bos_token_id is None:
                        raise ValueError("test tokenizer cannot add a missing BOS")
                    return [self.bos_token_id, *base]
                return base

        qwen: FakeTokenizer = FakeTokenizer(None, False)
        llama: FakeTokenizer = FakeTokenizer(1, True)
        with patch(
            "benchmark_scripts.seal_open_provenance.AutoTokenizer.from_pretrained",
            return_value=qwen,
        ):
            self.assertEqual(
                {
                    "effective_bos_count": 0,
                    "no_duplicate_special_tokens": True,
                    "verification_method": (
                        "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false"
                    ),
                },
                _verify_rendered_chat_tokenization("qwen2.5-7b-instruct", "/m", True),
            )
        with patch(
            "benchmark_scripts.seal_open_provenance.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(1, True),
        ):
            with self.assertRaisesRegex(ValueError, "no-op special-token"):
                _verify_rendered_chat_tokenization("qwen2.5-7b-instruct", "/m", True)
        with patch(
            "benchmark_scripts.seal_open_provenance.AutoTokenizer.from_pretrained",
            return_value=llama,
        ):
            self.assertEqual(
                1,
                _verify_rendered_chat_tokenization(
                    "llama-3.1-8b-instruct", "/m", False
                )["effective_bos_count"],
            )
            with self.assertRaisesRegex(ValueError, "single-BOS"):
                _verify_rendered_chat_tokenization("llama-3.1-8b-instruct", "/m", True)

    def test_release_source_correction_preserves_execution_history(self) -> None:
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
                "benchmark_scripts.provenance_sources."
                "GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256",
                execution,
                clear=True,
            ),
            patch.dict(
                "benchmark_scripts.provenance_sources."
                "GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS",
                correction,
                clear=True,
            ),
            patch(
                "benchmark_scripts.provenance_sources.OPEN_EXECUTION_SOURCE_FILES",
                ("source.py",),
            ),
        ):
            self.assertEqual(
                correction,
                _release_source_corrections(execution, release),
            )
            self.assertEqual({}, _release_source_corrections(release, release))
            revised = _release_source_corrections(
                execution, {"source.py": "3" * 64}
            )
            self.assertEqual("1" * 64, revised["source.py"]["execution_sha256"])
            self.assertEqual("3" * 64, revised["source.py"]["release_sha256"])
            self.assertIn("review_packaging", revised["source.py"]["scope"])

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
                        "software": {
                            "numpy": "2.2.1",
                            "pandas": "2.2.3",
                            "torch": "2.15.0a0+fb",
                        },
                        "parameters": {},
                        "complete_source_sha256": {"stale": "metadata"},
                    },
                    output,
                )

            model_manifest_sha256: str = canonical_file_hash_manifest_sha256(
                _model_artifact_hashes(model_dir)
            )
            qwen_tokenization: dict[str, Any] = {
                "effective_bos_count": 0,
                "no_duplicate_special_tokens": True,
                "verification_method": (
                    "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false"
                ),
            }
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                patch(
                    "benchmark_scripts.seal_open_provenance."
                    "_verify_rendered_chat_tokenization",
                    return_value=qwen_tokenization,
                ),
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
                build_release_source_corrections(
                    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
                    metadata["release_source_sha256"],
                ),
                metadata["release_source_corrections"],
            )
            self.assertEqual(5, metadata["schema_version"])
            self.assertEqual(
                GOLD_OPEN_MODEL_REPOSITORIES[model],
                metadata["execution_model_source"],
            )
            self.assertEqual("run_completion", metadata["execution_source_hash_timing"])
            self.assertEqual("post_run_seal", metadata["model_artifact_hash_timing"])
            self.assertEqual({"numpy", "pandas", "torch"}, set(metadata["software"]))
            self.assertEqual(
                metadata["model_revision"],
                "7ae557604adf67be50417f59c2c2f167def9a775",
            )
            self.assertEqual(
                metadata["model_artifact_manifest_sha256"], model_manifest_sha256
            )
            self.assertIs(
                metadata["parameters"]["rendered_chat_add_special_tokens"], True
            )
            self.assertEqual(
                "post_run_reconstruction_not_execution_attested",
                metadata["execution_dependency_hash_timing"],
            )
            self.assertEqual(
                GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
                metadata["execution_dependency_sha256"],
            )
            self.assertEqual(
                qwen_tokenization,
                metadata["tokenization_verification"],
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

    def test_current_execution_preserves_explicit_special_token_parameter(self) -> None:
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
            repository_root: str = os.path.dirname(os.path.dirname(__file__))
            execution_hashes: dict[str, str] = {
                path: _sha256(os.path.join(repository_root, path))
                for path in OPEN_EXECUTION_SOURCE_FILES
            }
            metadata_path: str = os.path.join(config_dir, f"{model}_run.json")
            with open(metadata_path, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "model": model,
                        "model_source": GOLD_OPEN_MODEL_REPOSITORIES[model],
                        "source_sha256": execution_hashes,
                        "model_identity_files_sha256": {
                            "config.json": _model_artifact_hashes(model_dir)[
                                "config.json"
                            ]
                        },
                        "software": {
                            "numpy": "2.2.1",
                            "pandas": "2.2.3",
                            "torch": "2.15.0a0+fb",
                        },
                        "parameters": {"rendered_chat_add_special_tokens": False},
                    },
                    output,
                )
            model_manifest_sha256: str = canonical_file_hash_manifest_sha256(
                _model_artifact_hashes(model_dir)
            )
            qwen_tokenization: dict[str, Any] = {
                "effective_bos_count": 0,
                "no_duplicate_special_tokens": True,
                "verification_method": (
                    "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false"
                ),
            }
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                patch(
                    "benchmark_scripts.seal_open_provenance."
                    "_verify_rendered_chat_tokenization",
                    return_value=qwen_tokenization,
                ),
            ):
                self.assertEqual(seal(results_dir, model, model_dir), 1)
            with open(metadata_path, encoding="utf-8") as source:
                metadata = json.load(source)
            self.assertIs(
                metadata["parameters"]["rendered_chat_add_special_tokens"], False
            )
            self.assertEqual({}, metadata["release_source_corrections"])
            self.assertEqual(execution_hashes, metadata["execution_dependency_sha256"])
            self.assertEqual(
                "qwen_rendered_token_ids_equal_with_special_tokens_true_or_false",
                metadata["tokenization_verification"]["verification_method"],
            )

    def test_rejects_legacy_duplicated_bos_llama_execution(self) -> None:
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
            with (
                patch.dict(
                    "benchmark_scripts.seal_open_provenance."
                    "GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256",
                    {model: model_manifest_sha256},
                ),
                self.assertRaisesRegex(ValueError, "duplicated-BOS Llama"),
            ):
                seal(results_dir, model, model_dir)
