# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
from collections.abc import Mapping
import json
import os
import tempfile
from types import MappingProxyType
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
from benchmark_scripts.consolidate_results import consolidate
from benchmark_scripts import provenance_sources, run_benchmark as runner
from benchmark_scripts.run_benchmark import (
    _attention_implementation_for_phase,
    _ensure_model_outputs_writable,
    _load_phase_model,
    _merge_segment_rows,
    _write_run_metadata,
    _write_per_model_outputs,
    EXECUTION_SOURCE_FILES,
)
from benchmark_scripts.run_q05_boolq_word_eager_diagnostic import _eager_backend


class TestConsolidate(TestCase):
    def test_execution_source_inventory_matches_provenance_contract(self) -> None:
        self.assertEqual(
            set(EXECUTION_SOURCE_FILES),
            set(provenance_sources.OPEN_EXECUTION_SOURCE_FILES),
        )

    def test_run_captures_one_source_snapshot_before_loading_data(self) -> None:
        snapshot = MappingProxyType({"source.py": "1" * 64})
        events: list[str] = []

        def capture() -> Mapping[str, str]:
            events.append("snapshot")
            return snapshot

        def stop_at_dataset(*_args: object, **_kwargs: object) -> object:
            events.append("dataset")
            raise RuntimeError("stop")

        with (
            patch.object(
                runner, "_snapshot_execution_source_hashes", side_effect=capture
            ) as source_snapshot,
            patch.object(runner, "load_benchmark_dataset", side_effect=stop_at_dataset),
        ):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                asyncio.run(runner.run_benchmark("boolq", "sentence"))
        source_snapshot.assert_called_once_with()
        self.assertEqual(events, ["snapshot", "dataset"])

    def test_run_metadata_uses_supplied_source_snapshot_without_rehashing(self) -> None:
        snapshot = MappingProxyType({"source.py": "1" * 64})
        dataset: pd.DataFrame = pd.DataFrame(
            [{"question": "q", "answer": True, "passage": "p"}]
        )
        with (
            tempfile.TemporaryDirectory() as output_dir,
            patch.object(runner, "_sha256_file") as source_hasher,
        ):
            _write_run_metadata(
                output_dir=output_dir,
                model_name="model",
                model_source="org/model",
                model_path=os.path.join(output_dir, "missing-model"),
                spec=runner.BENCHMARKS["boolq"],
                pregrouper="sentence",
                phases={"attention", "ablation"},
                batch_size=8,
                max_samples=None,
                max_forward_passes=None,
                seed=42,
                dataset_file=None,
                dataset=dataset,
                execution_source_sha256=snapshot,
            )
            source_hasher.assert_not_called()
            with open(os.path.join(output_dir, "model_run.json")) as source:
                metadata: dict[str, object] = json.load(source)
        self.assertEqual(metadata["source_sha256"], dict(snapshot))

    def test_run_passes_one_source_snapshot_to_every_sidecar(self) -> None:
        snapshot = MappingProxyType({"source.py": "1" * 64})
        dataset: pd.DataFrame = pd.DataFrame(
            [{"question": "q", "answer": True, "passage": "p"}]
        )
        with (
            tempfile.TemporaryDirectory() as results_dir,
            patch.object(
                runner, "_snapshot_execution_source_hashes", return_value=snapshot
            ) as source_snapshot,
            patch.object(runner, "load_benchmark_dataset", return_value=dataset),
            patch.object(
                runner,
                "MODEL_SETS",
                {"test": [("model-a", "org/a"), ("model-b", "org/b")]},
            ),
            patch.object(runner, "_write_segment_manifest"),
            patch.object(runner, "resolve_model_path", return_value="/missing-model"),
            patch.object(runner, "_write_per_model_outputs"),
            patch.object(runner, "_write_run_metadata") as metadata_writer,
        ):
            asyncio.run(
                runner.run_benchmark(
                    "boolq",
                    "sentence",
                    phases=set(),
                    model_set="test",
                    results_dir=results_dir,
                )
            )

        source_snapshot.assert_called_once_with()
        self.assertEqual(metadata_writer.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["execution_source_sha256"] is snapshot
                for call in metadata_writer.call_args_list
            )
        )

    def test_model_output_policy_rejects_implicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            stale_path: str = os.path.join(output_dir, "model_tokens.tsv.gz")
            with open(stale_path, "wb") as output:
                output.write(b"stale")

            with self.assertRaisesRegex(FileExistsError, "--overwrite-existing"):
                _ensure_model_outputs_writable(
                    output_dir,
                    "model",
                    overwrite_existing=False,
                )
            _ensure_model_outputs_writable(
                output_dir,
                "model",
                overwrite_existing=True,
            )

    def test_attention_only_overwrite_removes_stale_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            stale_compressed: str = os.path.join(output_dir, "model_tokens.tsv.gz")
            stale_legacy: str = os.path.join(output_dir, "model_tokens.tsv")
            for stale_path in (stale_compressed, stale_legacy):
                with open(stale_path, "wb") as output:
                    output.write(b"stale")

            _write_per_model_outputs(
                output_dir,
                "model",
                [{"prompt_idx": 0, "seg_idx": 0, "attention_mean": 1.0}],
                [],
                [],
                overwrite_existing=True,
            )

            self.assertFalse(os.path.exists(stale_compressed))
            self.assertFalse(os.path.exists(stale_legacy))
            self.assertTrue(
                os.path.exists(os.path.join(output_dir, "model_segment.tsv.gz"))
            )

    def test_scoring_phases_use_distinct_attention_backends(self) -> None:
        self.assertEqual(_attention_implementation_for_phase("attention"), "eager")
        self.assertEqual(_attention_implementation_for_phase("ablation"), "sdpa")
        with self.assertRaisesRegex(ValueError, "Unknown phase"):
            _attention_implementation_for_phase("unknown")

    def test_eager_diagnostic_forces_only_known_phases(self) -> None:
        self.assertEqual(_eager_backend("attention"), "eager")
        self.assertEqual(_eager_backend("ablation"), "eager")
        with self.assertRaisesRegex(ValueError, "Unknown phase"):
            _eager_backend("unknown")

    def test_gold_script_requests_both_phases_explicitly(self) -> None:
        script_path: str = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "benchmark_scripts",
            "run_all_benchmarks.sh",
        )
        with open(script_path, encoding="utf-8") as source:
            script: str = source.read()
        self.assertIn('--phases "attention,ablation"', script)

    def test_phase_model_loader_requests_backend_before_load(self) -> None:
        with patch("benchmark_scripts.run_benchmark.TransformersModel") as model_cls:
            loaded: MagicMock = MagicMock()
            model_cls.return_value.load.return_value = loaded

            self.assertIs(_load_phase_model("model", "/model", "attention"), loaded)
            model_cls.assert_called_once_with(
                model_name="model",
                model_path="/model",
                attn_implementation="eager",
            )

        with patch("benchmark_scripts.run_benchmark.TransformersModel") as model_cls:
            loaded = MagicMock()
            model_cls.return_value.load.return_value = loaded

            self.assertIs(_load_phase_model("model", "/model", "ablation"), loaded)
            model_cls.assert_called_once_with(
                model_name="model",
                model_path="/model",
                attn_implementation="sdpa",
            )

    def test_rejects_misaligned_attention_and_ablation_segments(self) -> None:
        attention_rows = [
            {"prompt_idx": 0, "seg_idx": segment_idx} for segment_idx in range(2)
        ]
        ablation_rows = [{"prompt_idx": 0, "seg_idx": 0}]

        with self.assertRaisesRegex(
            ValueError, "Attention and ablation segments are misaligned"
        ):
            _merge_segment_rows(attention_rows, ablation_rows)

    def test_subsampled_merge_restricts_attention_to_ablation_keys(self) -> None:
        attention_rows = [
            {"prompt_idx": 0, "seg_idx": segment_idx, "n_segments": 3}
            for segment_idx in range(3)
        ]
        ablation_rows = [
            {"prompt_idx": 0, "seg_idx": 1, "n_segments": 3, "w_norm": 2.0}
        ]

        merged = _merge_segment_rows(
            attention_rows,
            ablation_rows,
            restrict_to_ablation=True,
        )

        self.assertEqual(merged[["prompt_idx", "seg_idx"]].values.tolist(), [[0, 1]])

    def test_rejects_duplicate_segment_keys(self) -> None:
        rows = [
            {"prompt_idx": 0, "seg_idx": 0},
            {"prompt_idx": 0, "seg_idx": 0},
        ]

        with self.assertRaisesRegex(ValueError, "Duplicate"):
            _merge_segment_rows(rows, [])

    def test_rejects_mismatched_segment_metadata(self) -> None:
        attention_rows = [{"prompt_idx": 0, "seg_idx": 0, "n_segments": 1, "answer": 1}]
        ablation_rows = [{"prompt_idx": 0, "seg_idx": 0, "n_segments": 2, "answer": 1}]

        with self.assertRaisesRegex(ValueError, "metadata differ"):
            _merge_segment_rows(attention_rows, ablation_rows)

    def test_concat_two_models_segment_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            in_dir: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(in_dir)

            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "n_segments": 1,
                        "answer": 1,
                        "attention_mean": 0.1,
                        "w_norm": 2.0,
                    }
                ]
            ).to_csv(os.path.join(in_dir, "qa_segment.tsv"), sep="\t", index=False)
            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "n_segments": 1,
                        "answer": 1,
                        "attention_mean": 0.2,
                        "w_norm": 3.0,
                    }
                ]
            ).to_csv(os.path.join(in_dir, "qb_segment.tsv"), sep="\t", index=False)

            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "kind": "ablated",
                        "label": "true",
                        "token": "true",
                        "logprob": -0.2,
                    }
                ]
            ).to_csv(os.path.join(in_dir, "qa_tokens.tsv"), sep="\t", index=False)

            consolidate("boolq", "sentence", results_dir=results_dir)

            seg_df = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_segments.tsv"), sep="\t"
            )
            self.assertEqual(set(seg_df["model"].unique()), {"qa", "qb"})
            self.assertEqual(len(seg_df), 2)

            tok_df = pd.read_csv(
                os.path.join(results_dir, "boolq_sentence_tokens.tsv"), sep="\t"
            )
            self.assertEqual(set(tok_df["model"].unique()), {"qa"})
            self.assertEqual(len(tok_df), 1)

    def test_no_tokens_emits_only_segments_tsv(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            in_dir: str = os.path.join(results_dir, "lambada", "sentence")
            os.makedirs(in_dir)

            pd.DataFrame(
                [
                    {
                        "prompt_idx": 0,
                        "seg_idx": 0,
                        "n_segments": 1,
                        "orig_completion_logprob": -1.5,
                    }
                ]
            ).to_csv(os.path.join(in_dir, "qa_segment.tsv"), sep="\t", index=False)

            consolidate("lambada", "sentence", results_dir=results_dir)

            self.assertTrue(
                os.path.exists(
                    os.path.join(results_dir, "lambada_sentence_segments.tsv")
                )
            )
            self.assertFalse(
                os.path.exists(os.path.join(results_dir, "lambada_sentence_tokens.tsv"))
            )

    def test_rejects_compressed_and_uncompressed_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as results_dir:
            in_dir: str = os.path.join(results_dir, "boolq", "sentence")
            os.makedirs(in_dir)
            frame = pd.DataFrame([{"prompt_idx": 0, "seg_idx": 0, "n_segments": 1}])
            frame.to_csv(
                os.path.join(in_dir, "model_segment.tsv"), sep="\t", index=False
            )
            frame.to_csv(
                os.path.join(in_dir, "model_segment.tsv.gz"), sep="\t", index=False
            )

            with self.assertRaisesRegex(ValueError, "Duplicate outputs"):
                consolidate("boolq", "sentence", results_dir=results_dir)
