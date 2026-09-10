# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from unittest import TestCase
from unittest.mock import patch

from benchmark_scripts.benchmark_config import (
    BENCHMARKS,
    MODELS_LLAMA31_INSTRUCT,
    load_benchmark_dataset,
    resolve_model_path,
)


class TestBenchmarkConfig(TestCase):
    def test_llama_uses_canonical_hub_ids(self) -> None:
        self.assertEqual(
            [source for _, source in MODELS_LLAMA31_INSTRUCT],
            [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-70B-Instruct",
            ],
        )

    def test_canonical_llama_ids_reuse_legacy_cache_directories(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark_scripts.benchmark_config.LOCAL_MODEL_DIR", directory),
        ):
            for size in ("8B", "70B"):
                cache: str = os.path.join(directory, f"Meta-Llama-3.1-{size}-Instruct")
                os.makedirs(cache)
                with open(os.path.join(cache, "config.json"), "w") as output:
                    output.write("{}")
                self.assertEqual(
                    resolve_model_path(f"meta-llama/Llama-3.1-{size}-Instruct"),
                    cache,
                )

    def test_canonical_llama_hub_fallback_keeps_canonical_id(self) -> None:
        hub_id: str = "meta-llama/Llama-3.1-8B-Instruct"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("benchmark_scripts.benchmark_config.LOCAL_MODEL_DIR", directory),
            patch(
                "huggingface_hub.snapshot_download",
                side_effect=OSError("offline"),
            ) as download,
        ):
            self.assertEqual(resolve_model_path(hub_id), hub_id)
            download.assert_called_once_with(
                repo_id=hub_id,
                local_dir=os.path.join(directory, "Meta-Llama-3.1-8B-Instruct"),
                token=None,
            )

    def test_frozen_boolq_preserves_source_index_and_normalizes_answer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "boolq.tsv")
            with open(path, "w", encoding="utf-8") as output:
                output.write("\tquestion\tanswer\tpassage\n")
                output.write("17\tquestion\tTrue\tpassage\n")

            result = load_benchmark_dataset(BENCHMARKS["boolq"], path)

            self.assertEqual(result.index.tolist(), [17])
            self.assertEqual(bool(result.loc[17, "answer"]), True)

    def test_frozen_lambada_without_serialized_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path: str = os.path.join(directory, "lambada.tsv")
            with open(path, "w", encoding="utf-8") as output:
                output.write("context\ttarget\nhello\tworld\n")

            result = load_benchmark_dataset(BENCHMARKS["lambada"], path)

            self.assertEqual(result.index.tolist(), [0])
            self.assertEqual(result.loc[0, "context"], "hello")
