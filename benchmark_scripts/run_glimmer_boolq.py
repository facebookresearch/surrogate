# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Run the public Muse Glimmer BoolQ robustness experiment.

This is intentionally separate from the paper's canonical five-open-model
cohort. It reuses the audited benchmark pipeline while recording the model's
direct-response routing protocol in each run receipt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any, cast

import pandas as pd
import transformers

from benchmark_scripts import run_benchmark as benchmark_runner
from benchmark_scripts.benchmark_config import LOCAL_MODEL_DIR, MODEL_SETS
from surrogate.muse_glimmer_model import (
    DIRECT_RESPONSE_PREFIX,
    TEXT_ONLY_DEVICE_MAP,
    MuseGlimmerTransformersModel,
)
from surrogate.text_augmentation import PregrouperID


MODEL_NAME: str = "muse-glimmer-30b"
MODEL_SOURCE: str = "meta-models/Muse-Glimmer-30B"
MODEL_REVISION: str = "a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
MODEL_SET: str = "Muse-Glimmer"
MODEL_CACHE_DIR: str = os.path.join(LOCAL_MODEL_DIR, "Muse-Glimmer-30B")
MODEL_FILE_SHA256: dict[str, str] = {
    "chat_template.jinja": "cfc67e5f349f37690dfd31ed1f18bc4442a9dd32fe39a648f993cb4eb3cae678",
    "config.json": "5a9df2d8a385b3d361ab6ae68d73586f4e775033933bd0cd863fb7f3820e6a14",
    "generation_config.json": "1fa51889b1f8d3659802dedaa27e005b81e5c58483f13ecf13f2d97306bc6e35",
    "model-00001-of-00002.safetensors": "8eef61530e1283642c77ce2e6721feb5c6f348fa055c00e90f2844a136372694",
    "model-00002-of-00002.safetensors": "b58cc2144ba1ba1af4420f67f4ca3ced7f09298510b80464cc75018a0be14381",
    "model.safetensors.index.json": "7d817b4dccb1b123fc6c1939356c65cee3a0ad462a5b821ac88280990a27d1ba",
    "processor_config.json": "97e2a486dd9866b81f40cf4b8bc0c9ced9a7cd8a5bc65aa4cc2f4de0712dae77",
    "tokenizer.json": "c9dbee66967b58f31a7c27f723c3760da3526ccd0427578e8905b0abb0031c4d",
    "tokenizer_config.json": "781e6c74f571642c71202167b67d9255b28cc439bdda1582ff31346182f5a9c5",
}
EXTRA_EXECUTION_SOURCES: tuple[str, ...] = (
    "benchmark_scripts/run_glimmer_boolq.py",
    "surrogate/muse_glimmer_model.py",
)


def _validate_cached_revision(model_path: str) -> None:
    """Require Hugging Face's local tree record for the pinned revision."""
    tree_path: str = os.path.join(
        model_path,
        ".cache",
        "huggingface",
        "trees",
        f"{MODEL_REVISION}.json",
    )
    if not os.path.isfile(tree_path):
        raise RuntimeError(
            f"Checkpoint cache lacks the pinned revision record {tree_path}"
        )
    with open(tree_path, encoding="utf-8") as source:
        tree: Any = json.load(source)
    files: Any = tree.get("files") if isinstance(tree, dict) else None
    if not isinstance(files, dict) or not set(MODEL_FILE_SHA256).issubset(files):
        raise RuntimeError(f"Invalid Hugging Face revision record {tree_path}")
    missing: list[str] = []
    wrong_digest: list[str] = []
    for relative_path, expected_sha256 in MODEL_FILE_SHA256.items():
        path: str = os.path.join(model_path, relative_path)
        if not os.path.isfile(path):
            missing.append(relative_path)
            continue
        if _sha256_file(path) != expected_sha256:
            wrong_digest.append(relative_path)
    if missing or wrong_digest:
        raise RuntimeError(
            "Pinned Muse Glimmer cache is incomplete or inconsistent: "
            f"missing={missing}, wrong_digest={wrong_digest}"
        )


def _resolve_pinned_model_path(hf_hub_id: str) -> str:
    """Download and validate the exact public Muse Glimmer revision."""
    if hf_hub_id != MODEL_SOURCE:
        raise ValueError(f"Unexpected model source {hf_hub_id!r}")
    tree_path: str = os.path.join(
        MODEL_CACHE_DIR,
        ".cache",
        "huggingface",
        "trees",
        f"{MODEL_REVISION}.json",
    )
    if not os.path.isfile(tree_path):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=MODEL_SOURCE,
            revision=MODEL_REVISION,
            local_dir=MODEL_CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        )
    _validate_cached_revision(MODEL_CACHE_DIR)
    return MODEL_CACHE_DIR


def _install_adapter() -> None:
    """Register the isolated model set and wrapper for this process."""
    MODEL_SETS[MODEL_SET] = [(MODEL_NAME, MODEL_SOURCE)]
    benchmark_runner.TransformersModel = MuseGlimmerTransformersModel
    benchmark_runner.resolve_model_path = _resolve_pinned_model_path
    benchmark_runner.EXECUTION_SOURCE_FILES = tuple(
        dict.fromkeys(
            (*benchmark_runner.EXECUTION_SOURCE_FILES, *EXTRA_EXECUTION_SOURCES)
        )
    )


def _sha256_file(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(results_dir: str, reference_path: str) -> str:
    """Require exact segment identity with the committed BoolQ universe."""
    generated_path: str = os.path.join(
        results_dir, "boolq", "sentence", "segments.tsv.gz"
    )
    generated: pd.DataFrame = pd.read_csv(
        generated_path, sep="\t", keep_default_na=False
    )
    reference: pd.DataFrame = pd.read_csv(
        reference_path, sep="\t", keep_default_na=False
    )
    reference = reference[
        reference["prompt_idx"].isin(generated["prompt_idx"].unique())
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(generated, reference, check_dtype=False)
    return _sha256_file(reference_path)


def _annotate_receipt(
    results_dir: str,
    phases: set[str],
    reference_manifest_path: str,
    reference_manifest_sha256: str,
) -> None:
    """Record the public routing and output-head semantics in the receipt."""
    path: str = os.path.join(results_dir, "boolq", "sentence", f"{MODEL_NAME}_run.json")
    with open(path, encoding="utf-8") as source:
        metadata: dict[str, Any] = json.load(source)
    metadata["model_revision"] = MODEL_REVISION
    metadata["model_revision_resolution"] = "embedded_sha256_manifest"
    metadata["model_artifact_files_sha256"] = MODEL_FILE_SHA256
    metadata["model_protocol"] = {
        "assistant_response_prefix": DIRECT_RESPONSE_PREFIX,
        "loader": "AutoModelForImageTextToText",
        "text_only_device_map": TEXT_ONLY_DEVICE_MAP,
        "native_generation_boundary_scores_routing_tokens": True,
        "output_head": {
            "kind": "scaled_tanh_softcap",
            "intermediate_representation_warning": (
                "Fixed unembedding projections are pre-softcap diagnostics, "
                "not exact output-logit differences."
            ),
        },
    }
    metadata["dataset"]["reference_segment_manifest"] = os.path.basename(
        reference_manifest_path
    )
    metadata["dataset"]["reference_segment_manifest_sha256"] = reference_manifest_sha256
    metadata["parameters"]["phases"] = sorted(phases)
    metadata["software"]["transformers"] = transformers.__version__
    with open(path, "w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    """Parse CLI arguments and run the frozen-snapshot robustness study."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument(
        "--reference-segment-manifest",
        default="results/boolq/sentence/segments.tsv.gz",
    )
    parser.add_argument(
        "--phases", default="attention,ablation", help="Comma-separated phases"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-existing", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    phases: set[str] = {value.strip() for value in args.phases.split(",")}
    _install_adapter()
    asyncio.run(
        benchmark_runner.run_benchmark(
            "boolq",
            pregrouper=cast(PregrouperID, "sentence"),
            phases=phases,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            seed=args.seed,
            model_set=MODEL_SET,
            results_dir=args.results_dir,
            dataset_file=args.dataset_file,
            overwrite_existing=args.overwrite_existing,
        )
    )
    manifest_sha256: str = _validate_manifest(
        args.results_dir, args.reference_segment_manifest
    )
    _annotate_receipt(
        args.results_dir,
        phases,
        args.reference_segment_manifest,
        manifest_sha256,
    )


if __name__ == "__main__":
    main()
