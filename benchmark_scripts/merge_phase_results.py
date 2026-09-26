# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Merge independently executed attention and ablation benchmark phases."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import shutil
from typing import Any

import pandas as pd

from benchmark_scripts.normalize_segment_outputs import normalize_file
from benchmark_scripts.run_benchmark import GZIP_COMPRESSION, _merge_segment_rows


def _sha256(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_dir(root: str, benchmark: str, pregrouper: str) -> str:
    return os.path.join(root, benchmark, pregrouper)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        value: Any = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _validate_receipts(
    attention: dict[str, Any],
    ablation: dict[str, Any],
) -> None:
    """Require two receipts to describe the same model and prompt universe."""
    common_fields: tuple[str, ...] = (
        "schema_version",
        "benchmark",
        "pregrouper",
        "segmentation_scope",
        "model",
        "model_source",
        "model_revision",
        "model_revision_resolution",
        "model_artifact_files_sha256",
        "model_identity_files_sha256",
        "model_protocol",
        "dataset",
        "software",
        "source_sha256",
    )
    disagreements: list[str] = [
        field for field in common_fields if attention.get(field) != ablation.get(field)
    ]
    if disagreements:
        raise ValueError(f"Phase receipts disagree on {disagreements}")
    attention_parameters: Any = attention.get("parameters")
    ablation_parameters: Any = ablation.get("parameters")
    if not isinstance(attention_parameters, dict) or not isinstance(
        ablation_parameters, dict
    ):
        raise ValueError("Phase receipts must contain parameter objects")
    if attention_parameters.get("phases") != ["attention"]:
        raise ValueError("Attention receipt does not describe only attention")
    if ablation_parameters.get("phases") != ["ablation"]:
        raise ValueError("Ablation receipt does not describe only ablation")
    if attention_parameters.get("phase_attention_implementation") != {
        "attention": "eager"
    }:
        raise ValueError("Attention receipt does not use the eager backend")
    if ablation_parameters.get("phase_attention_implementation") != {
        "ablation": "sdpa"
    }:
        raise ValueError("Ablation receipt does not use the SDPA backend")
    ignored: set[str] = {"phases", "phase_attention_implementation"}
    for key in set(attention_parameters) | set(ablation_parameters):
        if key not in ignored and attention_parameters.get(
            key
        ) != ablation_parameters.get(key):
            raise ValueError(f"Phase parameters disagree on {key!r}")


def merge(
    attention_results_dir: str,
    ablation_results_dir: str,
    output_results_dir: str,
    benchmark: str,
    pregrouper: str,
    model: str,
    overwrite_existing: bool = False,
) -> None:
    """Merge phase outputs and preserve both complete execution receipts.

    Args:
        attention_results_dir: Result root containing the attention-only run.
        ablation_results_dir: Result root containing the ablation-only run.
        output_results_dir: Result root for the combined public artifact.
        benchmark: Benchmark identifier.
        pregrouper: Segment granularity.
        model: Model artifact prefix.
        overwrite_existing: Whether to replace existing model outputs.
    """
    attention_dir: str = _config_dir(attention_results_dir, benchmark, pregrouper)
    ablation_dir: str = _config_dir(ablation_results_dir, benchmark, pregrouper)
    output_dir: str = _config_dir(output_results_dir, benchmark, pregrouper)
    if os.path.abspath(output_dir) in {
        os.path.abspath(attention_dir),
        os.path.abspath(ablation_dir),
    }:
        raise ValueError("Output directory must be distinct from both phase inputs")
    attention_segment_path: str = os.path.join(attention_dir, f"{model}_segment.tsv.gz")
    ablation_segment_path: str = os.path.join(ablation_dir, f"{model}_segment.tsv.gz")
    token_path: str = os.path.join(ablation_dir, f"{model}_tokens.tsv.gz")
    attention_receipt_path: str = os.path.join(attention_dir, f"{model}_run.json")
    ablation_receipt_path: str = os.path.join(ablation_dir, f"{model}_run.json")
    required: tuple[str, ...] = (
        attention_segment_path,
        ablation_segment_path,
        token_path,
        attention_receipt_path,
        ablation_receipt_path,
    )
    for path in required:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    attention_manifest_path: str = os.path.join(attention_dir, "segments.tsv.gz")
    ablation_manifest_path: str = os.path.join(ablation_dir, "segments.tsv.gz")
    attention_manifest: pd.DataFrame = pd.read_csv(
        attention_manifest_path, sep="\t", keep_default_na=False
    )
    ablation_manifest: pd.DataFrame = pd.read_csv(
        ablation_manifest_path, sep="\t", keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        attention_manifest, ablation_manifest, check_dtype=False
    )

    attention_receipt: dict[str, Any] = _read_json(attention_receipt_path)
    ablation_receipt: dict[str, Any] = _read_json(ablation_receipt_path)
    _validate_receipts(attention_receipt, ablation_receipt)

    attention_rows: list[dict[str, Any]] = pd.read_csv(
        attention_segment_path, sep="\t"
    ).to_dict("records")
    ablation_rows: list[dict[str, Any]] = pd.read_csv(
        ablation_segment_path, sep="\t"
    ).to_dict("records")
    merged: pd.DataFrame = _merge_segment_rows(attention_rows, ablation_rows)

    output_paths: tuple[str, ...] = (
        os.path.join(output_dir, f"{model}_segment.tsv.gz"),
        os.path.join(output_dir, f"{model}_tokens.tsv.gz"),
        os.path.join(output_dir, f"{model}_run.json"),
    )
    existing: list[str] = [path for path in output_paths if os.path.exists(path)]
    if existing and not overwrite_existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    os.makedirs(output_dir, exist_ok=True)
    output_manifest_path: str = os.path.join(output_dir, "segments.tsv.gz")
    if os.path.isfile(output_manifest_path):
        output_manifest: pd.DataFrame = pd.read_csv(
            output_manifest_path, sep="\t", keep_default_na=False
        )
        pd.testing.assert_frame_equal(
            attention_manifest, output_manifest, check_dtype=False
        )
    else:
        shutil.copyfile(attention_manifest_path, output_manifest_path)

    output_segment_path, output_token_path, output_receipt_path = output_paths
    merged.to_csv(
        output_segment_path,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION,
    )
    normalize_file(output_segment_path, attention_manifest)
    shutil.copyfile(token_path, output_token_path)

    combined: dict[str, Any] = deepcopy(attention_receipt)
    combined_parameters: dict[str, Any] = deepcopy(attention_receipt["parameters"])
    combined_parameters["phases"] = ["ablation", "attention"]
    combined_parameters["phase_attention_implementation"] = {
        **attention_receipt["parameters"]["phase_attention_implementation"],
        **ablation_receipt["parameters"]["phase_attention_implementation"],
    }
    combined["parameters"] = combined_parameters
    combined["assembly"] = {
        "kind": "parallel_independent_phases",
        "generator": "benchmark_scripts.merge_phase_results",
        "generator_source_sha256": _sha256(__file__),
        "supporting_source_sha256": {
            "benchmark_scripts/normalize_segment_outputs.py": _sha256(
                os.path.join(os.path.dirname(__file__), "normalize_segment_outputs.py")
            ),
        },
        "phase_inputs": {
            "attention": {
                "receipt": attention_receipt,
                "segment_sha256": _sha256(attention_segment_path),
            },
            "ablation": {
                "receipt": ablation_receipt,
                "segment_sha256": _sha256(ablation_segment_path),
                "tokens_sha256": _sha256(token_path),
            },
        },
    }
    combined["output_sha256"] = {
        "segment": _sha256(output_segment_path),
        "tokens": _sha256(output_token_path),
    }
    with open(output_receipt_path, "w", encoding="utf-8") as output:
        json.dump(combined, output, indent=2, sort_keys=True)
        output.write("\n")


def main() -> None:
    """Parse CLI arguments and merge a pair of phase runs."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--attention-results-dir", required=True)
    parser.add_argument("--ablation-results-dir", required=True)
    parser.add_argument("--output-results-dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pregrouper", required=True, choices=["word", "sentence"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--overwrite-existing", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    merge(
        args.attention_results_dir,
        args.ablation_results_dir,
        args.output_results_dir,
        args.benchmark,
        args.pregrouper,
        args.model,
        args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
