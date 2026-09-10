# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Build an audit receipt for a separate BoolQ-word attention replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import numpy as np
import pandas as pd

from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
)


QWEN_MODELS: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
)
ATTENTION_COLUMNS: tuple[str, ...] = (
    "attention_mean",
    "attention_max",
    "attention_rollout",
)
IDENTITY_COLUMNS: tuple[str, ...] = (
    "prompt_idx",
    "seg_idx",
    "message_idx",
    "message_role",
    "message_seg_idx",
    "segment_text",
    "n_segments",
)
EXPECTED_SOURCE_DIFFERENCES: tuple[str, ...] = ()
MAX_ABS_ERROR: float = 1e-7
MIN_PEARSON_R: float = 0.999999
EXPECTED_REPLAY_ROWS: int = 476_154
EXPECTED_REPLAY_PROMPTS: int = 3_270
EXPECTED_REPLAY_MANIFEST_SHA256: str = (
    "ffc97bf4a3316e834eb30f4023f87c63789a040ccb1ee04711288f3963eda882"
)
PURPOSE: str = (
    "Separate uncapped full-corpus execution of corrected Qwen word attention "
    "after identifying sentence-attention contamination in the archive; "
    "comparisons use the fixed release coordinates."
)
EXPECTED_DATASET: dict[str, Any] = {
    "hf_name": "boolq",
    "hf_path": "aps/super_glue",
    "hf_split": "validation",
    "normalized_frame_sha256": (
        "8ef3b29d17186097b1857dcb6552f404b24157b7edd670d0eaa0121245167a0b"
    ),
    "rows": 3_270,
    "snapshot_filename": "google_boolq_validation.tsv",
    "snapshot_sha256": (
        "80040aa10f18e5b01082386dae3bdde48931a0311e807f6cb10f7173995f346a"
    ),
}


def _sha256(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_run(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        value: Any = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid run metadata: {path}")
    return value


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _model_receipt(results_dir: str, replay_dir: str, model: str) -> dict[str, Any]:
    gold_directory: str = os.path.join(results_dir, "boolq", "word")
    replay_directory: str = os.path.join(replay_dir, "boolq", "word")
    gold_path: str = os.path.join(gold_directory, f"{model}_segment.tsv.gz")
    replay_path: str = os.path.join(replay_directory, f"{model}_segment.tsv.gz")
    gold_run_path: str = os.path.join(gold_directory, f"{model}_run.json")
    replay_run_path: str = os.path.join(replay_directory, f"{model}_run.json")
    replay_manifest_path: str = os.path.join(replay_directory, "segments.tsv.gz")
    gold: pd.DataFrame = pd.read_csv(
        gold_path,
        sep="\t",
        usecols=[*IDENTITY_COLUMNS, *ATTENTION_COLUMNS],
        keep_default_na=False,
        na_values=[""],
    )
    replay: pd.DataFrame = pd.read_csv(
        replay_path,
        sep="\t",
        usecols=[*IDENTITY_COLUMNS, *ATTENTION_COLUMNS],
        keep_default_na=False,
        na_values=[""],
    )
    replay_manifest: pd.DataFrame = pd.read_csv(
        replay_manifest_path,
        sep="\t",
        usecols=IDENTITY_COLUMNS,
        keep_default_na=False,
        na_values=[""],
    )
    keys: list[str] = ["prompt_idx", "seg_idx"]
    if (
        len(gold) != 10_000
        or gold.duplicated(keys).any()
        or replay.duplicated(keys).any()
        or replay_manifest.duplicated(keys).any()
        or len(replay) != EXPECTED_REPLAY_ROWS
        or len(replay_manifest) != EXPECTED_REPLAY_ROWS
        or replay["prompt_idx"].nunique() != EXPECTED_REPLAY_PROMPTS
    ):
        raise ValueError(f"Invalid replay coordinate inventory for {model}")
    replay_identity: pd.DataFrame = replay[list(IDENTITY_COLUMNS)].merge(
        replay_manifest,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_result", "_manifest"),
        indicator=True,
    )
    if not replay_identity["_merge"].eq("both").all():
        raise ValueError(f"Replay disagrees with its full manifest for {model}")
    for column in IDENTITY_COLUMNS[2:]:
        if not replay_identity[f"{column}_result"].equals(
            replay_identity[f"{column}_manifest"]
        ):
            raise ValueError(f"Replay manifest identity mismatch for {model}:{column}")
    merged: pd.DataFrame = gold.merge(
        replay,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("_gold", "_replay"),
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError(f"Replay is missing gold coordinates for {model}")
    for column in IDENTITY_COLUMNS[2:]:
        if not merged[f"{column}_gold"].equals(merged[f"{column}_replay"]):
            raise ValueError(f"Replay identity mismatch for {model}:{column}")

    metric_checks: dict[str, dict[str, float]] = {}
    for column in ATTENTION_COLUMNS:
        gold_values: np.ndarray = pd.to_numeric(
            merged[f"{column}_gold"], errors="coerce"
        ).to_numpy(float)
        replay_values: np.ndarray = pd.to_numeric(
            merged[f"{column}_replay"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(gold_values).all() or not np.isfinite(replay_values).all():
            raise ValueError(f"Replay contains non-finite values for {model}:{column}")
        maximum_error: float = float(np.max(np.abs(gold_values - replay_values)))
        pearson_r: float = _pearson(gold_values, replay_values)
        if (
            maximum_error > MAX_ABS_ERROR
            or not np.isfinite(pearson_r)
            or pearson_r < MIN_PEARSON_R
        ):
            raise ValueError(
                f"Attention replay mismatch for {model}:{column}: "
                f"max_abs_error={maximum_error}, pearson_r={pearson_r}"
            )
        metric_checks[column] = {
            "max_abs_error": maximum_error,
            "pearson_r": pearson_r,
        }

    gold_run: dict[str, Any] = _load_run(gold_run_path)
    replay_run: dict[str, Any] = _load_run(replay_run_path)
    replay_manifest_sha256: str = _sha256(replay_manifest_path)
    if (
        replay_run.get("model") != model
        or replay_run.get("benchmark") != "boolq"
        or replay_run.get("pregrouper") != "word"
        or replay_run.get("segmentation_scope") != "full_dialog_in_message_order"
        or replay_run.get("parameters", {}).get("phases") != ["attention"]
        or replay_run.get("parameters", {}).get("phase_attention_implementation")
        != {"attention": "eager"}
        or replay_run.get("parameters", {}).get("max_forward_passes") is not None
        or replay_run.get("parameters", {}).get("max_samples") is not None
        or replay_run.get("dataset") != EXPECTED_DATASET
        or gold_run.get("dataset") != EXPECTED_DATASET
        or replay_manifest_sha256 != EXPECTED_REPLAY_MANIFEST_SHA256
        or replay_run.get("model_identity_files_sha256")
        != gold_run.get("model_identity_files_sha256")
    ):
        raise ValueError(f"Attention replay provenance mismatch for {model}")
    gold_sources: Any = gold_run.get("source_sha256")
    replay_sources: Any = replay_run.get("source_sha256")
    if not isinstance(gold_sources, dict) or not isinstance(replay_sources, dict):
        raise ValueError(f"Attention replay source hashes are missing for {model}")
    source_differences: list[str] = sorted(
        path
        for path in set(gold_sources) | set(replay_sources)
        if gold_sources.get(path) != replay_sources.get(path)
    )
    if tuple(source_differences) != EXPECTED_SOURCE_DIFFERENCES:
        raise ValueError(
            f"Unexpected replay source differences for {model}: {source_differences}"
        )
    return {
        "gold_execution_source_sha256": gold_sources,
        "gold_segment_sha256": _sha256(gold_path),
        "matched_rows": len(merged),
        "metrics": metric_checks,
        "replay_execution_source_sha256": replay_sources,
        "replay_dataset": EXPECTED_DATASET,
        "replay_manifest_sha256": replay_manifest_sha256,
        "replay_prompts": int(replay["prompt_idx"].nunique()),
        "replay_rows": len(replay),
        "replay_segment_sha256": _sha256(replay_path),
        "source_differences": source_differences,
    }


def build_receipt(
    results_dir: str,
    replay_dir: str,
    observation_path: str,
    replay_completion_path: str,
) -> dict[str, Any]:
    """Validate four separate replay executions and return a portable receipt."""
    observation: dict[str, Any] = _load_run(observation_path)
    completion: dict[str, Any] = _load_run(replay_completion_path)
    source_files: Any = observation.get("source_files")
    if (
        not isinstance(source_files, dict)
        or {
            str(path): str(record.get("sha256"))
            for path, record in source_files.items()
            if isinstance(record, dict)
        }
        != GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
    ):
        raise ValueError("Replay source observation disagrees with the gold pins")
    if any(
        not isinstance(record, dict) or set(record) != {"mtime_ns", "sha256"}
        for record in source_files.values()
    ):
        raise ValueError("Replay source observation fields disagree")
    if (
        completion.get("schema_version") != 1
        or type(completion.get("started_epoch")) is not int
        or type(completion.get("completed_epoch")) is not int
        or completion["started_epoch"] >= completion["completed_epoch"]
    ):
        raise ValueError("Replay completion timing is invalid")
    replay_start_ns: int = int(completion["started_epoch"]) * 1_000_000_000
    if any(
        int(record["mtime_ns"]) >= replay_start_ns for record in source_files.values()
    ):
        raise ValueError("An observed replay source does not predate the replay")
    return {
        "benchmark": "boolq",
        "models": {
            model: _model_receipt(results_dir, replay_dir, model)
            for model in QWEN_MODELS
        },
        "pregrouper": "word",
        "purpose": PURPOSE,
        "replay_completed_epoch": completion["completed_epoch"],
        "replay_started_epoch": completion["started_epoch"],
        "schema_version": 1,
        "source_files": source_files,
    }


def validate_receipt(receipt: dict[str, Any], results_dir: str) -> None:
    """Verify a shipped replay receipt against the gold artifacts and source."""
    if set(receipt) != {
        "benchmark",
        "models",
        "pregrouper",
        "purpose",
        "replay_completed_epoch",
        "replay_started_epoch",
        "schema_version",
        "source_files",
    } or (
        receipt.get("schema_version") != 1
        or receipt.get("benchmark") != "boolq"
        or receipt.get("pregrouper") != "word"
        or receipt.get("purpose") != PURPOSE
    ):
        raise ValueError("Attention replay receipt metadata disagrees")
    started_epoch: Any = receipt.get("replay_started_epoch")
    completed_epoch: Any = receipt.get("replay_completed_epoch")
    source_files: Any = receipt.get("source_files")
    if (
        type(started_epoch) is not int
        or type(completed_epoch) is not int
        or started_epoch >= completed_epoch
        or not isinstance(source_files, dict)
        or {
            str(path): str(record.get("sha256"))
            for path, record in source_files.items()
            if isinstance(record, dict)
        }
        != GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
        or any(
            not isinstance(record, dict)
            or set(record) != {"mtime_ns", "sha256"}
            or int(record["mtime_ns"]) >= started_epoch * 1_000_000_000
            for record in source_files.values()
        )
    ):
        raise ValueError("Attention replay source/timing evidence disagrees")
    models: Any = receipt.get("models")
    if not isinstance(models, dict) or set(models) != set(QWEN_MODELS):
        raise ValueError("Attention replay model inventory disagrees")
    expected_replay_sources: dict[str, str] = GOLD_OPEN_EXECUTION_SOURCE_SHA256
    for model, record in models.items():
        if not isinstance(record, dict) or set(record) != {
            "gold_execution_source_sha256",
            "gold_segment_sha256",
            "matched_rows",
            "metrics",
            "replay_execution_source_sha256",
            "replay_dataset",
            "replay_manifest_sha256",
            "replay_prompts",
            "replay_rows",
            "replay_segment_sha256",
            "source_differences",
        }:
            raise ValueError(f"Attention replay record fields disagree for {model}")
        if (
            record["gold_execution_source_sha256"] != GOLD_OPEN_EXECUTION_SOURCE_SHA256
            or record["replay_execution_source_sha256"] != expected_replay_sources
            or tuple(record["source_differences"]) != EXPECTED_SOURCE_DIFFERENCES
            or record["matched_rows"] != 10_000
            or record["replay_rows"] != EXPECTED_REPLAY_ROWS
            or record["replay_prompts"] != EXPECTED_REPLAY_PROMPTS
            or record["replay_dataset"] != EXPECTED_DATASET
            or record["replay_manifest_sha256"] != EXPECTED_REPLAY_MANIFEST_SHA256
            or any(
                not _is_sha256(record[field])
                for field in (
                    "gold_segment_sha256",
                    "replay_manifest_sha256",
                    "replay_segment_sha256",
                )
            )
        ):
            raise ValueError(f"Attention replay provenance disagrees for {model}")
        gold_path: str = os.path.join(
            results_dir,
            "boolq",
            "word",
            f"{model}_segment.tsv.gz",
        )
        if (
            not os.path.isfile(gold_path)
            or _sha256(gold_path) != record["gold_segment_sha256"]
        ):
            raise ValueError(f"Attention replay gold artifact disagrees for {model}")
        metrics: Any = record.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(ATTENTION_COLUMNS):
            raise ValueError(f"Attention replay metric inventory disagrees for {model}")
        for column, values in metrics.items():
            if (
                not isinstance(values, dict)
                or set(values) != {"max_abs_error", "pearson_r"}
                or not np.isfinite(float(values["max_abs_error"]))
                or float(values["max_abs_error"]) > MAX_ABS_ERROR
                or not np.isfinite(float(values["pearson_r"]))
                or float(values["pearson_r"]) < MIN_PEARSON_R
            ):
                raise ValueError(
                    f"Attention replay numerical check disagrees for {model}:{column}"
                )


def main() -> None:
    """Build and write the replay receipt."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--replay-completion", required=True)
    parser.add_argument("--output", required=True)
    args: argparse.Namespace = parser.parse_args()
    receipt: dict[str, Any] = build_receipt(
        os.path.abspath(args.results_dir),
        os.path.abspath(args.replay_dir),
        os.path.abspath(args.observation),
        os.path.abspath(args.replay_completion),
    )
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
