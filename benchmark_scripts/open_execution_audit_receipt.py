# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Build and verify the portable audit receipt for corrected open-model runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Any

import pandas as pd

from benchmark_scripts.f_table import OPEN_MODELS
from benchmark_scripts.provenance_sources import (
    GOLD_OPEN_EXECUTION_MODEL_SOURCES,
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    Q05_BOOLQ_WORD_EAGER_DIAGNOSTIC_WRAPPER_SHA256,
)


CONFIGURATIONS: tuple[tuple[str, str], ...] = (
    ("boolq", "sentence"),
    ("anli_r1", "sentence"),
    ("anli_r2", "sentence"),
    ("anli_r3", "sentence"),
    ("winogrande", "sentence"),
    ("race", "sentence"),
    ("boolq", "word"),
    ("lambada", "word"),
)
QUEUE_MODELS: dict[str, tuple[str, ...]] = {
    "gpu0": ("qwen2.5-14b-instruct", "qwen2.5-3b-instruct"),
    "gpu1": (
        "llama-3.1-8b-instruct",
        "qwen2.5-7b-instruct",
        "qwen2.5-0.5b-instruct",
    ),
}
NUMERICAL_HASH_FIELDS: tuple[str, ...] = (
    "manifest_sha256",
    "segment_sha256",
    "tokens_sha256",
)
EXPECTED_PHASE_BACKENDS: dict[str, str] = {
    "ablation": "sdpa",
    "attention": "eager",
}
Q05_BOOLQ_WORD_CELL: tuple[str, str, str] = (
    "qwen2.5-0.5b-instruct",
    "boolq",
    "word",
)
Q05_BOOLQ_WORD_CROSS_MODEL_ALIASES: dict[str, str] = {
    "l3.1-8b-instruct": "llama-3.1-8b-instruct",
    "q14b-instruct": "qwen2.5-14b-instruct",
    "q3b-instruct": "qwen2.5-3b-instruct",
    "q7b-instruct": "qwen2.5-7b-instruct",
}
Q05_BOOLQ_WORD_CROSS_MODELS: frozenset[str] = frozenset(
    Q05_BOOLQ_WORD_CROSS_MODEL_ALIASES.values()
)
Q05_BOOLQ_WORD_PAIR_MODELS: frozenset[str] = frozenset(
    {
        "llama-3.1-8b-instruct",
        "qwen2.5-14b-instruct",
        "qwen2.5-3b-instruct",
        "qwen2.5-7b-instruct",
    }
)
Q05_BOOLQ_WORD_SCOPE_ROWS: dict[str, int] = {
    "all": 10_000,
    "system": 2_859,
    "user": 7_141,
}
Q05_BOOLQ_WORD_PAIR_COMMON_ROWS: dict[str, dict[str, int]] = {
    "llama-3.1-8b-instruct": Q05_BOOLQ_WORD_SCOPE_ROWS,
    "qwen2.5-14b-instruct": {
        "all": 9_903,
        "system": 2_835,
        "user": 7_068,
    },
    "qwen2.5-3b-instruct": Q05_BOOLQ_WORD_SCOPE_ROWS,
    "qwen2.5-7b-instruct": Q05_BOOLQ_WORD_SCOPE_ROWS,
}
Q05_BOOLQ_WORD_STANDARD_CORRELATION_FIELDS: tuple[str, ...] = (
    "archive_original_pearson_r",
    "archive_delta_norm_postnorm_pearson_r",
    "archive_w_dot_delta_z_postnorm_pearson_r",
    "archive_attention_mean_sentence_contamination_pearson_r",
    "archive_attention_max_sentence_contamination_pearson_r",
    "archive_attention_rollout_sentence_contamination_pearson_r",
)
STANDARD_REPRESENTATION_CORRELATION_FIELDS: tuple[str, ...] = (
    "archive_delta_norm_postnorm_pearson_r",
    "archive_w_dot_delta_z_postnorm_pearson_r",
)
STANDARD_ATTENTION_CORRELATION_FIELDS: tuple[str, ...] = (
    "archive_attention_mean_pearson_r",
    "archive_attention_max_pearson_r",
    "archive_attention_rollout_pearson_r",
)
CONTAMINATED_ATTENTION_CORRELATION_FIELDS: tuple[str, ...] = (
    "archive_attention_mean_sentence_contamination_pearson_r",
    "archive_attention_max_sentence_contamination_pearson_r",
    "archive_attention_rollout_sentence_contamination_pearson_r",
)
ABLATION_SEGMENT_COLUMNS: tuple[str, ...] = (
    "prompt_idx",
    "answer",
    "seg_idx",
    "message_idx",
    "message_role",
    "message_seg_idx",
    "segment_text",
    "n_segments",
    "w_norm",
    "delta_norm_prenorm",
    "delta_norm_postnorm",
    "cossim_prenorm",
    "cossim_postnorm",
    "w_dot_delta_z_postnorm",
    "z_orig_norm_prenorm",
    "z_pert_norm_prenorm",
    "z_orig_norm_postnorm",
    "z_pert_norm_postnorm",
    "w_dot_z_orig_postnorm",
    "w_dot_z_pert_postnorm",
    "w_dot_delta_z_prenorm",
    "w_dot_z_orig_prenorm",
    "w_dot_z_pert_prenorm",
)
ATTENTION_SEGMENT_COLUMNS: tuple[str, ...] = (
    "attention_mean",
    "attention_max",
    "attention_rollout",
)
COMPLETION_SEGMENT_COLUMNS: tuple[str, ...] = (
    "orig_completion_logprob",
    "ablated_completion_logprob",
)
AVAILABILITY_COLUMNS: frozenset[str] = frozenset(
    {"original_result_available", "segment_result_available"}
)
EXECUTION_RUN_FIELDS: frozenset[str] = frozenset(
    {
        "benchmark",
        "dataset",
        "model",
        "model_identity_files_sha256",
        "model_source",
        "parameters",
        "pregrouper",
        "schema_version",
        "segmentation_scope",
        "software",
        "source_sha256",
    }
)
OBSERVATION_FIELDS: frozenset[str] = frozenset(
    {"models", "schema_version", "source_files"}
)
PRIVATE_QUEUE_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_sha256",
        "cells_per_model",
        "completed_epoch",
        "models",
        "phase_attention_implementation",
        "queue",
        "schema_version",
        "source_sha256",
        "started_epoch",
    }
)
PORTABLE_QUEUE_FIELDS: frozenset[str] = frozenset(
    {
        "artifact_sha256",
        "completed_epoch",
        "models",
        "phase_attention_implementation",
        "started_epoch",
    }
)
RECEIPT_FIELDS: frozenset[str] = frozenset(
    {
        "archive_comparison_role",
        "artifact_integrity_status",
        "execution_source_hash_timing",
        "model_artifact_hash_timing",
        "models",
        "numerical_cells",
        "queues",
        "schema_version",
        "segment_packaging",
        "source_files",
    }
)


def _expected_queue_artifacts(queue_name: str) -> set[str]:
    return {
        f"{benchmark}/{pregrouper}/{model}_{suffix}"
        for model in QUEUE_MODELS[queue_name]
        for benchmark, pregrouper in CONFIGURATIONS
        for suffix in (
            ("run.json", "segment.tsv.gz")
            if benchmark == "lambada"
            else ("run.json", "segment.tsv.gz", "tokens.tsv.gz")
        )
    }


def _expected_numerical_fields(
    row: dict[str, Any],
    *,
    portable: bool,
) -> set[str]:
    """Return the exact allowed schema for one numerical audit row."""
    benchmark: str = str(row.get("benchmark"))
    model: str = str(row.get("model"))
    pregrouper: str = str(row.get("pregrouper"))
    fields: set[str] = {
        "benchmark",
        "manifest_sha256",
        "model",
        "pregrouper",
        "prompts",
        "segments",
    }
    if portable:
        fields.update(
            {
                "execution_numerical_audit_status",
                "execution_run_metadata",
                "execution_run_sha256",
                "execution_segment_sha256",
                "historical_replication_status",
                "release_segment_projection_sha256",
                "release_segment_sha256",
                "structural_validation_status",
            }
        )
    else:
        fields.update({"run_sha256", "segment_sha256", "status"})
    if benchmark != "lambada":
        fields.add("tokens_sha256")
    if benchmark == "race":
        fields.update(
            {
                "archive_comparison",
                "archive_race_w_norm_max_abs_error",
                "archive_race_w_norm_prompts",
            }
        )
        return fields

    fields.update(
        {
            "archive_attribution_mae",
            "archive_attribution_pearson_r",
            "archive_comparison",
            "archive_original_mae",
            "archive_original_pearson_r",
            "archive_representation_comparison",
        }
    )
    is_qwen_word: bool = (
        benchmark == "boolq" and pregrouper == "word" and model.startswith("qwen")
    )
    if is_qwen_word:
        fields.update(
            {
                "archive_attention_comparison",
                "archive_attention_contaminated_coordinates",
                "archive_attention_max_direct_pearson_r",
                "archive_attention_max_sentence_contamination_pearson_r",
                "archive_attention_mean_direct_pearson_r",
                "archive_attention_mean_sentence_contamination_pearson_r",
                "archive_attention_rollout_direct_pearson_r",
                "archive_attention_rollout_sentence_contamination_pearson_r",
            }
        )
    else:
        fields.update(STANDARD_ATTENTION_CORRELATION_FIELDS)
    if not (
        model == "qwen2.5-14b-instruct"
        and benchmark == "boolq"
        and pregrouper == "word"
    ):
        fields.update(
            {
                "archive_delta_norm_postnorm_pearson_r",
                "archive_w_dot_delta_z_postnorm_pearson_r",
                "archive_w_norm_mae",
            }
        )
    if (model, benchmark, pregrouper) == Q05_BOOLQ_WORD_CELL:
        fields.update(
            {
                "archive_ablated_pearson_r",
                "archive_ablated_rmse",
                "archive_attribution_normalized_rmse",
                "archive_attribution_rmse",
                "archive_attribution_scope_diagnostics",
                "archive_attribution_std",
                "archive_attribution_validation",
                "archive_cross_model_attribution_max_abs_pearson_r",
                "archive_cross_model_attribution_pearson_r",
                "archive_eager_attribution_mae",
                "archive_eager_attribution_pearson_r",
                "eager_backend_diagnostic",
                "pairwise_f_attr_max_abs_delta",
                "pairwise_f_attr_sensitivity",
                "sdpa_repeatability",
            }
        )
    return fields


def _validate_numerical_row_schema(
    row: dict[str, Any],
    *,
    portable: bool,
) -> None:
    expected: set[str] = _expected_numerical_fields(row, portable=portable)
    if set(row) != expected:
        raise ValueError(
            "Open numerical audit row fields disagree: "
            f"missing={sorted(expected - set(row))}, "
            f"extra={sorted(set(row) - expected)}"
        )
    hash_fields: set[str] = {"manifest_sha256", "tokens_sha256"}
    hash_fields.update(
        {
            "execution_run_sha256",
            "execution_segment_sha256",
            "release_segment_projection_sha256",
            "release_segment_sha256",
        }
        if portable
        else {"run_sha256", "segment_sha256"}
    )
    if any(field in row and not _is_sha256(row[field]) for field in hash_fields):
        raise ValueError("Open numerical audit contains an invalid SHA-256")
    if (
        type(row.get("prompts")) is not int
        or int(row["prompts"]) <= 0
        or type(row.get("segments")) is not int
        or int(row["segments"]) <= 0
    ):
        raise ValueError("Open numerical audit counts disagree")


def _sha256(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as source:
        value: Any = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_numerical_rows(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            value: Any = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Invalid numerical-audit row in {path}")
            rows.append(value)
    return rows


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_object_sha256(value: dict[str, Any]) -> str:
    payload: bytes = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _segment_projection_sha256(path: str) -> str:
    frame: pd.DataFrame = pd.read_csv(path, sep="\t")
    if not set(ABLATION_SEGMENT_COLUMNS).issubset(frame.columns):
        raise ValueError(f"Ablation segment projection is incomplete: {path}")
    payload: bytes = (
        frame[list(ABLATION_SEGMENT_COLUMNS)]
        .to_csv(
            index=False,
            lineterminator="\n",
        )
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _release_equivalent_segment_projection_sha256(
    path: str,
    benchmark: str,
) -> str:
    """Hash every substantive segment column, excluding availability flags."""
    frame: pd.DataFrame = pd.read_csv(
        path, sep="\t", keep_default_na=False, na_values=[""]
    )
    columns: tuple[str, ...] = (
        *ABLATION_SEGMENT_COLUMNS[:8],
        *ATTENTION_SEGMENT_COLUMNS,
        *ABLATION_SEGMENT_COLUMNS[8:],
        *(COMPLETION_SEGMENT_COLUMNS if benchmark == "lambada" else ()),
    )
    expected: set[str] = set(columns)
    if set(frame.columns) - AVAILABILITY_COLUMNS != expected:
        raise ValueError(f"Release-equivalent segment columns disagree: {path}")
    payload: bytes = (
        frame[list(columns)]
        .to_csv(
            index=False,
            lineterminator="\n",
        )
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_release_segment(
    path: str,
    manifest_path: str,
    row: dict[str, Any],
) -> str:
    """Validate a normalized open segment file and return its data projection."""
    projection_sha256: str = _release_equivalent_segment_projection_sha256(
        path,
        str(row["benchmark"]),
    )
    frame: pd.DataFrame = pd.read_csv(
        path, sep="\t", keep_default_na=False, na_values=[""]
    )
    manifest: pd.DataFrame = pd.read_csv(
        manifest_path, sep="\t", keep_default_na=False, na_values=[""]
    )
    identity_columns: list[str] = [
        "prompt_idx",
        "seg_idx",
        "answer",
        "message_idx",
        "message_role",
        "message_seg_idx",
        "segment_text",
        "n_segments",
    ]
    if (
        len(frame) != row.get("segments")
        or frame["prompt_idx"].nunique() != row.get("prompts")
        or frame.duplicated(["prompt_idx", "seg_idx"]).any()
        or len(frame) != len(manifest)
    ):
        raise ValueError(f"Open release segment population disagrees: {path}")
    try:
        pd.testing.assert_frame_equal(
            frame[identity_columns].reset_index(drop=True),
            manifest[identity_columns].reset_index(drop=True),
            check_dtype=False,
            check_like=False,
        )
    except AssertionError as error:
        raise ValueError(f"Open release segment identity disagrees: {path}") from error
    if set(frame.columns) != (
        set(frame.columns) - AVAILABILITY_COLUMNS
    ) | AVAILABILITY_COLUMNS or not all(
        frame[column].eq(True).all() for column in AVAILABILITY_COLUMNS
    ):
        raise ValueError(f"Open release availability schema disagrees: {path}")
    metric_columns: set[str] = (
        set(frame.columns) - set(identity_columns) - AVAILABILITY_COLUMNS
    )
    if any(
        not pd.to_numeric(frame[column], errors="coerce").map(math.isfinite).all()
        for column in metric_columns
    ):
        raise ValueError(f"Open release segment metrics are incomplete: {path}")
    return projection_sha256


def _validate_recheck_metadata(
    run: Any,
    main_run: dict[str, Any],
    backend: str,
) -> None:
    if not isinstance(run, dict) or set(run) != EXECUTION_RUN_FIELDS:
        raise ValueError(f"{backend} recheck execution metadata fields disagree")
    expected_parameters: dict[str, Any] = {
        "batch_size": 8,
        "max_forward_passes": 10_000,
        "max_samples": None,
        "phase_attention_implementation": {"ablation": backend},
        "phases": ["ablation"],
        "seed": 42,
    }
    shared_fields: tuple[str, ...] = (
        "benchmark",
        "dataset",
        "model",
        "model_identity_files_sha256",
        "model_source",
        "pregrouper",
        "schema_version",
        "segmentation_scope",
        "software",
        "source_sha256",
    )
    if run.get("parameters") != expected_parameters or any(
        run.get(field) != main_run.get(field) for field in shared_fields
    ):
        raise ValueError(f"{backend} recheck execution metadata disagrees")


def _require_correlations(
    row: dict[str, Any],
    fields: tuple[str, ...],
    minimum: float,
) -> None:
    if any(
        not isinstance(row.get(field), (int, float))
        or not math.isfinite(float(row[field]))
        or not -1.0 <= float(row[field]) <= 1.0
        or (minimum > -1.0 and float(row[field]) < minimum)
        for field in fields
    ):
        raise ValueError(f"Open archive correlation diagnostics disagree: {row}")


def _validate_standard_archive_diagnostics(row: dict[str, Any]) -> None:
    """Validate historical comparison diagnostics without treating them as truth."""
    _require_correlations(row, ("archive_original_pearson_r",), -1.0)
    _require_correlations(row, ("archive_attribution_pearson_r",), -1.0)
    for field in ("archive_original_mae", "archive_attribution_mae"):
        value: Any = row.get(field)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"Open archive error diagnostic disagrees: {row}")

    is_qwen_word: bool = (
        row["benchmark"] == "boolq"
        and row["pregrouper"] == "word"
        and str(row["model"]).startswith("qwen")
    )
    if is_qwen_word:
        if (
            row.get("archive_attention_comparison")
            != "not_comparable_archive_sentence_contamination"
            or row.get("archive_attention_contaminated_coordinates") != 546
        ):
            raise ValueError(f"BoolQ-word attention diagnostic disagrees: {row}")
        _require_correlations(row, CONTAMINATED_ATTENTION_CORRELATION_FIELDS, -1.0)
        for aggregation in ("mean", "max", "rollout"):
            direct: Any = row.get(f"archive_attention_{aggregation}_direct_pearson_r")
            if (
                not isinstance(direct, (int, float))
                or not math.isfinite(float(direct))
                or not -1.0 <= float(direct) <= 1.0
            ):
                raise ValueError(
                    f"BoolQ-word direct attention diagnostic differs: {row}"
                )
    else:
        if row.get("archive_attention_comparison") is not None:
            raise ValueError(f"Unexpected archive attention status: {row}")
        _require_correlations(row, STANDARD_ATTENTION_CORRELATION_FIELDS, -1.0)

    unavailable_representation: bool = (
        row["model"] == "qwen2.5-14b-instruct"
        and row["benchmark"] == "boolq"
        and row["pregrouper"] == "word"
    )
    expected_representation: str = (
        "not_available" if unavailable_representation else "same_coordinates"
    )
    if row.get("archive_representation_comparison") != expected_representation:
        raise ValueError(f"Open archive representation status disagrees: {row}")
    if not unavailable_representation:
        if row.get("archive_w_norm_mae") != 0.0:
            raise ValueError(f"Open archive w_norm diagnostic disagrees: {row}")
        _require_correlations(row, STANDARD_REPRESENTATION_CORRELATION_FIELDS, -1.0)
    elif any(
        field in row
        for field in (
            "archive_w_norm_mae",
            *STANDARD_REPRESENTATION_CORRELATION_FIELDS,
        )
    ):
        raise ValueError(f"Unavailable representation diagnostics are populated: {row}")


def _validate_q05_boolq_word_diagnostics(
    row: dict[str, Any],
    results_dir: str,
    portable: bool,
) -> None:
    if row.get("archive_comparison") != (
        "same_coordinates_component_agreement_attribution_below_standard_threshold"
    ) or row.get("archive_attribution_validation") != (
        "new_artifact_not_archive_equivalence"
    ):
        raise ValueError("Qwen-0.5B BoolQ-word archive status disagrees")
    if (
        any(
            not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            or float(row[field]) < 0.95
            for field in Q05_BOOLQ_WORD_STANDARD_CORRELATION_FIELDS
        )
        or row.get("archive_w_norm_mae") != 0.0
        or row.get("archive_attention_comparison")
        != "not_comparable_archive_sentence_contamination"
        or row.get("archive_attention_contaminated_coordinates") != 546
    ):
        raise ValueError("Qwen-0.5B BoolQ-word standard diagnostics disagree")

    cross_model: Any = row.get("archive_cross_model_attribution_pearson_r")
    scope_diagnostics: Any = row.get("archive_attribution_scope_diagnostics")
    pairwise: Any = row.get("pairwise_f_attr_sensitivity")
    maximum_pairwise: Any = row.get("pairwise_f_attr_max_abs_delta")
    scalar_fields: tuple[str, ...] = (
        "archive_original_pearson_r",
        "archive_ablated_pearson_r",
        "archive_attribution_pearson_r",
        "archive_attribution_mae",
        "archive_attribution_rmse",
        "archive_attribution_std",
        "archive_attribution_normalized_rmse",
        "archive_eager_attribution_pearson_r",
        "archive_eager_attribution_mae",
        "archive_ablated_rmse",
        "archive_cross_model_attribution_max_abs_pearson_r",
    )
    cross_model_keys: set[str] = (
        set(cross_model) if isinstance(cross_model, dict) else set()
    )
    expected_cross_model_keys: set[frozenset[str]] = (
        {Q05_BOOLQ_WORD_CROSS_MODELS}
        if portable
        else {
            Q05_BOOLQ_WORD_CROSS_MODELS,
            frozenset(Q05_BOOLQ_WORD_CROSS_MODEL_ALIASES),
        }
    )
    if (
        not isinstance(cross_model, dict)
        or frozenset(cross_model_keys) not in expected_cross_model_keys
        or not isinstance(scope_diagnostics, dict)
        or set(scope_diagnostics) != set(Q05_BOOLQ_WORD_SCOPE_ROWS)
        or not isinstance(pairwise, dict)
        or set(pairwise) != Q05_BOOLQ_WORD_PAIR_MODELS
        or not isinstance(maximum_pairwise, dict)
        or set(maximum_pairwise) != {"pearson_r2", "spearman"}
        or any(
            not isinstance(row.get(field), (int, float))
            or not math.isfinite(float(row[field]))
            for field in scalar_fields
        )
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in cross_model.values()
        )
    ):
        raise ValueError("Qwen-0.5B BoolQ-word diagnostic fields disagree")

    attribution_r: float = float(row["archive_attribution_pearson_r"])
    eager_r: float = float(row["archive_eager_attribution_pearson_r"])
    attribution_mae: float = float(row["archive_attribution_mae"])
    eager_mae: float = float(row["archive_eager_attribution_mae"])
    cross_maximum: float = max(abs(float(value)) for value in cross_model.values())
    if (
        float(row["archive_original_pearson_r"]) < 0.98
        or float(row["archive_ablated_pearson_r"]) < 0.98
        or not 0.90 <= attribution_r < 0.95
        or not -1.0 <= eager_r <= 1.0
        or not all(-1.0 <= float(value) <= 1.0 for value in cross_model.values())
        or float(row["archive_cross_model_attribution_max_abs_pearson_r"])
        not in [abs(float(value)) for value in cross_model.values()]
        or attribution_r <= cross_maximum
        or float(row["archive_cross_model_attribution_max_abs_pearson_r"])
        != cross_maximum
        or eager_r > attribution_r + 0.001
        or eager_mae < attribution_mae - 0.001
        or attribution_mae < 0.0
        or eager_mae < 0.0
        or float(row["archive_ablated_rmse"]) < 0.0
        or float(row["archive_attribution_rmse"]) < 0.0
        or float(row["archive_attribution_std"]) <= 0.0
        or float(row["archive_attribution_normalized_rmse"]) < 0.0
        or not math.isclose(
            float(row["archive_attribution_normalized_rmse"]),
            float(row["archive_attribution_rmse"])
            / float(row["archive_attribution_std"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Qwen-0.5B BoolQ-word archive diagnostics disagree")

    for scope, expected_rows in Q05_BOOLQ_WORD_SCOPE_ROWS.items():
        values: Any = scope_diagnostics.get(scope)
        if (
            not isinstance(values, dict)
            or set(values)
            != {
                "archive_std",
                "mae",
                "normalized_rmse",
                "pearson_r",
                "rmse",
                "rows",
            }
            or values.get("rows") != expected_rows
            or any(
                not isinstance(values.get(field), (int, float))
                or not math.isfinite(float(values[field]))
                for field in (
                    "archive_std",
                    "mae",
                    "normalized_rmse",
                    "pearson_r",
                    "rmse",
                )
            )
            or not math.isclose(
                float(values["normalized_rmse"]),
                float(values["rmse"]) / float(values["archive_std"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not -1.0 <= float(values["pearson_r"]) <= 1.0
            or float(values["archive_std"]) <= 0.0
            or float(values["mae"]) < 0.0
            or float(values["rmse"]) < 0.0
            or float(values["normalized_rmse"]) < 0.0
        ):
            raise ValueError(f"Qwen-0.5B BoolQ-word scope diagnostic differs: {scope}")
    if any(
        not math.isclose(
            float(scope_diagnostics["all"][scope_field]),
            float(row[row_field]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for scope_field, row_field in (
            ("pearson_r", "archive_attribution_pearson_r"),
            ("mae", "archive_attribution_mae"),
            ("rmse", "archive_attribution_rmse"),
            ("archive_std", "archive_attribution_std"),
            ("normalized_rmse", "archive_attribution_normalized_rmse"),
        )
    ):
        raise ValueError("Qwen-0.5B BoolQ-word all-scope diagnostics disagree")

    computed_maxima: dict[str, float] = {"pearson_r2": 0.0, "spearman": 0.0}
    for model, scopes in pairwise.items():
        if not isinstance(scopes, dict) or set(scopes) != set(
            Q05_BOOLQ_WORD_SCOPE_ROWS
        ):
            raise ValueError(f"Qwen-0.5B pairwise scopes disagree: {model}")
        for scope, expected_rows in Q05_BOOLQ_WORD_PAIR_COMMON_ROWS[model].items():
            values: Any = scopes.get(scope)
            fields: set[str] = {
                "archive_pearson_r2",
                "archive_spearman",
                "current_pearson_r2",
                "current_spearman",
                "n_common",
                "pearson_r2_delta",
                "spearman_delta",
            }
            if (
                not isinstance(values, dict)
                or set(values) != fields
                or values.get("n_common") != expected_rows
                or any(
                    not isinstance(values.get(field), (int, float))
                    or not math.isfinite(float(values[field]))
                    for field in fields - {"n_common"}
                )
                or not math.isclose(
                    float(values["pearson_r2_delta"]),
                    float(values["current_pearson_r2"])
                    - float(values["archive_pearson_r2"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(values["spearman_delta"]),
                    float(values["current_spearman"])
                    - float(values["archive_spearman"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not 0.0 <= float(values["archive_pearson_r2"]) <= 1.0
                or not 0.0 <= float(values["current_pearson_r2"]) <= 1.0
                or not -1.0 <= float(values["archive_spearman"]) <= 1.0
                or not -1.0 <= float(values["current_spearman"]) <= 1.0
            ):
                raise ValueError(
                    f"Qwen-0.5B pairwise diagnostic differs: {model}/{scope}"
                )
            computed_maxima["pearson_r2"] = max(
                computed_maxima["pearson_r2"], abs(float(values["pearson_r2_delta"]))
            )
            computed_maxima["spearman"] = max(
                computed_maxima["spearman"], abs(float(values["spearman_delta"]))
            )
    if any(
        not 0.0 <= float(maximum_pairwise[metric]) <= 0.025
        or not math.isclose(
            float(maximum_pairwise[metric]),
            computed_maxima[metric],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for metric in computed_maxima
    ):
        raise ValueError("Qwen-0.5B pairwise sensitivity is material or inconsistent")

    recorded_main_run: Any = row.get("execution_run_metadata")
    main_run: dict[str, Any] = (
        recorded_main_run
        if isinstance(recorded_main_run, dict)
        else _load_object(_run_path(results_dir, row))
    )
    segment_path: str = _artifact_path(results_dir, row, "segment_sha256")
    gold_projection_sha256: str = _segment_projection_sha256(segment_path)
    sdpa: Any = row.get("sdpa_repeatability")
    eager: Any = row.get("eager_backend_diagnostic")
    common_fields: set[str] = {
        "completed_epoch",
        "execution_run_metadata",
        "execution_run_sha256",
        "manifest_sha256",
        "schema_version",
        "segment_projection_sha256",
        "started_epoch",
        "tokens_sha256",
    }
    if (
        not isinstance(sdpa, dict)
        or set(sdpa) != common_fields | {"gold_segment_projection_sha256"}
        or not isinstance(eager, dict)
        or set(eager) != common_fields | {"diagnostic_wrapper_sha256"}
    ):
        raise ValueError("Qwen-0.5B recheck evidence fields disagree")
    for backend, evidence in (("sdpa", sdpa), ("eager", eager)):
        if (
            evidence.get("schema_version") != 1
            or type(evidence.get("started_epoch")) is not int
            or type(evidence.get("completed_epoch")) is not int
            or evidence["started_epoch"] >= evidence["completed_epoch"]
            or not _is_sha256(evidence.get("execution_run_sha256"))
            or not _is_sha256(evidence.get("segment_projection_sha256"))
            or not _is_sha256(evidence.get("tokens_sha256"))
        ):
            raise ValueError(f"Qwen-0.5B {backend} recheck evidence disagrees")
        _validate_recheck_metadata(
            evidence.get("execution_run_metadata"), main_run, backend
        )
        if evidence["execution_run_sha256"] != _json_object_sha256(
            evidence["execution_run_metadata"]
        ):
            raise ValueError(f"Qwen-0.5B {backend} recheck run hash disagrees")
    if (
        sdpa["completed_epoch"] > eager["started_epoch"]
        or sdpa.get("manifest_sha256") != row.get("manifest_sha256")
        or eager.get("manifest_sha256") != row.get("manifest_sha256")
        or sdpa.get("tokens_sha256") != row.get("tokens_sha256")
        or sdpa.get("gold_segment_projection_sha256") != gold_projection_sha256
        or sdpa.get("segment_projection_sha256") != gold_projection_sha256
        or eager.get("diagnostic_wrapper_sha256")
        != Q05_BOOLQ_WORD_EAGER_DIAGNOSTIC_WRAPPER_SHA256
    ):
        raise ValueError("Qwen-0.5B recheck artifact binding disagrees")


def _expected_grid() -> set[tuple[str, str, str]]:
    return {
        (model, benchmark, pregrouper)
        for model in OPEN_MODELS
        for benchmark, pregrouper in CONFIGURATIONS
    }


def _artifact_path(
    results_dir: str,
    row: dict[str, Any],
    hash_field: str,
) -> str:
    directory: str = os.path.join(
        results_dir,
        str(row["benchmark"]),
        str(row["pregrouper"]),
    )
    if hash_field == "manifest_sha256":
        return os.path.join(directory, "segments.tsv.gz")
    suffix: str = (
        "segment.tsv.gz" if hash_field == "segment_sha256" else "tokens.tsv.gz"
    )
    return os.path.join(directory, f"{row['model']}_{suffix}")


def _run_path(results_dir: str, row: dict[str, Any]) -> str:
    return os.path.join(
        results_dir,
        str(row["benchmark"]),
        str(row["pregrouper"]),
        f"{row['model']}_run.json",
    )


def _validate_execution_run(
    run: dict[str, Any],
    row: dict[str, Any],
    path: str,
) -> None:
    """Validate the immutable execution-time projection of one run record."""
    model: str = str(row["model"])
    if set(run) != EXECUTION_RUN_FIELDS:
        raise ValueError(f"Execution run fields disagree: {path}")
    parameters: Any = run.get("parameters")
    if (
        run.get("schema_version") != 2
        or run.get("model") != model
        or run.get("benchmark") != row["benchmark"]
        or run.get("pregrouper") != row["pregrouper"]
        or run.get("segmentation_scope") != "full_dialog_in_message_order"
        or run.get("model_source") != GOLD_OPEN_EXECUTION_MODEL_SOURCES[model]
        or run.get("source_sha256") != GOLD_OPEN_EXECUTION_SOURCE_SHA256
        or not isinstance(parameters, dict)
        or parameters.get("phases") != ["ablation", "attention"]
        or parameters.get("phase_attention_implementation") != EXPECTED_PHASE_BACKENDS
    ):
        raise ValueError(f"Execution run metadata disagrees: {path}")


def _validate_sealed_run_projection(
    results_dir: str,
    row: dict[str, Any],
) -> None:
    """Require the sealed sidecar to preserve its audited execution fields."""
    execution_run: Any = row.get("execution_run_metadata")
    execution_sha256: Any = row.get("execution_run_sha256")
    if not isinstance(execution_run, dict) or not isinstance(execution_sha256, str):
        raise ValueError("Execution run binding is missing from numerical audit")
    if _json_object_sha256(execution_run) != execution_sha256:
        raise ValueError("Execution run metadata hash disagrees with numerical audit")
    path: str = _run_path(results_dir, row)
    _validate_execution_run(execution_run, row, path)
    sealed: dict[str, Any] = _load_object(path)
    direct_fields: tuple[str, ...] = (
        "benchmark",
        "dataset",
        "model",
        "model_identity_files_sha256",
        "parameters",
        "pregrouper",
        "segmentation_scope",
        "source_sha256",
    )
    if any(sealed.get(field) != execution_run[field] for field in direct_fields):
        raise ValueError(f"Sealed run changed execution metadata: {path}")
    if sealed.get("execution_model_source") != execution_run["model_source"]:
        raise ValueError(f"Sealed run changed its execution model source: {path}")
    execution_software: Any = execution_run.get("software")
    sealed_software: Any = sealed.get("software")
    if (
        not isinstance(execution_software, dict)
        or not isinstance(sealed_software, dict)
        or any(
            sealed_software.get(name) != version
            for name, version in execution_software.items()
        )
    ):
        raise ValueError(f"Sealed run changed execution software metadata: {path}")


def _validate_numerical_rows(
    rows: list[dict[str, Any]],
    results_dir: str,
    portable_rows: bool,
    require_execution_segment_bytes: bool,
) -> None:
    keys: list[tuple[str, str, str]] = [
        (str(row.get("model")), str(row.get("benchmark")), str(row.get("pregrouper")))
        for row in rows
    ]
    if len(rows) != 40 or len(set(keys)) != 40 or set(keys) != _expected_grid():
        raise ValueError("Open numerical audit does not cover the exact 5x8 grid")
    for row in rows:
        _validate_numerical_row_schema(
            row,
            portable=portable_rows,
        )
        if portable_rows:
            if (
                row.get("execution_numerical_audit_status") != "passed"
                or row.get("structural_validation_status") != "passed"
                or row.get("historical_replication_status")
                not in {
                    "within_release_regression_tolerance",
                    "documented_historical_difference",
                    "not_comparable_prompt_version",
                }
            ):
                raise ValueError(f"Open numerical audit status is invalid: {row}")
        elif row.get("status") != "passed":
            raise ValueError(f"Open execution-time numerical audit failed: {row}")
        cell: tuple[str, str, str] = (
            str(row["model"]),
            str(row["benchmark"]),
            str(row["pregrouper"]),
        )
        if cell == Q05_BOOLQ_WORD_CELL:
            _validate_q05_boolq_word_diagnostics(row, results_dir, portable_rows)
        elif row["benchmark"] == "race":
            if (
                row.get("archive_comparison") != "not_comparable_prompt_version"
                or row.get("archive_race_w_norm_prompts") != 4_934
                or row.get("archive_race_w_norm_max_abs_error") != 0.0
            ):
                raise ValueError(
                    f"RACE historical replication diagnostic is missing: {row}"
                )
        elif row.get("archive_comparison") != "same_coordinates":
            raise ValueError(f"Open archive numerical status disagrees: {row}")
        else:
            _validate_standard_archive_diagnostics(row)
        for hash_field in NUMERICAL_HASH_FIELDS:
            path: str = _artifact_path(results_dir, row, hash_field)
            recorded_field: str = (
                "execution_segment_sha256"
                if hash_field == "segment_sha256" and portable_rows
                else hash_field
            )
            if recorded_field not in row:
                if row["benchmark"] == "lambada" and hash_field == "tokens_sha256":
                    continue
                raise ValueError(f"Missing {recorded_field} in open numerical audit")
            if hash_field == "segment_sha256" and not require_execution_segment_bytes:
                actual_projection: str = _validate_release_segment(
                    path,
                    _artifact_path(results_dir, row, "manifest_sha256"),
                    row,
                )
                if portable_rows and (
                    _sha256(path) != row.get("release_segment_sha256")
                    or actual_projection != row.get("release_segment_projection_sha256")
                ):
                    raise ValueError(
                        f"Open numerical audit release segment disagrees: {path}"
                    )
                continue
            if not os.path.isfile(path) or _sha256(path) != row[recorded_field]:
                raise ValueError(
                    f"Open numerical audit artifact hash disagrees: {path}"
                )


def build_receipt(
    results_dir: str,
    observation_path: str,
    queue_receipt_paths: list[str],
    numerical_log_path: str,
) -> dict[str, Any]:
    """Validate private execution evidence and emit a portable receipt."""
    observation: dict[str, Any] = _load_object(observation_path)
    if set(observation) != OBSERVATION_FIELDS or observation.get("schema_version") != 1:
        raise ValueError("Unsupported execution observation schema")
    source_files: Any = observation.get("source_files")
    models: Any = observation.get("models")
    if not isinstance(source_files, dict) or not isinstance(models, dict):
        raise ValueError("Execution observation is incomplete")
    observed_source_hashes: dict[str, str] = {
        str(path): str(record.get("sha256"))
        for path, record in source_files.items()
        if isinstance(record, dict)
    }
    if observed_source_hashes != GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256:
        raise ValueError("Observed execution source hashes disagree with the gold pins")
    if any(
        not isinstance(record, dict) or set(record) != {"mtime_ns", "sha256"}
        for record in source_files.values()
    ):
        raise ValueError("Execution source observations contain unexpected fields")

    queues: dict[str, dict[str, Any]] = {}
    model_queue: dict[str, str] = {}
    for path in queue_receipt_paths:
        queue: dict[str, Any] = _load_object(path)
        queue_name: str = str(queue.get("queue"))
        if queue_name not in QUEUE_MODELS or queue_name in queues:
            raise ValueError(f"Unexpected or duplicate queue receipt: {queue_name}")
        if (
            set(queue) != PRIVATE_QUEUE_FIELDS
            or queue.get("schema_version") != 2
            or tuple(queue.get("models", [])) != QUEUE_MODELS[queue_name]
            or queue.get("cells_per_model") != len(CONFIGURATIONS)
            or queue.get("source_sha256") != GOLD_OPEN_EXECUTION_SOURCE_SHA256
            or queue.get("phase_attention_implementation") != EXPECTED_PHASE_BACKENDS
        ):
            raise ValueError(f"Queue receipt metadata disagrees: {queue_name}")
        started_epoch: int = int(queue["started_epoch"])
        completed_epoch: int = int(queue["completed_epoch"])
        if started_epoch >= completed_epoch:
            raise ValueError(f"Queue timing is invalid: {queue_name}")
        portable_artifacts: dict[str, str] = {
            str(locator): str(digest)
            for locator, digest in queue.get("artifact_sha256", {}).items()
        }
        if set(portable_artifacts) != _expected_queue_artifacts(queue_name) or any(
            not _is_sha256(digest) for digest in portable_artifacts.values()
        ):
            raise ValueError(f"Queue artifact inventory disagrees: {queue_name}")
        queues[queue_name] = {
            "artifact_sha256": portable_artifacts,
            "completed_epoch": completed_epoch,
            "models": list(QUEUE_MODELS[queue_name]),
            "phase_attention_implementation": EXPECTED_PHASE_BACKENDS,
            "started_epoch": started_epoch,
        }
        for model in QUEUE_MODELS[queue_name]:
            model_queue[model] = queue_name
    if set(queues) != set(QUEUE_MODELS):
        raise ValueError("Both open-model queue receipts are required")

    earliest_start_ns: int = (
        min(int(queue["started_epoch"]) for queue in queues.values()) * 1_000_000_000
    )
    if any(
        not isinstance(record, dict)
        or int(record.get("mtime_ns", earliest_start_ns)) >= earliest_start_ns
        for record in source_files.values()
    ):
        raise ValueError("An observed execution source does not predate the queues")
    if set(models) != set(OPEN_MODELS):
        raise ValueError("Execution observation model inventory disagrees")
    for model, record in models.items():
        if not isinstance(record, dict) or set(record) != {
            "artifact_manifest_sha256",
            "latest_selected_artifact_mtime_ns",
        }:
            raise ValueError(f"Invalid model observation for {model}")
        queue_start_ns: int = (
            int(queues[model_queue[str(model)]]["started_epoch"]) * 1_000_000_000
        )
        if (
            record.get("artifact_manifest_sha256")
            != GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256.get(str(model))
            or int(record.get("latest_selected_artifact_mtime_ns", queue_start_ns))
            >= queue_start_ns
        ):
            raise ValueError(f"Model observation disagrees for {model}")

    numerical_rows: list[dict[str, Any]] = _load_numerical_rows(numerical_log_path)
    _validate_numerical_rows(
        numerical_rows,
        results_dir,
        portable_rows=False,
        require_execution_segment_bytes=False,
    )
    q05_word_row: dict[str, Any] = next(
        row
        for row in numerical_rows
        if (str(row["model"]), str(row["benchmark"]), str(row["pregrouper"]))
        == Q05_BOOLQ_WORD_CELL
    )
    if int(q05_word_row["sdpa_repeatability"]["started_epoch"]) < max(
        int(queue["completed_epoch"]) for queue in queues.values()
    ):
        raise ValueError("Qwen-0.5B repeatability run overlapped an original queue")
    q05_cross_model: Any = q05_word_row.get("archive_cross_model_attribution_pearson_r")
    if not isinstance(q05_cross_model, dict):
        raise ValueError("Qwen-0.5B cross-model diagnostics are missing")
    q05_word_row["archive_cross_model_attribution_pearson_r"] = {
        Q05_BOOLQ_WORD_CROSS_MODEL_ALIASES.get(str(model), str(model)): value
        for model, value in q05_cross_model.items()
    }
    for row in numerical_rows:
        queue_name = model_queue[str(row["model"])]
        queue_artifacts: dict[str, str] = queues[queue_name]["artifact_sha256"]
        for hash_field in ("segment_sha256", "tokens_sha256", "run_sha256"):
            if hash_field not in row:
                if hash_field == "run_sha256":
                    raise ValueError("Numerical audit lacks its execution run hash")
                continue
            suffix: str = {
                "segment_sha256": "segment.tsv.gz",
                "tokens_sha256": "tokens.tsv.gz",
                "run_sha256": "run.json",
            }[hash_field]
            locator: str = (
                f"{row['benchmark']}/{row['pregrouper']}/{row['model']}_{suffix}"
            )
            if queue_artifacts.get(locator) != row[hash_field]:
                raise ValueError(f"Queue/numerical audit hash mismatch: {locator}")
        run_path: str = _run_path(results_dir, row)
        if _sha256(run_path) != row["run_sha256"]:
            raise ValueError(f"Execution run hash disagrees: {run_path}")
        execution_run: dict[str, Any] = _load_object(run_path)
        _validate_execution_run(execution_run, row, run_path)
        segment_path: str = _artifact_path(results_dir, row, "segment_sha256")
        row["execution_segment_sha256"] = row.pop("segment_sha256")
        row["release_segment_sha256"] = _sha256(segment_path)
        row["release_segment_projection_sha256"] = (
            _release_equivalent_segment_projection_sha256(
                segment_path,
                str(row["benchmark"]),
            )
        )
        row["execution_run_sha256"] = row.pop("run_sha256")
        row["execution_run_metadata"] = execution_run
        row["execution_numerical_audit_status"] = row.pop("status")
        row["structural_validation_status"] = "passed"
        cell: tuple[str, str, str] = (
            str(row["model"]),
            str(row["benchmark"]),
            str(row["pregrouper"]),
        )
        row["historical_replication_status"] = (
            "documented_historical_difference"
            if cell == Q05_BOOLQ_WORD_CELL
            else (
                "not_comparable_prompt_version"
                if row["benchmark"] == "race"
                else "within_release_regression_tolerance"
            )
        )

    return {
        "archive_comparison_role": (
            "historical_replication_diagnostic_not_scientific_correctness"
        ),
        "artifact_integrity_status": "passed",
        "execution_source_hash_timing": "run_completion",
        "model_artifact_hash_timing": "post_run_seal",
        "models": models,
        "numerical_cells": sorted(
            numerical_rows,
            key=lambda row: (
                str(row["model"]),
                str(row["benchmark"]),
                str(row["pregrouper"]),
            ),
        ),
        "queues": queues,
        "schema_version": 1,
        "segment_packaging": {
            "name": "normalize_segment_outputs",
            "producer_to_release_evidence": (
                "producer hashes plus audited deterministic public transformation; "
                "not a cryptographic pre/post equivalence proof"
            ),
            "source_sha256": _sha256(
                os.path.join(os.path.dirname(__file__), "normalize_segment_outputs.py")
            ),
        },
        "source_files": source_files,
    }


def validate_receipt(receipt: dict[str, Any], results_dir: str) -> None:
    """Verify a shipped receipt against its current result artifacts."""
    if (
        set(receipt) != RECEIPT_FIELDS
        or receipt.get("schema_version") != 1
        or receipt.get("archive_comparison_role")
        != "historical_replication_diagnostic_not_scientific_correctness"
        or receipt.get("artifact_integrity_status") != "passed"
        or receipt.get("execution_source_hash_timing") != "run_completion"
        or receipt.get("model_artifact_hash_timing") != "post_run_seal"
    ):
        raise ValueError("Open execution audit receipt metadata disagrees")
    segment_packaging: Any = receipt.get("segment_packaging")
    if (
        not isinstance(segment_packaging, dict)
        or set(segment_packaging)
        != {"name", "producer_to_release_evidence", "source_sha256"}
        or segment_packaging.get("name") != "normalize_segment_outputs"
        or segment_packaging.get("producer_to_release_evidence")
        != (
            "producer hashes plus audited deterministic public transformation; "
            "not a cryptographic pre/post equivalence proof"
        )
        or segment_packaging.get("source_sha256")
        != _sha256(
            os.path.join(os.path.dirname(__file__), "normalize_segment_outputs.py")
        )
    ):
        raise ValueError("Open segment packaging evidence disagrees")
    source_files: Any = receipt.get("source_files")
    if (
        not isinstance(source_files, dict)
        or {
            str(path): str(record.get("sha256"))
            for path, record in source_files.items()
            if isinstance(record, dict)
        }
        != GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
    ):
        raise ValueError("Open execution audit source hashes disagree")
    if any(
        not isinstance(record, dict) or set(record) != {"mtime_ns", "sha256"}
        for record in source_files.values()
    ):
        raise ValueError("Open execution audit source fields disagree")
    models: Any = receipt.get("models")
    if not isinstance(models, dict) or set(models) != set(OPEN_MODELS):
        raise ValueError("Open execution audit model inventory disagrees")
    for model, record in models.items():
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "artifact_manifest_sha256",
                "latest_selected_artifact_mtime_ns",
            }
            or record.get("artifact_manifest_sha256")
            != GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256.get(str(model))
        ):
            raise ValueError(f"Open execution audit model seal disagrees: {model}")
    queues: Any = receipt.get("queues")
    if not isinstance(queues, dict) or set(queues) != set(QUEUE_MODELS):
        raise ValueError("Open execution audit queue inventory disagrees")
    model_queue: dict[str, str] = {}
    for queue_name, queue in queues.items():
        if (
            not isinstance(queue, dict)
            or set(queue) != PORTABLE_QUEUE_FIELDS
            or tuple(queue.get("models", [])) != QUEUE_MODELS[str(queue_name)]
            or queue.get("phase_attention_implementation") != EXPECTED_PHASE_BACKENDS
            or int(queue.get("started_epoch", 0))
            >= int(queue.get("completed_epoch", 0))
        ):
            raise ValueError(
                f"Open execution audit queue metadata disagrees: {queue_name}"
            )
        artifacts: Any = queue.get("artifact_sha256")
        if (
            not isinstance(artifacts, dict)
            or set(artifacts) != _expected_queue_artifacts(str(queue_name))
            or any(not _is_sha256(value) for value in artifacts.values())
        ):
            raise ValueError(
                f"Open execution audit queue artifacts disagree: {queue_name}"
            )
        for model in QUEUE_MODELS[str(queue_name)]:
            model_queue[model] = str(queue_name)
    earliest_start_ns: int = (
        min(int(queue["started_epoch"]) for queue in queues.values()) * 1_000_000_000
    )
    if any(
        int(record["mtime_ns"]) >= earliest_start_ns for record in source_files.values()
    ):
        raise ValueError("Open execution audit source timing disagrees")
    for model, record in models.items():
        queue_start_ns: int = (
            int(queues[model_queue[str(model)]]["started_epoch"]) * 1_000_000_000
        )
        if int(record["latest_selected_artifact_mtime_ns"]) >= queue_start_ns:
            raise ValueError(f"Open execution audit model timing disagrees: {model}")
    rows: Any = receipt.get("numerical_cells")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("Open execution numerical audit is missing")
    _validate_numerical_rows(
        rows,
        results_dir,
        portable_rows=True,
        require_execution_segment_bytes=False,
    )
    q05_word_row: dict[str, Any] = next(
        row
        for row in rows
        if (str(row["model"]), str(row["benchmark"]), str(row["pregrouper"]))
        == Q05_BOOLQ_WORD_CELL
    )
    if int(q05_word_row["sdpa_repeatability"]["started_epoch"]) < max(
        int(queue["completed_epoch"]) for queue in queues.values()
    ):
        raise ValueError("Qwen-0.5B repeatability timing disagrees")
    for row in rows:
        queue_name: str = model_queue[str(row["model"])]
        queue_artifacts: dict[str, str] = queues[queue_name]["artifact_sha256"]
        stem: str = f"{row['benchmark']}/{row['pregrouper']}/{row['model']}"
        expected_queue_hashes: dict[str, Any] = {
            f"{stem}_segment.tsv.gz": row.get("execution_segment_sha256"),
            f"{stem}_run.json": row.get("execution_run_sha256"),
        }
        if row["benchmark"] != "lambada":
            expected_queue_hashes[f"{stem}_tokens.tsv.gz"] = row.get("tokens_sha256")
        for locator, expected_sha256 in expected_queue_hashes.items():
            if queue_artifacts.get(locator) != expected_sha256:
                raise ValueError(f"Execution queue hash mismatch: {locator}")
        _validate_sealed_run_projection(results_dir, row)


def main() -> None:
    """Build a portable receipt from the queue and numerical audit records."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--queue-receipt", action="append", required=True)
    parser.add_argument("--numerical-log", required=True)
    parser.add_argument("--output", required=True)
    args: argparse.Namespace = parser.parse_args()
    receipt: dict[str, Any] = build_receipt(
        os.path.abspath(args.results_dir),
        os.path.abspath(args.observation),
        [os.path.abspath(path) for path in args.queue_receipt],
        os.path.abspath(args.numerical_log),
    )
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write("\n")


if __name__ == "__main__":
    main()
