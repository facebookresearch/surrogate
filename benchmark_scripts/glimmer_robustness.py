# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Derive the BoolQ fidelity extension for the public Muse Glimmer model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from benchmark_scripts.compute_logodds import compute
from benchmark_scripts.consolidate_results import consolidate
from benchmark_scripts.derived_provenance import (
    derived_supporting_source_paths,
    sha256_file,
    write_derived_provenance,
)
from benchmark_scripts.f_table import (
    API_MODELS,
    OPEN_MODELS,
    PAPER_MODELS,
    _analysis_rng,
    _derived_input_paths,
    _process_benchmark,
)
from benchmark_scripts.merge_phase_results import _validate_receipts
from benchmark_scripts.run_benchmark import EXECUTION_SOURCE_FILES
from benchmark_scripts.run_glimmer_boolq import (
    DIRECT_RESPONSE_PREFIX,
    EXTRA_EXECUTION_SOURCES,
    MODEL_FILE_SHA256,
    MODEL_REVISION,
    MODEL_SOURCE,
    TEXT_ONLY_DEVICE_MAP,
)
from benchmark_scripts.validate_results import (
    OPEN_SEGMENT_COLUMNS,
    _validate_derived_sidecar,
)
from surrogate.eval_constants import BOOLQ_CONFIG


MODEL_NAME: str = "muse-glimmer-30b"
COHORT_NAME: str = "recent_model_robustness"
METRICS: set[str] = {"F_pred", "F_attr"}
OUTPUT_COLUMNS: list[str] = [
    "cohort",
    "pair_population",
    "benchmark",
    "pregrouper",
    "scope",
    "requested_scope",
    "resolved_scope",
    "contrast",
    "requested_contrast",
    "resolved_source_contrast",
    "resolved_target_contrast",
    "readout_contrast",
    "availability_status",
    "unavailable_reason",
    "api_infinity_policy",
    "aggregation",
    "model_s",
    "model_t",
    "metric",
    "statistic",
    "n_observations",
    "expected_observations",
    "observation_coverage",
    "n_prompts",
    "expected_prompts",
    "prompt_coverage",
    "f_point",
    "f_lo",
    "f_hi",
]


def _pair_population(row: pd.Series) -> str:
    """Identify the canonical reference population paired with Glimmer."""
    reference: str = str(row["model_s"])
    if reference in OPEN_MODELS:
        return "glimmer_open"
    if reference in API_MODELS:
        return "glimmer_hosted"
    raise ValueError(f"Unexpected Glimmer reference model {reference!r}")


def _validate_extension_receipt(
    receipt_path: str,
    segment_path: str,
    token_path: str,
    manifest_path: str,
) -> None:
    """Validate the merged raw artifact and both embedded phase receipts."""
    with open(receipt_path, encoding="utf-8") as source:
        receipt: Any = json.load(source)
    if not isinstance(receipt, dict):
        raise ValueError(f"Expected a JSON object in {receipt_path}")
    expected_identity: dict[str, Any] = {
        "benchmark": "boolq",
        "pregrouper": "sentence",
        "segmentation_scope": "full_dialog_in_message_order",
        "model": MODEL_NAME,
        "model_source": MODEL_SOURCE,
        "model_revision": MODEL_REVISION,
        "model_revision_resolution": "embedded_sha256_manifest",
        "model_artifact_files_sha256": MODEL_FILE_SHA256,
        "model_identity_files_sha256": {
            filename: MODEL_FILE_SHA256[filename]
            for filename in (
                "config.json",
                "generation_config.json",
                "model.safetensors.index.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        },
    }
    disagreements: list[str] = [
        field
        for field, expected in expected_identity.items()
        if receipt.get(field) != expected
    ]
    if disagreements:
        raise ValueError(f"Glimmer receipt identity disagrees on {disagreements}")
    expected_protocol: dict[str, Any] = {
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
    if receipt.get("model_protocol") != expected_protocol:
        raise ValueError("Glimmer scoring protocol disagrees")
    expected_parameters: dict[str, Any] = {
        "phases": ["ablation", "attention"],
        "rendered_chat_add_special_tokens": False,
        "phase_attention_implementation": {
            "ablation": "sdpa",
            "attention": "eager",
        },
        "batch_size": 2,
        "max_samples": None,
        "max_forward_passes": None,
        "seed": 42,
    }
    if receipt.get("parameters") != expected_parameters:
        raise ValueError("Glimmer scoring parameters disagree")
    software: Any = receipt.get("software")
    if not isinstance(software, dict) or software.get("transformers") != "5.17.0":
        raise ValueError("Glimmer Transformers version disagrees")
    dataset: Any = receipt.get("dataset")
    if (
        not isinstance(dataset, dict)
        or dataset.get("rows") != 3270
        or dataset.get("reference_segment_manifest_sha256")
        != sha256_file(manifest_path)
    ):
        raise ValueError("Glimmer receipt dataset identity disagrees")
    output_hashes: Any = receipt.get("output_sha256")
    if output_hashes != {
        "segment": sha256_file(segment_path),
        "tokens": sha256_file(token_path),
    }:
        raise ValueError("Glimmer receipt output hashes disagree")
    assembly: Any = receipt.get("assembly")
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    merge_source_path: str = os.path.join(
        repository_root, "benchmark_scripts", "merge_phase_results.py"
    )
    normalize_source_path: str = os.path.join(
        repository_root, "benchmark_scripts", "normalize_segment_outputs.py"
    )
    if (
        not isinstance(assembly, dict)
        or assembly.get("kind") != "parallel_independent_phases"
        or assembly.get("generator") != "benchmark_scripts.merge_phase_results"
        or assembly.get("generator_source_sha256") != sha256_file(merge_source_path)
        or assembly.get("supporting_source_sha256")
        != {
            "benchmark_scripts/normalize_segment_outputs.py": sha256_file(
                normalize_source_path
            )
        }
    ):
        raise ValueError("Glimmer phase-assembly provenance disagrees")
    phase_inputs: Any = assembly.get("phase_inputs")
    if not isinstance(phase_inputs, dict):
        raise ValueError("Glimmer phase receipts are missing")
    attention: Any = phase_inputs.get("attention")
    ablation: Any = phase_inputs.get("ablation")
    if not isinstance(attention, dict) or not isinstance(ablation, dict):
        raise ValueError("Glimmer phase receipts are invalid")
    attention_receipt: Any = attention.get("receipt")
    ablation_receipt: Any = ablation.get("receipt")
    if not isinstance(attention_receipt, dict) or not isinstance(
        ablation_receipt, dict
    ):
        raise ValueError("Glimmer embedded phase receipts are invalid")
    _validate_receipts(attention_receipt, ablation_receipt)
    for phase_receipt in (attention_receipt, ablation_receipt):
        for field in (
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
        ):
            if phase_receipt.get(field) != receipt.get(field):
                raise ValueError(
                    f"Glimmer combined receipt disagrees with a phase on {field}"
                )
    source_hashes: Any = receipt.get("source_sha256")
    expected_source_paths: set[str] = set(EXECUTION_SOURCE_FILES) | set(
        EXTRA_EXECUTION_SOURCES
    )
    if (
        not isinstance(source_hashes, dict)
        or set(source_hashes) != expected_source_paths
    ):
        raise ValueError("Glimmer execution source hashes are missing")
    for relative_path, expected_sha256 in source_hashes.items():
        source_path: str = os.path.join(repository_root, str(relative_path))
        if (
            not os.path.isfile(source_path)
            or sha256_file(source_path) != expected_sha256
        ):
            raise ValueError(f"Glimmer execution source disagrees: {relative_path}")


def _validate_extension_outputs(
    segment_path: str,
    token_path: str,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require complete, unique Glimmer rows on the canonical coordinates."""
    segments: pd.DataFrame = pd.read_csv(segment_path, sep="\t")
    identity_columns: list[str] = manifest.columns.tolist()
    missing_identity: set[str] = set(identity_columns) - set(segments.columns)
    if missing_identity:
        raise ValueError(f"Glimmer segment identities are missing {missing_identity}")
    pd.testing.assert_frame_equal(
        segments[identity_columns], manifest, check_dtype=False
    )
    if segments.duplicated(["prompt_idx", "seg_idx"]).any():
        raise ValueError("Glimmer segment coordinates are not unique")
    for column in ("original_result_available", "segment_result_available"):
        if column not in segments or not segments[column].eq(True).all():
            raise ValueError(f"Glimmer coverage is incomplete in {column}")
    required_metrics: set[str] = set(OPEN_SEGMENT_COLUMNS)
    if not required_metrics.issubset(segments.columns):
        raise ValueError("Glimmer segment metrics are incomplete")
    if not np.isfinite(segments[list(required_metrics)].to_numpy(dtype=float)).all():
        raise ValueError("Glimmer segment metrics contain non-finite values")

    tokens: pd.DataFrame = pd.read_csv(
        token_path,
        sep="\t",
        dtype={"label": str, "token": str, "kind": str},
    )
    required_token_columns: set[str] = {
        "prompt_idx",
        "seg_idx",
        "kind",
        "answer",
        "label",
        "token",
        "logprob",
    }
    if not required_token_columns.issubset(tokens.columns):
        raise ValueError("Glimmer token columns are incomplete")
    aliases: set[tuple[str, str]] = {
        (label, report_token.alias)
        for label, report_tokens in BOOLQ_CONFIG.report_tokens.items()
        for report_token in report_tokens
    }
    tokens["seg_idx_key"] = tokens["seg_idx"].fillna(-1).astype(int)
    token_keys: list[str] = [
        "prompt_idx",
        "seg_idx_key",
        "kind",
        "label",
        "token",
    ]
    if tokens.duplicated(token_keys).any():
        raise ValueError("Glimmer token coordinates are not unique")
    if set(tokens["kind"]) != {"orig", "ablated"}:
        raise ValueError("Glimmer token kinds are invalid")
    prompt_ids: set[int] = set(manifest["prompt_idx"].astype(int))
    manifest_keys: set[tuple[int, int]] = set(
        manifest[["prompt_idx", "seg_idx"]].itertuples(index=False, name=None)
    )
    original: pd.DataFrame = tokens[tokens["kind"].eq("orig")]
    ablated: pd.DataFrame = tokens[tokens["kind"].eq("ablated")]
    if (
        set(original["prompt_idx"].astype(int)) != prompt_ids
        or not original["seg_idx"].isna().all()
    ):
        raise ValueError("Glimmer original-token coordinates are incomplete")
    ablated_keys: set[tuple[int, int]] = set(
        ablated[["prompt_idx", "seg_idx_key"]].itertuples(index=False, name=None)
    )
    if ablated_keys != manifest_keys:
        raise ValueError("Glimmer ablated-token coordinates are incomplete")
    expected_rows: int = (len(prompt_ids) + len(manifest_keys)) * len(aliases)
    if (
        len(tokens) != expected_rows
        or set(tokens[["label", "token"]].itertuples(index=False, name=None)) != aliases
    ):
        raise ValueError("Glimmer label-token grid is incomplete")
    coordinate_sizes: pd.Series = tokens.groupby(
        ["prompt_idx", "seg_idx_key", "kind"]
    ).size()
    if (
        len(coordinate_sizes) != len(prompt_ids) + len(manifest_keys)
        or not (coordinate_sizes == len(aliases)).all()
    ):
        raise ValueError("Glimmer label-token rows are incomplete by coordinate")
    expected_alias_count: dict[tuple[str, str], int] = {
        alias: len(prompt_ids) + len(manifest_keys) for alias in aliases
    }
    actual_alias_count: dict[tuple[str, str], int] = {
        (str(label), str(token)): int(count)
        for (label, token), count in tokens.groupby(["label", "token"]).size().items()
    }
    if actual_alias_count != expected_alias_count:
        raise ValueError("Glimmer label-token aliases have incomplete coverage")
    tokens["numeric_logprob"] = pd.to_numeric(tokens["logprob"], errors="coerce")
    finite_label_groups: pd.Series = tokens.groupby(
        ["prompt_idx", "seg_idx_key", "kind", "label"]
    )["numeric_logprob"].apply(lambda values: np.isfinite(values).any())
    if not finite_label_groups.all():
        raise ValueError("Glimmer has a label group without a finite token score")
    expected_answers: pd.Series = manifest.groupby("prompt_idx")["answer"].first()
    token_answers: pd.DataFrame = tokens[["prompt_idx", "answer"]].drop_duplicates()
    if token_answers.duplicated("prompt_idx").any() or not token_answers.set_index(
        "prompt_idx"
    )["answer"].equals(expected_answers):
        raise ValueError("Glimmer token answers disagree with the manifest")
    return segments, tokens.drop(columns=["numeric_logprob", "seg_idx_key"])


def derive(
    results_dir: str,
    extension_results_dir: str,
    output_path: str,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
    verify_existing: bool = False,
) -> pd.DataFrame:
    """Build and seal Glimmer-vs-paper-cohort BoolQ fidelity estimates.

    Args:
        results_dir: Root containing the canonical paper artifacts.
        extension_results_dir: Root containing the Glimmer raw artifact.
        output_path: Destination for the long-format robustness table.
        bootstrap_resamples: Number of prompt-cluster bootstrap samples.
        confidence_level: Central bootstrap interval coverage.
        seed: Deterministic bootstrap seed.
        verify_existing: Validate the existing table and sidecar without writes.

    Returns:
        The table written to ``output_path``.
    """
    consolidate("boolq", "sentence", results_dir)
    canonical_config_dir: str = os.path.join(results_dir, "boolq", "sentence")
    extension_config_dir: str = os.path.join(extension_results_dir, "boolq", "sentence")
    canonical_manifest_path: str = os.path.join(canonical_config_dir, "segments.tsv.gz")
    extension_manifest_path: str = os.path.join(extension_config_dir, "segments.tsv.gz")
    canonical_manifest: pd.DataFrame = pd.read_csv(
        canonical_manifest_path, sep="\t", keep_default_na=False
    )
    extension_manifest: pd.DataFrame = pd.read_csv(
        extension_manifest_path, sep="\t", keep_default_na=False
    )
    pd.testing.assert_frame_equal(
        canonical_manifest, extension_manifest, check_dtype=False
    )

    extension_segment_path: str = os.path.join(
        extension_config_dir, f"{MODEL_NAME}_segment.tsv.gz"
    )
    extension_token_path: str = os.path.join(
        extension_config_dir, f"{MODEL_NAME}_tokens.tsv.gz"
    )
    extension_run_path: str = os.path.join(
        extension_config_dir, f"{MODEL_NAME}_run.json"
    )
    for path in (extension_segment_path, extension_token_path, extension_run_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    _validate_extension_receipt(
        extension_run_path,
        extension_segment_path,
        extension_token_path,
        extension_manifest_path,
    )

    extension_segments, extension_tokens = _validate_extension_outputs(
        extension_segment_path,
        extension_token_path,
        canonical_manifest,
    )
    canonical_segments: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "boolq_sentence_segments.tsv"), sep="\t"
    )
    canonical_tokens: pd.DataFrame = pd.read_csv(
        os.path.join(results_dir, "boolq_sentence_tokens.tsv"),
        sep="\t",
        dtype={"label": str, "token": str, "kind": str},
    )
    extension_segments = extension_segments.assign(model=MODEL_NAME)
    extension_tokens = extension_tokens.assign(model=MODEL_NAME)
    cohort: tuple[str, ...] = (*PAPER_MODELS, MODEL_NAME)
    with TemporaryDirectory() as temporary_results_dir:
        temporary_config_dir: str = os.path.join(
            temporary_results_dir, "boolq", "sentence"
        )
        os.makedirs(temporary_config_dir)
        shutil.copyfile(
            canonical_manifest_path,
            os.path.join(temporary_config_dir, "segments.tsv.gz"),
        )
        pd.concat([canonical_segments, extension_segments], ignore_index=True).to_csv(
            os.path.join(temporary_results_dir, "boolq_sentence_segments.tsv"),
            sep="\t",
            index=False,
        )
        pd.concat([canonical_tokens, extension_tokens], ignore_index=True).to_csv(
            os.path.join(temporary_results_dir, "boolq_sentence_tokens.tsv"),
            sep="\t",
            index=False,
        )
        compute("boolq", "sentence", temporary_results_dir)
        rows: list[dict[str, str | float]] = []
        # Process one reference pair at a time. ``_process_benchmark`` emits
        # every pair in its cohort, so passing the full twelve-model cohort
        # would spend most of the bootstrap work on reference-reference pairs
        # that are not part of this extension.
        for reference_model in PAPER_MODELS:
            rows.extend(
                _process_benchmark(
                    "boolq",
                    "sentence",
                    temporary_results_dir,
                    bootstrap_resamples,
                    confidence_level,
                    _analysis_rng(
                        seed,
                        "boolq",
                        "sentence",
                        reference_model,
                        MODEL_NAME,
                    ),
                    (reference_model, MODEL_NAME),
                    "user",
                    "canonical",
                    "pairwise_complete",
                    "row_pooled",
                    seed,
                    METRICS,
                )
            )
    frame: pd.DataFrame = pd.DataFrame(rows)
    frame = frame[frame["model_t"].eq(MODEL_NAME)].copy().reset_index(drop=True)
    expected_rows: int = len(PAPER_MODELS) * len(METRICS) * 3
    if len(frame) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} Glimmer comparison rows, found {len(frame)}"
        )
    if set(frame["model_s"]) != set(PAPER_MODELS):
        raise ValueError("Glimmer table does not cover the exact paper cohort")
    expected_keys: set[tuple[str, str, str]] = {
        (model, metric, statistic)
        for model in PAPER_MODELS
        for metric in METRICS
        for statistic in ("spearman", "pearson_r", "pearson_r2")
    }
    actual_keys: set[tuple[str, str, str]] = set(
        frame[["model_s", "metric", "statistic"]].itertuples(index=False, name=None)
    )
    if (
        actual_keys != expected_keys
        or frame.duplicated(["model_s", "model_t", "metric", "statistic"]).any()
    ):
        raise ValueError("Glimmer fidelity key grid is incomplete")
    expected_user_segments: int = int(
        canonical_manifest["message_role"].eq("user").sum()
    )
    expected_observations: pd.Series = frame["metric"].map(
        {
            "F_pred": len(canonical_manifest["prompt_idx"].unique()),
            "F_attr": expected_user_segments,
        }
    )
    if not frame["expected_observations"].eq(expected_observations).all():
        raise ValueError("Glimmer fidelity expected-observation counts disagree")
    if not frame["expected_prompts"].eq(3270).all():
        raise ValueError("Glimmer fidelity expected-prompt counts disagree")
    if (
        not frame["availability_status"].eq("available").all()
        or not np.isfinite(
            frame[["f_point", "f_lo", "f_hi"]].to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError("Glimmer fidelity contains unavailable estimates")
    frame["cohort"] = COHORT_NAME
    frame["pair_population"] = frame.apply(_pair_population, axis=1)
    frame = frame[OUTPUT_COLUMNS]

    supporting_sources: dict[str, str] = derived_supporting_source_paths()
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    f_table_source: str = "benchmark_scripts/f_table.py"
    supporting_sources[f_table_source] = os.path.join(repository_root, f_table_source)
    input_paths: dict[str, str] = _derived_input_paths(
        results_dir,
        [("boolq", "sentence")],
        PAPER_MODELS,
        set(),
    )
    for name, path in {
        "segments.tsv.gz": extension_manifest_path,
        f"{MODEL_NAME}_segment.tsv.gz": extension_segment_path,
        f"{MODEL_NAME}_tokens.tsv.gz": extension_token_path,
        f"{MODEL_NAME}_run.json": extension_run_path,
    }.items():
        input_paths[f"robustness/muse_glimmer/{name}"] = path
    provenance_root: str = os.path.commonpath(
        [
            os.path.abspath(results_dir),
            os.path.abspath(extension_results_dir),
            os.path.abspath(output_path),
        ]
    )
    parameters: dict[str, Any] = {
        "benchmark": "boolq",
        "pregrouper": "sentence",
        "scope": "user",
        "contrast": "canonical",
        "metrics": sorted(METRICS),
        "api_infinity_policy": "pairwise_complete",
        "aggregation": "row_pooled",
        "bootstrap_resamples": bootstrap_resamples,
        "confidence_level": confidence_level,
        "seed": seed,
        "bootstrap_rng": "sha256_cell_key_v1",
        "extension_model": MODEL_NAME,
        "reference_models": list(PAPER_MODELS),
    }
    if verify_existing:
        if not os.path.isfile(output_path):
            raise FileNotFoundError(output_path)
        recorded: pd.DataFrame = pd.read_csv(
            output_path,
            sep="\t",
            keep_default_na=False,
        )
        pd.testing.assert_frame_equal(
            recorded,
            frame,
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
        recorded_parameters: dict[str, Any] = _validate_derived_sidecar(
            provenance_root,
            output_path,
            "benchmark_scripts.glimmer_robustness",
            [("boolq", "sentence")],
            cohort,
            expected_input_paths=input_paths,
            supporting_source_files=tuple(supporting_sources),
        )
        if recorded_parameters != parameters:
            raise ValueError("Glimmer derived parameters disagree")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        frame.to_csv(output_path, sep="\t", index=False)
        write_derived_provenance(
            output_path,
            generator_name="benchmark_scripts.glimmer_robustness",
            generator_path=__file__,
            input_paths=input_paths,
            parameters=parameters,
            root_dir=provenance_root,
            supporting_source_paths=supporting_sources,
        )
    return frame


def main() -> None:
    """Parse CLI arguments and derive the Glimmer robustness table."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--extension-results-dir", default="robustness/muse_glimmer")
    parser.add_argument(
        "--output",
        default="robustness/muse_glimmer/glimmer_boolq_fidelity.tsv",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-existing", action="store_true")
    args: argparse.Namespace = parser.parse_args()
    derive(
        args.results_dir,
        args.extension_results_dir,
        args.output,
        args.bootstrap_resamples,
        args.confidence_level,
        args.seed,
        args.verify_existing,
    )


if __name__ == "__main__":
    main()
