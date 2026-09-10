# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Reconcile published, archived, and corrected scalar fidelity results.

The paper tables were produced from wide, consolidated TSVs.  The corrected
pipeline emits long tables with explicit scope and contrast metadata.  This
module independently evaluates the historical estimands on the archived TSVs,
summarizes compatible corrected rows, and records both beside the fixed values
printed in Tables 1 and 2.

Comparisons are keyed by a complete ``estimand_id``.  In particular, a result
from the corrected RACE segment grid is retained but is never compared with a
number from the historical 136,313-row grid. Historical BoolQ-word attention
rows contaminated with sentence-level values are likewise retained under a
distinct, explicitly invalid estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
import pandas as pd

from benchmark_scripts.provenance_sources import (
    RECONCILIATION_ARCHIVE_CHECKS_SHA256,
    RECONCILIATION_ARCHIVE_LEDGER_SHA256,
    RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
    RECONCILIATION_ARCHIVE_SHA256,
    RECONCILIATION_ARCHIVE_SIZE_BYTES,
    RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
    RECONCILIATION_RACE_RAW_SHA256,
    RECONCILIATION_RACE_RAW_SIZE_BYTES,
)
from benchmark_scripts.race_multiclass import (
    MAGNITUDE_METRICS,
    VECTOR_METRICS,
    ModelSignals,
    _aligned_moments,
    _attribution_features,
    _evaluate,
    _prediction_features,
)
from surrogate.eval_constants import RACE_CONFIG
from surrogate.utils import segment_text


SCHEMA_VERSION: int = 2
PAIRWISE_DROP: str = "pairwise_drop_nonfinite"
STATISTIC: str = "pearson_r2"

OPEN: tuple[str, ...] = (
    "qwen2.5-0.5b-instruct",
    "qwen2.5-3b-instruct",
    "llama-3.1-8b-instruct",
    "qwen2.5-7b-instruct",
    "qwen2.5-14b-instruct",
)
HOSTED: tuple[str, ...] = (
    "llama3.1-70b-instruct",
    "llama3.3-70b-instruct",
    "llama4-maverick-17b-128e-instruct",
    "gpt-4o",
    "gpt-4-1",
    "gemini-2-5-flash-lite-vertex",
)
ALL: tuple[str, ...] = OPEN + HOSTED

METRICS: tuple[str, ...] = (
    "F_pred",
    "F_attr",
    "F_attn_rollout",
    "F_attn_mean",
    "F_attn_max",
    "F_mag",
    "F_align",
    "F_align_to_attr",
    "F_mag_to_attr",
    "F_attn_rollout_to_attr",
    "F_attn_mean_to_attr",
    "F_attn_max_to_attr",
)
REPRESENTATION_METRICS: frozenset[str] = frozenset(
    {"F_attn_rollout", "F_attn_mean", "F_attn_max", "F_mag", "F_align"}
)
TRANSFER_METRICS: frozenset[str] = frozenset(
    {
        "F_align_to_attr",
        "F_mag_to_attr",
        "F_attn_rollout_to_attr",
        "F_attn_mean_to_attr",
        "F_attn_max_to_attr",
    }
)
BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS: frozenset[str] = frozenset(
    {
        "F_attn_rollout",
        "F_attn_mean",
        "F_attn_max",
        "F_attn_rollout_to_attr",
        "F_attn_mean_to_attr",
        "F_attn_max_to_attr",
    }
)
BOOLQ_WORD_ATTENTION_NOTE: str = (
    "The historical BoolQ-word Qwen attention columns are sentence-level "
    "values joined onto the word grid by integer segment index; this row "
    "reproduces that archived computation but is not a valid word-level "
    "attention estimand."
)


@dataclass(frozen=True)
class BenchmarkSpec:
    """A paper Table 2 benchmark configuration and its historical artifact."""

    benchmark: str
    pregrouper: str
    filename: str
    paper_label: str
    historical_rows: int


@dataclass(frozen=True)
class LegacyCohort:
    """External mapping from archive column prefixes to public model IDs."""

    archive_to_public: dict[str, str]
    open_models: tuple[str, ...]
    hosted_models: tuple[str, ...]

    @property
    def all_models(self) -> tuple[str, ...]:
        """Return the exact ordered 11-model paper cohort."""
        return self.open_models + self.hosted_models

    def archive_prefix(self, public_model: str) -> str:
        """Return the archive column prefix for a public model identifier."""
        matches: list[str] = [
            prefix
            for prefix, model in self.archive_to_public.items()
            if model == public_model
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one archive prefix for {public_model!r}")
        return matches[0]


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        "boolq", "sentence", "boolq_sentence_consolidated.tsv", "BoolQ", 27516
    ),
    BenchmarkSpec(
        "anli_r1", "sentence", "anli_r1_sentence_consolidated.tsv", "ANLI R1", 8181
    ),
    BenchmarkSpec(
        "anli_r2", "sentence", "anli_r2_sentence_consolidated.tsv", "ANLI R2", 8163
    ),
    BenchmarkSpec(
        "anli_r3", "sentence", "anli_r3_sentence_consolidated.tsv", "ANLI R3", 10028
    ),
    BenchmarkSpec(
        "winogrande",
        "sentence",
        "winogrande_sentence_consolidated.tsv",
        "WinoGrande",
        9135,
    ),
    BenchmarkSpec("boolq", "word", "boolq_word_consolidated.tsv", "BoolQ word", 10000),
    BenchmarkSpec("lambada", "word", "lambada_word_consolidated.tsv", "LAMBADA", 10000),
    BenchmarkSpec("race", "sentence", "race_sentence_consolidated.tsv", "RACE", 136313),
)

# Exact values as printed in the paper.  These are deliberately strings so the
# checks preserve the publication's three-decimal precision.
PUBLISHED_TABLE1: dict[str, dict[str, tuple[str, str, str]]] = {
    "F_pred": {
        "open": (".177", ".652", ".795"),
        "open_to_hosted": (".108", ".682", ".845"),
        "all": (".108", ".709", ".956"),
    },
    "F_attr": {
        "open": (".091", ".432", ".560"),
        "open_to_hosted": (".032", ".420", ".557"),
        "all": (".032", ".460", ".828"),
    },
    "F_attn_mean": {"open": (".762", ".905", ".984")},
    "F_mag": {"open": (".641", ".855", ".903")},
    "F_attn_rollout": {"open": (".443", ".848", ".938")},
    "F_attn_max": {"open": (".641", ".842", ".910")},
    "F_align": {"open": (".058", ".200", ".294")},
    "F_align_to_attr": {
        "open": (".060", ".259", ".381"),
        "open_to_hosted": (".032", ".262", ".362"),
        "all": (".032", ".259", ".381"),
    },
    "F_mag_to_attr": {
        "open": (".000", ".074", ".229"),
        "open_to_hosted": (".000", ".122", ".282"),
        "all": (".000", ".109", ".282"),
    },
    "F_attn_mean_to_attr": {
        "open": (".000", ".004", ".035"),
        "open_to_hosted": (".000", ".003", ".024"),
        "all": (".000", ".003", ".035"),
    },
    "F_attn_rollout_to_attr": {
        "open": (".000", ".003", ".069"),
        "open_to_hosted": (".000", ".002", ".013"),
        "all": (".000", ".002", ".069"),
    },
    "F_attn_max_to_attr": {
        "open": (".000", ".002", ".032"),
        "open_to_hosted": (".000", ".002", ".019"),
        "all": (".000", ".002", ".032"),
    },
}

_T2: dict[str, tuple[str, ...]] = {
    "F_pred": (".709", ".354", ".282", ".343", ".225", ".708", ".108", ".462"),
    "F_attr": (".460", ".280", ".223", ".241", ".150", ".175", ".115", ".524"),
    "F_attn_rollout": (".848", ".852", ".853", ".854", ".850", ".857", ".690"),
    "F_attn_mean": (".905", ".953", ".951", ".954", ".871", ".881", ".561"),
    "F_attn_max": (".842", ".939", ".937", ".937", ".808", ".627", ".332"),
    "F_mag": (".855", ".867", ".875", ".874", ".671", ".524", ".342"),
    "F_align": (".200", ".078", ".071", ".060", ".018", ".012", ".024"),
    "F_align_to_attr": (".259", ".128", ".113", ".114", ".051", ".026", ".040"),
    "F_mag_to_attr": (".109", ".297", ".268", ".305", ".228", ".173", ".159"),
    "F_attn_rollout_to_attr": (".002", ".021", ".016", ".011", ".017", ".004", ".007"),
    "F_attn_mean_to_attr": (".003", ".013", ".013", ".022", ".018", ".003", ".099"),
    "F_attn_max_to_attr": (".002", ".013", ".016", ".010", ".022", ".002", ".069"),
}
PUBLISHED_TABLE2: dict[str, dict[str, str]] = {
    metric: {spec.paper_label: value for spec, value in zip(BENCHMARKS, values)}
    for metric, values in _T2.items()
}

HISTORICAL_NUMERIC_SENTINELS: dict[tuple[str, str, str], float] = {
    ("boolq", "sentence", "F_pred"): 0.7091681295097776,
    ("boolq", "sentence", "F_attr"): 0.4601916121912448,
    ("race", "sentence", "F_pred"): 0.4618473276927767,
    ("race", "sentence", "F_attr"): 0.5238982090704335,
}

RACE_RAW_ARCHIVE_BASENAMES: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "race_q05_ablation.json",
    "qwen2.5-3b-instruct": "race_qwen2.5-3b-instruct.json",
    "llama-3.1-8b-instruct": "race_llama-3.1-8b-instruct.json",
    "qwen2.5-7b-instruct": "race_qwen2.5-7b-instruct.json",
    "qwen2.5-14b-instruct": "race_qwen2.5-14b-instruct.json",
    "llama3.1-70b-instruct": "race_llama3.1-70b-instruct.json",
    "llama3.3-70b-instruct": "race_llama3.3-70b-instruct.json",
    "llama4-maverick-17b-128e-instruct": (
        "race_llama4-maverick-17b-128e-instruct.json"
    ),
    "gpt-4o": "race_gpt-4o.json",
    "gpt-4-1": "race_gpt-4-1.json",
    "gemini-2-5-flash-lite-vertex": "race_gemini-2-5-flash-lite-vertex.json",
}

CLAIM_COLUMNS: tuple[str, ...] = (
    "claim_id",
    "claim_group",
    "source_location",
    "source_state",
    "benchmark",
    "pregrouper",
    "segment_grid_id",
    "cohort",
    "scope",
    "contrast",
    "representation",
    "metric",
    "statistic",
    "aggregation",
    "missingness_policy",
    "pair_set",
    "model_s",
    "model_t",
    "value_component",
    "expected_display",
    "expected_value",
    "tolerance",
    "expected_relation",
    "actual_value",
    "n_pairs",
    "n_observations",
    "n_prompts",
    "effective_pairs",
    "input_locators",
    "passed",
    "note",
)

MANUSCRIPT_DISPOSITION_COLUMNS: tuple[str, ...] = (
    "issue_id",
    "paper_reference_id",
    "paper_location",
    "affected_claims",
    "audited_behavior",
    "candidate_replacement",
    "revision_action",
    "selection_status",
    "selected_candidate_id",
    "interpretation",
)

FINITE_EXTREME_EXPECTATIONS: dict[
    tuple[str, str], tuple[float, float, float, float]
] = {
    ("boolq", "sentence"): (0.709168, 0.460192, 0.728851, 0.162042),
    ("anli_r1", "sentence"): (0.354206, 0.280300, 0.491683, 0.198400),
    ("anli_r2", "sentence"): (0.282462, 0.223453, 0.400229, 0.176328),
    ("anli_r3", "sentence"): (0.342835, 0.241122, 0.450067, 0.171539),
    ("winogrande", "sentence"): (0.224576, 0.149737, 0.222998, 0.098401),
    ("boolq", "word"): (0.708474, 0.174603, 0.731158, 0.069118),
}

PROMPT_EQUAL_EXPECTATIONS: dict[str, float] = {
    "F_align_to_attr": 0.409,
    "F_mag_to_attr": 0.158,
    "F_attn_rollout_to_attr": 0.163,
    "F_attn_mean_to_attr": 0.304,
    "F_attn_max_to_attr": 0.199,
}

EXPANDED_RACE_EXPECTATIONS: dict[tuple[str, str], tuple[float, float, float]] = {
    ("prediction", "signed_frobenius_r"): (0.825, 0.516, 0.931),
    ("prediction", "direction_cosine"): (0.781, 0.404, 0.900),
    ("prediction", "linear_cka"): (0.669, 0.268, 0.856),
    ("attribution", "signed_frobenius_r"): (0.610, 0.215, 0.762),
    ("attribution", "direction_cosine"): (0.188, 0.082, 0.291),
    ("attribution", "linear_cka"): (0.385, 0.064, 0.600),
    ("attribution", "aitchison_magnitude_r"): (0.723, 0.524, 0.853),
    ("attribution", "fisher_rao_magnitude_r"): (0.575, 0.392, 0.707),
    ("attribution", "fisher_local_magnitude_r"): (0.122, 0.060, 0.267),
}

MAIN_TEXT_PAIR_EXPECTATIONS: tuple[tuple[str, str, float, float], ...] = (
    ("llama3.1-70b-instruct", "llama3.3-70b-instruct", 0.910, 0.956),
    ("gpt-4o", "gpt-4-1", 0.678, 0.849),
    ("llama3.1-70b-instruct", "qwen2.5-14b-instruct", 0.684, 0.819),
)

FIGURE13_EXPECTATIONS: dict[str, float] = {
    "prediction": 0.734720,
    "attribution": 0.431266,
}

RACE_NOTE: str = (
    "The RACE F_pred/F_attr cells printed in Table 2 are scalar, "
    "answer-conditioned correct-vs-rest Pearson r-squared values on the "
    "historical 136,313-row grid, although the paper describes RACE as an RV "
    "comparison. Corrected-grid scalar output is compatibility-only; use the "
    "multiclass RACE outputs for the stated estimand."
)
PAPER_REFERENCE_ID: str = "arxiv:2606.32008:unversioned-audited-claim-set"
PAPER_IDENTITY_STATUS: str = "unversioned_reference_not_byte_pinned"
PAPER_CLAIM_CANONICALIZATION: str = "utf8_json_sorted_keys_claim_id_v1"
MODEL_IDENTITY_NOTE: str = (
    "Hosted endpoint names do not establish immutable provider snapshots. "
    "A corrected result without explicit snapshot identifiers is only "
    "nominally comparable under the same method."
)

LEDGER_COLUMNS: tuple[str, ...] = (
    "reconciliation_id",
    "source_state",
    "source_locator",
    "source_sha256",
    "paper_location",
    "benchmark",
    "pregrouper",
    "segment_grid_id",
    "declared_scope",
    "executed_scope",
    "declared_contrast",
    "executed_contrast",
    "cohort",
    "pair_set",
    "metric",
    "statistic",
    "aggregation",
    "missingness_policy",
    "value_component",
    "value",
    "display_value",
    "n_pairs",
    "effective_pairs",
    "data_status",
    "specification_status",
    "revision_action",
    "reported_metric",
    "reported_statistic",
    "computed_metric",
    "computed_statistic",
    "numerical_reproduction_status",
    "model_identity_status",
    "model_snapshot_identity",
    "method_id",
    "estimand_id",
    "comparison_status",
    "delta_from_published",
    "note",
)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_ledger_sha256(path: str) -> str:
    """Hash the canonical sanitized archive-row projection of a ledger TSV."""
    frame: pd.DataFrame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if tuple(frame.columns) != LEDGER_COLUMNS:
        raise ValueError(f"Unexpected reconciliation ledger schema in {path}")
    rows: list[dict[str, str]] = (
        frame[frame["source_state"] == "archive"]
        .sort_values("reconciliation_id", kind="stable")
        .to_dict(orient="records")
    )
    serialized: bytes = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def archive_checks_sha256(checks: list[dict[str, Any]]) -> str:
    """Hash all deterministic checks derived from the historical archive."""
    archive_checks: list[dict[str, Any]] = sorted(
        (check for check in checks if check.get("kind") != "corrected_provenance"),
        key=lambda check: str(check["check_id"]),
    )
    serialized: bytes = json.dumps(
        archive_checks,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def historical_claims_sha256(path: str) -> str:
    """Hash the canonical row projection of the historical-claims TSV."""
    frame: pd.DataFrame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    if tuple(frame.columns) != CLAIM_COLUMNS:
        raise ValueError(f"Unexpected historical-claims schema in {path}")
    rows: list[dict[str, str]] = frame.sort_values("claim_id", kind="stable").to_dict(
        orient="records"
    )
    serialized: bytes = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def audited_claim_set_sha256(
    published_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]
) -> str:
    """Hash only immutable transcriptions of claims in the audited manuscript."""
    published_ids: list[str] = [
        str(row.get("reconciliation_id", "")) for row in published_rows
    ]
    paper_claim_rows: list[dict[str, Any]] = [
        row
        for row in claim_rows
        if str(row.get("source_location", "")).startswith("paper/")
    ]
    other_ids: list[str] = [str(row.get("claim_id", "")) for row in paper_claim_rows]
    if (
        not all(published_ids)
        or len(published_ids) != len(set(published_ids))
        or not all(other_ids)
        or len(other_ids) != len(set(other_ids))
    ):
        raise ValueError("Audited manuscript claim identifiers must be unique")
    table_fields: tuple[str, ...] = (
        "reconciliation_id",
        "paper_location",
        "benchmark",
        "pregrouper",
        "declared_scope",
        "declared_contrast",
        "reported_metric",
        "reported_statistic",
        "pair_set",
        "value_component",
        "display_value",
    )
    claim_fields: tuple[str, ...] = (
        "claim_id",
        "claim_group",
        "source_location",
        "benchmark",
        "pregrouper",
        "scope",
        "contrast",
        "representation",
        "metric",
        "statistic",
        "aggregation",
        "missingness_policy",
        "pair_set",
        "model_s",
        "model_t",
        "value_component",
        "expected_display",
        "expected_value",
        "expected_relation",
    )
    payload: dict[str, list[dict[str, str]]] = {
        "table_claims": sorted(
            (
                {field: str(row[field]) for field in table_fields}
                for row in published_rows
            ),
            key=lambda row: row["reconciliation_id"],
        ),
        "other_claims": sorted(
            (
                {field: str(row[field]) for field in claim_fields}
                for row in paper_claim_rows
            ),
            key=lambda row: row["claim_id"],
        ),
    }
    serialized: bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def manuscript_dispositions() -> list[dict[str, str]]:
    """Return explicit recommendations and unresolved choices for a revision."""
    rows: tuple[tuple[str, ...], ...] = (
        (
            "segment_scope",
            "Tables 1-2 and segment-level prose",
            "all segment-level fidelity and transfer values",
            "Paper prose describes user content; executed tables pool system and user segments.",
            "f_table_revision_candidate_{pairwise_drop,finite_extreme}.tsv",
            "undecided",
            "author_decision_required",
            "",
            "Choose a user-coordinate candidate before revising headline values.",
        ),
        (
            "anli_contrast",
            "ANLI methods and Table 2",
            "ANLI F_pred, F_attr, and attribution-target transfer values",
            "Paper prose describes entailment minus contradiction; executed tables use entailment minus neutral.",
            "f_table_revision_candidate_{pairwise_drop,finite_extreme}.tsv",
            "undecided",
            "author_decision_required",
            "",
            "Both ordered contrasts remain published; the candidates use entailment minus contradiction.",
        ),
        (
            "hosted_nonfinite_policy",
            "Methods and hosted scalar results",
            "all hosted-inclusive log-odds metrics",
            "Executed tables use pair-specific finite-row deletion; prose describes finite-extreme replacement.",
            "f_table_revision_candidate_{pairwise_drop,finite_extreme}.tsv",
            "undecided",
            "author_decision_required",
            "",
            "Select by estimand rationale and coverage, not observed effect size.",
        ),
        (
            "race_metric_label",
            "Table 2 RACE F_pred/F_attr",
            "RACE F_pred=.462 and F_attr=.524",
            "Printed values are scalar correct-vs-rest Pearson r-squared, not RV coefficients.",
            "race_rv_open.tsv and race_rv_paper_complete_case.tsv",
            "replace",
            "ready",
            "race_rv_open_full_coverage_or_paper_mnar_sensitivity",
            "Use genuine multivariate RV and state the hosted complete-case limitation.",
        ),
        (
            "boolq_word_attention_grid",
            "Table 2 BoolQ-word attention and attention-transfer cells",
            "BoolQ-word F_attn_* and F_attn_*_to_attr",
            "Historical sentence-level attention was joined to word ablations by integer segment index.",
            "corrected BoolQ-word rows in the revised-candidate tables",
            "replace",
            "ready",
            "corrected_boolq_word_full_dialog",
            "Do not use the contaminated historical cells as evidence.",
        ),
        (
            "cross_level_aggregation",
            "Tables 1-2 cross-level cells",
            "F_align_to_attr, F_mag_to_attr, and F_attn_*_to_attr",
            "The PDF values are row-pooled; a later notebook used prompt-equal means.",
            "f_table_prompt_equal_transfer.tsv",
            "retain_and_clarify",
            "ready",
            "historical_row_pooled_with_prompt_equal_sensitivity",
            "Name both estimands and do not conflate them.",
        ),
        (
            "main_text_pair_values",
            "Main-text prediction-fidelity examples",
            "three named BoolQ F_pred examples",
            "Two values reproduce legacy columns; the third provenance remains unresolved.",
            "historical_claims.tsv",
            "replace",
            "ready",
            "correct_original_prompt_logodds_values",
            "Replace the examples and disclose the unresolved third historical value.",
        ),
        (
            "q05_boolq_word_residual",
            "Corrected BoolQ-word reproducibility note",
            "Qwen-0.5B BoolQ-word attribution",
            "One corrected Qwen-0.5B attribution differs from the archived value for an unresolved historical reason.",
            "open_execution_audit_receipt.json",
            "disclose",
            "ready",
            "corrected_q05_new_artifact",
            "Treat archive similarity as a replication diagnostic, not a correctness oracle.",
        ),
    )
    return [
        dict(
            zip(
                MANUSCRIPT_DISPOSITION_COLUMNS,
                (issue_id, PAPER_REFERENCE_ID, *values),
            )
        )
        for issue_id, *values in rows
    ]


def _json_digest(value: dict[str, str]) -> str:
    payload: bytes = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_legacy_cohort(path: str) -> LegacyCohort:
    """Load the archive-prefix mapping kept outside the public source tree.

    The JSON object must contain ``archive_to_public_model`` plus ordered
    ``open_models`` and ``hosted_models`` arrays of public model identifiers.
    Archive-only routing aliases are consumed in memory and never written to a
    reconciliation artifact.
    """
    with open(path, encoding="utf-8") as source:
        payload: Any = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("Legacy model configuration must be a JSON object")
    mapping_raw: Any = payload.get("archive_to_public_model")
    open_raw: Any = payload.get("open_models")
    hosted_raw: Any = payload.get("hosted_models")
    if (
        not isinstance(mapping_raw, dict)
        or not isinstance(open_raw, list)
        or not isinstance(hosted_raw, list)
    ):
        raise ValueError(
            "Legacy model configuration requires archive_to_public_model, "
            "open_models, and hosted_models"
        )
    mapping: dict[str, str] = {
        str(prefix): str(model) for prefix, model in mapping_raw.items()
    }
    open_models: tuple[str, ...] = tuple(str(model) for model in open_raw)
    hosted_models: tuple[str, ...] = tuple(str(model) for model in hosted_raw)
    if len(open_models) != 5 or len(hosted_models) != 6:
        raise ValueError(
            "The historical paper cohort must contain 5 open and 6 hosted models"
        )
    all_models: tuple[str, ...] = open_models + hosted_models
    if len(set(all_models)) != 11 or set(mapping.values()) != set(all_models):
        raise ValueError(
            "Legacy model mapping must map one prefix to each of 11 public model IDs"
        )
    if len(mapping) != 11 or any(
        not prefix or not model for prefix, model in mapping.items()
    ):
        raise ValueError("Legacy model mapping must be one-to-one and nonempty")
    return LegacyCohort(mapping, open_models, hosted_models)


def _scope_and_contrast(benchmark: str, metric: str) -> tuple[str, str, str, str]:
    declared_scope: str = "not_applicable" if metric == "F_pred" else "user"
    executed_scope: str = "not_applicable" if metric == "F_pred" else "all"
    label_dependent: bool = (
        metric in {"F_pred", "F_attr", "F_align"} or metric in TRANSFER_METRICS
    )
    if benchmark.startswith("anli_") and label_dependent:
        return (
            declared_scope,
            executed_scope,
            "entailment_contradiction",
            "entailment_neutral",
        )
    if benchmark == "race" and metric in {"F_pred", "F_attr"}:
        return (
            declared_scope,
            executed_scope,
            "multiclass_rv",
            "answer_conditioned_correct_vs_rest",
        )
    if label_dependent:
        return declared_scope, executed_scope, "canonical", "canonical"
    return declared_scope, executed_scope, "not_applicable", "not_applicable"


def _historical_aggregation(metric: str) -> str:
    """Return the aggregation used by the cached paper table cells."""
    del metric
    return "row_pooled_then_model_pair_distribution"


def _corrected_aggregation(raw_aggregation: str) -> str:
    """Normalize an explicitly recorded corrected-table aggregation."""
    mapping: dict[str, str] = {
        "row_pooled": "row_pooled_then_model_pair_distribution",
        "prompt_equal_mean_r2": ("prompt_equal_mean_r2_then_model_pair_distribution"),
    }
    if raw_aggregation not in mapping:
        raise ValueError(f"Unknown corrected aggregation {raw_aggregation!r}")
    return mapping[raw_aggregation]


def _grid_id(benchmark: str, pregrouper: str, source_state: str, metric: str) -> str:
    if (
        benchmark == "boolq"
        and pregrouper == "word"
        and source_state in {"published", "archive"}
        and metric in BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
    ):
        return "boolq_word_historical_sentence_contaminated_attention"
    if benchmark == "race":
        return (
            "race_sentence_historical_136313"
            if source_state in {"published", "archive"}
            else "race_sentence_corrected_145544"
        )
    return f"{benchmark}_{pregrouper}_paper_full_dialog"


def _estimand_fields(row: dict[str, Any]) -> dict[str, str]:
    keys: tuple[str, ...] = (
        "benchmark",
        "pregrouper",
        "segment_grid_id",
        "executed_scope",
        "executed_contrast",
        "cohort",
        "pair_set",
        "metric",
        "statistic",
        "aggregation",
        "missingness_policy",
        "value_component",
        "n_pairs",
        "effective_pairs",
        "model_snapshot_identity",
    )
    return {key: str(row[key]) for key in keys}


def method_id(row: dict[str, Any]) -> str:
    """Return the estimand ID with model snapshot identity deliberately omitted."""
    fields: dict[str, str] = _estimand_fields(row)
    del fields["model_snapshot_identity"]
    return "method-v1:" + _json_digest(fields)[:20]


def _pair_id(model_s: str, model_t: str, directed: bool) -> str:
    if directed:
        return f"{model_s}->{model_t}"
    left, right = sorted((model_s, model_t))
    return f"{left}<->{right}"


def _expected_pairs(metric: str, pair_set: str) -> tuple[str, ...]:
    directed: bool = metric in TRANSFER_METRICS
    if directed:
        targets: tuple[str, ...] = {
            "open": OPEN,
            "open_to_hosted": HOSTED,
            "all": ALL,
        }[pair_set]
        return tuple(
            sorted(
                _pair_id(source, target, True)
                for source in OPEN
                for target in targets
                if source != target
            )
        )
    if metric in REPRESENTATION_METRICS:
        if pair_set != "open" and pair_set != "all":
            return ()
        members_a, members_b = OPEN, OPEN
    elif pair_set == "open":
        members_a, members_b = OPEN, OPEN
    elif pair_set == "open_to_hosted":
        members_a, members_b = OPEN, HOSTED
    else:
        members_a, members_b = ALL, ALL
    if members_a == members_b:
        return tuple(
            sorted(
                _pair_id(left, right, False)
                for index, left in enumerate(members_a)
                for right in members_a[index + 1 :]
            )
        )
    return tuple(
        sorted(
            _pair_id(left, right, False) for left in members_a for right in members_b
        )
    )


def _historical_expected_pairs(
    benchmark: str, pregrouper: str, metric: str, pair_set: str
) -> tuple[str, ...]:
    """Return the effective model pairs that actually populated a paper cell."""
    if benchmark == "boolq" and pregrouper == "word":
        representation_subset: tuple[str, ...] = OPEN[:-1]
        if metric in {"F_align", "F_mag"}:
            return tuple(
                sorted(
                    _pair_id(left, right, False)
                    for index, left in enumerate(representation_subset)
                    for right in representation_subset[index + 1 :]
                )
            )
        if metric in {"F_align_to_attr", "F_mag_to_attr"}:
            return tuple(
                sorted(
                    _pair_id(source, target, True)
                    for source in representation_subset
                    for target in ALL
                    if source != target
                )
            )
    if benchmark == "lambada" and pregrouper == "word":
        available: tuple[str, ...] = ALL[:-1]
        if metric in {"F_pred", "F_attr"}:
            return tuple(
                sorted(
                    _pair_id(left, right, False)
                    for index, left in enumerate(available)
                    for right in available[index + 1 :]
                )
            )
        if metric in TRANSFER_METRICS:
            return tuple(
                sorted(
                    _pair_id(source, target, True)
                    for source in OPEN
                    for target in available
                    if source != target
                )
            )
    return _expected_pairs(metric, pair_set)


def _encode_pairs(pairs: tuple[str, ...] | list[str]) -> str:
    return json.dumps(sorted(pairs), separators=(",", ":"))


def estimand_id(row: dict[str, Any]) -> str:
    """Return a stable identifier for all value-defining analysis choices."""
    fields: dict[str, str] = _estimand_fields(row)
    return "estimand-v1:" + _json_digest(fields)[:20]


def comparable_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    """Subtract two results only if their full estimands are identical.

    Raises:
        ValueError: If either estimand ID is absent or the IDs differ.
    """
    if not left.get("estimand_id") or left.get("estimand_id") != right.get(
        "estimand_id"
    ):
        raise ValueError("Cannot compare results from nonmatching estimands")
    return float(left["value"]) - float(right["value"])


def _corr_r2(x: pd.Series, y: pd.Series) -> float | None:
    pair: pd.DataFrame = (
        pd.DataFrame({"x": x.to_numpy(copy=False), "y": y.to_numpy(copy=False)})
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(pair) < 3 or pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return None
    value: float = float(pair["x"].corr(pair["y"]))
    return value * value if np.isfinite(value) else None


def _finite_correlation(
    x: np.ndarray[Any, Any], y: np.ndarray[Any, Any]
) -> float | None:
    """Return Pearson r for already finite one-dimensional arrays."""
    if len(x) < 3:
        return None
    x_centered: np.ndarray[Any, Any] = x - x.mean()
    y_centered: np.ndarray[Any, Any] = y - y.mean()
    denominator: float = float(
        np.sqrt(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered))
    )
    if denominator <= 0.0:
        return None
    return float(np.sum(x_centered * y_centered) / denominator)


def _finite_extreme_signal(signal: pd.Series) -> pd.Series:
    """Replace infinities just beyond a signal's model-specific finite range."""
    finite: pd.Series = signal.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return signal
    low: float = float(finite.min() - abs(finite.min()) * 0.01 - 0.01)
    high: float = float(finite.max() + abs(finite.max()) * 0.01 + 0.01)
    return signal.replace({np.inf: high, -np.inf: low})


def _legacy_signals(
    df: pd.DataFrame,
    cohort: LegacyCohort,
    finite_extreme_api: bool = False,
) -> dict[str, dict[str, pd.Series]]:
    signals: dict[str, dict[str, pd.Series]] = {metric: {} for metric in METRICS}
    for model in cohort.all_models:
        prefix: str = cohort.archive_prefix(model)
        pred_col: str = f"{prefix}_orig_logodds"
        if pred_col in df.columns and df[pred_col].notna().any():
            signals["F_pred"][model] = cast(
                pd.Series, df.groupby("prompt_idx", sort=False)[pred_col].first()
            )
        elif model in cohort.hosted_models and prefix in df.columns:
            signals["F_pred"][model] = cast(
                pd.Series, df.groupby("prompt_idx", sort=False)[prefix].first()
            )
        ablation_col: str = f"{prefix}_ablation"
        if ablation_col in df.columns and df[ablation_col].notna().any():
            signals["F_attr"][model] = df[ablation_col]
    suffixes: dict[str, str] = {
        "F_attn_rollout": "_attn_rollout",
        "F_attn_mean": "_attn_mean",
        "F_attn_max": "_attn_max",
        "F_mag": "_dn_postnorm",
    }
    for metric, suffix in suffixes.items():
        for model in cohort.open_models:
            column: str = f"{cohort.archive_prefix(model)}{suffix}"
            if column in df.columns and df[column].notna().any():
                signals[metric][model] = df[column]
    for model in cohort.open_models:
        prefix = cohort.archive_prefix(model)
        columns: tuple[str, str, str] = (
            f"{prefix}_wdz_postnorm",
            f"{prefix}_w_norm",
            f"{prefix}_dn_postnorm",
        )
        if all(column in df.columns for column in columns):
            signals["F_align"][model] = df[columns[0]] / (
                df[columns[1]] * df[columns[2]]
            )
    if finite_extreme_api:
        for metric_signals in signals.values():
            for model in cohort.hosted_models:
                if model in metric_signals:
                    metric_signals[model] = _finite_extreme_signal(
                        metric_signals[model]
                    )
    return signals


def _symmetric_values(
    signals: dict[str, pd.Series], models_a: tuple[str, ...], models_b: tuple[str, ...]
) -> dict[str, float]:
    if models_a == models_b:
        pairs: list[tuple[str, str]] = [
            (left, right)
            for index, left in enumerate(models_a)
            for right in models_a[index + 1 :]
        ]
    else:
        pairs = [(left, right) for left in models_a for right in models_b]
    values: dict[str, float] = {}
    for left, right in pairs:
        if left not in signals or right not in signals:
            continue
        x: pd.Series = signals[left]
        y: pd.Series = signals[right]
        if x.index.name == "prompt_idx" and y.index.name == "prompt_idx":
            common: pd.Index = x.index.intersection(y.index)
            x, y = x.loc[common], y.loc[common]
        value: float | None = _corr_r2(x, y)
        if value is not None:
            values[_pair_id(left, right, False)] = value
    return values


def _transfer_values(
    df: pd.DataFrame,
    cohort: LegacyCohort,
    signal: dict[str, pd.Series],
    source_models: tuple[str, ...],
    target_models: tuple[str, ...],
    signed: bool,
    aggregation: str = "row_pooled",
) -> dict[str, float]:
    if aggregation not in {"row_pooled", "prompt_equal_mean_r2"}:
        raise ValueError(f"Unknown historical transfer aggregation {aggregation!r}")
    values: dict[str, float] = {}
    for source in source_models:
        if source not in signal:
            continue
        source_values: np.ndarray[Any, Any] = signal[source].to_numpy(dtype=float)
        for target in target_models:
            if source == target:
                continue
            column: str = f"{cohort.archive_prefix(target)}_ablation"
            if column not in df.columns:
                continue
            target_values: np.ndarray[Any, Any] = df[column].to_numpy(dtype=float)
            if not signed:
                target_values = np.abs(target_values)
            finite: np.ndarray[Any, Any] = np.isfinite(source_values) & np.isfinite(
                target_values
            )
            if not finite.any():
                continue
            if aggregation == "row_pooled":
                correlation: float | None = _finite_correlation(
                    source_values[finite], target_values[finite]
                )
                if correlation is not None:
                    values[_pair_id(source, target, True)] = correlation * correlation
                continue
            prompt_frame: pd.DataFrame = pd.DataFrame(
                {
                    "prompt_idx": df["prompt_idx"].to_numpy()[finite],
                    "source": source_values[finite],
                    "target": target_values[finite],
                }
            )
            prompt_frame["source_square"] = prompt_frame["source"] ** 2
            prompt_frame["target_square"] = prompt_frame["target"] ** 2
            prompt_frame["cross"] = prompt_frame["source"] * prompt_frame["target"]
            moments: pd.DataFrame = prompt_frame.groupby("prompt_idx", sort=False).agg(
                n=("source", "size"),
                source_sum=("source", "sum"),
                target_sum=("target", "sum"),
                source_square=("source_square", "sum"),
                target_square=("target_square", "sum"),
                cross=("cross", "sum"),
            )
            covariance: pd.Series = (
                moments["cross"]
                - moments["source_sum"] * moments["target_sum"] / moments["n"]
            )
            source_variance: pd.Series = (
                moments["source_square"] - moments["source_sum"] ** 2 / moments["n"]
            )
            target_variance: pd.Series = (
                moments["target_square"] - moments["target_sum"] ** 2 / moments["n"]
            )
            denominator: pd.Series = np.sqrt(source_variance * target_variance)
            prompt_r2: pd.Series = (covariance / denominator) ** 2
            valid_prompts: pd.Series = (
                moments["n"].ge(3)
                & source_variance.gt(0.0)
                & target_variance.gt(0.0)
                & np.isfinite(prompt_r2)
            )
            if valid_prompts.any():
                values[_pair_id(source, target, True)] = float(
                    prompt_r2.loc[valid_prompts].mean()
                )
    return values


def recompute_archive(
    df: pd.DataFrame,
    cohort: LegacyCohort,
    metrics: tuple[str, ...] = METRICS,
    finite_extreme_api: bool = False,
    transfer_aggregation: str = "row_pooled",
) -> dict[str, dict[str, float]]:
    """Recompute every historical per-pair Pearson-r-squared estimand."""
    signals: dict[str, dict[str, pd.Series]] = _legacy_signals(
        df, cohort, finite_extreme_api
    )
    output: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if metric in TRANSFER_METRICS:
            base: str = metric.removesuffix("_to_attr")
            signed: bool = metric != "F_mag_to_attr"
            for pair_set, targets in (
                ("open", cohort.open_models),
                ("open_to_hosted", cohort.hosted_models),
                ("all", cohort.all_models),
            ):
                output[f"{metric}:{pair_set}"] = _transfer_values(
                    df,
                    cohort,
                    signals[base],
                    cohort.open_models,
                    targets,
                    signed,
                    transfer_aggregation,
                )
        else:
            for pair_set, left, right in (
                ("open", cohort.open_models, cohort.open_models),
                ("open_to_hosted", cohort.open_models, cohort.hosted_models),
                ("all", cohort.all_models, cohort.all_models),
            ):
                output[f"{metric}:{pair_set}"] = _symmetric_values(
                    signals[metric], left, right
                )
    return output


def _claim_row(
    *,
    claim_id: str,
    claim_group: str,
    source_location: str,
    source_state: str,
    benchmark: str,
    pregrouper: str,
    segment_grid_id: str,
    cohort: str,
    scope: str,
    contrast: str,
    representation: str,
    metric: str,
    statistic: str,
    aggregation: str,
    missingness_policy: str,
    pair_set: str,
    model_s: str,
    model_t: str,
    value_component: str,
    expected_display: str,
    expected_value: float,
    tolerance: float,
    actual_value: float,
    input_locators: list[str],
    expected_relation: str = "within_tolerance",
    n_pairs: int | str = "",
    n_observations: int | str = "",
    n_prompts: int | str = "",
    effective_pairs: list[str] | tuple[str, ...] = (),
    note: str = "",
) -> dict[str, Any]:
    """Build one self-describing, executable historical claim."""
    if expected_relation not in {"within_tolerance", "outside_tolerance"}:
        raise ValueError(f"Unknown historical claim relation {expected_relation!r}")
    finite_actual: bool = bool(np.isfinite(actual_value))
    difference: float = abs(actual_value - expected_value)
    return {
        "claim_id": claim_id,
        "claim_group": claim_group,
        "source_location": source_location,
        "source_state": source_state,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "segment_grid_id": segment_grid_id,
        "cohort": cohort,
        "scope": scope,
        "contrast": contrast,
        "representation": representation,
        "metric": metric,
        "statistic": statistic,
        "aggregation": aggregation,
        "missingness_policy": missingness_policy,
        "pair_set": pair_set,
        "model_s": model_s,
        "model_t": model_t,
        "value_component": value_component,
        "expected_display": expected_display,
        "expected_value": expected_value,
        "tolerance": tolerance,
        "expected_relation": expected_relation,
        "actual_value": actual_value if finite_actual else "",
        "n_pairs": n_pairs,
        "n_observations": n_observations,
        "n_prompts": n_prompts,
        "effective_pairs": _encode_pairs(effective_pairs),
        "input_locators": json.dumps(sorted(input_locators), separators=(",", ":")),
        "passed": finite_actual
        and (
            difference <= tolerance
            if expected_relation == "within_tolerance"
            else difference > tolerance
        ),
        "note": note,
    }


def _race_raw_paths(archive_dir: str) -> dict[str, tuple[str, str]]:
    """Resolve paper-era RACE JSONs to sanitized logical locators."""
    output: dict[str, tuple[str, str]] = {}
    for model, legacy_basename in RACE_RAW_ARCHIVE_BASENAMES.items():
        locator: str = f"archive/race_raw/{model}.json"
        neutral_path: str = os.path.join(archive_dir, "race_raw", f"{model}.json")
        legacy_path: str = os.path.join(archive_dir, legacy_basename)
        path: str = neutral_path if os.path.isfile(neutral_path) else legacy_path
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing archived RACE raw response for public model {model!r}"
            )
        output[model] = (locator, path)
    return output


def _label_values(value: Any) -> list[float]:
    """Return A--D log scores, using NaN for absent or invalid coordinates."""
    if not isinstance(value, dict):
        return [float("nan")] * 4
    output: list[float] = []
    for label in ("A", "B", "C", "D"):
        coordinate: Any = value.get(label)
        try:
            output.append(float(coordinate))
        except (TypeError, ValueError):
            output.append(float("nan"))
    return output


def _historical_race_signal(path: str) -> ModelSignals:
    """Load one archived raw RACE JSON into shared multiclass coordinates."""
    with open(path, encoding="utf-8") as source:
        payload: Any = json.load(source)
    if not isinstance(payload, list):
        raise ValueError(f"Archived RACE response is not a list: {path}")
    prompt_ids: list[int] = []
    original_values: list[list[float]] = []
    attribution_keys: list[tuple[int, int]] = []
    baseline_values: list[list[float]] = []
    ablated_values: list[list[float]] = []
    raw_attribution_keys: set[tuple[int, int]] = set()
    for result in payload:
        if not isinstance(result, dict):
            raise ValueError(f"Archived RACE response contains a non-object: {path}")
        prompt_idx: int = int(result["prompt_idx"])
        prompt_ids.append(prompt_idx)
        original: list[float] = _label_values(result.get("orig_label_logprobs"))
        original_values.append(original)
        n_segments: int = int(result.get("n_segments", 0))
        ablated_payload: Any = result.get("ablated_label_logprobs")
        if ablated_payload is None:
            ablated: list[Any] = []
        elif isinstance(ablated_payload, list):
            ablated = ablated_payload
        else:
            raise ValueError(f"Invalid archived RACE ablation list: {path}")
        if n_segments and len(ablated) != n_segments:
            raise ValueError(
                f"Prompt {prompt_idx} has {len(ablated)}/{n_segments} RACE ablations"
            )
        for segment_idx, values in enumerate(ablated):
            key: tuple[int, int] = (prompt_idx, segment_idx)
            raw_attribution_keys.add(key)
            candidate: list[float] = _label_values(values)
            if np.isfinite(original).all() and np.isfinite(candidate).all():
                attribution_keys.append(key)
                baseline_values.append(original)
                ablated_values.append(candidate)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError(f"Archived RACE response has duplicate prompts: {path}")
    original_array: np.ndarray = np.asarray(original_values, dtype=np.float64)
    original_finite: np.ndarray = np.isfinite(original_array).all(axis=1)
    prediction_vectors: np.ndarray = _prediction_features(
        original_array[original_finite]
    )
    prediction: pd.DataFrame = pd.DataFrame(
        prediction_vectors,
        index=np.asarray(prompt_ids, dtype=np.int64)[original_finite],
        columns=[f"vector_{index}" for index in range(prediction_vectors.shape[1])],
    )
    prediction.index.name = "prompt_idx"
    if baseline_values:
        attribution_vectors, magnitudes = _attribution_features(
            np.asarray(baseline_values, dtype=np.float64),
            np.asarray(ablated_values, dtype=np.float64),
        )
    else:
        attribution_vectors = np.empty((0, 3), dtype=np.float64)
        magnitudes = np.empty((0, 3), dtype=np.float64)
    attribution: pd.DataFrame = pd.DataFrame(
        np.column_stack((attribution_vectors, magnitudes)),
        index=pd.MultiIndex.from_tuples(
            attribution_keys, names=["prompt_idx", "seg_idx"]
        ),
        columns=[
            "vector_0",
            "vector_1",
            "vector_2",
            *MAGNITUDE_METRICS,
        ],
    )
    return ModelSignals(
        prediction=prediction,
        attribution=attribution,
        raw_prediction_keys=frozenset(prompt_ids),
        raw_attribution_keys=frozenset(raw_attribution_keys),
    )


def _race_pair_values(
    signals: dict[str, ModelSignals],
    models: tuple[str, ...],
    kind: str,
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Evaluate row-pooled historical multiclass metrics for every model pair."""
    if kind not in {"prediction", "attribution"}:
        raise ValueError(f"Unknown RACE signal kind {kind!r}")
    if kind == "prediction":
        allowed: set[int] | set[tuple[int, int]] = set().union(
            *(set(signals[model].raw_prediction_keys) for model in models)
        )
        scalar_names: tuple[str, ...] = ()
        metric_names: tuple[str, ...] = VECTOR_METRICS
    else:
        allowed = set().union(
            *(set(signals[model].raw_attribution_keys) for model in models)
        )
        scalar_names = MAGNITUDE_METRICS
        metric_names = (*VECTOR_METRICS, *MAGNITUDE_METRICS)
    values: dict[str, dict[str, float]] = {metric: {} for metric in metric_names}
    observations: dict[str, int] = {}
    for index, model_s in enumerate(models):
        for model_t in models[index + 1 :]:
            left: pd.DataFrame = getattr(signals[model_s], kind)
            right: pd.DataFrame = getattr(signals[model_t], kind)
            moments = _aligned_moments(left, right, allowed, scalar_names)
            estimates: dict[str, float] = _evaluate(
                moments,
                np.ones(len(moments.prompt_ids), dtype=np.float64),
                "row_pooled",
            )
            pair: str = _pair_id(model_s, model_t, False)
            observations[pair] = int(moments.counts.sum())
            for metric in metric_names:
                values[metric][pair] = estimates[metric]
    return values, observations


def _race_claim_rows(
    archive_dir: str,
    allow_unpinned_archive: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Recompute Figure 13 and expanded RACE claims from raw archived JSON."""
    resolved: dict[str, tuple[str, str]] = _race_raw_paths(archive_dir)
    checks: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for model, (locator, path) in resolved.items():
        del model
        inputs[locator] = path
        digest: str = _sha256(path)
        size: int = os.path.getsize(path)
        expected_digest: str = RECONCILIATION_RACE_RAW_SHA256[locator]
        expected_size: int = RECONCILIATION_RACE_RAW_SIZE_BYTES[locator]
        checks.extend(
            [
                {
                    "check_id": f"archive_byte_sha256:{locator.removeprefix('archive/')}",
                    "kind": "archive_provenance",
                    "locator": locator,
                    "expected": expected_digest,
                    "actual": digest,
                    "tolerance": 0,
                    "bypassed": allow_unpinned_archive and digest != expected_digest,
                    "passed": digest == expected_digest or allow_unpinned_archive,
                },
                {
                    "check_id": f"archive_size_bytes:{locator.removeprefix('archive/')}",
                    "kind": "archive_provenance",
                    "locator": locator,
                    "expected": expected_size,
                    "actual": size,
                    "tolerance": 0,
                    "bypassed": allow_unpinned_archive and size != expected_size,
                    "passed": size == expected_size or allow_unpinned_archive,
                },
            ]
        )
    signals: dict[str, ModelSignals] = {
        model: _historical_race_signal(path) for model, (_, path) in resolved.items()
    }
    input_locators: list[str] = sorted(inputs)
    rows: list[dict[str, Any]] = []
    paper_values: dict[str, dict[str, float]] = {}
    paper_observations: dict[str, dict[str, int]] = {}
    for kind in ("prediction", "attribution"):
        values, observations = _race_pair_values(signals, ALL, kind)
        paper_values[kind] = values["linear_cka"]
        paper_observations[kind] = observations
        pair_values: dict[str, float] = values["linear_cka"]
        rows.append(
            _claim_row(
                claim_id=f"paper.figure13.race.{kind}.rv.median",
                claim_group="race_figure13",
                source_location="paper/figure_13",
                source_state="archive",
                benchmark="race",
                pregrouper="sentence",
                segment_grid_id="race_sentence_historical_136313",
                cohort="paper_11",
                scope="not_applicable" if kind == "prediction" else "all",
                contrast="all_pairwise_logodds",
                representation="all_pairs",
                metric=f"F_{'pred' if kind == 'prediction' else 'attr'}_rv",
                statistic="rv",
                aggregation="row_pooled_then_model_pair_median",
                missingness_policy="pair_specific_complete_case",
                pair_set="all",
                model_s="",
                model_t="",
                value_component="median",
                expected_display=f"{FIGURE13_EXPECTATIONS[kind]:.6f}",
                expected_value=FIGURE13_EXPECTATIONS[kind],
                tolerance=0.0000005000001,
                actual_value=float(np.median(list(pair_values.values()))),
                input_locators=input_locators,
                n_pairs=len(pair_values),
                n_observations=sum(paper_observations[kind].values()),
                n_prompts=4934,
                effective_pairs=list(pair_values),
                note="Centered RV equals linear CKA on the shared centered subspace.",
            )
        )

    for kind in ("prediction", "attribution"):
        values, observations = _race_pair_values(signals, OPEN, kind)
        for metric, expected_values in EXPANDED_RACE_EXPECTATIONS.items():
            expected_kind, metric_name = metric
            if expected_kind != kind:
                continue
            pair_values = values[metric_name]
            actual_values: tuple[float, float, float] = (
                float(np.median(list(pair_values.values()))),
                float(np.min(list(pair_values.values()))),
                float(np.max(list(pair_values.values()))),
            )
            for component, expected, actual in zip(
                ("median", "min", "max"), expected_values, actual_values
            ):
                rows.append(
                    _claim_row(
                        claim_id=(
                            f"corrections.race_expanded.{kind}.{metric_name}."
                            f"{component}"
                        ),
                        claim_group="race_expanded_multiclass",
                        source_location="corrections/race_metric_and_prompt_version",
                        source_state="archive",
                        benchmark="race",
                        pregrouper="sentence",
                        segment_grid_id="race_sentence_historical_136313",
                        cohort="open_5",
                        scope="not_applicable" if kind == "prediction" else "all",
                        contrast="all_pairwise_logodds",
                        representation="shared_orthonormal_clr",
                        metric=metric_name,
                        statistic=(
                            "mean_cosine"
                            if metric_name == "direction_cosine"
                            else "cka" if metric_name == "linear_cka" else "pearson_r"
                        ),
                        aggregation="row_pooled_then_model_pair_distribution",
                        missingness_policy="strict_complete",
                        pair_set="open",
                        model_s="",
                        model_t="",
                        value_component=component,
                        expected_display=f"{expected:.3f}",
                        expected_value=expected,
                        tolerance=0.0005000001,
                        actual_value=actual,
                        input_locators=[resolved[model][0] for model in OPEN],
                        n_pairs=len(pair_values),
                        n_observations=sum(observations.values()),
                        n_prompts=4934,
                        effective_pairs=list(pair_values),
                    )
                )
    return rows, checks, inputs


def _claim_frame(
    path: str, cohort: LegacyCohort, transfer: bool = False
) -> pd.DataFrame:
    """Read only archived wide columns needed for scalar historical claims."""
    columns: list[str] = list(pd.read_csv(path, sep="\t", nrows=0).columns)
    wanted: set[str] = {"prompt_idx", "segment_idx", "n_segments"}
    suffixes: tuple[str, ...] = ("_orig_logodds", "_ablation")
    if transfer:
        suffixes += (
            "_w_norm",
            "_dn_postnorm",
            "_wdz_postnorm",
            "_attn_rollout",
            "_attn_mean",
            "_attn_max",
        )
    for model in cohort.all_models:
        prefix: str = cohort.archive_prefix(model)
        wanted.update(column for column in columns if column == prefix)
        wanted.update(f"{prefix}{suffix}" for suffix in suffixes)
    return pd.read_csv(path, sep="\t", usecols=lambda column: column in wanted)


def _boolq_word_attention_contamination_claims(
    archive_dir: str,
    cohort: LegacyCohort,
) -> list[dict[str, Any]]:
    """Prove that archived Qwen word attention was copied from sentence rows."""
    locators: list[str] = [
        "archive/boolq_word_consolidated.tsv",
        "archive/boolq_sentence_consolidated.tsv",
    ]
    qwen_models: tuple[str, ...] = tuple(
        model for model in cohort.open_models if model.startswith("qwen2.5-")
    )
    if len(qwen_models) != 4:
        raise ValueError("Expected four Qwen models in the historical open cohort")
    attention_columns: list[str] = [
        f"{cohort.archive_prefix(model)}_attn_{variant}"
        for model in qwen_models
        for variant in ("rollout", "mean", "max")
    ]
    coordinate_columns: list[str] = ["prompt_idx", "segment_idx"]
    word: pd.DataFrame = pd.read_csv(
        os.path.join(archive_dir, "boolq_word_consolidated.tsv"),
        sep="\t",
        usecols=[*coordinate_columns, *attention_columns],
    )
    sentence: pd.DataFrame = pd.read_csv(
        os.path.join(archive_dir, "boolq_sentence_consolidated.tsv"),
        sep="\t",
        usecols=[*coordinate_columns, *attention_columns],
    )
    word_keys: set[tuple[int, int]] = set(
        word[coordinate_columns].itertuples(index=False, name=None)
    )
    sentence_keys: set[tuple[int, int]] = set(
        sentence[coordinate_columns].itertuples(index=False, name=None)
    )
    overlap_keys: set[tuple[int, int]] = word_keys & sentence_keys
    masks_match: bool = True
    correlations: list[float] = []
    for column in attention_columns:
        finite_word: pd.DataFrame = word[
            np.isfinite(pd.to_numeric(word[column], errors="coerce"))
        ][[*coordinate_columns, column]]
        finite_keys: set[tuple[int, int]] = set(
            finite_word[coordinate_columns].itertuples(index=False, name=None)
        )
        masks_match = masks_match and finite_keys == overlap_keys
        paired: pd.DataFrame = finite_word.merge(
            sentence[[*coordinate_columns, column]],
            on=coordinate_columns,
            how="inner",
            validate="one_to_one",
            suffixes=("_word", "_sentence"),
        )
        correlation: float | None = _finite_correlation(
            paired[f"{column}_word"].to_numpy(dtype=float),
            paired[f"{column}_sentence"].to_numpy(dtype=float),
        )
        if correlation is None:
            raise ValueError(f"No finite BoolQ attention overlap for {column}")
        correlations.append(correlation)
    observed_coordinates: float = float(len(overlap_keys)) if masks_match else math.nan
    observed_prompts: float = (
        float(len({prompt_idx for prompt_idx, _ in overlap_keys}))
        if masks_match
        else math.nan
    )
    return [
        _claim_row(
            claim_id="corrections.boolq_word.qwen_attention.contaminated_coordinates",
            claim_group="boolq_word_attention_contamination",
            source_location="paper/table_2",
            source_state="archive",
            benchmark="boolq",
            pregrouper="word",
            segment_grid_id="boolq_word_historical_sentence_contaminated_attention",
            cohort="qwen_open_4",
            scope="all",
            contrast="not_applicable",
            representation="attention",
            metric="finite_coordinate_count",
            statistic="count",
            aggregation="all_qwen_attention_columns_exact_same_mask",
            missingness_policy="finite_archive_values",
            pair_set="not_applicable",
            model_s="",
            model_t="",
            value_component="coordinates",
            expected_display="546",
            expected_value=546.0,
            tolerance=0.0,
            actual_value=observed_coordinates,
            input_locators=locators,
            n_observations=int(observed_coordinates) if masks_match else "",
            n_prompts=int(observed_prompts) if masks_match else "",
            note=BOOLQ_WORD_ATTENTION_NOTE,
        ),
        _claim_row(
            claim_id="corrections.boolq_word.qwen_attention.sentence_match_min_pearson",
            claim_group="boolq_word_attention_contamination",
            source_location="paper/table_2",
            source_state="archive",
            benchmark="boolq",
            pregrouper="word",
            segment_grid_id="boolq_word_historical_sentence_contaminated_attention",
            cohort="qwen_open_4",
            scope="all",
            contrast="not_applicable",
            representation="attention",
            metric="sentence_match",
            statistic="pearson_r",
            aggregation="minimum_over_4_models_x_3_attention_variants",
            missingness_policy="finite_archive_values",
            pair_set="not_applicable",
            model_s="",
            model_t="",
            value_component="minimum",
            expected_display="0.996761721984",
            expected_value=0.9967617219843782,
            tolerance=1e-12,
            actual_value=float(min(correlations)),
            input_locators=locators,
            n_pairs=len(correlations),
            n_observations=len(overlap_keys) * len(correlations),
            n_prompts=len({prompt_idx for prompt_idx, _ in overlap_keys}),
            note=BOOLQ_WORD_ATTENTION_NOTE,
        ),
    ]


def _main_text_pair_claims(
    frame: pd.DataFrame,
    cohort: LegacyCohort,
    locator: str,
) -> list[dict[str, Any]]:
    """Recompute the three main-text BoolQ pair examples under both columns."""
    rows: list[dict[str, Any]] = []
    for (
        model_s,
        model_t,
        legacy_expected,
        corrected_expected,
    ) in MAIN_TEXT_PAIR_EXPECTATIONS:
        pair: str = _pair_id(model_s, model_t, False)
        for variant, expected in (
            ("legacy_hosted_column", legacy_expected),
            ("original_prompt_logodds", corrected_expected),
        ):
            expected_relation: str = (
                "outside_tolerance"
                if variant == "legacy_hosted_column"
                and model_s == "llama3.1-70b-instruct"
                and model_t == "qwen2.5-14b-instruct"
                else "within_tolerance"
            )
            series: list[pd.Series] = []
            for model in (model_s, model_t):
                prefix: str = cohort.archive_prefix(model)
                column: str = (
                    prefix
                    if variant == "legacy_hosted_column"
                    and model in cohort.hosted_models
                    and prefix in frame.columns
                    else f"{prefix}_orig_logodds"
                )
                series.append(
                    cast(
                        pd.Series,
                        frame.groupby("prompt_idx", sort=False)[column].first(),
                    )
                )
            paired: pd.DataFrame = (
                pd.concat(series, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
            )
            correlation: float | None = _finite_correlation(
                paired.iloc[:, 0].to_numpy(dtype=float),
                paired.iloc[:, 1].to_numpy(dtype=float),
            )
            if correlation is None:
                raise ValueError(f"No finite observations for main-text pair {pair}")
            rows.append(
                _claim_row(
                    claim_id=f"paper.main_text.boolq.F_pred.{pair}.{variant}",
                    claim_group="main_text_pair_values",
                    source_location="paper/main_text",
                    source_state="archive",
                    benchmark="boolq",
                    pregrouper="sentence",
                    segment_grid_id="boolq_sentence_paper_full_dialog",
                    cohort="single_pair",
                    scope="not_applicable",
                    contrast="canonical",
                    representation="scalar_logodds",
                    metric="F_pred",
                    statistic="pearson_r2",
                    aggregation="prompt_level_single_pair",
                    missingness_policy="pairwise_drop_nonfinite",
                    pair_set="single_pair",
                    model_s=model_s,
                    model_t=model_t,
                    value_component="point",
                    expected_display=f"{expected:.3f}",
                    expected_value=expected,
                    tolerance=0.0005000001,
                    actual_value=correlation * correlation,
                    input_locators=[locator],
                    expected_relation=expected_relation,
                    n_pairs=1,
                    n_observations=len(paired),
                    n_prompts=len(paired),
                    effective_pairs=[pair],
                    note=(
                        "Tested legacy-column hypothesis; it does not reproduce "
                        "the prose example."
                        if variant == "legacy_hosted_column"
                        and expected_relation == "outside_tolerance"
                        else (
                            "Legacy hosted score column used by the prose example."
                            if variant == "legacy_hosted_column"
                            else "Correct original-prompt log-odds column."
                        )
                    ),
                )
            )
        if model_s == "llama3.1-70b-instruct" and model_t == "qwen2.5-14b-instruct":
            source_prefix: str = cohort.archive_prefix(model_s)
            target_prefix: str = cohort.archive_prefix(model_t)
            source_column: str = (
                source_prefix
                if source_prefix in frame.columns
                else f"{source_prefix}_orig_logodds"
            )
            row_weighted: pd.DataFrame = (
                frame[[source_column, f"{target_prefix}_orig_logodds"]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            row_correlation: float | None = _finite_correlation(
                row_weighted.iloc[:, 0].to_numpy(dtype=float),
                row_weighted.iloc[:, 1].to_numpy(dtype=float),
            )
            if row_correlation is None:
                raise ValueError(f"No finite row-weighted observations for {pair}")
            rows.append(
                _claim_row(
                    claim_id=f"paper.main_text.boolq.F_pred.{pair}.legacy_row_weighted",
                    claim_group="main_text_pair_values",
                    source_location="paper/main_text",
                    source_state="archive",
                    benchmark="boolq",
                    pregrouper="sentence",
                    segment_grid_id="boolq_sentence_paper_full_dialog",
                    cohort="single_pair",
                    scope="not_applicable",
                    contrast="canonical",
                    representation="scalar_logodds",
                    metric="F_pred",
                    statistic="pearson_r2",
                    aggregation="segment_row_weighted_single_pair",
                    missingness_policy="pairwise_drop_nonfinite",
                    pair_set="single_pair",
                    model_s=model_s,
                    model_t=model_t,
                    value_component="point",
                    expected_display=".684",
                    expected_value=0.684,
                    tolerance=0.0005000001,
                    actual_value=row_correlation * row_correlation,
                    input_locators=[locator],
                    expected_relation="outside_tolerance",
                    n_pairs=1,
                    n_observations=len(row_weighted),
                    n_prompts=frame["prompt_idx"].nunique(),
                    effective_pairs=[pair],
                    note=(
                        "Ruled-out alternative: weighting repeated segment rows also "
                        "does not reproduce the paper-reported value."
                    ),
                )
            )
    return rows


def _finite_extreme_claims(
    archive_dir: str,
    cohort: LegacyCohort,
) -> list[dict[str, Any]]:
    """Recompute the paper-executed and prose-described non-finite policies."""
    rows: list[dict[str, Any]] = []
    specs: dict[tuple[str, str], BenchmarkSpec] = {
        (spec.benchmark, spec.pregrouper): spec for spec in BENCHMARKS
    }
    for config, expected_values in FINITE_EXTREME_EXPECTATIONS.items():
        benchmark, pregrouper = config
        spec: BenchmarkSpec = specs[config]
        path: str = os.path.join(archive_dir, spec.filename)
        locator: str = f"archive/{spec.filename}"
        frame: pd.DataFrame = _claim_frame(path, cohort)
        by_policy: tuple[tuple[str, bool, tuple[float, float]], ...] = (
            ("pairwise_drop_nonfinite", False, expected_values[:2]),
            ("model_specific_finite_extreme_replacement", True, expected_values[2:]),
        )
        for policy, finite_extreme, expected_pair in by_policy:
            values: dict[str, dict[str, float]] = recompute_archive(
                frame,
                cohort,
                ("F_pred", "F_attr"),
                finite_extreme_api=finite_extreme,
            )
            for metric, expected in zip(("F_pred", "F_attr"), expected_pair):
                pair_values: dict[str, float] = values[f"{metric}:all"]
                rows.append(
                    _claim_row(
                        claim_id=(
                            f"corrections.nonfinite.{benchmark}.{pregrouper}."
                            f"{metric}.{policy}"
                        ),
                        claim_group="hosted_nonfinite_policy",
                        source_location="corrections/hosted_nonfinite_policy",
                        source_state="archive",
                        benchmark=benchmark,
                        pregrouper=pregrouper,
                        segment_grid_id=f"{benchmark}_{pregrouper}_paper_full_dialog",
                        cohort="paper_11",
                        scope="not_applicable" if metric == "F_pred" else "all",
                        contrast=(
                            "entailment_neutral"
                            if benchmark.startswith("anli_")
                            else "canonical"
                        ),
                        representation="scalar_logodds",
                        metric=metric,
                        statistic="pearson_r2",
                        aggregation="model_pair_median",
                        missingness_policy=policy,
                        pair_set="all",
                        model_s="",
                        model_t="",
                        value_component="median",
                        expected_display=f"{expected:.6f}",
                        expected_value=expected,
                        tolerance=0.0000005000001,
                        actual_value=float(np.median(list(pair_values.values()))),
                        input_locators=[locator],
                        n_pairs=len(pair_values),
                        effective_pairs=list(pair_values),
                    )
                )
    return rows


def _prompt_equal_claims(
    frame: pd.DataFrame,
    cohort: LegacyCohort,
    locator: str,
) -> list[dict[str, Any]]:
    """Recompute later-notebook BoolQ prompt-equal transfer summaries."""
    values: dict[str, dict[str, float]] = recompute_archive(
        frame,
        cohort,
        tuple(PROMPT_EQUAL_EXPECTATIONS),
        transfer_aggregation="prompt_equal_mean_r2",
    )
    rows: list[dict[str, Any]] = []
    for metric, expected in PROMPT_EQUAL_EXPECTATIONS.items():
        pair_values: dict[str, float] = values[f"{metric}:all"]
        rows.append(
            _claim_row(
                claim_id=f"corrections.prompt_equal.boolq.sentence.{metric}.median",
                claim_group="cross_level_aggregation",
                source_location="corrections/cross_level_aggregation",
                source_state="archive",
                benchmark="boolq",
                pregrouper="sentence",
                segment_grid_id="boolq_sentence_paper_full_dialog",
                cohort="paper_11",
                scope="all",
                contrast="canonical",
                representation="scalar",
                metric=metric,
                statistic="pearson_r2",
                aggregation="prompt_equal_mean_r2_then_model_pair_median",
                missingness_policy="pairwise_drop_nonfinite",
                pair_set="all",
                model_s="",
                model_t="",
                value_component="median",
                expected_display=f"{expected:.3f}",
                expected_value=expected,
                tolerance=0.0005000001,
                actual_value=float(np.median(list(pair_values.values()))),
                input_locators=[locator],
                n_pairs=len(pair_values),
                effective_pairs=list(pair_values),
            )
        )
    return rows


def _race_prompt_segment_count(
    row: pd.Series, normalize_punctuation: bool, system_segments: int
) -> int:
    """Reconstruct one historical or punctuation-normalized RACE prompt size."""

    def normalize(value: Any) -> str:
        text: str = str(value).rstrip()
        if not normalize_punctuation or not text or text[-1] in ".!?;":
            return text
        return text + "."

    prompt_text: str = (
        f"Article:\n{row['article']}\n\n"
        f"Question:\n{normalize(row['question'])}\n\n"
        f"A. {normalize(row['A'])}\n"
        f"B. {normalize(row['B'])}\n"
        f"C. {normalize(row['C'])}\n"
        f"D. {normalize(row['D'])}"
    )
    return system_segments + len(segment_text(prompt_text, level="sentence")[0])


def _validate_race_grid_structure(
    historical: pd.DataFrame,
    corrected: pd.DataFrame,
    expected_prompt_ids: set[int],
) -> None:
    """Verify key, metadata, template, and message-scope RACE invariants."""
    historical_counts: pd.Series = historical.groupby("prompt_idx").size()
    corrected_counts: pd.Series = corrected.groupby("prompt_idx").size()
    historical_prompt_ids: set[int] = set(historical_counts.index.astype(int))
    corrected_prompt_ids: set[int] = set(corrected_counts.index.astype(int))
    if historical_prompt_ids != expected_prompt_ids:
        raise ValueError("Historical RACE prompt IDs disagree with the expected grid")
    if corrected_prompt_ids != expected_prompt_ids:
        raise ValueError("Corrected RACE prompt IDs disagree with the expected grid")
    historical_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(segment_idx))
        for prompt_idx, segment_idx in historical[
            ["prompt_idx", "segment_idx"]
        ].itertuples(index=False, name=None)
    }
    corrected_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in corrected[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }
    if len(historical_keys) != len(historical):
        raise ValueError("Historical RACE coordinate keys are not unique")
    if len(corrected_keys) != len(corrected):
        raise ValueError("Corrected RACE coordinate keys are not unique")
    if not historical_keys.issubset(corrected_keys):
        raise ValueError("Historical RACE coordinates are not retained")

    historical_declared_unique: pd.Series = historical.groupby("prompt_idx")[
        "n_segments"
    ].nunique()
    corrected_declared_unique: pd.Series = corrected.groupby("prompt_idx")[
        "n_segments"
    ].nunique()
    if (
        not historical_declared_unique.eq(1).all()
        or not corrected_declared_unique.eq(1).all()
    ):
        raise ValueError("RACE per-prompt n_segments metadata is inconsistent")
    historical_declared: pd.Series = historical.groupby("prompt_idx")[
        "n_segments"
    ].first()
    corrected_declared: pd.Series = corrected.groupby("prompt_idx")[
        "n_segments"
    ].first()
    if not historical_counts.eq(historical_declared.astype(int)).all():
        raise ValueError("Historical RACE coordinate counts disagree with metadata")
    if not corrected_counts.eq(corrected_declared.astype(int)).all():
        raise ValueError("Corrected RACE coordinate counts disagree with metadata")

    metadata_columns: list[str] = [
        "prompt_idx",
        "article",
        "question",
        "A",
        "B",
        "C",
        "D",
    ]
    prompt_metadata: pd.DataFrame = historical[metadata_columns].drop_duplicates()
    if prompt_metadata["prompt_idx"].duplicated().any():
        raise ValueError("Historical RACE prompt text is inconsistent across rows")
    prompt_metadata = prompt_metadata.set_index("prompt_idx").sort_index()
    system_segments: int = len(
        segment_text(RACE_CONFIG.system_prompt, level="sentence")[0]
    )
    reconstructed_historical: pd.Series = prompt_metadata.apply(
        lambda row: _race_prompt_segment_count(row, False, system_segments), axis=1
    ).astype(int)
    reconstructed_corrected: pd.Series = prompt_metadata.apply(
        lambda row: _race_prompt_segment_count(row, True, system_segments), axis=1
    ).astype(int)
    if not reconstructed_historical.eq(historical_counts.astype(int)).all():
        raise ValueError("Historical RACE prompt template was not reconstructed")
    if not reconstructed_corrected.eq(corrected_counts.astype(int)).all():
        raise ValueError("Corrected RACE prompt template was not reconstructed")
    if (corrected_counts.astype(int) - historical_counts.astype(int) < 0).any():
        raise ValueError("A corrected RACE prompt unexpectedly lost segments")

    system_rows: pd.DataFrame = corrected[
        corrected["message_role"].astype(str).eq("system")
    ]
    system_counts: pd.Series = system_rows.groupby("prompt_idx").size()
    if (
        system_segments != 3
        or len(system_rows) != system_segments * len(expected_prompt_ids)
        or set(system_counts.index.astype(int)) != expected_prompt_ids
        or not system_counts.eq(system_segments).all()
    ):
        raise ValueError("Corrected RACE system-message grid is not three per prompt")


def _race_grid_claims(
    archive_dir: str,
    corrected_results_dir: str,
    enforce_gold_grid: bool,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Recompute every numerical RACE prompt-version statement."""
    archive_path: str = os.path.join(archive_dir, "race_sentence_consolidated.tsv")
    archive_locator: str = "archive/race_sentence_consolidated.tsv"
    corrected_path: str = os.path.join(
        corrected_results_dir, "race", "sentence", "segments.tsv.gz"
    )
    corrected_locator: str = "corrected/race/sentence/segments.tsv.gz"
    if not os.path.isfile(corrected_path):
        raise FileNotFoundError(corrected_path)
    historical_columns: list[str] = ["prompt_idx", "segment_idx", "n_segments"]
    corrected_columns: list[str] = ["prompt_idx", "seg_idx", "n_segments"]
    if enforce_gold_grid:
        historical_columns.extend(["article", "question", "A", "B", "C", "D"])
        corrected_columns.append("message_role")
    historical: pd.DataFrame = pd.read_csv(
        archive_path, sep="\t", usecols=historical_columns
    )
    corrected: pd.DataFrame = pd.read_csv(
        corrected_path, sep="\t", usecols=corrected_columns
    )
    historical_counts: pd.Series = historical.groupby("prompt_idx").size()
    historical_declared: pd.Series = historical.groupby("prompt_idx")[
        "n_segments"
    ].first()
    corrected_counts: pd.Series = corrected.groupby("prompt_idx").size()
    common: pd.Index = historical_counts.index.intersection(corrected_counts.index)
    historical_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(segment_idx))
        for prompt_idx, segment_idx in historical[
            ["prompt_idx", "segment_idx"]
        ].itertuples(index=False, name=None)
    }
    corrected_keys: set[tuple[int, int]] = {
        (int(prompt_idx), int(seg_idx))
        for prompt_idx, seg_idx in corrected[["prompt_idx", "seg_idx"]].itertuples(
            index=False, name=None
        )
    }
    omitted_by_integer_join: int = len(corrected_keys - historical_keys)
    if enforce_gold_grid:
        _validate_race_grid_structure(historical, corrected, set(range(4_934)))
        count_delta: pd.Series = corrected_counts.astype(
            int
        ) - historical_counts.astype(int)
        if (
            len(historical) != 136_313
            or len(corrected) != 145_544
            or omitted_by_integer_join != 9_231
            or int(count_delta.ne(0).sum()) != 3_182
        ):
            raise ValueError("RACE prompt-version difference no longer reproduces")
    values: tuple[tuple[str, str, float, float, str], ...] = (
        ("historical_rows", "136,313", 136313.0, float(len(historical)), "archive"),
        ("corrected_rows", "145,544", 145544.0, float(len(corrected)), "corrected"),
        (
            "historical_prompt_counts_reproduced",
            "4,934",
            4934.0,
            float((historical_counts == historical_declared).sum()),
            "archive",
        ),
        (
            "rows_omitted_by_integer_join",
            "9,231",
            9231.0,
            float(omitted_by_integer_join),
            "archive_and_corrected",
        ),
        (
            "prompts_with_different_segment_counts",
            "3,182",
            3182.0,
            float(
                (historical_counts.loc[common] != corrected_counts.loc[common]).sum()
            ),
            "archive_and_corrected",
        ),
    )
    rows: list[dict[str, Any]] = []
    for claim_name, display, expected, actual, state in values:
        locators: list[str] = (
            [archive_locator]
            if state == "archive"
            else (
                [corrected_locator]
                if state == "corrected"
                else [archive_locator, corrected_locator]
            )
        )
        rows.append(
            _claim_row(
                claim_id=f"corrections.race_grid.{claim_name}",
                claim_group="race_prompt_version",
                source_location="corrections/race_metric_and_prompt_version",
                source_state=state,
                benchmark="race",
                pregrouper="sentence",
                segment_grid_id="race_historical_vs_corrected",
                cohort="not_applicable",
                scope="all",
                contrast="not_applicable",
                representation="segment_coordinates",
                metric=claim_name,
                statistic="count",
                aggregation="count",
                missingness_policy="not_applicable",
                pair_set="not_applicable",
                model_s="",
                model_t="",
                value_component="count",
                expected_display=display,
                expected_value=expected,
                tolerance=0.0,
                actual_value=actual,
                input_locators=locators,
                n_observations=int(actual),
                n_prompts=len(common),
            )
        )
    return rows, {corrected_locator: corrected_path}


def _race_w_norm_claims(
    archive_dir: str,
    corrected_results_dir: str,
    cohort: LegacyCohort,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Bind every corrected open RACE run to an archive-stable model invariant."""
    archive_path: str = os.path.join(archive_dir, "race_sentence_consolidated.tsv")
    archive_locator: str = "archive/race_sentence_consolidated.tsv"
    archive_columns: dict[str, str] = {
        model: f"{cohort.archive_prefix(model)}_w_norm" for model in cohort.open_models
    }
    archive: pd.DataFrame = pd.read_csv(
        archive_path,
        sep="\t",
        usecols=["prompt_idx", *archive_columns.values()],
    )
    rows: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    expected_prompt_ids: set[int] = set(range(4_934))
    for model, archive_column in archive_columns.items():
        corrected_path: str = os.path.join(
            corrected_results_dir,
            "race",
            "sentence",
            f"{model}_segment.tsv.gz",
        )
        corrected_locator: str = f"corrected/race/sentence/{model}_segment.tsv.gz"
        corrected: pd.DataFrame = pd.read_csv(
            corrected_path,
            sep="\t",
            usecols=["prompt_idx", "w_norm"],
        )
        archive_unique: pd.Series = archive.groupby("prompt_idx")[
            archive_column
        ].nunique(dropna=False)
        corrected_unique: pd.Series = corrected.groupby("prompt_idx")["w_norm"].nunique(
            dropna=False
        )
        if not archive_unique.eq(1).all() or not corrected_unique.eq(1).all():
            raise ValueError(f"RACE w_norm is not prompt-constant for {model}")
        archive_by_prompt: pd.Series = archive.groupby("prompt_idx", sort=True)[
            archive_column
        ].first()
        corrected_by_prompt: pd.Series = corrected.groupby("prompt_idx", sort=True)[
            "w_norm"
        ].first()
        if set(
            archive_by_prompt.index.astype(int)
        ) != expected_prompt_ids or not archive_by_prompt.index.equals(
            corrected_by_prompt.index
        ):
            raise ValueError(f"RACE w_norm prompt grid disagrees for {model}")
        archive_values: np.ndarray = pd.to_numeric(
            archive_by_prompt, errors="coerce"
        ).to_numpy(float)
        corrected_values: np.ndarray = pd.to_numeric(
            corrected_by_prompt, errors="coerce"
        ).to_numpy(float)
        if (
            not np.isfinite(archive_values).all()
            or not np.isfinite(corrected_values).all()
        ):
            raise ValueError(f"RACE w_norm contains non-finite values for {model}")
        maximum_error: float = float(np.max(np.abs(corrected_values - archive_values)))
        rows.append(
            _claim_row(
                claim_id=f"corrections.race_grid.{model}.w_norm_max_abs_error",
                claim_group="race_prompt_version_model_invariant",
                source_location="corrections/race_metric_and_prompt_version",
                source_state="archive_and_corrected",
                benchmark="race",
                pregrouper="sentence",
                segment_grid_id="race_historical_vs_corrected_prompt_identity",
                cohort="open_5",
                scope="all",
                contrast="answer_conditioned_correct_vs_rest",
                representation="readout_norm",
                metric="w_norm",
                statistic="max_abs_error",
                aggregation="first_per_prompt",
                missingness_policy="require_finite_complete_prompt_grid",
                pair_set="not_applicable",
                model_s=model,
                model_t="",
                value_component="maximum",
                expected_display="0",
                expected_value=0.0,
                tolerance=1e-9,
                actual_value=maximum_error,
                input_locators=[archive_locator, corrected_locator],
                n_observations=len(corrected_by_prompt),
                n_prompts=len(corrected_by_prompt),
                note=(
                    "The prompt text changed, but the answer-conditioned linear "
                    "readout norm is prompt-version independent and exactly binds "
                    "the corrected run to the archived model direction."
                ),
            )
        )
        inputs[corrected_locator] = corrected_path
    return rows, inputs


def _race_scalar_claims(
    archive_dir: str,
    cohort: LegacyCohort,
) -> list[dict[str, Any]]:
    """Expose the two scalar RACE Table 2 cells in the claims ledger."""
    spec: BenchmarkSpec = next(item for item in BENCHMARKS if item.benchmark == "race")
    locator: str = f"archive/{spec.filename}"
    frame: pd.DataFrame = _claim_frame(os.path.join(archive_dir, spec.filename), cohort)
    values: dict[str, dict[str, float]] = recompute_archive(
        frame, cohort, ("F_pred", "F_attr")
    )
    rows: list[dict[str, Any]] = []
    for metric in ("F_pred", "F_attr"):
        expected: float = HISTORICAL_NUMERIC_SENTINELS[("race", "sentence", metric)]
        pair_values: dict[str, float] = values[f"{metric}:all"]
        rows.append(
            _claim_row(
                claim_id=f"paper.table2.race.{metric}.scalar_median",
                claim_group="race_scalar_metric_label",
                source_location="paper/table_2",
                source_state="archive",
                benchmark="race",
                pregrouper="sentence",
                segment_grid_id="race_sentence_historical_136313",
                cohort="paper_11",
                scope="not_applicable" if metric == "F_pred" else "all",
                contrast="answer_conditioned_correct_vs_rest",
                representation="scalar_logodds",
                metric=metric,
                statistic="pearson_r2",
                aggregation="row_pooled_then_model_pair_median",
                missingness_policy="pairwise_drop_nonfinite",
                pair_set="all",
                model_s="",
                model_t="",
                value_component="median",
                expected_display=f"{expected:.10f}",
                expected_value=expected,
                tolerance=1e-12,
                actual_value=float(np.median(list(pair_values.values()))),
                input_locators=[locator],
                n_pairs=len(pair_values),
                effective_pairs=list(pair_values),
                note="Scalar compatibility result mislabeled as RV in Table 2.",
            )
        )
    return rows


def _historical_claim_rows(
    archive_dir: str,
    corrected_results_dir: str,
    cohort: LegacyCohort,
    allow_unpinned_archive: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Recompute all exact numerical claims outside Tables 1 and 2."""
    race_rows, provenance_checks, raw_inputs = _race_claim_rows(
        archive_dir, allow_unpinned_archive
    )
    grid_rows, corrected_inputs = _race_grid_claims(
        archive_dir,
        corrected_results_dir,
        enforce_gold_grid=not allow_unpinned_archive,
    )
    if not allow_unpinned_archive:
        race_w_norm_rows, race_w_norm_inputs = _race_w_norm_claims(
            archive_dir,
            corrected_results_dir,
            cohort,
        )
        grid_rows.extend(race_w_norm_rows)
        corrected_inputs.update(race_w_norm_inputs)
    boolq_spec: BenchmarkSpec = BENCHMARKS[0]
    boolq_locator: str = f"archive/{boolq_spec.filename}"
    boolq_frame: pd.DataFrame = _claim_frame(
        os.path.join(archive_dir, boolq_spec.filename), cohort, transfer=True
    )
    rows: list[dict[str, Any]] = [
        *race_rows,
        *grid_rows,
        *_race_scalar_claims(archive_dir, cohort),
    ]
    rows.extend(_main_text_pair_claims(boolq_frame, cohort, boolq_locator))
    rows.extend(_finite_extreme_claims(archive_dir, cohort))
    rows.extend(_prompt_equal_claims(boolq_frame, cohort, boolq_locator))
    if not allow_unpinned_archive:
        rows.extend(_boolq_word_attention_contamination_claims(archive_dir, cohort))
    identifiers: list[str] = [str(row["claim_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Historical claim identifiers are not unique")
    claim_checks: list[dict[str, Any]] = [
        {
            "check_id": f"historical_claim:{row['claim_id']}",
            "kind": "historical_claim",
            "locator": "reconciliation/historical_claims.tsv",
            "expected": row["expected_value"],
            "actual": row["actual_value"],
            "tolerance": row["tolerance"],
            "passed": row["passed"],
        }
        for row in rows
    ]
    return (
        rows,
        [*provenance_checks, *claim_checks],
        {
            **raw_inputs,
            **corrected_inputs,
        },
    )


def _base_row(
    *,
    source_state: str,
    source_locator: str,
    source_sha256: str,
    paper_location: str,
    benchmark: str,
    pregrouper: str,
    pair_set: str,
    metric: str,
    aggregation: str,
    value_component: str,
    value: float | None,
    display_value: str,
    n_pairs: int | None,
    effective_pairs: tuple[str, ...] | list[str],
    data_status: str,
    model_identity_status: str | None = None,
    model_snapshot_identity: str | None = None,
) -> dict[str, Any]:
    declared_scope, executed_scope, declared_contrast, executed_contrast = (
        _scope_and_contrast(benchmark, metric)
    )
    identity_status: str = model_identity_status or (
        "version_not_verifiable"
        if source_state == "corrected"
        else "historical_archived_snapshot"
    )
    snapshot_identity: str = model_snapshot_identity or (
        "unverified" if source_state == "corrected" else "historical_archived_snapshot"
    )
    notes: list[str] = []
    if benchmark == "race":
        notes.append(RACE_NOTE)
    boolq_word_attention_contaminated: bool = (
        benchmark == "boolq"
        and pregrouper == "word"
        and source_state in {"published", "archive"}
        and metric in BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
    )
    if boolq_word_attention_contaminated:
        notes.append(BOOLQ_WORD_ATTENTION_NOTE)
    if source_state == "corrected" and identity_status == "version_not_verifiable":
        notes.append(MODEL_IDENTITY_NOTE)
    specification_mismatches: list[str] = []
    if declared_scope != executed_scope:
        specification_mismatches.append("scope")
    if declared_contrast != executed_contrast and not (
        benchmark == "race" and metric in {"F_pred", "F_attr"}
    ):
        specification_mismatches.append("contrast")
    race_metric_mislabeled: bool = benchmark == "race" and metric in {
        "F_pred",
        "F_attr",
    }
    if race_metric_mislabeled:
        specification_mismatches.append("metric_label")
    specification_status: str = (
        "matches_declared_method"
        if not specification_mismatches
        else (
            f"{specification_mismatches[0]}_mismatch"
            if len(specification_mismatches) == 1
            else "multiple_mismatches"
        )
    )
    revision_action: str = (
        "replace"
        if race_metric_mislabeled or boolq_word_attention_contaminated
        else ("undecided" if specification_mismatches else "retain")
    )
    reported_statistic: str = "rv_coefficient" if race_metric_mislabeled else STATISTIC
    computed_statistic: str = (
        "pearson_r2_answer_conditioned_correct_vs_rest"
        if race_metric_mislabeled
        else STATISTIC
    )
    row: dict[str, Any] = {
        "reconciliation_id": (
            f"{paper_location}:{benchmark}:{pregrouper}:{metric}:"
            f"{pair_set}:{value_component}"
        ),
        "source_state": source_state,
        "source_locator": source_locator,
        "source_sha256": source_sha256,
        "paper_location": paper_location,
        "benchmark": benchmark,
        "pregrouper": pregrouper,
        "segment_grid_id": _grid_id(benchmark, pregrouper, source_state, metric),
        "declared_scope": declared_scope,
        "executed_scope": executed_scope,
        "declared_contrast": declared_contrast,
        "executed_contrast": executed_contrast,
        "cohort": "paper_11",
        "pair_set": pair_set,
        "metric": metric,
        "statistic": STATISTIC,
        "aggregation": aggregation,
        "missingness_policy": PAIRWISE_DROP,
        "value_component": value_component,
        "value": value,
        "display_value": display_value,
        "n_pairs": n_pairs,
        "effective_pairs": _encode_pairs(effective_pairs),
        "data_status": data_status,
        "specification_status": specification_status,
        "revision_action": revision_action,
        "reported_metric": metric,
        "reported_statistic": reported_statistic,
        "computed_metric": metric,
        "computed_statistic": computed_statistic,
        "numerical_reproduction_status": (
            "reference" if source_state == "published" else "not_evaluated"
        ),
        "model_identity_status": identity_status,
        "model_snapshot_identity": snapshot_identity,
        "method_id": "",
        "estimand_id": "",
        "comparison_status": "reference" if source_state == "published" else "pending",
        "delta_from_published": None,
        "note": " ".join(notes),
    }
    row["method_id"] = method_id(row)
    row["estimand_id"] = estimand_id(row)
    return row


def _published_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, pair_sets in PUBLISHED_TABLE1.items():
        for pair_set, values in pair_sets.items():
            for component, display in zip(("min", "median", "max"), values):
                rows.append(
                    _base_row(
                        source_state="published",
                        source_locator="audited_claim_set/table_1",
                        source_sha256="",
                        paper_location="Table 1",
                        benchmark="boolq",
                        pregrouper="sentence",
                        pair_set=pair_set,
                        metric=metric,
                        aggregation=_historical_aggregation(metric),
                        value_component=component,
                        value=float(display),
                        display_value=display,
                        n_pairs=len(
                            _historical_expected_pairs(
                                "boolq", "sentence", metric, pair_set
                            )
                        ),
                        effective_pairs=_historical_expected_pairs(
                            "boolq", "sentence", metric, pair_set
                        ),
                        data_status="published",
                    )
                )
    for metric, values in PUBLISHED_TABLE2.items():
        for spec in BENCHMARKS:
            if spec.paper_label not in values:
                continue
            display: str = values[spec.paper_label]
            rows.append(
                _base_row(
                    source_state="published",
                    source_locator="audited_claim_set/table_2",
                    source_sha256="",
                    paper_location="Table 2",
                    benchmark=spec.benchmark,
                    pregrouper=spec.pregrouper,
                    pair_set="all",
                    metric=metric,
                    aggregation=_historical_aggregation(metric),
                    value_component="median",
                    value=float(display),
                    display_value=display,
                    n_pairs=len(
                        _historical_expected_pairs(
                            spec.benchmark, spec.pregrouper, metric, "all"
                        )
                    ),
                    effective_pairs=_historical_expected_pairs(
                        spec.benchmark, spec.pregrouper, metric, "all"
                    ),
                    data_status=(
                        "published_invalid_sentence_attention_contamination"
                        if spec.benchmark == "boolq"
                        and spec.pregrouper == "word"
                        and metric in BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
                        else (
                            "published_metric_mislabeled"
                            if spec.benchmark == "race"
                            and metric in {"F_pred", "F_attr"}
                            else "published"
                        )
                    ),
                )
            )
    return rows


def _archive_rows(
    archive_dir: str, cohort: LegacyCohort, allow_unpinned_archive: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    inputs: dict[str, str] = {}
    for spec in BENCHMARKS:
        path: str = os.path.join(archive_dir, spec.filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        locator: str = f"archive/{spec.filename}"
        digest: str = _sha256(path)
        inputs[locator] = path
        expected_digest: str = RECONCILIATION_ARCHIVE_SHA256[locator]
        checks.append(
            {
                "check_id": f"archive_byte_sha256:{spec.benchmark}:{spec.pregrouper}",
                "kind": "archive_provenance",
                "locator": locator,
                "expected": expected_digest,
                "actual": digest,
                "tolerance": 0,
                "bypassed": allow_unpinned_archive and digest != expected_digest,
                "passed": digest == expected_digest or allow_unpinned_archive,
            }
        )
        df: pd.DataFrame = pd.read_csv(path, sep="\t")
        checks.append(
            {
                "check_id": f"archive_rows:{spec.benchmark}:{spec.pregrouper}",
                "kind": "historical_sentinel",
                "locator": locator,
                "expected": spec.historical_rows,
                "actual": len(df),
                "tolerance": 0,
                "passed": len(df) == spec.historical_rows,
            }
        )
        metrics: tuple[str, ...] = (
            ("F_pred", "F_attr") if spec.benchmark == "race" else METRICS
        )
        values: dict[str, dict[str, float]] = recompute_archive(df, cohort, metrics)
        for metric in metrics:
            sentinel: float | None = HISTORICAL_NUMERIC_SENTINELS.get(
                (spec.benchmark, spec.pregrouper, metric)
            )
            if sentinel is None:
                continue
            actual: float = float(np.median(list(values[f"{metric}:all"].values())))
            checks.append(
                {
                    "check_id": (
                        f"historical_numeric:{spec.benchmark}:{spec.pregrouper}:{metric}"
                    ),
                    "kind": "historical_sentinel",
                    "locator": locator,
                    "expected": sentinel,
                    "actual": actual,
                    "tolerance": 1e-12,
                    "passed": abs(actual - sentinel) <= 1e-12,
                }
            )
        if spec.benchmark == "boolq" and spec.pregrouper == "sentence":
            pair: str = _pair_id("qwen2.5-7b-instruct", "qwen2.5-14b-instruct", False)
            actual_pair: float | None = values["F_attr:all"].get(pair)
            expected_pair: float = 0.5600893341848812
            checks.append(
                {
                    "check_id": "historical_numeric:boolq:qwen7b_qwen14b:F_attr",
                    "kind": "historical_sentinel",
                    "locator": locator,
                    "expected": expected_pair,
                    "actual": actual_pair,
                    "tolerance": 1e-12,
                    "passed": actual_pair is not None
                    and abs(actual_pair - expected_pair) <= 1e-12,
                }
            )
        if spec.benchmark == "boolq" and spec.pregrouper == "sentence":
            for metric, pair_sets in PUBLISHED_TABLE1.items():
                for pair_set in pair_sets:
                    pair_values: dict[str, float] = values[f"{metric}:{pair_set}"]
                    metric_values: list[float] = list(pair_values.values())
                    observed_pairs: tuple[str, ...] = tuple(sorted(pair_values))
                    expected_pairs: tuple[str, ...] = _historical_expected_pairs(
                        spec.benchmark, spec.pregrouper, metric, pair_set
                    )
                    for grid_component in ("min", "median", "max"):
                        checks.append(
                            {
                                "check_id": (
                                    f"archive_pair_grid:Table1:{spec.benchmark}:"
                                    f"{spec.pregrouper}:{metric}:{pair_set}:"
                                    f"{grid_component}"
                                ),
                                "kind": "historical_pair_grid",
                                "locator": locator,
                                "expected": list(expected_pairs),
                                "actual": list(observed_pairs),
                                "tolerance": 0,
                                "passed": observed_pairs == expected_pairs,
                            }
                        )
                    if not metric_values:
                        continue
                    for component, value in zip(
                        ("min", "median", "max"),
                        (
                            float(np.min(metric_values)),
                            float(np.median(metric_values)),
                            float(np.max(metric_values)),
                        ),
                    ):
                        rows.append(
                            _base_row(
                                source_state="archive",
                                source_locator=locator,
                                source_sha256=digest,
                                paper_location="Table 1",
                                benchmark=spec.benchmark,
                                pregrouper=spec.pregrouper,
                                pair_set=pair_set,
                                metric=metric,
                                aggregation=_historical_aggregation(metric),
                                value_component=component,
                                value=value,
                                display_value=f"{value:.12g}",
                                n_pairs=len(metric_values),
                                effective_pairs=list(pair_values),
                                data_status=(
                                    "complete"
                                    if observed_pairs == expected_pairs
                                    else "incomplete_pair_set"
                                ),
                            )
                        )
        for metric in METRICS:
            if spec.benchmark == "race" and metric not in {"F_pred", "F_attr"}:
                continue
            pair_values = values[f"{metric}:all"]
            metric_values = list(pair_values.values())
            observed_pairs = tuple(sorted(pair_values))
            expected_pairs = _historical_expected_pairs(
                spec.benchmark, spec.pregrouper, metric, "all"
            )
            checks.append(
                {
                    "check_id": (
                        f"archive_pair_grid:Table2:{spec.benchmark}:"
                        f"{spec.pregrouper}:{metric}:all"
                    ),
                    "kind": "historical_pair_grid",
                    "locator": locator,
                    "expected": list(expected_pairs),
                    "actual": list(observed_pairs),
                    "tolerance": 0,
                    "passed": observed_pairs == expected_pairs,
                }
            )
            if not metric_values:
                continue
            value = float(np.median(metric_values))
            rows.append(
                _base_row(
                    source_state="archive",
                    source_locator=locator,
                    source_sha256=digest,
                    paper_location="Table 2",
                    benchmark=spec.benchmark,
                    pregrouper=spec.pregrouper,
                    pair_set="all",
                    metric=metric,
                    aggregation=_historical_aggregation(metric),
                    value_component="median",
                    value=value,
                    display_value=f"{value:.12g}",
                    n_pairs=len(metric_values),
                    effective_pairs=list(pair_values),
                    data_status=(
                        "reproduced_invalid_sentence_attention_contamination"
                        if spec.benchmark == "boolq"
                        and spec.pregrouper == "word"
                        and metric in BOOLQ_WORD_CONTAMINATED_ATTENTION_METRICS
                        else (
                            (
                                "reproduced_metric_mislabeled"
                                if spec.benchmark == "race"
                                and metric in {"F_pred", "F_attr"}
                                else "complete"
                            )
                            if observed_pairs == expected_pairs
                            else "incomplete_pair_set"
                        )
                    ),
                )
            )
    return rows, checks, inputs


CORRECTED_TABLES: tuple[str, ...] = (
    "f_table.tsv",
    "f_table_prompt_equal_transfer.tsv",
    "f_table_revision_candidate_pairwise_drop.tsv",
    "f_table_revision_candidate_finite_extreme.tsv",
    "race_scalar_paper_compatibility.tsv",
)


def _corrected_input_paths(results_dir: str) -> dict[str, str]:
    return {
        f"corrected/{filename}": os.path.join(results_dir, filename)
        for filename in CORRECTED_TABLES
        if os.path.isfile(os.path.join(results_dir, filename))
    }


def _verify_derived_sidecar(
    results_dir: str, locator: str, table_path: str
) -> tuple[bool, list[dict[str, Any]], dict[str, str]]:
    """Verify a derived table sidecar and every file hash it binds."""
    sidecar_path: str = f"{table_path}.provenance.json"
    sidecar_locator: str = f"{locator}.provenance.json"
    inputs: dict[str, str] = {}
    checks: list[dict[str, Any]] = []
    if not os.path.isfile(sidecar_path):
        checks.append(
            {
                "check_id": f"derived_sidecar:{locator}",
                "kind": "corrected_provenance",
                "locator": sidecar_locator,
                "expected": "present",
                "actual": "missing",
                "tolerance": 0,
                "passed": False,
            }
        )
        return False, checks, inputs
    inputs[sidecar_locator] = sidecar_path
    valid: bool = True
    try:
        with open(sidecar_path, encoding="utf-8") as source:
            payload: Any = json.load(source)
        output_record: Any = (
            payload.get("output") if isinstance(payload, dict) else None
        )
        recorded_output_hash: Any = (
            output_record.get("sha256") if isinstance(output_record, dict) else None
        )
        actual_output_hash: str = _sha256(table_path)
        output_valid: bool = recorded_output_hash == actual_output_hash
        valid = valid and output_valid
        checks.append(
            {
                "check_id": f"derived_sidecar_output:{locator}",
                "kind": "corrected_provenance",
                "locator": sidecar_locator,
                "expected": recorded_output_hash,
                "actual": actual_output_hash,
                "tolerance": 0,
                "passed": output_valid,
            }
        )
        bound_inputs: Any = payload.get("inputs") if isinstance(payload, dict) else None
        if not isinstance(bound_inputs, dict) or not bound_inputs:
            valid = False
            checks.append(
                {
                    "check_id": f"derived_sidecar_inputs:{locator}",
                    "kind": "corrected_provenance",
                    "locator": sidecar_locator,
                    "expected": "one_or_more_bound_inputs",
                    "actual": "missing_or_empty",
                    "tolerance": 0,
                    "passed": False,
                }
            )
        else:
            for identifier, record in sorted(bound_inputs.items()):
                relative: Any = record.get("path") if isinstance(record, dict) else None
                expected_hash: Any = (
                    record.get("sha256") if isinstance(record, dict) else None
                )
                safe_relative: bool = (
                    isinstance(relative, str)
                    and not os.path.isabs(relative)
                    and relative != os.pardir
                    and not relative.startswith(os.pardir + "/")
                )
                input_path: str = (
                    os.path.join(results_dir, relative) if safe_relative else ""
                )
                actual_hash: str | None = (
                    _sha256(input_path)
                    if input_path and os.path.isfile(input_path)
                    else None
                )
                input_valid: bool = (
                    safe_relative
                    and isinstance(expected_hash, str)
                    and actual_hash == expected_hash
                )
                valid = valid and input_valid
                checks.append(
                    {
                        "check_id": f"derived_sidecar_input:{locator}:{identifier}",
                        "kind": "corrected_provenance",
                        "locator": sidecar_locator,
                        "expected": expected_hash,
                        "actual": actual_hash,
                        "tolerance": 0,
                        "passed": input_valid,
                    }
                )
    except (json.JSONDecodeError, OSError, TypeError, AttributeError) as error:
        valid = False
        checks.append(
            {
                "check_id": f"derived_sidecar_parse:{locator}",
                "kind": "corrected_provenance",
                "locator": sidecar_locator,
                "expected": "valid_derived_provenance",
                "actual": type(error).__name__,
                "tolerance": 0,
                "passed": False,
            }
        )
    return valid, checks, inputs


def _validate_corrected_schema(frame: pd.DataFrame, locator: str) -> None:
    required: set[str] = {
        "artifact_role",
        "benchmark",
        "candidate_id",
        "cohort",
        "pregrouper",
        "scope",
        "contrast",
        "requested_scope",
        "resolved_scope",
        "requested_contrast",
        "resolved_source_contrast",
        "resolved_target_contrast",
        "readout_contrast",
        "api_infinity_policy",
        "aggregation",
        "availability_status",
        "unavailable_reason",
        "model_s",
        "model_t",
        "metric",
        "statistic",
        "f_point",
    }
    missing: list[str] = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"{locator} lacks estimand metadata required for reconciliation: {missing}"
        )


def _models_match_pair_set(
    model_s: str, model_t: str, metric: str, pair_set: str
) -> bool:
    source_open: bool = model_s in OPEN
    target_open: bool = model_t in OPEN
    source_hosted: bool = model_s in HOSTED
    target_hosted: bool = model_t in HOSTED
    if metric in TRANSFER_METRICS:
        if pair_set == "open":
            return source_open and target_open and model_s != model_t
        if pair_set == "open_to_hosted":
            return source_open and target_hosted
        return source_open and (target_open or target_hosted) and model_s != model_t
    if pair_set == "open":
        return source_open and target_open
    if pair_set == "open_to_hosted":
        return (source_open and target_hosted) or (source_hosted and target_open)
    if metric in REPRESENTATION_METRICS:
        return source_open and target_open
    return (source_open or source_hosted) and (target_open or target_hosted)


def _summarize_corrected(
    frame: pd.DataFrame,
    locator: str,
    digest: str,
    spec: BenchmarkSpec,
    paper_location: str,
    pair_set: str,
    metric: str,
    component: str,
) -> dict[str, Any] | None:
    contrast: str = "canonical"
    subset: pd.DataFrame = frame[
        (frame["benchmark"] == spec.benchmark)
        & (frame["pregrouper"] == spec.pregrouper)
        & (frame["scope"] == "all")
        & (frame["contrast"] == contrast)
        & (frame["api_infinity_policy"] == "drop")
        & (frame["metric"] == metric)
        & (frame["statistic"] == STATISTIC)
    ].copy()
    if subset.empty:
        return None
    subset = subset[
        subset.apply(
            lambda row: _models_match_pair_set(
                str(row["model_s"]), str(row["model_t"]), metric, pair_set
            ),
            axis=1,
        )
    ]
    if subset.empty:
        return None
    raw_aggregations: list[str] = sorted(
        str(value) for value in subset["aggregation"].dropna().unique()
    )
    if subset["aggregation"].isna().any() or len(raw_aggregations) != 1:
        raise ValueError(
            f"{locator} has missing or mixed aggregation values for "
            f"{spec.benchmark}/{spec.pregrouper}/{metric}/{pair_set}: "
            f"{raw_aggregations}"
        )
    aggregation: str = _corrected_aggregation(raw_aggregations[0])
    subset["_value"] = pd.to_numeric(subset["f_point"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    directed: bool = metric in TRANSFER_METRICS
    subset["_pair"] = subset.apply(
        lambda row: _pair_id(str(row["model_s"]), str(row["model_t"]), directed),
        axis=1,
    )
    if subset["_pair"].duplicated().any():
        duplicates: list[str] = sorted(
            subset.loc[subset["_pair"].duplicated(keep=False), "_pair"].unique()
        )
        raise ValueError(
            f"Duplicate corrected model pairs in {locator}: {duplicates[:5]}"
        )
    finite_frame: pd.DataFrame = subset.dropna(subset=["_value"])
    finite: pd.Series = finite_frame["_value"]
    functions: dict[str, Callable[[pd.Series], float]] = {
        "min": lambda values: float(values.min()),
        "median": lambda values: float(values.median()),
        "max": lambda values: float(values.max()),
    }
    value: float | None = functions[component](finite) if not finite.empty else None
    observed_pairs: tuple[str, ...] = tuple(sorted(finite_frame["_pair"].tolist()))
    expected_pairs: tuple[str, ...] = _expected_pairs(metric, pair_set)
    status: str = (
        "unsupported"
        if not observed_pairs
        else "complete" if observed_pairs == expected_pairs else "incomplete_pair_set"
    )
    snapshot_status: str = "version_not_verifiable"
    snapshot_identity: str = "unverified"
    if "model_snapshot_id" in finite_frame.columns and not finite_frame.empty:
        snapshot_values: pd.Series = finite_frame["model_snapshot_id"].astype("string")
        if snapshot_values.notna().all() and snapshot_values.str.len().gt(0).all():
            snapshot_status = "recorded"
            snapshot_identity = _encode_pairs(
                tuple(sorted(snapshot_values.astype(str).unique().tolist()))
            )
    return _base_row(
        source_state="corrected",
        source_locator=locator,
        source_sha256=digest,
        paper_location=paper_location,
        benchmark=spec.benchmark,
        pregrouper=spec.pregrouper,
        pair_set=pair_set,
        metric=metric,
        aggregation=aggregation,
        value_component=component,
        value=value,
        display_value="" if value is None else f"{value:.12g}",
        n_pairs=len(finite),
        effective_pairs=observed_pairs,
        data_status=status,
        model_identity_status=snapshot_status,
        model_snapshot_identity=snapshot_identity,
    )


def _corrected_rows(
    results_dir: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    table_inputs: dict[str, str] = _corrected_input_paths(results_dir)
    inputs: dict[str, str] = dict(table_inputs)
    checks: list[dict[str, Any]] = []
    frames: list[tuple[str, str, pd.DataFrame]] = []
    for filename in CORRECTED_TABLES:
        locator: str = f"corrected/{filename}"
        if locator not in table_inputs:
            checks.append(
                {
                    "check_id": f"corrected_table:{locator}",
                    "kind": "corrected_provenance",
                    "locator": locator,
                    "expected": "present",
                    "actual": "missing",
                    "tolerance": 0,
                    "passed": False,
                }
            )
    for locator, path in table_inputs.items():
        provenance_valid, provenance_checks, provenance_inputs = (
            _verify_derived_sidecar(results_dir, locator, path)
        )
        checks.extend(provenance_checks)
        inputs.update(provenance_inputs)
        if not provenance_valid:
            continue
        frame: pd.DataFrame = pd.read_csv(path, sep="\t")
        _validate_corrected_schema(frame, locator)
        frames.append((locator, _sha256(path), frame))

    rows: list[dict[str, Any]] = []
    boolq_spec: BenchmarkSpec = BENCHMARKS[0]
    for locator, digest, frame in frames:
        filename: str = locator.removeprefix("corrected/")
        if filename == "f_table_prompt_equal_transfer.tsv":
            table1_metrics: tuple[str, ...] = tuple(TRANSFER_METRICS)
            table2_metrics: tuple[str, ...] = tuple(TRANSFER_METRICS)
            table2_specs: tuple[BenchmarkSpec, ...] = tuple(
                spec for spec in BENCHMARKS if spec.benchmark != "race"
            )
        elif filename.startswith("f_table_revision_candidate_"):
            table1_metrics = ()
            table2_metrics = ()
            table2_specs = ()
        elif filename == "race_scalar_paper_compatibility.tsv":
            table1_metrics = ()
            table2_metrics = ("F_pred", "F_attr")
            table2_specs = tuple(
                spec for spec in BENCHMARKS if spec.benchmark == "race"
            )
        else:
            table1_metrics = METRICS
            table2_metrics = METRICS
            table2_specs = tuple(
                spec for spec in BENCHMARKS if spec.benchmark != "race"
            )
        for metric in table1_metrics:
            pair_sets: dict[str, tuple[str, str, str]] | None = PUBLISHED_TABLE1.get(
                metric
            )
            if pair_sets is None:
                continue
            for pair_set in pair_sets:
                for component in ("min", "median", "max"):
                    row: dict[str, Any] | None = _summarize_corrected(
                        frame,
                        locator,
                        digest,
                        boolq_spec,
                        "Table 1",
                        pair_set,
                        metric,
                        component,
                    )
                    if row is not None:
                        rows.append(row)
        for spec in table2_specs:
            for metric in table2_metrics:
                row = _summarize_corrected(
                    frame, locator, digest, spec, "Table 2", "all", metric, "median"
                )
                if row is not None:
                    rows.append(row)
    # A configuration must come from exactly one corrected table.  Duplicate
    # rows would make the source of the aggregate ambiguous.
    keys: list[str] = [
        f"{row['reconciliation_id']}:{row['estimand_id']}" for row in rows
    ]
    duplicates: list[str] = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate corrected reconciliation cells: {duplicates[:5]}")
    return rows, checks, inputs


def _attach_comparisons(rows: list[dict[str, Any]]) -> None:
    published: dict[str, dict[str, Any]] = {
        str(row["reconciliation_id"]): row
        for row in rows
        if row["source_state"] == "published"
    }
    for row in rows:
        if row["source_state"] == "published":
            continue
        reference: dict[str, Any] | None = published.get(str(row["reconciliation_id"]))
        if reference is None:
            row["comparison_status"] = "no_published_reference"
            row["numerical_reproduction_status"] = "no_published_reference"
        elif row["data_status"] == "unsupported":
            row["comparison_status"] = "unsupported"
            row["numerical_reproduction_status"] = "not_available"
        elif row["data_status"] == "incomplete_pair_set":
            row["comparison_status"] = "incomplete_pair_set"
            row["numerical_reproduction_status"] = "not_comparable"
        elif (
            row["source_state"] == "corrected"
            and row["method_id"] == reference["method_id"]
            and row["model_identity_status"] == "version_not_verifiable"
        ):
            row["comparison_status"] = "nominally_comparable_same_method"
            row["delta_from_published"] = None
            row["numerical_reproduction_status"] = "not_comparable_model_snapshot"
        elif row["estimand_id"] != reference["estimand_id"]:
            row["comparison_status"] = "nonmatching_estimand"
            row["delta_from_published"] = None
            if row["source_state"] == "archive":
                difference: float = abs(float(row["value"]) - float(reference["value"]))
                row["numerical_reproduction_status"] = (
                    "matches_published_number"
                    if difference <= 0.0005000001
                    else "differs_from_published_number"
                )
            else:
                row["numerical_reproduction_status"] = "not_comparable_estimand"
        else:
            row["comparison_status"] = "comparable"
            row["delta_from_published"] = comparable_delta(row, reference)
            row["numerical_reproduction_status"] = (
                "matches_published_number"
                if abs(float(row["value"]) - float(reference["value"])) <= 0.0005000001
                else "differs_from_published_number"
            )


def _reproduction_checks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {
        str(row["reconciliation_id"]): row
        for row in rows
        if row["source_state"] == "published"
    }
    checks: list[dict[str, Any]] = []
    for row in rows:
        if row["source_state"] != "archive":
            continue
        reference: dict[str, Any] | None = references.get(str(row["reconciliation_id"]))
        if reference is None or row["estimand_id"] != reference["estimand_id"]:
            continue
        expected: float = float(reference["value"])
        actual: float = float(row["value"])
        tolerance: float = 0.0005000001
        checks.append(
            {
                "check_id": f"published_rounding:{row['reconciliation_id']}",
                "kind": "published_sentinel",
                "locator": str(row["source_locator"]),
                "expected": expected,
                "actual": actual,
                "tolerance": tolerance,
                "passed": abs(actual - expected) <= tolerance,
            }
        )
    return checks


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for distribution in ("numpy", "pandas"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _write_json(path: str, payload: dict[str, Any]) -> None:
    directory: str = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    serialized: str = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, delete=False
    ) as output:
        temporary: str = output.name
        output.write(serialized)
    os.replace(temporary, path)


def reconcile(
    archive_dir: str,
    corrected_results_dir: str,
    output_dir: str,
    legacy_model_config: str | None = None,
    allow_unpinned_archive: bool = False,
) -> tuple[str, str, str, str]:
    """Build the reconciliation ledger, historical claims, manifest, and checks.

    Args:
        archive_dir: Directory containing the eight legacy consolidated TSVs.
        corrected_results_dir: Directory containing corrected long result TSVs.
        output_dir: Destination for the reconciliation artifacts.
        legacy_model_config: External archive-prefix to public-model mapping.
        allow_unpinned_archive: Permit noncanonical archive bytes for miniature
            tests. Gold reconciliation must leave this false.

    Returns:
        Paths to ``reconciliation.tsv``, ``historical_claims.tsv``, the
        reconciliation manifest, and checks JSON. The manuscript-disposition
        table is written alongside them and sealed by the manifest.
    """
    config_path: str = legacy_model_config or os.path.join(
        archive_dir, "reconciliation_models.json"
    )
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Missing external legacy model mapping: {config_path}. See --legacy-model-config."
        )
    cohort: LegacyCohort = load_legacy_cohort(config_path)
    published_rows: list[dict[str, Any]] = _published_rows()
    archive_rows, checks, archive_inputs = _archive_rows(
        archive_dir, cohort, allow_unpinned_archive
    )
    config_digest: str = _sha256(config_path)
    expected_config_digest: str = RECONCILIATION_ARCHIVE_SHA256[
        "archive/reconciliation_models.json"
    ]
    checks.append(
        {
            "check_id": "archive_byte_sha256:reconciliation_models",
            "kind": "archive_provenance",
            "locator": "archive/reconciliation_models.json",
            "expected": expected_config_digest,
            "actual": config_digest,
            "tolerance": 0,
            "bypassed": allow_unpinned_archive
            and config_digest != expected_config_digest,
            "passed": config_digest == expected_config_digest or allow_unpinned_archive,
        }
    )
    corrected_rows, corrected_checks, corrected_inputs = _corrected_rows(
        corrected_results_dir
    )
    checks.extend(corrected_checks)
    claim_rows, claim_checks, claim_inputs = _historical_claim_rows(
        archive_dir,
        corrected_results_dir,
        cohort,
        allow_unpinned_archive,
    )
    checks.extend(claim_checks)
    rows: list[dict[str, Any]] = published_rows + archive_rows + corrected_rows
    _attach_comparisons(rows)
    checks.extend(_reproduction_checks(rows))
    checks.append(
        {
            "check_id": "nonmatching_estimands_have_no_delta",
            "kind": "comparison_guard",
            "locator": "reconciliation.tsv",
            "expected": 0,
            "actual": sum(
                1
                for row in rows
                if row["comparison_status"] == "nonmatching_estimand"
                and row["delta_from_published"] is not None
            ),
            "tolerance": 0,
            "passed": all(
                row["delta_from_published"] is None
                for row in rows
                if row["comparison_status"] == "nonmatching_estimand"
            ),
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    table_path: str = os.path.join(output_dir, "reconciliation.tsv")
    claims_path: str = os.path.join(output_dir, "historical_claims.tsv")
    disposition_path: str = os.path.join(output_dir, "manuscript_disposition.tsv")
    manifest_path: str = os.path.join(output_dir, "reconciliation_manifest.json")
    checks_path: str = os.path.join(output_dir, "reconciliation_checks.json")
    frame: pd.DataFrame = pd.DataFrame(rows, columns=list(LEDGER_COLUMNS)).sort_values(
        [
            "paper_location",
            "benchmark",
            "pregrouper",
            "metric",
            "pair_set",
            "value_component",
            "source_state",
        ],
        kind="stable",
    )
    frame.to_csv(table_path, sep="\t", index=False)
    claims_frame: pd.DataFrame = pd.DataFrame(
        claim_rows, columns=list(CLAIM_COLUMNS)
    ).sort_values("claim_id", kind="stable")
    claims_frame.to_csv(claims_path, sep="\t", index=False)
    disposition_frame: pd.DataFrame = pd.DataFrame(
        manuscript_dispositions(), columns=list(MANUSCRIPT_DISPOSITION_COLUMNS)
    ).sort_values("issue_id", kind="stable")
    disposition_frame.to_csv(disposition_path, sep="\t", index=False)
    claims_digest: str = historical_claims_sha256(claims_path)
    audited_claim_digest: str = audited_claim_set_sha256(published_rows, claim_rows)
    checks.append(
        {
            "check_id": "historical_claims_sha256",
            "kind": "archive_provenance",
            "locator": "reconciliation/historical_claims.tsv",
            "expected": RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
            "actual": claims_digest,
            "tolerance": 0,
            "bypassed": allow_unpinned_archive
            and claims_digest != RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
            "passed": (
                claims_digest == RECONCILIATION_HISTORICAL_CLAIMS_SHA256
                or allow_unpinned_archive
            ),
        }
    )
    checks.append(
        {
            "check_id": "audited_claim_set_sha256",
            "kind": "paper_reference",
            "locator": "reconciliation/reconciliation_manifest.json",
            "expected": RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
            "actual": audited_claim_digest,
            "tolerance": 0,
            "bypassed": allow_unpinned_archive
            and audited_claim_digest != RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
            "passed": (
                audited_claim_digest == RECONCILIATION_AUDITED_CLAIM_SET_SHA256
                or allow_unpinned_archive
            ),
        }
    )
    archive_ledger_digest: str = archive_ledger_sha256(table_path)
    checks.append(
        {
            "check_id": "archive_ledger_sha256",
            "kind": "archive_provenance",
            "locator": "reconciliation.tsv",
            "expected": RECONCILIATION_ARCHIVE_LEDGER_SHA256,
            "actual": archive_ledger_digest,
            "tolerance": 0,
            "bypassed": allow_unpinned_archive
            and archive_ledger_digest != RECONCILIATION_ARCHIVE_LEDGER_SHA256,
            "passed": (
                archive_ledger_digest == RECONCILIATION_ARCHIVE_LEDGER_SHA256
                or allow_unpinned_archive
            ),
        }
    )
    archive_checks_digest: str = archive_checks_sha256(checks)
    if (
        not allow_unpinned_archive
        and archive_checks_digest != RECONCILIATION_ARCHIVE_CHECKS_SHA256
    ):
        raise ValueError(
            "Sanitized archive check digest does not match the audited snapshot"
        )
    _write_json(
        checks_path,
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "reconciliation_checks",
            "all_passed": all(bool(check["passed"]) for check in checks),
            "checks": checks,
        },
    )
    input_paths: dict[str, str] = {
        "archive/reconciliation_models.json": config_path,
        **archive_inputs,
        **claim_inputs,
        **corrected_inputs,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "result_reconciliation",
        "generator": {
            "locator": "benchmark_scripts/reconcile_results.py",
            "sha256": _sha256(__file__),
        },
        "inputs": {
            locator: {"sha256": _sha256(path), "size_bytes": os.path.getsize(path)}
            for locator, path in sorted(input_paths.items())
        },
        "outputs": {
            "reconciliation.tsv": {
                "sha256": _sha256(table_path),
                "size_bytes": os.path.getsize(table_path),
            },
            "historical_claims.tsv": {
                "sha256": _sha256(claims_path),
                "size_bytes": os.path.getsize(claims_path),
            },
            "manuscript_disposition.tsv": {
                "sha256": _sha256(disposition_path),
                "size_bytes": os.path.getsize(disposition_path),
            },
            "reconciliation_checks.json": {
                "sha256": _sha256(checks_path),
                "size_bytes": os.path.getsize(checks_path),
            },
        },
        "historical_estimand": {
            "statistic": STATISTIC,
            "missingness_policy": PAIRWISE_DROP,
            "prediction_scope": "not_applicable",
            "segment_level_declared_scope": "user",
            "segment_level_executed_scope": "all",
            "anli_declared_contrast": "entailment_contradiction",
            "anli_executed_contrast": "entailment_neutral",
        },
        "paper_reference": {
            "reference_id": PAPER_REFERENCE_ID,
            "work_id": "arXiv:2606.32008",
            "version": None,
            "pdf_sha256": None,
            "identity_status": PAPER_IDENTITY_STATUS,
            "claim_canonicalization": PAPER_CLAIM_CANONICALIZATION,
            "audited_claim_set_sha256": audited_claim_digest,
            "claim_scope": (
                "Tables 1-2 plus explicitly enumerated Figure 13 and "
                "main-text numerical claims"
            ),
        },
        "archive_provenance": {
            "kind": "raw_archive_snapshot_bytes",
            "enforced": not allow_unpinned_archive,
            "trusted_sha256": dict(sorted(RECONCILIATION_ARCHIVE_SHA256.items())),
            "trusted_size_bytes": dict(
                sorted(RECONCILIATION_ARCHIVE_SIZE_BYTES.items())
            ),
            "trusted_race_raw_sha256": dict(
                sorted(RECONCILIATION_RACE_RAW_SHA256.items())
            ),
            "trusted_race_raw_size_bytes": dict(
                sorted(RECONCILIATION_RACE_RAW_SIZE_BYTES.items())
            ),
            "sanitized_ledger_sha256": RECONCILIATION_ARCHIVE_LEDGER_SHA256,
            "sanitized_checks_sha256": RECONCILIATION_ARCHIVE_CHECKS_SHA256,
            "historical_claims_sha256": RECONCILIATION_HISTORICAL_CLAIMS_SHA256,
            "audited_claim_set_sha256": RECONCILIATION_AUDITED_CLAIM_SET_SHA256,
        },
        "race_caveat": RACE_NOTE,
        "boolq_word_attention_caveat": BOOLQ_WORD_ATTENTION_NOTE,
        "model_identity_caveat": MODEL_IDENTITY_NOTE,
        "software": _software_versions(),
    }
    _write_json(manifest_path, manifest)
    return table_path, claims_path, manifest_path, checks_path


def main() -> None:
    """Run reconciliation from command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--corrected-results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--legacy-model-config",
        default=None,
        help=(
            "External JSON mapping archive column prefixes to public model IDs "
            "(default: ARCHIVE_DIR/reconciliation_models.json)."
        ),
    )
    parser.add_argument(
        "--allow-unpinned-archive",
        action="store_true",
        help="Permit noncanonical archive bytes for miniature tests only.",
    )
    args: argparse.Namespace = parser.parse_args()
    table, claims, manifest, checks = reconcile(
        args.archive_dir,
        args.corrected_results_dir,
        args.output_dir,
        args.legacy_model_config,
        args.allow_unpinned_archive,
    )
    print(table)
    print(claims)
    print(manifest)
    print(checks)
    with open(checks, encoding="utf-8") as source:
        check_payload: Any = json.load(source)
    if (
        not isinstance(check_payload, dict)
        or check_payload.get("all_passed") is not True
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
