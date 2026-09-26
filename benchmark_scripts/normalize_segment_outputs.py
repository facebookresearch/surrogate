# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Align per-model segment outputs to a canonical segment manifest."""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import numpy as np
import pandas as pd

GZIP_COMPRESSION: dict[str, Any] = {
    "method": "gzip",
    "compresslevel": 9,
    "mtime": 0,
}
KEY_COLUMNS: list[str] = ["prompt_idx", "seg_idx"]
MANIFEST_COLUMNS: list[str] = [
    "answer",
    "message_idx",
    "message_role",
    "message_seg_idx",
    "segment_text",
    "n_segments",
]


def normalize_file(
    path: str,
    manifest: pd.DataFrame,
    allow_missing: bool = False,
    legacy_identity_attestation: str | None = None,
) -> None:
    """Align one model result, preserving missingness and segment identity."""
    frame: pd.DataFrame = pd.read_csv(
        path, sep="\t", keep_default_na=False, na_values=[""]
    )
    for name, candidate in (("manifest", manifest), (path, frame)):
        if candidate.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"{name} has duplicate segment keys")
    manifest_keys: pd.MultiIndex = pd.MultiIndex.from_frame(manifest[KEY_COLUMNS])
    frame_keys: pd.MultiIndex = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
    if not frame_keys.isin(manifest_keys).all():
        raise ValueError(f"{path} contains keys outside the segment manifest")
    missing_count: int = int((~manifest_keys.isin(frame_keys)).sum())
    if missing_count and not allow_missing:
        raise ValueError(
            f"{path} is missing {missing_count} canonical segment keys; "
            "use --allow-missing only for independently verified hosted outputs"
        )

    missing_identity: set[str] = set(MANIFEST_COLUMNS) - set(frame.columns)
    if missing_identity and legacy_identity_attestation is None:
        raise ValueError(
            f"{path} lacks identity columns {sorted(missing_identity)}; a canonical "
            "manifest cannot prove that legacy integer keys refer to the same text"
        )
    if (
        legacy_identity_attestation is not None
        and not legacy_identity_attestation.strip()
    ):
        raise ValueError("Legacy identity attestation must be non-empty")

    duplicate_metadata: list[str] = [
        column for column in MANIFEST_COLUMNS if column in frame.columns
    ]
    if duplicate_metadata:
        comparison: pd.DataFrame = manifest[[*KEY_COLUMNS, *duplicate_metadata]].merge(
            frame[[*KEY_COLUMNS, *duplicate_metadata]],
            on=KEY_COLUMNS,
            how="inner",
            suffixes=("_manifest", "_result"),
            validate="one_to_one",
        )
        for column in duplicate_metadata:
            expected: pd.Series = comparison[f"{column}_manifest"]
            actual: pd.Series = comparison[f"{column}_result"]
            equal: pd.Series = expected.eq(actual) | (expected.isna() & actual.isna())
            if not equal.all():
                raise ValueError(f"{path} has conflicting {column!r} metadata")
    canonical_key_order: list[tuple[int, int]] = list(
        manifest[KEY_COLUMNS].itertuples(index=False, name=None)
    )
    frame_key_order: list[tuple[int, int]] = list(
        frame[KEY_COLUMNS].itertuples(index=False, name=None)
    )
    is_completion: bool = "ablated_completion_logprob" in frame.columns
    completion_flags_match: bool = False
    if is_completion and "orig_completion_logprob" not in frame.columns:
        raise ValueError(f"{path} lacks orig_completion_logprob")
    if is_completion and {
        "segment_result_available",
        "original_result_available",
    }.issubset(frame.columns):
        expected_segment_available: pd.Series = np.isfinite(
            pd.to_numeric(frame["ablated_completion_logprob"], errors="coerce")
        )
        expected_original_available: pd.Series = np.isfinite(
            pd.to_numeric(frame["orig_completion_logprob"], errors="coerce")
        )
        completion_flags_match = bool(
            frame["segment_result_available"].eq(expected_segment_available).all()
            and frame["original_result_available"].eq(expected_original_available).all()
        )
    if (
        not missing_count
        and not missing_identity
        and {
            "segment_result_available",
            "original_result_available",
        }.issubset(frame.columns)
        and frame_key_order == canonical_key_order
        and (not is_completion or completion_flags_match)
    ):
        return
    payload: pd.DataFrame = frame.drop(columns=duplicate_metadata).copy()
    if is_completion:
        payload["segment_result_available"] = np.isfinite(
            pd.to_numeric(payload["ablated_completion_logprob"], errors="coerce")
        )
        payload["original_result_available"] = np.isfinite(
            pd.to_numeric(payload["orig_completion_logprob"], errors="coerce")
        )
    elif "segment_result_available" not in payload.columns:
        payload["segment_result_available"] = True
    if not is_completion and "original_result_available" not in payload.columns:
        # Classification runners score the original together with every
        # observed segment. A missing segment row therefore does not make the
        # prompt-level original unavailable; propagate that observation after
        # joining to the canonical manifest.
        payload["original_result_available"] = True
    if (
        not is_completion
        and payload.groupby("prompt_idx")["original_result_available"]
        .nunique(dropna=False)
        .gt(1)
        .any()
    ):
        raise ValueError(f"{path} has inconsistent original availability by prompt")
    original_available_by_prompt: pd.Series = payload.groupby("prompt_idx")[
        "original_result_available"
    ].first()
    normalized: pd.DataFrame = manifest.merge(
        payload,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    normalized["segment_result_available"] = normalized["segment_result_available"].eq(
        True
    )
    if not is_completion:
        normalized["original_result_available"] = (
            normalized["prompt_idx"].map(original_available_by_prompt).eq(True)
        )
    normalized.to_csv(
        path,
        sep="\t",
        index=False,
        compression=GZIP_COMPRESSION if path.endswith(".gz") else None,
    )


def normalize_directory(
    directory: str,
    allow_missing: bool = False,
    legacy_identity_attestation: str | None = None,
) -> None:
    """Normalize every per-model segment TSV in a result directory."""
    manifest_path: str = os.path.join(directory, "segments.tsv.gz")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Missing segment manifest: {manifest_path}")
    manifest: pd.DataFrame = pd.read_csv(
        manifest_path, sep="\t", keep_default_na=False, na_values=[""]
    )
    paths: list[str] = sorted(
        set(
            glob.glob(os.path.join(directory, "*_segment.tsv"))
            + glob.glob(os.path.join(directory, "*_segment.tsv.gz"))
        )
    )
    for path in paths:
        normalize_file(
            path,
            manifest,
            allow_missing=allow_missing,
            legacy_identity_attestation=legacy_identity_attestation,
        )


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Materialize missing manifest keys with availability=false. Use "
            "only for hosted outputs whose segment identity was independently "
            "verified before import."
        ),
    )
    parser.add_argument(
        "--attest-legacy-identity",
        help=(
            "Explicit audit statement permitting identity columns to be added to "
            "legacy hosted files. This is an attestation, not automatic proof."
        ),
    )
    args: argparse.Namespace = parser.parse_args()
    normalize_directory(
        args.directory,
        allow_missing=args.allow_missing,
        legacy_identity_attestation=args.attest_legacy_identity,
    )


if __name__ == "__main__":
    main()
