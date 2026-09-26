# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Build the frozen BoolQ layer-control direction specification.

The production specification is deliberately derived only from exact,
single-token surfaces shared by all five pinned public tokenizers.  It contains
no corpus-, frequency-, or model-score-dependent filtering.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from benchmark_scripts.provenance_sources import (
    canonical_file_hash_manifest_sha256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
)


SCHEMA_VERSION: int = 1
NAMESPACE: str = "surrogate_fidelity.boolq_token_control.v1"
PRODUCTION_DRAW_COUNT: int = 256
PRODUCTION_P9_COUNT: int = 278
PRODUCTION_P8_COUNT: int = 399
BASE_PATTERN: re.Pattern[str] = re.compile(r"[a-z]{3,16}")
EXCLUDED_BASE_SURFACES: frozenset[str] = frozenset({"true", "false", "yes", "no"})
P9_PROFILE: str = "111111111"
P8_PROFILE: str = "111111101"
TOKENIZER_IDENTITY_FILENAMES: frozenset[str] = frozenset(
    {
        "added_tokens.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
    }
)
MODEL_METADATA_FIELDS: frozenset[str] = frozenset(
    {
        "model_artifact_manifest_sha256",
        "model_revision",
        "model_source",
        "tokenizer_class",
        "tokenizer_files_sha256",
        "tokenizer_manifest_sha256",
        "vocabulary_size_including_added_tokens",
    }
)

# This order is part of the sampling contract.  It matches the public
# cross-model layer analysis; canonical JSON serialization still sorts mapping
# keys so the output bytes do not depend on dictionary insertion order.
PRODUCTION_MODELS: tuple[tuple[str, str], ...] = (
    ("qwen2.5-0.5b-instruct", "Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-3b-instruct", "Qwen2.5-3B-Instruct"),
    ("llama-3.1-8b-instruct", "Meta-Llama-3.1-8B-Instruct"),
    ("qwen2.5-7b-instruct", "Qwen2.5-7B-Instruct"),
    ("qwen2.5-14b-instruct", "Qwen2.5-14B-Instruct"),
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: str) -> str:
    digest: Any = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tokenizer_identity_hashes(model_path: str) -> dict[str, str]:
    hashes: dict[str, str] = {
        name: _sha256_file(os.path.join(model_path, name))
        for name in sorted(TOKENIZER_IDENTITY_FILENAMES)
        if os.path.isfile(os.path.join(model_path, name))
    }
    if not hashes or not any(
        name in hashes for name in ("tokenizer.json", "tokenizer.model")
    ):
        raise ValueError(f"No tokenizer identity artifact found in {model_path}")
    return hashes


def _model_artifact_hashes(model_path: str) -> dict[str, str]:
    """Hash the same local model-artifact inventory as layer production."""
    names: list[str] = sorted(
        name
        for name in os.listdir(model_path)
        if name.endswith((".safetensors", ".bin", ".json", ".model", ".txt"))
        and os.path.isfile(os.path.join(model_path, name))
    )
    if not any(name.endswith((".safetensors", ".bin")) for name in names):
        raise ValueError(f"No model weight artifact found in {model_path}")
    return {name: _sha256_file(os.path.join(model_path, name)) for name in names}


def _ordered_variants(base_surface: str) -> list[str]:
    capitalized: str = base_surface.capitalize()
    upper: str = base_surface.upper()
    return [
        base_surface,
        capitalized,
        upper,
        f" {base_surface}",
        f" {capitalized}",
        f" {upper}",
        f"_{base_surface}",
        f"_{capitalized}",
        f"_{upper}",
    ]


def _exact_singleton_surfaces(tokenizer: Any) -> dict[str, int]:
    """Map NFC-exact singleton round trips to IDs, excluding added/special IDs."""
    special_ids: set[int] = {int(value) for value in tokenizer.all_special_ids}
    added_ids: set[int] = {int(value) for value in tokenizer.get_added_vocab().values()}
    excluded_ids: set[int] = special_ids | added_ids
    surfaces: dict[str, int] = {}
    for token_id in range(len(tokenizer)):
        if token_id in excluded_ids:
            continue
        surface: Any = tokenizer.decode(
            [token_id],
            clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        if (
            not isinstance(surface, str)
            or not surface
            or unicodedata.normalize("NFC", surface) != surface
        ):
            continue
        encoded: list[int] = [
            int(value) for value in tokenizer.encode(surface, add_special_tokens=False)
        ]
        if encoded != [token_id]:
            continue
        previous: int | None = surfaces.get(surface)
        if previous is not None and previous != token_id:
            raise ValueError(
                f"Surface {surface!r} round-trips to multiple singleton IDs"
            )
        surfaces[surface] = token_id
    return surfaces


def _basis_entry(
    base_surface: str,
    variants: Sequence[str],
    surfaces_by_model: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    variant_list: list[str] = list(variants)
    return {
        "base_surface": base_surface,
        "variants": variant_list,
        "token_ids_by_model": {
            model: [int(surfaces[surface]) for surface in variant_list]
            for model, surfaces in surfaces_by_model.items()
        },
    }


def _enumerate_common_basis_pools(
    surfaces_by_model: Mapping[str, Mapping[str, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not surfaces_by_model:
        raise ValueError("At least one tokenizer surface map is required")
    shared_surfaces: set[str] = set.intersection(
        *(set(surfaces) for surfaces in surfaces_by_model.values())
    )
    bases: list[str] = sorted(
        surface
        for surface in shared_surfaces
        if BASE_PATTERN.fullmatch(surface) is not None
        and surface.casefold().strip() not in EXCLUDED_BASE_SURFACES
    )
    p9_pool: list[dict[str, Any]] = []
    p8_pool: list[dict[str, Any]] = []
    for base_surface in bases:
        variants: list[str] = _ordered_variants(base_surface)
        profile: str = "".join(
            (
                "1"
                if all(
                    surface in model_surfaces
                    for model_surfaces in surfaces_by_model.values()
                )
                else "0"
            )
            for surface in variants
        )
        if profile == P9_PROFILE:
            p9_pool.append(_basis_entry(base_surface, variants, surfaces_by_model))
        elif profile == P8_PROFILE:
            accepted_variants: list[str] = [
                surface
                for present, surface in zip(profile, variants, strict=True)
                if present == "1"
            ]
            p8_pool.append(
                _basis_entry(base_surface, accepted_variants, surfaces_by_model)
            )
    return p9_pool, p8_pool


def _hash_rank(pool_tag: str, base_surface: str) -> tuple[bytes, bytes]:
    payload: bytes = b"\0".join(
        (
            NAMESPACE.encode("ascii"),
            pool_tag.encode("ascii"),
            base_surface.encode("utf-8"),
        )
    )
    return hashlib.sha256(payload).digest(), base_surface.encode("utf-8")


def _select_paired_bases(
    p9_pool: Sequence[Mapping[str, Any]],
    p8_pool: Sequence[Mapping[str, Any]],
    draw_count: int,
) -> list[dict[str, Any]]:
    if isinstance(draw_count, bool) or not isinstance(draw_count, int):
        raise TypeError("draw_count must be an integer")
    if draw_count <= 0 or draw_count > PRODUCTION_DRAW_COUNT:
        raise ValueError(f"draw_count must be in [1, {PRODUCTION_DRAW_COUNT}]")
    if draw_count > min(len(p9_pool), len(p8_pool)):
        raise ValueError("draw_count exceeds an eligible basis pool")
    positive: list[str] = sorted(
        (str(entry["base_surface"]) for entry in p9_pool),
        key=lambda base: _hash_rank("p9", base),
    )[:draw_count]
    negative: list[str] = sorted(
        (str(entry["base_surface"]) for entry in p8_pool),
        key=lambda base: _hash_rank("p8", base),
    )[:draw_count]
    return [
        {
            "draw_idx": draw_idx,
            "positive_base_surface": positive_base,
            "negative_base_surface": negative_base,
        }
        for draw_idx, (positive_base, negative_base) in enumerate(
            zip(positive, negative, strict=True)
        )
    ]


def _isotropic_seeds(
    model_names: Sequence[str], draw_count: int
) -> dict[str, list[int]]:
    seeds_by_model: dict[str, list[int]] = {}
    all_seeds: set[int] = set()
    for model_name in model_names:
        seeds: list[int] = []
        for draw_idx in range(draw_count):
            payload: bytes = b"\0".join(
                (
                    NAMESPACE.encode("ascii"),
                    b"isotropic_seed",
                    model_name.encode("utf-8"),
                    str(draw_idx).encode("ascii"),
                )
            )
            seed: int = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
                2**63 - 1
            )
            seeds.append(seed)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Isotropic seed collision for {model_name}")
        if all_seeds & set(seeds):
            raise ValueError("Isotropic seed collision across models")
        all_seeds.update(seeds)
        seeds_by_model[model_name] = seeds
    return seeds_by_model


def _selection_metadata(
    draw_count: int,
    p9_pool: Sequence[Mapping[str, Any]],
    p8_pool: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "algorithm": "sha256_rank_without_replacement_pair_by_rank_v1",
        "base_surface_regex": BASE_PATTERN.pattern,
        "draw_count": draw_count,
        "excluded_base_surfaces": sorted(EXCLUDED_BASE_SURFACES),
        "hash_encoding": "namespace_nul_pool_tag_nul_utf8_base_surface",
        "isotropic_seed_algorithm": (
            "sha256(namespace_nul_isotropic_seed_nul_model_nul_decimal_draw)_"
            "first_u64_be_mask_int64_v1"
        ),
        "linear_surface_rule": "single ASCII space + lowercase base_surface",
        "linear_control_role": (
            "secondary_legacy_single_token_true_false_direction_bridge"
        ),
        "p8_profile": P8_PROFILE,
        "p9_profile": P9_PROFILE,
        "pool_sha256": {
            "p8": _canonical_sha256(p8_pool),
            "p9": _canonical_sha256(p9_pool),
        },
        "production_draw_count": PRODUCTION_DRAW_COUNT,
        "production_pool_counts": {
            "p8": PRODUCTION_P8_COUNT,
            "p9": PRODUCTION_P9_COUNT,
        },
        "singleton_contract": (
            "non_special_non_added_nfc_exact_decode_and_singleton_reencode_v1"
        ),
        "spec_hash_contract": (
            "sha256_compact_sorted_utf8_json_with_selection_algorithm_"
            "spec_sha256_absent"
        ),
        "variant_order": [
            "plain_lower",
            "plain_capitalized",
            "plain_upper",
            "space_lower",
            "space_capitalized",
            "space_upper",
            "underscore_lower",
            "underscore_capitalized",
            "underscore_upper",
        ],
    }


def _validate_production_model_metadata(
    model_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    for model_name, record in model_metadata.items():
        if set(record) != MODEL_METADATA_FIELDS:
            raise ValueError(
                f"Production model metadata fields disagree for {model_name}"
            )
        if (
            record["model_source"] != GOLD_OPEN_MODEL_REPOSITORIES[model_name]
            or record["model_revision"] != GOLD_OPEN_MODEL_REVISIONS[model_name]
            or record["model_artifact_manifest_sha256"]
            != GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[model_name]
        ):
            raise ValueError(
                f"Production pinned model identity disagrees for {model_name}"
            )
        tokenizer_class: Any = record["tokenizer_class"]
        tokenizer_hashes: Any = record["tokenizer_files_sha256"]
        vocabulary_size: Any = record["vocabulary_size_including_added_tokens"]
        if not isinstance(tokenizer_class, str) or not tokenizer_class:
            raise ValueError(f"Production tokenizer class is invalid for {model_name}")
        if not isinstance(tokenizer_hashes, dict) or not tokenizer_hashes:
            raise ValueError(f"Production tokenizer hashes are absent for {model_name}")
        for filename, digest in tokenizer_hashes.items():
            if not isinstance(filename, str) or not filename:
                raise ValueError(
                    f"Production tokenizer filename is invalid for {model_name}"
                )
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(
                    f"Production tokenizer hash is invalid for {model_name}"
                )
        if record["tokenizer_manifest_sha256"] != (
            canonical_file_hash_manifest_sha256(tokenizer_hashes)
        ):
            raise ValueError(
                f"Production tokenizer manifest disagrees for {model_name}"
            )
        if (
            isinstance(vocabulary_size, bool)
            or not isinstance(vocabulary_size, int)
            or vocabulary_size <= 0
        ):
            raise ValueError(f"Production vocabulary size is invalid for {model_name}")


def build_control_spec(
    tokenizers: Mapping[str, Any],
    model_metadata: Mapping[str, Mapping[str, Any]],
    draw_count: int = PRODUCTION_DRAW_COUNT,
    *,
    require_production_contract: bool = True,
) -> dict[str, Any]:
    """Build and self-hash a deterministic control specification."""
    if set(tokenizers) != set(model_metadata):
        raise ValueError("Tokenizer and model-metadata names differ")
    expected_names: set[str] = {name for name, _basename in PRODUCTION_MODELS}
    if require_production_contract and set(tokenizers) != expected_names:
        raise ValueError("Production specification requires all five pinned models")
    if require_production_contract:
        _validate_production_model_metadata(model_metadata)
    surfaces_by_model: dict[str, dict[str, int]] = {
        model_name: _exact_singleton_surfaces(tokenizers[model_name])
        for model_name in tokenizers
    }
    p9_pool, p8_pool = _enumerate_common_basis_pools(surfaces_by_model)
    if require_production_contract and (
        len(p9_pool) != PRODUCTION_P9_COUNT or len(p8_pool) != PRODUCTION_P8_COUNT
    ):
        raise ValueError(
            "Production basis-pool counts disagree: "
            f"expected P9/P8={PRODUCTION_P9_COUNT}/{PRODUCTION_P8_COUNT}, "
            f"got {len(p9_pool)}/{len(p8_pool)}"
        )
    selected: list[dict[str, Any]] = _select_paired_bases(p9_pool, p8_pool, draw_count)
    metadata: dict[str, Any] = _selection_metadata(draw_count, p9_pool, p8_pool)
    metadata["production_pool_counts_verified"] = require_production_contract
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "models": {model: dict(model_metadata[model]) for model in tokenizers},
        "namespace": NAMESPACE,
        "selection_algorithm": metadata,
        "p9_basis_pool": p9_pool,
        "p8_basis_pool": p8_pool,
        "selected_paired_bases": selected,
        "isotropic_seeds_by_model": _isotropic_seeds(list(tokenizers), draw_count),
    }
    metadata["spec_sha256"] = _canonical_sha256(spec)
    return spec


def validate_production_control_spec(spec: Mapping[str, Any]) -> None:
    """Validate the complete deterministic production specification contract.

    This check does not reopen tokenizer files. It verifies all reproducible
    structure encoded in a serialized spec: pinned model metadata, basis-shape
    rules, pool hashes, SHA-ranked selection, isotropic seeds, and the spec's
    self-hash.

    Args:
        spec: Parsed control-specification mapping.

    Raises:
        ValueError: If any production-contract field disagrees.
    """
    if spec.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Production control spec schema version disagrees")
    if spec.get("namespace") != NAMESPACE:
        raise ValueError("Production control spec namespace disagrees")
    models_value: Any = spec.get("models")
    if not isinstance(models_value, dict) or set(models_value) != {
        name for name, _basename in PRODUCTION_MODELS
    }:
        raise ValueError("Production control spec model set disagrees")
    model_metadata: dict[str, Mapping[str, Any]] = {}
    for model_name, record in models_value.items():
        if not isinstance(model_name, str) or not isinstance(record, dict):
            raise ValueError("Production control spec model metadata is invalid")
        model_metadata[model_name] = record
    _validate_production_model_metadata(model_metadata)

    pools: list[tuple[str, int, Any]] = [
        ("p9_basis_pool", 9, spec.get("p9_basis_pool")),
        ("p8_basis_pool", 8, spec.get("p8_basis_pool")),
    ]
    validated_pools: dict[str, list[Mapping[str, Any]]] = {}
    for field, width, value in pools:
        expected_count: int = PRODUCTION_P9_COUNT if width == 9 else PRODUCTION_P8_COUNT
        if not isinstance(value, list) or len(value) != expected_count:
            raise ValueError(f"Production {field} size disagrees")
        entries: list[Mapping[str, Any]] = []
        for entry in value:
            if not isinstance(entry, dict):
                raise ValueError(f"Production {field} contains a non-object")
            base: Any = entry.get("base_surface")
            if (
                not isinstance(base, str)
                or BASE_PATTERN.fullmatch(base) is None
                or base in EXCLUDED_BASE_SURFACES
            ):
                raise ValueError(f"Production {field} has an invalid base surface")
            ordered: list[str] = _ordered_variants(base)
            expected_variants: list[str] = (
                ordered if width == 9 else ordered[:7] + ordered[8:]
            )
            if entry.get("variants") != expected_variants:
                raise ValueError(f"Production {field} variant profile disagrees")
            entries.append(entry)
        validated_pools[field] = entries

    p9_pool: list[Mapping[str, Any]] = validated_pools["p9_basis_pool"]
    p8_pool: list[Mapping[str, Any]] = validated_pools["p8_basis_pool"]
    expected_selection: list[dict[str, Any]] = _select_paired_bases(
        p9_pool, p8_pool, PRODUCTION_DRAW_COUNT
    )
    if spec.get("selected_paired_bases") != expected_selection:
        raise ValueError("Production control selection is not the SHA-ranked sample")
    expected_seeds: dict[str, list[int]] = _isotropic_seeds(
        sorted(model_metadata), PRODUCTION_DRAW_COUNT
    )
    if spec.get("isotropic_seeds_by_model") != expected_seeds:
        raise ValueError("Production isotropic seeds disagree")

    selection_value: Any = spec.get("selection_algorithm")
    if not isinstance(selection_value, dict):
        raise ValueError("Production selection metadata is invalid")
    selection: dict[str, Any] = dict(selection_value)
    declared_spec_sha256: Any = selection.pop("spec_sha256", None)
    expected_metadata: dict[str, Any] = _selection_metadata(
        PRODUCTION_DRAW_COUNT, p9_pool, p8_pool
    )
    expected_metadata["production_pool_counts_verified"] = True
    if selection != expected_metadata:
        raise ValueError("Production selection metadata disagrees")
    unhashed_spec: dict[str, Any] = copy.deepcopy(dict(spec))
    unhashed_selection: Any = unhashed_spec.get("selection_algorithm")
    if not isinstance(unhashed_selection, dict):
        raise ValueError("Production selection metadata is invalid")
    unhashed_selection.pop("spec_sha256", None)
    if declared_spec_sha256 != _canonical_sha256(unhashed_spec):
        raise ValueError("Production control spec self-hash disagrees")


def _load_production_inputs(
    model_root: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    # Keep the pure specification helpers usable in lightweight CPU tests.
    from transformers import AutoTokenizer

    tokenizers: dict[str, Any] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for model_name, local_basename in PRODUCTION_MODELS:
        model_path: str = os.path.join(model_root, local_basename)
        if not os.path.isdir(model_path):
            raise FileNotFoundError(model_path)
        repository: str = GOLD_OPEN_MODEL_REPOSITORIES[model_name]
        revision: str = GOLD_OPEN_MODEL_REVISIONS[model_name]
        expected_model_manifest: str = GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[
            model_name
        ]
        artifact_hashes: dict[str, str] = _model_artifact_hashes(model_path)
        observed_model_manifest: str = canonical_file_hash_manifest_sha256(
            artifact_hashes
        )
        if observed_model_manifest != expected_model_manifest:
            raise ValueError(
                f"Pinned model artifact manifest disagrees for {model_name}: "
                f"expected {expected_model_manifest}, got {observed_model_manifest}"
            )
        tokenizer_hashes: dict[str, str] = _tokenizer_identity_hashes(model_path)
        tokenizer: Any = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        tokenizers[model_name] = tokenizer
        metadata[model_name] = {
            "model_artifact_manifest_sha256": observed_model_manifest,
            "model_revision": revision,
            "model_source": repository,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_files_sha256": tokenizer_hashes,
            "tokenizer_manifest_sha256": canonical_file_hash_manifest_sha256(
                tokenizer_hashes
            ),
            "vocabulary_size_including_added_tokens": len(tokenizer),
        }
    return tokenizers, metadata


def write_control_spec(spec: Mapping[str, Any], output_path: str) -> None:
    output_directory: str = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_directory,
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        delete=False,
    ) as output:
        temporary_path: str = output.name
        json.dump(
            spec,
            output,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")
    os.replace(temporary_path, output_path)


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-root", default="/tmp/models")
    parser.add_argument(
        "--draw-count",
        type=int,
        default=PRODUCTION_DRAW_COUNT,
        help="Number of paired controls to select (1-256); use fewer for a canary.",
    )
    args: argparse.Namespace = parser.parse_args()
    tokenizers, metadata = _load_production_inputs(args.model_root)
    spec: dict[str, Any] = build_control_spec(
        tokenizers,
        metadata,
        draw_count=args.draw_count,
        require_production_contract=True,
    )
    write_control_spec(spec, args.output)
    print(
        f"Wrote {args.output}: {len(spec['selected_paired_bases'])} draws, "
        f"spec_sha256={spec['selection_algorithm']['spec_sha256']}"
    )


if __name__ == "__main__":
    main()
