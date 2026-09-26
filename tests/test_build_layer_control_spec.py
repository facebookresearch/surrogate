# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import itertools
import json
import os
import tempfile
import unicodedata
from typing import Any
from unittest import TestCase

from benchmark_scripts.build_layer_control_spec import (
    _canonical_sha256,
    _enumerate_common_basis_pools,
    _exact_singleton_surfaces,
    _isotropic_seeds,
    _ordered_variants,
    _select_paired_bases,
    build_control_spec,
    NAMESPACE,
    PRODUCTION_MODELS,
    validate_production_control_spec,
    write_control_spec,
)
from benchmark_scripts.provenance_sources import (
    canonical_file_hash_manifest_sha256,
    GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256,
    GOLD_OPEN_MODEL_REPOSITORIES,
    GOLD_OPEN_MODEL_REVISIONS,
)


class _FakeTokenizer:
    def __init__(
        self,
        surfaces: list[str],
        *,
        special_ids: set[int] | None = None,
        added_ids: set[int] | None = None,
        reencode_overrides: dict[str, list[int]] | None = None,
    ) -> None:
        self._surfaces: list[str] = surfaces
        self.all_special_ids: list[int] = sorted(special_ids or set())
        self._added_ids: set[int] = added_ids or set()
        self._reencode_overrides: dict[str, list[int]] = reencode_overrides or {}
        self._ids: dict[str, int] = {
            surface: token_id for token_id, surface in enumerate(surfaces)
        }

    def __len__(self) -> int:
        return len(self._surfaces)

    def get_added_vocab(self) -> dict[str, int]:
        return {self._surfaces[token_id]: token_id for token_id in self._added_ids}

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return self._surfaces[token_ids[0]]

    def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("test contract must disable special tokens")
        return self._reencode_overrides.get(surface, [self._ids[surface]])


def _surface_map(*, p9_bases: list[str], p8_bases: list[str]) -> dict[str, int]:
    surfaces: list[str] = []
    for base in p9_bases:
        surfaces.extend(_ordered_variants(base))
    for base in p8_bases:
        variants: list[str] = _ordered_variants(base)
        surfaces.extend(variants[:7])
        surfaces.append(variants[8])
    return {
        surface: token_id for token_id, surface in enumerate(dict.fromkeys(surfaces))
    }


def _production_metadata(model_names: list[str]) -> dict[str, dict[str, Any]]:
    tokenizer_hashes: dict[str, str] = {"tokenizer.json": "a" * 64}
    return {
        model_name: {
            "model_artifact_manifest_sha256": (
                GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256[model_name]
            ),
            "model_revision": GOLD_OPEN_MODEL_REVISIONS[model_name],
            "model_source": GOLD_OPEN_MODEL_REPOSITORIES[model_name],
            "tokenizer_class": "FakeTokenizer",
            "tokenizer_files_sha256": tokenizer_hashes,
            "tokenizer_manifest_sha256": canonical_file_hash_manifest_sha256(
                tokenizer_hashes
            ),
            "vocabulary_size_including_added_tokens": 1,
        }
        for model_name in model_names
    }


class TestBuildLayerControlSpec(TestCase):
    def test_singleton_inventory_excludes_special_added_non_nfc_and_multitoken(
        self,
    ) -> None:
        decomposed: str = unicodedata.normalize("NFD", "café")
        tokenizer: _FakeTokenizer = _FakeTokenizer(
            ["alpha", "special", "added", decomposed, "multi", "omega"],
            special_ids={1},
            added_ids={2},
            reencode_overrides={"multi": [4, 5]},
        )
        self.assertEqual({"alpha": 0, "omega": 5}, _exact_singleton_surfaces(tokenizer))

    def test_enumerates_exact_common_p9_and_false_like_p8_profiles(self) -> None:
        first: dict[str, int] = _surface_map(
            p9_bases=["alpha", "true"], p8_bases=["bravo", "false", "yes"]
        )
        second: dict[str, int] = {
            surface: token_id + 1000 for surface, token_id in first.items()
        }
        # A partial profile and a too-short P9 basis must not enter either pool.
        for surface in _ordered_variants("odd")[:3]:
            first[surface] = len(first)
            second[surface] = len(second) + 1000
        for surface in _ordered_variants("no"):
            first[surface] = len(first)
            second[surface] = len(second) + 1000
        p9, p8 = _enumerate_common_basis_pools({"m1": first, "m2": second})
        self.assertEqual(["alpha"], [entry["base_surface"] for entry in p9])
        self.assertEqual(["bravo"], [entry["base_surface"] for entry in p8])
        self.assertEqual(9, len(p9[0]["variants"]))
        self.assertEqual(8, len(p8[0]["variants"]))
        self.assertNotIn("_Bravo", p8[0]["variants"])
        self.assertEqual(
            [first[surface] for surface in p9[0]["variants"]],
            p9[0]["token_ids_by_model"]["m1"],
        )

    def test_selection_and_isotropic_seeds_are_deterministic_unique_and_bounded(
        self,
    ) -> None:
        p9: list[dict[str, str]] = [
            {"base_surface": base} for base in ["alpha", "bravo", "charlie"]
        ]
        p8: list[dict[str, str]] = [
            {"base_surface": base} for base in ["delta", "echo", "foxtrot"]
        ]
        selected: list[dict[str, Any]] = _select_paired_bases(p9, p8, 2)
        self.assertEqual(
            [
                {
                    "draw_idx": 0,
                    "positive_base_surface": "bravo",
                    "negative_base_surface": "echo",
                },
                {
                    "draw_idx": 1,
                    "positive_base_surface": "charlie",
                    "negative_base_surface": "delta",
                },
            ],
            selected,
        )
        self.assertEqual(selected, _select_paired_bases(list(reversed(p9)), p8, 2))
        self.assertEqual([0, 1], [row["draw_idx"] for row in selected])
        self.assertEqual(2, len({row["positive_base_surface"] for row in selected}))
        self.assertEqual(2, len({row["negative_base_surface"] for row in selected}))
        seeds: dict[str, list[int]] = _isotropic_seeds(["m1", "m2"], 3)
        self.assertEqual(
            [8076712966392717691, 5571843486624340112, 4179931952906937108],
            seeds["m1"],
        )
        self.assertEqual(seeds, _isotropic_seeds(["m1", "m2"], 3))
        self.assertEqual(3, len(set(seeds["m1"])))
        self.assertNotEqual(seeds["m1"], seeds["m2"])
        self.assertTrue(all(0 <= seed <= 2**63 - 1 for seed in seeds["m1"]))

    def test_builds_self_hashed_schema_and_writes_deterministic_json(self) -> None:
        surface_ids: dict[str, int] = _surface_map(
            p9_bases=["alpha", "bravo"], p8_bases=["charlie", "delta"]
        )
        surfaces: list[str] = [
            surface
            for surface, _token_id in sorted(
                surface_ids.items(), key=lambda row: row[1]
            )
        ]
        tokenizers: dict[str, _FakeTokenizer] = {
            "m1": _FakeTokenizer(surfaces),
            "m2": _FakeTokenizer(surfaces),
        }
        metadata: dict[str, dict[str, str]] = {
            "m1": {"revision": "one"},
            "m2": {"revision": "two"},
        }
        spec: dict[str, Any] = build_control_spec(
            tokenizers,
            metadata,
            draw_count=2,
            require_production_contract=False,
        )
        self.assertEqual(
            {
                "schema_version",
                "models",
                "namespace",
                "selection_algorithm",
                "p9_basis_pool",
                "p8_basis_pool",
                "selected_paired_bases",
                "isotropic_seeds_by_model",
            },
            set(spec),
        )
        self.assertEqual(NAMESPACE, spec["namespace"])
        stored_hash: str = spec["selection_algorithm"].pop("spec_sha256")
        self.assertEqual(stored_hash, _canonical_sha256(spec))
        spec["selection_algorithm"]["spec_sha256"] = stored_hash
        with tempfile.TemporaryDirectory() as directory:
            first_path: str = os.path.join(directory, "one.json")
            second_path: str = os.path.join(directory, "two.json")
            write_control_spec(spec, first_path)
            write_control_spec(spec, second_path)
            with open(first_path, "rb") as first, open(second_path, "rb") as second:
                self.assertEqual(first.read(), second.read())
            with open(first_path, encoding="utf-8") as source:
                self.assertEqual(spec, json.load(source))

    def test_validates_complete_production_selection_contract(self) -> None:
        bases: list[str] = [
            "".join(letters)
            for letters in itertools.product("abcdefghijklmnopqrstuvwxyz", repeat=3)
        ]
        p9_bases: list[str] = bases[:278]
        p8_bases: list[str] = bases[278 : 278 + 399]
        surface_ids: dict[str, int] = _surface_map(p9_bases=p9_bases, p8_bases=p8_bases)
        surfaces: list[str] = [
            surface
            for surface, _token_id in sorted(
                surface_ids.items(), key=lambda row: row[1]
            )
        ]
        model_names: list[str] = [name for name, _basename in PRODUCTION_MODELS]
        spec: dict[str, Any] = build_control_spec(
            {model_name: _FakeTokenizer(surfaces) for model_name in model_names},
            _production_metadata(model_names),
        )
        validate_production_control_spec(spec)
        altered: dict[str, Any] = copy.deepcopy(spec)
        altered["selection_algorithm"]["algorithm"] = "unverified"
        with self.assertRaisesRegex(ValueError, "selection metadata"):
            validate_production_control_spec(altered)

    def test_production_contract_rejects_missing_models_before_enumeration(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "all five pinned models"):
            build_control_spec(
                {"m1": _FakeTokenizer(["alpha"])},
                {"m1": {}},
                draw_count=1,
                require_production_contract=True,
            )
        self.assertEqual(5, len(PRODUCTION_MODELS))

    def test_production_contract_rejects_wrong_audited_pool_counts(self) -> None:
        model_names: list[str] = [name for name, _basename in PRODUCTION_MODELS]
        surfaces: list[str] = _ordered_variants("alpha")
        tokenizers: dict[str, _FakeTokenizer] = {
            model_name: _FakeTokenizer(surfaces) for model_name in model_names
        }
        metadata: dict[str, dict[str, Any]] = _production_metadata(model_names)
        with self.assertRaisesRegex(ValueError, "expected P9/P8=278/399"):
            build_control_spec(
                tokenizers,
                metadata,
                draw_count=1,
                require_production_contract=True,
            )

    def test_production_contract_rejects_unpinned_model_metadata(self) -> None:
        model_names: list[str] = [name for name, _basename in PRODUCTION_MODELS]
        tokenizers: dict[str, _FakeTokenizer] = {
            model_name: _FakeTokenizer(["alpha"]) for model_name in model_names
        }
        metadata: dict[str, dict[str, Any]] = _production_metadata(model_names)
        metadata[model_names[0]]["model_revision"] = "untrusted"
        with self.assertRaisesRegex(ValueError, "pinned model identity disagrees"):
            build_control_spec(
                tokenizers,
                metadata,
                draw_count=1,
                require_production_contract=True,
            )
