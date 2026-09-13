# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Canonical source-file sets used to seal public result provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


def canonical_file_hash_manifest_sha256(file_hashes: Mapping[str, str]) -> str:
    """Hash a filename-to-SHA256 mapping with a stable public encoding."""

    payload: bytes = json.dumps(
        dict(file_hashes), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Immutable public Hugging Face revisions whose files were used for the
# release open-model artifacts. The aggregate digests are SHA-256 over compact,
# sorted JSON mappings from artifact basename to artifact SHA-256.
GOLD_OPEN_MODEL_REPOSITORIES: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
}
# Exact public repository locators present in the unsealed execution records.
# The Llama spelling was canonicalized for the release only after its local
# weight bytes had been independently pinned; preserving this field prevents
# the packaging seal from silently rewriting execution metadata.
GOLD_OPEN_EXECUTION_MODEL_SOURCES: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
GOLD_OPEN_MODEL_REVISIONS: dict[str, str] = {
    "qwen2.5-0.5b-instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "qwen2.5-3b-instruct": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "qwen2.5-7b-instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "qwen2.5-14b-instruct": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "llama-3.1-8b-instruct": "0e9e39f249a16976918f6564b8830bc894c89659",
}
GOLD_OPEN_MODEL_ARTIFACT_MANIFEST_SHA256: dict[str, str] = {
    "qwen2.5-0.5b-instruct": (
        "74bf6dd08049d862b3bd56ac2a84a8e1980ae93acfad71612a9ea00548e887b7"
    ),
    "qwen2.5-3b-instruct": (
        "515cc4186a7431ffd2ae609f4d03157ebe3c23eb952d1a8e03cc835201d6620e"
    ),
    "qwen2.5-7b-instruct": (
        "60208359faf0425cc3788ec94b6341091bd5b30d76c87d2d5ee439966a4c1b41"
    ),
    "qwen2.5-14b-instruct": (
        "c35195e5d56a7eb65e98bbd7a458297e81f1f4509af5a1ebdce22d2751b51927"
    ),
    "llama-3.1-8b-instruct": (
        "6e5b9a49f9f077fa78d65de615b45fd56150ed900e7024a46d50ee8ce7988dea"
    ),
}
OPEN_MODEL_IDENTITY_FILENAMES: frozenset[str] = frozenset(
    {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)

OPEN_COMPLETE_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/compute_logodds.py",
    "benchmark_scripts/consolidate_results.py",
    "benchmark_scripts/layerwise_fidelity.py",
    "benchmark_scripts/dialog_identity.py",
    "benchmark_scripts/f_table.py",
    "benchmark_scripts/hosted_audit_receipt.py",
    "benchmark_scripts/hosted_completion_audit_receipt.py",
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/race_rv.py",
    "benchmark_scripts/run_benchmark.py",
    "benchmark_scripts/run_layerwise.py",
    "benchmark_scripts/validate_results.py",
    "surrogate/attention_scoring.py",
    "surrogate/eval_constants.py",
    "surrogate/layerwise_scoring.py",
    "surrogate/model_types.py",
    "surrogate/representation_scoring.py",
    "surrogate/text_augmentation.py",
    "surrogate/transformers_model.py",
    "surrogate/transformers_scoring.py",
    "surrogate/utils.py",
)

# Exact source bytes captured by every open-model run in this release. These
# deliberately remain fixed if the public files receive later documentation or
# packaging-only changes; ``release_source_sha256`` separately binds the final
# committed implementation.
GOLD_OPEN_EXECUTION_SOURCE_SHA256: dict[str, str] = {
    "benchmark_scripts/benchmark_config.py": (
        "2c0f28b30cd8779af076c6a6c5813356eeb5d8b0eb07bf883159a7ada64845aa"
    ),
    "benchmark_scripts/run_benchmark.py": (
        "f5b4549e1df21833d674d7425a613100d8bbb0a7cb606b6b9a81427dc07b2987"
    ),
    "surrogate/text_augmentation.py": (
        "0183846200a7d12c23259e3a0dda89d478bbb35e5cba7ef0e796126db19e3d16"
    ),
    "surrogate/utils.py": (
        "256c98c992e39f9bfa8862f112b0bf77bd4f1d91ebcabb9366e714f34d6964a0"
    ),
}

# The original run sidecars hash the four files above.  This broader snapshot
# also covers every public module imported by the scoring path.  Its file
# hashes were reconstructed from unchanged files after launch. They are
# corroborating evidence rather than a cryptographic start-time attestation.
GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256: dict[str, str] = {
    **GOLD_OPEN_EXECUTION_SOURCE_SHA256,
    "surrogate/attention_scoring.py": (
        "a8f6a8cce01767e4839b0c3684f58481bd1c434acf71707a33814321eaa4b538"
    ),
    "surrogate/eval_constants.py": (
        "ae46e73d9f14224a1368a2c496ee492f1eced33c8bc7262b67f828f5d51925a6"
    ),
    "surrogate/model_types.py": (
        "9bf381e60172217c0bfbc5eef79c07c9eed9ea8d8b0f28ad57f9d651fb3b75bd"
    ),
    "surrogate/representation_scoring.py": (
        "c574c6cfaa5c9d3b6380129d019ef259f06f19ccb0f01c07def1a60abc391b0c"
    ),
    "surrogate/transformers_model.py": (
        "b9fd08fe2ba2f0dbc4dd41b468c9875dfcac2c62373ccadeef1837409dbf10f9"
    ),
    "surrogate/transformers_scoring.py": (
        "18a894fd3f0153152162ee7ee4b028f1a673e62468b1a00762ddf8033f71e35e"
    ),
}

# Current runs hash this complete execution dependency set directly. Earlier
# Qwen run records contain the four-file subset above; the broader observed
# snapshot is separately labeled as post-run corroboration in their sidecars.
OPEN_EXECUTION_SOURCE_FILES: tuple[str, ...] = tuple(
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
)

# Exact public dependency set captured at the beginning of every layerwise
# model run. Keeping this inventory here prevents the producer, analyzer, and
# release validator from silently drifting apart.
LAYER_EXECUTION_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/run_layerwise.py",
    "surrogate/eval_constants.py",
    "surrogate/layerwise_scoring.py",
    "surrogate/model_types.py",
    "surrogate/text_augmentation.py",
    "surrogate/transformers_model.py",
    "surrogate/utils.py",
)

# Exact, narrowly scoped differences between the observed Qwen execution
# snapshot and the release sources.
GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS: dict[str, dict[str, str]] = {
    "benchmark_scripts/benchmark_config.py": {
        "execution_sha256": (
            "2c0f28b30cd8779af076c6a6c5813356eeb5d8b0eb07bf883159a7ada64845aa"
        ),
        "release_sha256": (
            "143c8c79c2cf864513a4974b63eeeeccd74ac2a53b448a54f43b42daabc0bb3a"
        ),
        "reason": "Canonicalize both public Llama repository locators.",
        "scope": "model_locator_only_no_numerical_change",
    },
    "benchmark_scripts/run_benchmark.py": {
        "execution_sha256": (
            "f5b4549e1df21833d674d7425a613100d8bbb0a7cb606b6b9a81427dc07b2987"
        ),
        "release_sha256": (
            "7e68f08b7ff115240e01cf5d718ec6ccdbfa816981dbb2e7f5783b10a5196980"
        ),
        "reason": (
            "Hash the complete execution dependency set at run start and "
            "record the single-BOS rendered-chat policy."
        ),
        "scope": "provenance_and_qwen_tokenization_noop",
    },
    "surrogate/utils.py": {
        "execution_sha256": (
            "256c98c992e39f9bfa8862f112b0bf77bd4f1d91ebcabb9366e714f34d6964a0"
        ),
        "release_sha256": (
            "9d8e6146a7c635fc0c2e89e08cb3b70bb6929b56eb7cc541b9df105dcc227638"
        ),
        "reason": (
            "Simplify sentence segmentation while preserving every frozen "
            "benchmark segment manifest exactly."
        ),
        "scope": "segmentation_refactor_manifest_identical",
    },
    "surrogate/attention_scoring.py": {
        "execution_sha256": (
            "a8f6a8cce01767e4839b0c3684f58481bd1c434acf71707a33814321eaa4b538"
        ),
        "release_sha256": (
            "63cb3535ab296ff952ea3caf57479391d399cf703170d2ece501e549d9393fa3"
        ),
        "reason": "Use the explicit rendered-chat tokenization policy.",
        "scope": "qwen_tokenization_equivalence_verified_no_numerical_change",
    },
    "surrogate/transformers_model.py": {
        "execution_sha256": (
            "b9fd08fe2ba2f0dbc4dd41b468c9875dfcac2c62373ccadeef1837409dbf10f9"
        ),
        "release_sha256": (
            "61d6e906b539354e798000dc055eb4c6e9bedb1c0605e4851585796bcaa8d67b"
        ),
        "reason": "Use the explicit rendered-chat tokenization policy.",
        "scope": "qwen_tokenization_equivalence_verified_no_numerical_change",
    },
}

# Fixed public text used in gold hosted run records. Producer-specific details
# belong in hash-bound structured fields, not in a free-form string that could
# accidentally disclose a private path or routing alias.
GOLD_HOSTED_IDENTITY_ATTESTATION: str = (
    "Hosted prompt and ablation coordinates were verified against the canonical "
    "public segment manifest."
)

# Exact canonical bytes of the independently audited 7-configuration by
# 7-model hosted-classification receipt shipped with the gold artifact.
GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256: str = (
    "6e50059b699e6de9d39eefb4602d74a86dd502d83b9c4aad3823321574f11cf4"
)

# Exact canonical bytes of the independently audited seven-model hosted
# completion receipt shipped with the gold artifact.
GOLD_HOSTED_COMPLETION_AUDIT_RECEIPT_SHA256: str = (
    "69d02b4b13318ee185b57b7287ea26ff87ccffdd8b1f66a7979717f0f1ab1888"
)

# Exact public transformation sources recorded when the hosted artifacts were
# imported.  These hashes are execution evidence and therefore remain fixed as
# the public validator and analysis code evolve.  Fresh imports record the
# current hashes of the same closed dependency inventory below.
GOLD_HOSTED_IMPORT_SOURCE_SHA256: dict[str, str] = {
    "benchmark_scripts/benchmark_config.py": (
        "2eab350439b48525eaf0f5b710856a494af61af2c89e1286ed9029514eb64dfb"
    ),
    "benchmark_scripts/dialog_identity.py": (
        "c259193fea2cd45a079b9c0ea1ff1f4b5ed6a030a965b7c43ee738551b306939"
    ),
    "benchmark_scripts/hosted_audit_receipt.py": (
        "08bd899c6e594e8d055cec247898361a16905a2c263f74219e2912d90852d04f"
    ),
    "benchmark_scripts/hosted_completion.py": (
        "b87a8e391f7532157a45b3c42c411f97f8ee2c349fb30b98b01756d091df4213"
    ),
    "benchmark_scripts/hosted_completion_audit_receipt.py": (
        "c0cf3091ce96f495bef093b3d386b2b0852fca288f4d94b18fe936337fb45dcc"
    ),
    "benchmark_scripts/import_hosted_results.py": (
        "ff29713e0d6f9688a0fea70ce7e75be31b5a4a52ee4467a0973a9cc1c59709d1"
    ),
    "benchmark_scripts/provenance_sources.py": (
        "db9b3469255d2c4724b94bc37d0b46e635f92101924a9de77c6c2c1ae2483e77"
    ),
    "benchmark_scripts/record_hosted_provenance.py": (
        "70b1c2347002fd3d96792b5b150eb263e2df21fb23bf82dc81744f339eba0e18"
    ),
    "surrogate/eval_constants.py": (
        "ae46e73d9f14224a1368a2c496ee492f1eced33c8bc7262b67f828f5d51925a6"
    ),
}

HOSTED_IMPORT_SOURCE_FILES: tuple[str, ...] = tuple(GOLD_HOSTED_IMPORT_SOURCE_SHA256)

HOSTED_RECORD_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/hosted_completion.py",
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/record_hosted_provenance.py",
)
