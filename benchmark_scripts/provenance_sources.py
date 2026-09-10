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
# corrected open-model runs. The aggregate digests are SHA-256 over compact,
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

# Exact byte hashes of the raw wide TSVs downloaded from the paper-era
# Archived snapshot. These are intentionally not hashes of normalized Drive
# exports, whose serialization can differ while containing similar records.
RECONCILIATION_ARCHIVE_SHA256: dict[str, str] = {
    "archive/boolq_sentence_consolidated.tsv": (
        "1309dcb70d961e394f9a832dc3ce05ccd85c0d81bdf3a303b9ede1dd5cfcd10b"
    ),
    "archive/anli_r1_sentence_consolidated.tsv": (
        "d6ef0c4d6c42e35e89001a1c37709a8345046bf85412ce2c17521763ec01f66a"
    ),
    "archive/anli_r2_sentence_consolidated.tsv": (
        "52af1f482e5d687d9019a8b61a457f023cf75ff223665925d40c4e8c9c8f8aa5"
    ),
    "archive/anli_r3_sentence_consolidated.tsv": (
        "0a84501fe564613fa3cb99e77f8688daacefca2fdef158c813fac8c9da05a494"
    ),
    "archive/winogrande_sentence_consolidated.tsv": (
        "09fa9d8268f684c68ff75993afad6fd9fb04317df21121549722fc24670d817d"
    ),
    "archive/boolq_word_consolidated.tsv": (
        "8d187bb42a00a16dfb4765ff68fee58a87ed2313eccdfaefd743cba4867c9e46"
    ),
    "archive/lambada_word_consolidated.tsv": (
        "d802a4bc33c5438abdd01d615ea0e06b7d44300f31bad59e006f99c6c71cbec2"
    ),
    "archive/race_sentence_consolidated.tsv": (
        "e36899afe20559f00fef530a623e4496fe9d14ab156eb364a0bed86f64cab77a"
    ),
    "archive/reconciliation_models.json": (
        "d9d4036452f0ed382d64f320d0a806c23fbdfd7bcf4cc9c50e1941f5fdbc5d8c"
    ),
}

RECONCILIATION_ARCHIVE_SIZE_BYTES: dict[str, int] = {
    "archive/boolq_sentence_consolidated.tsv": 64_458_815,
    "archive/anli_r1_sentence_consolidated.tsv": 15_741_412,
    "archive/anli_r2_sentence_consolidated.tsv": 15_767_909,
    "archive/anli_r3_sentence_consolidated.tsv": 19_242_932,
    "archive/winogrande_sentence_consolidated.tsv": 13_244_509,
    "archive/boolq_word_consolidated.tsv": 18_355_541,
    "archive/lambada_word_consolidated.tsv": 15_582_147,
    "archive/race_sentence_consolidated.tsv": 429_766_274,
    "archive/reconciliation_models.json": 944,
}

# The wide RACE table retains only the answer-conditioned scalar signal.  These
# paper-era raw response files are therefore required to independently recover
# the four-label Figure 13 and expanded multiclass audit claims.  Logical
# locators deliberately use public model identifiers rather than the filenames
# or routing aliases used by the original producer.
RECONCILIATION_RACE_RAW_SHA256: dict[str, str] = {
    "archive/race_raw/qwen2.5-0.5b-instruct.json": (
        "58225ee9d957b250346571abe2f740d28f09cd245fbd0efe171f5f6f202dd13b"
    ),
    "archive/race_raw/qwen2.5-3b-instruct.json": (
        "31251ed54d0b199ab288eb63248ab74d821ed7df2f378b48d7fc8a200c061078"
    ),
    "archive/race_raw/llama-3.1-8b-instruct.json": (
        "49c8ae9c3493ab23e6d5c5b914f6f0c25ced8409ccb684fda428bc411628913e"
    ),
    "archive/race_raw/qwen2.5-7b-instruct.json": (
        "9146b28e9a778a2908f39a538570200ebe95776552b7e8b25893b44fe4732fc8"
    ),
    "archive/race_raw/qwen2.5-14b-instruct.json": (
        "f5af4df813872b9e2a19dc6e6ababf3e033809847f046001c77ae5a2839e6cd6"
    ),
    "archive/race_raw/llama3.1-70b-instruct.json": (
        "427406414256f94c90af1a973dcf1ad75258ac895e1e55a189a126b900054d3e"
    ),
    "archive/race_raw/llama3.3-70b-instruct.json": (
        "653718b1bcb720c98c544fc27c3cc7553abeed24f09a3fe2a5f6c6a2e6eb17d7"
    ),
    "archive/race_raw/llama4-maverick-17b-128e-instruct.json": (
        "be1727b16ed1add1bb7f0804a44bc4334df5cdc1159ba91e807950d95e801c93"
    ),
    "archive/race_raw/gpt-4o.json": (
        "5e677c5b6f240cf18d1fd0e3d1c120be5a5a79fe52d56467a503e5e67396d281"
    ),
    "archive/race_raw/gpt-4-1.json": (
        "8d64d83c7db3dc9551eb90c093228f9e518d41e1248a0f9a80bbf92e86079258"
    ),
    "archive/race_raw/gemini-2-5-flash-lite-vertex.json": (
        "03d632fa4c6cc2b70dfae455673f9b2c0aa469e78db189cd1040d5231d7b5800"
    ),
}

RECONCILIATION_RACE_RAW_SIZE_BYTES: dict[str, int] = {
    "archive/race_raw/qwen2.5-0.5b-instruct.json": 75_799_808,
    "archive/race_raw/qwen2.5-3b-instruct.json": 74_899_641,
    "archive/race_raw/llama-3.1-8b-instruct.json": 75_772_209,
    "archive/race_raw/qwen2.5-7b-instruct.json": 74_299_515,
    "archive/race_raw/qwen2.5-14b-instruct.json": 73_280_526,
    "archive/race_raw/llama3.1-70b-instruct.json": 21_363_791,
    "archive/race_raw/llama3.3-70b-instruct.json": 17_208_477,
    "archive/race_raw/llama4-maverick-17b-128e-instruct.json": 17_525_205,
    "archive/race_raw/gpt-4o.json": 20_143_058,
    "archive/race_raw/gpt-4-1.json": 16_296_520,
    "archive/race_raw/gemini-2-5-flash-lite-vertex.json": 21_403_308,
}

# Canonical JSON digest of the 164 sanitized archive rows in reconciliation.tsv,
# sorted by reconciliation_id. This binds every recomputed historical value and
# metadata field without publishing the private wide-table column aliases.
RECONCILIATION_ARCHIVE_LEDGER_SHA256: str = (
    "1f090aaeb9f069e0e69c6e59e6d873411ad4a9945dba3902147f57ae5d896bab"
)

# Canonical JSON digest of every non-corrected-provenance check produced from
# the pinned archive and deterministic historical-claim audit (455 checks,
# sorted by check_id). Corrected-provenance checks are excluded.
RECONCILIATION_ARCHIVE_CHECKS_SHA256: str = (
    "ec6af91ec01f323598b88691ecb8c1a09d6992486b235a19b4d11e581d76a2b4"
)

# Canonical JSON digest of the historical-claims TSV.  This value is updated
# only after running the gold reconciliation against all pinned inputs.
RECONCILIATION_HISTORICAL_CLAIMS_SHA256: str = (
    "ac6fd0b788d3c749eef94652b9a8ca68894106597d27c325c07ecbdbc8a56577"
)

# Canonical digest of the immutable claim transcription for the audited,
# currently unversioned arXiv manuscript reference. Filled only after a full
# reconciliation run; it is distinct from recomputed audit outcomes.
RECONCILIATION_AUDITED_CLAIM_SET_SHA256: str = (
    "07b39fa99ba62247e231569806c2cb74a96d209d5a3f226c81e92016a9053831"
)

OPEN_COMPLETE_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/attention_replay_receipt.py",
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/compute_logodds.py",
    "benchmark_scripts/consolidate_results.py",
    "benchmark_scripts/dialog_identity.py",
    "benchmark_scripts/f_table.py",
    "benchmark_scripts/hosted_audit_receipt.py",
    "benchmark_scripts/hosted_completion_audit_receipt.py",
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/open_execution_audit_receipt.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/race_multiclass.py",
    "benchmark_scripts/race_rv.py",
    "benchmark_scripts/run_q05_boolq_word_eager_diagnostic.py",
    "benchmark_scripts/run_benchmark.py",
    "benchmark_scripts/validate_results.py",
    "surrogate/attention_scoring.py",
    "surrogate/eval_constants.py",
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
# hashes and modification times are preserved in the execution audit receipt;
# because it was reconstructed from unchanged files after launch, it is
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

# Future runs hash this complete execution dependency set directly.  The gold
# rerun predates that metadata-only hardening, hence the separate observed
# snapshot above.
OPEN_EXECUTION_SOURCE_FILES: tuple[str, ...] = tuple(
    GOLD_OPEN_OBSERVED_EXECUTION_SOURCE_SHA256
)

# Exact, narrowly scoped execution-to-release source changes made only after
# all corrected queues and diagnostic replays completed.
GOLD_OPEN_RELEASE_SOURCE_CORRECTIONS: dict[str, dict[str, str]] = {
    "benchmark_scripts/benchmark_config.py": {
        "execution_sha256": (
            "2c0f28b30cd8779af076c6a6c5813356eeb5d8b0eb07bf883159a7ada64845aa"
        ),
        "release_sha256": (
            "2eab350439b48525eaf0f5b710856a494af61af2c89e1286ed9029514eb64dfb"
        ),
        "reason": "Canonicalize both public Llama repository locators.",
        "scope": "model_locator_only_no_numerical_change",
    },
    "benchmark_scripts/run_benchmark.py": {
        "execution_sha256": (
            "f5b4549e1df21833d674d7425a613100d8bbb0a7cb606b6b9a81427dc07b2987"
        ),
        "release_sha256": (
            "c4d0555ce2d9687eeb4e84e93ebc8f53d67832989e6cea2b4c31af23d1e3700a"
        ),
        "reason": (
            "Hash the complete execution dependency set once at run start for "
            "future runs."
        ),
        "scope": "provenance_only_no_numerical_change",
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

# Filled from the deterministic portable receipts after the full corrected
# open-model grid and the separate BoolQ-word attention replay complete.
GOLD_OPEN_EXECUTION_AUDIT_RECEIPT_SHA256: str = (
    "569cbeb9655f13f60ff5fe611d184a3ad316b3b3959f19d677ebb75ce89b339f"
)
GOLD_OPEN_ATTENTION_REPLAY_RECEIPT_SHA256: str = (
    "d75ebf201c624e8d7e7e3c5eeb1067eae98ab4b091032b734ce95da9710ca782"
)

# The eager-backend Qwen-0.5B BoolQ-word check uses a dedicated public
# diagnostic runner so it cannot alter the frozen main execution sources.
# Its exact digest is included in the release receipt.
Q05_BOOLQ_WORD_EAGER_DIAGNOSTIC_WRAPPER_SHA256: str = (
    "ecf513dfb80ddf830679ac44975cd1ad9faf46e59de01c4a25c347205123ca90"
)

HOSTED_IMPORT_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/dialog_identity.py",
    "benchmark_scripts/hosted_audit_receipt.py",
    "benchmark_scripts/hosted_completion.py",
    "benchmark_scripts/hosted_completion_audit_receipt.py",
    "benchmark_scripts/import_hosted_results.py",
    "benchmark_scripts/provenance_sources.py",
    "benchmark_scripts/record_hosted_provenance.py",
    "surrogate/eval_constants.py",
)

HOSTED_RECORD_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/hosted_completion.py",
    "benchmark_scripts/normalize_segment_outputs.py",
    "benchmark_scripts/record_hosted_provenance.py",
)
