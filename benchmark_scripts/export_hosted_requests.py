# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Export the canonical hosted-classification request dialogs as JSONL.

This module contains no service client. It writes the exact original and
ablated dialogs that a hosted producer must score, and refuses to write them
unless their canonical identity digests equal those recorded in the gold
hosted-classification audit receipt. A producer that scores these lines
therefore sees byte-identical inputs to the audited hosted runs.

The first line is a header carrying the configuration, label-token sets, and
identity digests; each later line is one request with ``seg_idx`` null for the
original prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from typing import Any

import pandas as pd
from benchmark_scripts.benchmark_config import BENCHMARKS
from benchmark_scripts.dialog_identity import _canonical_line, _messages
from benchmark_scripts.hosted_audit_receipt import load_receipt
from benchmark_scripts.provenance_sources import (
    GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
)
from surrogate.model_types import Dialog, make_dialog
from surrogate.text_augmentation import segment_and_ablate


async def build_requests(
    benchmark: str,
    pregrouper: str,
    dataset_path: str,
    manifest_path: str,
) -> tuple[list[dict[str, Any]], str, str]:
    """
    Build every original and manifest-selected ablated request dialog.

    Args:
        benchmark: Classification benchmark key in ``BENCHMARKS``.
        pregrouper: Segmentation granularity of the manifest.
        dataset_path: Frozen dataset TSV snapshot.
        manifest_path: Public ``segments.tsv.gz`` coordinate manifest.

    Returns:
        ``(requests, prompt_identity_sha256, ablated_dialog_identity_sha256)``
        where the digests use the same canonical encoding as
        ``dialog_identity.compute_dialog_identity``.
    """
    spec = BENCHMARKS[benchmark]
    if spec.eval_config is None:
        raise ValueError("Hosted request export supports classification only")
    frame: pd.DataFrame = pd.read_csv(dataset_path, sep="\t")
    if spec.dataset_preprocessor is not None:
        frame = spec.dataset_preprocessor(frame.copy())
    manifest: pd.DataFrame = pd.read_csv(manifest_path, sep="\t")
    prompt_hash = hashlib.sha256()
    ablated_hash = hashlib.sha256()
    requests: list[dict[str, Any]] = []
    for prompt_idx, rows in sorted(manifest.groupby("prompt_idx", sort=False)):
        rows = rows.sort_values("seg_idx")
        answers: list[Any] = rows["answer"].unique().tolist()
        if len(answers) != 1:
            raise ValueError(f"Prompt {prompt_idx} has conflicting answers")
        answer: Any = answers[0].item() if hasattr(answers[0], "item") else answers[0]
        n_segments: int = int(rows["n_segments"].iloc[0])
        dialog: Dialog = make_dialog(
            spec.eval_config.system_prompt,
            spec.prompt_builder(frame.iloc[int(prompt_idx)]),
        )
        messages: list[list[str]] = _messages(dialog)
        prompt_hash.update(_canonical_line([int(prompt_idx), messages]))
        common: dict[str, Any] = {
            "prompt_idx": int(prompt_idx),
            "answer": answer,
            "n_segments": n_segments,
        }
        requests.append({**common, "seg_idx": None, "messages": messages})
        ablated: list[Dialog] = await segment_and_ablate(
            dialog, pregrouper_id=pregrouper
        )
        if len(ablated) != n_segments:
            raise ValueError(f"Ablation count mismatch at prompt {prompt_idx}")
        for seg_idx in rows["seg_idx"].astype(int):
            ablated_messages: list[list[str]] = _messages(ablated[seg_idx])
            ablated_hash.update(
                _canonical_line([int(prompt_idx), seg_idx, ablated_messages])
            )
            requests.append(
                {**common, "seg_idx": seg_idx, "messages": ablated_messages}
            )
    return requests, prompt_hash.hexdigest(), ablated_hash.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--pregrouper", choices=["sentence", "word"], required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    manifest_path: str = os.path.join(
        args.results_dir, args.benchmark, args.pregrouper, "segments.tsv.gz"
    )
    receipt = load_receipt(
        os.path.join(args.results_dir, "hosted_classification_audit_receipt.json"),
        expected_sha256=GOLD_HOSTED_CLASSIFICATION_AUDIT_RECEIPT_SHA256,
    )
    configuration: dict[str, Any] = next(
        row
        for row in receipt["configurations"]
        if (row["benchmark"], row["pregrouper"]) == (args.benchmark, args.pregrouper)
    )
    requests, prompt_sha256, ablated_sha256 = asyncio.run(
        build_requests(
            args.benchmark, args.pregrouper, args.dataset_file, manifest_path
        )
    )
    if (prompt_sha256, ablated_sha256) != (
        configuration["prompt_identity_sha256"],
        configuration["ablated_dialog_identity_sha256"],
    ):
        raise ValueError(
            "Rebuilt request dialogs disagree with the audited hosted identity"
        )
    eval_config = BENCHMARKS[args.benchmark].eval_config
    assert eval_config is not None
    header: dict[str, Any] = {
        "benchmark": args.benchmark,
        "pregrouper": args.pregrouper,
        "manifest_sha256": configuration["manifest_sha256"],
        "prompt_identity_sha256": prompt_sha256,
        "ablated_dialog_identity_sha256": ablated_sha256,
        "label_tokens": {
            label: sorted(tokens) for label, tokens in eval_config.label_tokens.items()
        },
        "n_requests": len(requests),
    }
    with open(args.output, "w", encoding="utf-8") as output:
        for line in [header, *requests]:
            output.write(json.dumps(line, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
