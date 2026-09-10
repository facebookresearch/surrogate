# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Run the Qwen-0.5B BoolQ-word ablation backend diagnostic."""

from __future__ import annotations

import argparse
import asyncio
import os

from benchmark_scripts import run_benchmark as runner


def _eager_backend(phase: str) -> str:
    """Force eager attention for the diagnostic ablation execution."""
    if phase not in runner.VALID_PHASES:
        raise ValueError(f"Unknown phase: {phase}")
    return "eager"


def main() -> None:
    """Execute the fixed diagnostic with caller-supplied portable paths."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--dataset-file", required=True)
    args: argparse.Namespace = parser.parse_args()

    os.environ["BENCHMARK_MODELS"] = "qwen2.5-0.5b-instruct"
    runner._attention_implementation_for_phase = _eager_backend
    asyncio.run(
        runner.run_benchmark(
            "boolq",
            pregrouper="word",
            phases={"ablation"},
            batch_size=8,
            max_forward_passes=10_000,
            seed=42,
            model_set="Qwen2.5-Instruct",
            results_dir=args.results_dir,
            dataset_file=args.dataset_file,
            overwrite_existing=True,
        )
    )


if __name__ == "__main__":
    main()
