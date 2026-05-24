#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Run the full benchmark suite across all available model sets.
#
# Model sets:
#   Qwen2.5-Instruct : 0.5B, 3B, 7B, 14B
#   Qwen2.5-Base     : 0.5B, 3B, 7B, 14B, 32B
#   Llama3.1-Instruct: 8B, 70B
#
# Results are saved to /tmp/{pregrouper}_{phase}/{benchmark}/{model}.json
# by run_benchmark.py's default behavior.

set -euo pipefail

cd "$(dirname "$0")/.."

BENCHMARKS=(boolq anli_r1 anli_r2 anli_r3 winogrande lambada)
MODEL_SETS=("Qwen2.5-Instruct" "Qwen2.5-Base" "Llama3.1-Instruct")

SUCCESSES=0
FAILURES=0
FAILED_RUNS=()

run_benchmark() {
    local benchmark="$1"
    local model_set="$2"
    local extra_args="${3:-}"

    echo ""
    echo "================================================================"
    echo "  Benchmark: ${benchmark} | Model set: ${model_set}"
    echo "================================================================"

    local cmd="python -m benchmark_scripts.run_benchmark --benchmark ${benchmark} --model-set ${model_set} ${extra_args}"

    if eval "$cmd"; then
        SUCCESSES=$((SUCCESSES + 1))
    else
        FAILURES=$((FAILURES + 1))
        FAILED_RUNS+=("${benchmark} / ${model_set}")
        echo "*** FAILED: ${benchmark} / ${model_set} ***"
    fi
}

echo "Starting full benchmark suite (all models, all benchmarks)"
echo "Start time: $(date)"
echo ""

for benchmark in "${BENCHMARKS[@]}"; do
    # lambada uses completion_logprob scoring and requires --max-forward-passes
    if [[ "$benchmark" == "lambada" ]]; then
        extra_args="--max-forward-passes 10000"
    else
        extra_args=""
    fi

    for model_set in "${MODEL_SETS[@]}"; do
        run_benchmark "$benchmark" "$model_set" "$extra_args"
    done
done

echo ""
echo "================================================================"
echo "  SUMMARY"
echo "================================================================"
echo "Successes: ${SUCCESSES}"
echo "Failures:  ${FAILURES}"
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
    echo ""
    echo "Failed runs:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - ${run}"
    done
fi
echo ""
echo "End time: $(date)"

if [[ "$FAILURES" -gt 0 ]]; then
    exit 1
fi
