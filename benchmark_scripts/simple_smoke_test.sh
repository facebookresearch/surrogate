#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#
# Quick smoke test: run all 6 benchmarks with only Qwen2.5-0.5B-Instruct
# and 10 samples each. Useful for validating the pipeline end-to-end
# without waiting for large models or full datasets.
#
# Usage:
#   bash benchmark_scripts/run_small_test.sh

set -euo pipefail

cd "$(dirname "$0")/.."

BENCHMARKS=(boolq anli_r1 anli_r2 anli_r3 winogrande lambada)
MODEL_FILTER="qwen2.5-0.5b-instruct"
MODEL_SET="Qwen2.5-Instruct"
MAX_SAMPLES=10

SUCCESSES=0
FAILURES=0
FAILED_RUNS=()

echo "Starting small smoke test (Qwen2.5-0.5B-Instruct, ${MAX_SAMPLES} samples)"
echo "Start time: $(date)"

for benchmark in "${BENCHMARKS[@]}"; do
    echo ""
    echo "================================================================"
    echo "  Benchmark: ${benchmark} | Model: ${MODEL_FILTER} | Samples: ${MAX_SAMPLES}"
    echo "================================================================"

    extra_args=""
    if [[ "$benchmark" == "lambada" ]]; then
        extra_args="--max-forward-passes 10000"
    fi

    if BENCHMARK_MODELS="$MODEL_FILTER" python -m benchmark_scripts.run_benchmark \
        --benchmark "$benchmark" \
        --model-set "$MODEL_SET" \
        --max-samples "$MAX_SAMPLES" \
        $extra_args; then
        SUCCESSES=$((SUCCESSES + 1))
    else
        FAILURES=$((FAILURES + 1))
        FAILED_RUNS+=("$benchmark")
        echo "*** FAILED: ${benchmark} ***"
    fi
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
