#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Download every model and dataset needed by surrogate benchmarks.
# Models -> /tmp/models/<basename>
# Datasets -> /tmp/datasets/<org>___<dataset>/
#
# Requires: pip install huggingface_hub datasets
# For Llama (gated): export HF_TOKEN=hf_xxx with accepted license on HF.

set -euo pipefail

mkdir -p /tmp/models /tmp/datasets

echo "==> Downloading datasets"
python3 - <<'PY'
from datasets import load_dataset
CACHE = "/tmp/datasets"
jobs = [
    ("aps/super_glue",            "boolq",          "validation"),
    ("facebook/anli",             None,             "test_r1"),
    ("facebook/anli",             None,             "test_r2"),
    ("facebook/anli",             None,             "test_r3"),
    ("allenai/winogrande",        "winogrande_xl",  "validation"),
    ("EleutherAI/lambada_openai", "default",        "test"),
]
for path, name, split in jobs:
    print(f"-- {path} {name} {split}")
    load_dataset(path, name, split=split, cache_dir=CACHE)
PY

echo "==> Downloading Qwen2.5 instruct + base"
QWEN=(
  Qwen2.5-0.5B-Instruct
  Qwen2.5-3B-Instruct
  Qwen2.5-7B-Instruct
  Qwen2.5-14B-Instruct
  Qwen2.5-0.5B
  Qwen2.5-3B
  Qwen2.5-7B
  Qwen2.5-14B
  Qwen2.5-32B
)
for m in "${QWEN[@]}"; do
  hf download "Qwen/$m" --local-dir "/tmp/models/$m" --max-workers 8
done

echo "==> Downloading Llama-3.1 (gated; needs HF_TOKEN)"
LLAMA=(
  Meta-Llama-3.1-8B-Instruct
  # Meta-Llama-3.1-70B-Instruct  # Skipped: too big for typical local GPUs;
                                  # score via a hosted API instead.
)
for m in "${LLAMA[@]}"; do
  hf download "meta-llama/$m" --local-dir "/tmp/models/$m" --max-workers 8
done

echo "==> Done"
du -sh /tmp/models/* /tmp/datasets/* 2>/dev/null || true
