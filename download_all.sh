#!/usr/bin/env bash
# Copyright (c) 2025 The Authors
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
    ("ehovy/race",                "all",             "test"),
    ("EleutherAI/lambada_openai", "default",        "test"),
]
for path, name, split in jobs:
    print(f"-- {path} {name} {split}")
    load_dataset(path, name, split=split, cache_dir=CACHE)
PY

echo "==> Downloading pinned open models (Llama requires HF_TOKEN)"
while read -r repository local_name revision; do
  hf download "$repository" --revision "$revision" \
    --local-dir "/tmp/models/$local_name" --max-workers 8
done <<'EOF'
Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-0.5B-Instruct 7ae557604adf67be50417f59c2c2f167def9a775
Qwen/Qwen2.5-3B-Instruct Qwen2.5-3B-Instruct aa8e72537993ba99e69dfaafa59ed015b17504d1
Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28
Qwen/Qwen2.5-14B-Instruct Qwen2.5-14B-Instruct cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
meta-llama/Llama-3.1-8B-Instruct Meta-Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659
EOF

echo "==> Done"
du -sh /tmp/models/* /tmp/datasets/* 2>/dev/null || true
