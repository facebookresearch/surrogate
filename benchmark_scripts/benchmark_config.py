# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark-specific configuration: how to load data, build prompts,
and configure scoring for each supported benchmark.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from surrogate.eval_constants import (
    ANLI_CONFIG,
    BOOLQ_CONFIG,
    EvalConfig,
    WINOGRANDE_CONFIG,
)

logger: logging.Logger = logging.getLogger(__name__)

# Lightweight config for completion-logprob benchmarks (no label tokens)
LAMBADA_SYSTEM_PROMPT: str = (
    "You are a language model. You will be shown a passage with the final "
    "word removed. Predict the missing final word."
)

LOCAL_MODEL_DIR: str = "/tmp/models/"
LOCAL_DATASET_DIR: str = "/tmp/datasets/"

MODELS_QWEN25_INSTRUCT: list[tuple[str, str]] = [
    ("qwen2.5-0.5b-instruct", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-3b-instruct", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen2.5-14b-instruct", "Qwen/Qwen2.5-14B-Instruct"),
]

MODELS_QWEN25_BASE: list[tuple[str, str]] = [
    ("qwen2.5-0.5b-base", "Qwen/Qwen2.5-0.5B"),
    ("qwen2.5-3b-base", "Qwen/Qwen2.5-3B"),
    ("qwen2.5-7b-base", "Qwen/Qwen2.5-7B"),
    ("qwen2.5-14b-base", "Qwen/Qwen2.5-14B"),
    ("qwen2.5-32b-base", "Qwen/Qwen2.5-32B"),
]

MODELS_LLAMA31_INSTRUCT: list[tuple[str, str]] = [
    ("llama-3.1-8b-instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ("llama-3.1-70b-instruct", "meta-llama/Meta-Llama-3.1-70B-Instruct"),
]


MODEL_SETS: dict[str, list[tuple[str, str]]] = {
    "Qwen2.5-Instruct": MODELS_QWEN25_INSTRUCT,
    "Qwen2.5-Base": MODELS_QWEN25_BASE,
    "Llama3.1-Instruct": MODELS_LLAMA31_INSTRUCT,
}


@dataclass
class BenchmarkSpec:
    """Everything needed to run a benchmark: data source, prompt builder, eval config."""

    name: str
    eval_config: EvalConfig | None  # None for completion-logprob benchmarks
    hf_dataset_path: str
    prompt_builder: Callable[[pd.Series], str]  # row -> prompt text
    answer_column: str  # column name for ground truth
    hf_dataset_name: str | None = None
    hf_split: str = "test"
    scoring_mode: str = "logit_difference"  # "logit_difference" or "completion_logprob"
    target_column: str | None = (
        None  # column for per-prompt target (completion_logprob)
    )
    system_prompt_override: str | None = None  # override eval_config system prompt
    dataset_preprocessor: Callable[[pd.DataFrame], pd.DataFrame] | None = None


def _boolq_prompt(row: pd.Series) -> str:
    return f"Context:\n{row['passage']}.\n\nQuestion:\n{row['question']}."


def _anli_prompt(row: pd.Series) -> str:
    return f"Premise:\n{row['premise']}.\n\nHypothesis:\n{row['hypothesis']}."


def _winogrande_prompt(row: pd.Series) -> str:
    return (
        f"Sentence:\n{row['sentence']}\n\n"
        f"Option 1:\n{row['option1']}\n\n"
        f"Option 2:\n{row['option2']}"
    )


def _lambada_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["text"].str.rsplit(n=1)
    df["context"] = parts.str[0]
    df["target"] = parts.str[1]
    return df


def _lambada_prompt(row: pd.Series) -> str:
    return str(row["context"])


BENCHMARKS: dict[str, BenchmarkSpec] = {
    "boolq": BenchmarkSpec(
        name="boolq",
        eval_config=BOOLQ_CONFIG,
        hf_dataset_path="aps/super_glue",
        hf_dataset_name="boolq",
        hf_split="validation",
        prompt_builder=_boolq_prompt,
        answer_column="label",
    ),
    "anli_r1": BenchmarkSpec(
        name="anli_r1",
        eval_config=ANLI_CONFIG,
        hf_dataset_path="facebook/anli",
        hf_split="test_r1",
        prompt_builder=_anli_prompt,
        answer_column="label",
    ),
    "anli_r2": BenchmarkSpec(
        name="anli_r2",
        eval_config=ANLI_CONFIG,
        hf_dataset_path="facebook/anli",
        hf_split="test_r2",
        prompt_builder=_anli_prompt,
        answer_column="label",
    ),
    "anli_r3": BenchmarkSpec(
        name="anli_r3",
        eval_config=ANLI_CONFIG,
        hf_dataset_path="facebook/anli",
        hf_split="test_r3",
        prompt_builder=_anli_prompt,
        answer_column="label",
    ),
    "winogrande": BenchmarkSpec(
        name="winogrande",
        eval_config=WINOGRANDE_CONFIG,
        hf_dataset_path="allenai/winogrande",
        hf_dataset_name="winogrande_xl",
        hf_split="validation",
        prompt_builder=_winogrande_prompt,
        answer_column="answer",
    ),
    "lambada": BenchmarkSpec(
        name="lambada",
        eval_config=None,
        hf_dataset_path="EleutherAI/lambada_openai",
        hf_dataset_name="default",
        hf_split="test",
        prompt_builder=_lambada_prompt,
        answer_column="target",
        scoring_mode="completion_logprob",
        target_column="target",
        system_prompt_override=LAMBADA_SYSTEM_PROMPT,
        dataset_preprocessor=_lambada_preprocess,
    ),
}


def resolve_model_path(hf_hub_id: str) -> str:
    """Return local path if model is cached under LOCAL_MODEL_DIR, else download.

    Checks ``/tmp/models/<basename>`` first (e.g.
    ``/tmp/models/Qwen2.5-0.5B-Instruct``). If the directory is empty or
    absent, attempts to download from HuggingFace Hub via ``snapshot_download``.
    Falls back to the raw *hf_hub_id* (which ``from_pretrained`` can resolve
    directly) if the download fails.

    For gated models (e.g. Llama), set the ``HF_TOKEN`` environment variable
    to a valid HuggingFace access token.
    """
    local_dir: str = os.path.join(LOCAL_MODEL_DIR, os.path.basename(hf_hub_id))
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        logger.info(f"Using cached model at {local_dir}")
        return local_dir

    try:
        from huggingface_hub import snapshot_download

        token: str | None = os.environ.get("HF_TOKEN")
        os.makedirs(local_dir, exist_ok=True)
        logger.info(f"Downloading {hf_hub_id} to {local_dir} ...")
        snapshot_download(repo_id=hf_hub_id, local_dir=local_dir, token=token)
        return local_dir
    except (ImportError, OSError) as e:
        logger.warning(
            f"Could not download {hf_hub_id} to {local_dir}: {e}. "
            f"Falling back to HF hub ID."
        )
        return hf_hub_id


def _dataset_cache_dir(spec: "BenchmarkSpec") -> str:
    """Return the expected HF datasets cache subdirectory for *spec*.

    HuggingFace ``datasets`` stores cached data under
    ``<cache_dir>/<org>___<dataset>/`` (slashes replaced with ``___``).
    """
    folder: str = spec.hf_dataset_path.replace("/", "___")
    return os.path.join(LOCAL_DATASET_DIR, folder)


def load_benchmark_dataset(spec: "BenchmarkSpec") -> pd.DataFrame:
    """Load benchmark data from local cache or HuggingFace Hub.

    Datasets are cached under ``LOCAL_DATASET_DIR`` (``/tmp/datasets/``).
    If the dataset is already cached it is loaded from disk; otherwise it is
    downloaded from HuggingFace Hub on first access.
    """
    from datasets import load_dataset

    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

    cache_subdir: str = _dataset_cache_dir(spec)
    if os.path.isdir(cache_subdir) and os.listdir(cache_subdir):
        logger.info(f"Using cached dataset for {spec.name} at {cache_subdir}")
    else:
        logger.info(
            f"Dataset {spec.name} not found at {cache_subdir}, "
            f"downloading from HuggingFace ({spec.hf_dataset_path}) ..."
        )

    ds: Any = load_dataset(
        spec.hf_dataset_path,
        spec.hf_dataset_name,
        split=spec.hf_split,
        cache_dir=LOCAL_DATASET_DIR,
    )
    df: pd.DataFrame = ds.to_pandas()
    if spec.dataset_preprocessor is not None:
        df = spec.dataset_preprocessor(df)
    logger.info(f"Loaded {spec.name}: {len(df)} rows (cache_dir={LOCAL_DATASET_DIR})")
    return df
