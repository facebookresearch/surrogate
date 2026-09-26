# Copyright (c) 2025 The Authors
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Write deterministic provenance sidecars for derived result tables."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from typing import Any

PROVENANCE_SCHEMA_VERSION: int = 1
SOFTWARE_DISTRIBUTIONS: tuple[str, ...] = ("numpy", "pandas", "scipy", "tqdm")
RAW_RESULT_SUFFIXES: tuple[str, ...] = (
    "_segment.tsv.gz",
    "_tokens.tsv.gz",
    "_run.json",
    "_canary.json",
)
DERIVED_SUPPORTING_SOURCE_FILES: tuple[str, ...] = (
    "benchmark_scripts/benchmark_config.py",
    "benchmark_scripts/compute_logodds.py",
    "benchmark_scripts/consolidate_results.py",
    "benchmark_scripts/derived_provenance.py",
    "benchmark_scripts/hosted_audit_receipt.py",
    "benchmark_scripts/run_all_benchmarks.sh",
    "surrogate/eval_constants.py",
)


def derived_supporting_source_paths() -> dict[str, str]:
    """Return the single canonical source inventory for derived tables."""
    repository_root: str = os.path.dirname(os.path.dirname(__file__))
    return {
        relative_path: os.path.join(repository_root, relative_path)
        for relative_path in DERIVED_SUPPORTING_SOURCE_FILES
    }


def sha256_file(path: str) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: str, root_dir: str) -> str:
    """Return a result-relative path without exposing machine-local prefixes."""
    absolute_path: str = os.path.abspath(path)
    absolute_root: str = os.path.abspath(root_dir)
    relative: str = os.path.relpath(absolute_path, absolute_root)
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        relative = os.path.basename(absolute_path)
    return relative.replace(os.sep, "/")


def _software_versions() -> dict[str, str]:
    """Return versions of the runtime and numerical packages used here."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    for distribution in SOFTWARE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def collect_result_inputs(
    results_dir: str,
    benchmark: str,
    pregrouper: str,
    immediate_inputs: dict[str, str],
    models: tuple[str, ...] | None = None,
) -> dict[str, str]:
    """Collect immediate tables and their committed raw reconstruction inputs.

    Args:
        results_dir: Root of the result artifact.
        benchmark: Benchmark identifier used in result paths.
        pregrouper: Segmentation granularity.
        immediate_inputs: Logical names and paths read directly by an analyzer.
        models: Optional exact model cohort whose raw files affect the output.

    Returns:
        Stable logical IDs mapped to every existing immediate input, canonical
        manifest, per-model raw table, run record, and canary record.
    """
    prefix: str = f"{benchmark}/{pregrouper}"
    inputs: dict[str, str] = {
        f"{prefix}/immediate/{name}": path
        for name, path in immediate_inputs.items()
        if os.path.isfile(path)
    }
    config_dir: str = os.path.join(results_dir, benchmark, pregrouper)
    if not os.path.isdir(config_dir):
        return inputs
    for name in sorted(os.listdir(config_dir)):
        path: str = os.path.join(config_dir, name)
        if not os.path.isfile(path):
            continue
        is_manifest: bool = name in {"segments.tsv", "segments.tsv.gz"}
        is_model_input: bool = name.endswith(RAW_RESULT_SUFFIXES) and (
            models is None or any(name.startswith(f"{model}_") for model in models)
        )
        if is_manifest or is_model_input:
            inputs[f"{prefix}/raw/{name}"] = path
    raw_inputs: dict[str, str] = {
        identifier: path for identifier, path in inputs.items() if "/raw/" in identifier
    }
    # Consolidated TSVs are deterministic scratch products and intentionally
    # excluded from release commits. Bind the sidecar to the complete shipped
    # raw reconstruction inputs whenever they are available; retain immediate
    # inputs only for standalone/custom invocations without a raw artifact.
    return raw_inputs or inputs


def write_derived_provenance(
    output_path: str,
    *,
    generator_name: str,
    generator_path: str,
    input_paths: dict[str, str],
    parameters: dict[str, Any],
    root_dir: str,
    supporting_source_paths: dict[str, str] | None = None,
) -> str:
    """Write a sidecar binding a derived output to inputs and parameters.

    Args:
        output_path: Derived table whose bytes are being sealed.
        generator_name: Public import path of the generating module.
        generator_path: Source file implementing the generator.
        input_paths: Mapping from stable logical input IDs to local paths.
        parameters: Complete JSON-serializable analysis parameters.
        root_dir: Root used to make artifact paths portable.
        supporting_source_paths: Public repository-relative source IDs mapped
            to files that prepare or orchestrate the derived inputs.

    Returns:
        The path to ``<output_path>.provenance.json``.

    Raises:
        FileNotFoundError: If the output, generator, or any declared input is
            absent.
        ValueError: If an input ID is empty.
        TypeError: If parameters are not JSON serializable.
    """
    source_paths: dict[str, str] = supporting_source_paths or {}
    required_paths: list[str] = [
        output_path,
        generator_path,
        *input_paths.values(),
        *source_paths.values(),
    ]
    for path in required_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    if any(not str(identifier).strip() for identifier in input_paths):
        raise ValueError("Derived provenance input IDs must be nonempty")
    if any(
        not identifier.strip()
        or os.path.isabs(identifier)
        or identifier == os.pardir
        or identifier.startswith(os.pardir + "/")
        for identifier in source_paths
    ):
        raise ValueError("Supporting source IDs must be public relative paths")

    inputs: dict[str, dict[str, str | int]] = {
        identifier: {
            "path": _portable_path(path, root_dir),
            "sha256": sha256_file(path),
            "size_bytes": os.path.getsize(path),
        }
        for identifier, path in sorted(input_paths.items())
    }
    payload: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "artifact_type": "derived_table",
        "generator": {
            "module": generator_name,
            "source_sha256": sha256_file(generator_path),
            "provenance_writer_sha256": sha256_file(__file__),
        },
        "supporting_sources": {
            identifier: {
                "sha256": sha256_file(path),
                "size_bytes": os.path.getsize(path),
            }
            for identifier, path in sorted(source_paths.items())
        },
        "output": {
            "path": _portable_path(output_path, root_dir),
            "sha256": sha256_file(output_path),
            "size_bytes": os.path.getsize(output_path),
        },
        "inputs": inputs,
        "parameters": parameters,
        "software": _software_versions(),
    }
    sidecar_path: str = f"{output_path}.provenance.json"
    output_directory: str = os.path.dirname(os.path.abspath(sidecar_path))
    os.makedirs(output_directory, exist_ok=True)
    serialized: str = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output_directory, delete=False
    ) as temporary:
        temporary_path: str = temporary.name
        temporary.write(serialized)
    os.replace(temporary_path, sidecar_path)
    return sidecar_path
