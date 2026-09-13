# Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?

Reference implementation and audited benchmark artifacts for measuring
prediction, attribution, attention, and representation fidelity across language
models.

Models always receive the complete dialog. The headline segment-level analysis
selects user-message coordinates from that shared full-dialog segmentation.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Method

The release uses these primary choices:

- User-message segment coordinates, with the full dialog retained as context.
- Entailment-minus-contradiction log odds for ANLI.
- Model-pair-specific complete cases for non-finite hosted scores.
- Row-pooled point estimates with prompt-cluster bootstrap intervals.
- Signed Spearman and Pearson correlations, plus Pearson \(r^2\).
- Pair-specific observation and prompt coverage reported with every estimate.

A successful hosted response whose requested label is absent from top-k is
stored as `-inf`. A request without a valid response is unavailable and becomes
`NaN` in derived scores. These states remain distinct in the raw artifacts.

The finite-extreme table is a non-primary sensitivity analysis. It replaces a
hosted model's infinities just outside that signal's observed finite range,
where such a range exists, while leaving unavailable values missing.

## Running benchmarks

The pipeline writes model-independent segment identities, raw model outputs,
deterministic derived tables, and provenance sidecars:

```text
run_benchmark       -> results/{benchmark}/{pregrouper}/segments.tsv.gz
                       results/{benchmark}/{pregrouper}/{model}_segment.tsv.gz
                       results/{benchmark}/{pregrouper}/{model}_tokens.tsv.gz
consolidate_results -> results/{benchmark}_{pregrouper}_segments.tsv
                       results/{benchmark}_{pregrouper}_tokens.tsv
compute_logodds     -> results/{benchmark}_{pregrouper}_logodds.tsv
f_table             -> results/f_table.tsv
race_rv             -> results/race_rv.tsv
```

### 1. Generate open-model outputs

`run_benchmark` supports BoolQ, ANLI R1-R3, WinoGrande, RACE, and LAMBADA.
For an audited reproduction, provide the frozen source TSV named in
[`results/README.md`](results/README.md). Live Hugging Face datasets are useful
for exploration but do not reproduce the frozen artifact byte-for-byte.

```bash
python -m benchmark_scripts.run_benchmark \
    --benchmark boolq \
    --dataset-file /path/to/google_boolq_validation.tsv \
    --results-dir results/rerun

python -m benchmark_scripts.run_benchmark \
    --benchmark boolq \
    --pregrouper word \
    --max-forward-passes 10000 \
    --dataset-file /path/to/google_boolq_validation.tsv \
    --results-dir results/rerun
```

Both attention and ablation use the same full-dialog segmentation. Every row
records global and message-local segment indices, role, and exact segment text;
the runner checks alignment before merging the phases.

The batch script runs the five open models across all eight configurations on
two GPUs:

```bash
DATASET_DIR=/path/to/frozen-tsvs \
RESULTS_DIR=results/rerun \
bash benchmark_scripts/run_all_benchmarks.sh
```

For a small functional check:

```bash
bash benchmark_scripts/simple_smoke_test.sh
```

### 2. Import hosted-model outputs

No hosted-service client or provider-specific routing code is included.
Portable JSON results can be imported with
`benchmark_scripts.import_hosted_results`. The importer validates each prompt
and ablation against `segments.tsv.gz`, preserves request-status metadata, and
requires a sanitized producer receipt for audited artifacts.

```bash
python -m benchmark_scripts.import_hosted_results \
    --input /path/to/model.json \
    --manifest results/boolq/sentence/segments.tsv.gz \
    --output-dir results/boolq/sentence \
    --model hosted-model-name \
    --benchmark boolq \
    --pregrouper sentence \
    --identity-attestation \
      "Hosted prompt and ablation coordinates were verified against the canonical public segment manifest." \
    --producer-revision 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --generated-at 2026-09-08T00:00:00Z \
    --served-model hosted-model-name \
    --request-parameters-json '{"max_tokens":1,"top_logprobs":19,"echo":false,"scoring":"first_generated_token_label_logprobs"}' \
    --producer-audit-receipt results/hosted_classification_audit_receipt.json
```

### 3. Consolidate and derive scores

```bash
python -m benchmark_scripts.normalize_segment_outputs results/boolq/sentence
python -m benchmark_scripts.consolidate_results --benchmark boolq
python -m benchmark_scripts.compute_logodds --benchmark boolq
```

Token tables retain every queried ANLI label, so all directed contrasts can be
computed without another model run. The headline analysis selects
`logodds_entailment_contradiction`.

### 4. Generate fidelity tables

```bash
# Canonical scalar analysis
python -m benchmark_scripts.f_table

# Non-primary missingness sensitivity
python -m benchmark_scripts.f_table \
    --api-infinity-policy finite_extreme \
    --output results/f_table_finite_extreme_sensitivity.tsv

# Canonical multivariate RACE analysis
python -m benchmark_scripts.race_rv
```

`F_pred` is prompt-level and therefore uses the full dialog. Other metrics use
the requested segment coordinates. ANLI E-C `F_align` and
`F_align_to_attr` are explicitly unavailable in the current artifact because
the stored open-model readout projection used a different contrast; the code
does not relabel those values.

Additional scopes and ANLI contrasts remain reproducible from the raw files:

```bash
python -m benchmark_scripts.f_table --scopes all system user
python -m benchmark_scripts.f_table \
    --benchmarks anli_r1 anli_r2 anli_r3 \
    --scopes all system user \
    --contrasts entailment_neutral entailment_contradiction \
    --anli-contrast entailment_contradiction \
    --output results/anli_sensitivity.tsv
```

RACE is evaluated as a multivariate four-label signal using centered RV. The
`all_pairs` representation is the six pairwise A-D margin system;
`anchor_a` is included as a three-dimensional sensitivity. Hosted comparisons
use model-pair-specific finite complete cases and report their coverage.

### 5. Validate and seal

```bash
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --skip-manifest
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --verify-manifest
```

Validation checks segment identity, complete model grids, raw artifact hashes,
hosted request-status consistency, coverage, derived-table provenance, and the
top-level release manifest.

## License

Surrogate is MIT licensed; see [LICENSE](LICENSE).
