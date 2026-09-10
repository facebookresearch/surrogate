# Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?

Benchmarking scripts and audited results from the paper.

> **Correction notice.** The initial public open-model runner segmented the
> system message for ablation but the user message for attention, while its
> hosted-model ablations used user-message segments. It then joined unrelated
> open-model ablation and attention segments by integer index. The paper-era
> BoolQ/ANLI/WinoGrande
> computations used the full dialog in both phases. This version restores that
> behavior, records segment identity explicitly, and validates every model
> against a model-independent segment manifest before producing tables.
> The audit also found RACE prompt-version and metric-labeling errors. The
> release artifacts preserve the historical and corrected analyses separately.

## Installation

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Or using conda
conda create -n surrogate python=3.11
conda activate surrogate
pip install -e .
```

### Development

```bash
pip install -e ".[dev]"
pytest tests/
```

## Running Benchmarks

The pipeline writes auditable raw files and deterministic derived tables:

```
run_benchmark         → results/{benchmark}/{pregrouper}/segments.tsv.gz
                       results/{benchmark}/{pregrouper}/{model}_segment.tsv.gz
                       results/{benchmark}/{pregrouper}/{model}_tokens.tsv.gz
validate_results      → identity, coverage, and provenance checks
consolidate_results   → results/{benchmark}_{pregrouper}_segments.tsv
                       results/{benchmark}_{pregrouper}_tokens.tsv
compute_logodds       → results/{benchmark}_{pregrouper}_logodds.tsv
f_table               → results/f_table.tsv
```

### Step 1: Generate per-model raw outputs

Use `run_benchmark` to score attention and ablation across supported benchmarks
(BoolQ, ANLI R1–R3, WinoGrande, RACE, and LAMBADA). Models and datasets are
downloaded from HuggingFace Hub automatically. For exact reproduction, pass a
frozen source TSV with `--dataset-file`. The script writes raw per-label token
log-probabilities, per-segment attention scores, per-segment representation
metrics, a canonical `segments.tsv.gz` manifest, and per-run provenance.

Both attention and ablation segment every dialog message (system, user, and
any additional messages) in message order. Consequently, `seg_idx` has the
same meaning in every model and scoring phase; the runner validates this
alignment before merging phase outputs.

The two phases intentionally use separate model instances. Attention requests
the eager attention implementation so attention matrices are available;
ablation requests SDPA, matching the numerical backend used by the archived
paper-era ablations. Each run sidecar records this phase-to-backend mapping.

```bash
# Sentence-level BoolQ with all Qwen2.5 instruct models (default)
python -m benchmark_scripts.run_benchmark --benchmark boolq --results-dir results/rerun

# Word-level ablation (requires --max-forward-passes to subsample)
python -m benchmark_scripts.run_benchmark --benchmark boolq --pregrouper word --max-forward-passes 10000 --results-dir results/rerun

# Run only ablation phase with base models into a fresh directory
python -m benchmark_scripts.run_benchmark --benchmark boolq --model-set Qwen2.5-Base --phases ablation --results-dir results/ablation-only

# LAMBADA uses word-level completion scoring and a fixed ablation-coordinate budget
python -m benchmark_scripts.run_benchmark --benchmark lambada --pregrouper word --max-forward-passes 10000

# Filter to specific models within a set
BENCHMARK_MODELS=qwen2.5-0.5b-instruct python -m benchmark_scripts.run_benchmark --benchmark boolq
```

Available benchmarks: `boolq`, `anli_r1`, `anli_r2`, `anli_r3`, `winogrande`, `race`, `lambada`

Available model sets: `Qwen2.5-Instruct`, `Qwen2.5-Base`, `Llama3.1-Instruct`

Per-model TSVs are written to `results/{benchmark}/{pregrouper}/`:

- `segments.tsv.gz` — model-independent segment identities, including global
  and message-local indices, role, and exact segment text.
- `{model}_segment.tsv.gz` — one row per `(prompt, segment)`, including segment
  metadata, attention, and representation metrics. LAMBADA also stores original
  and ablated completion log-probabilities; hosted rows retain explicit
  original/ablation availability reason codes.
- `{model}_tokens.tsv.gz` — one row per
  `(prompt, segment, kind, label, token variant)`. `kind` is `orig` (empty
  `seg_idx`) or `ablated`. This retains all queried ANLI labels, so
  entailment–neutral and entailment–contradiction can be selected after the run.
- `{model}_run.json` — model, dataset, parameter, software, and source hashes.

The runner fails before model loading if any selected model output already
exists. Pass `--overwrite-existing` only to deliberately replace the complete
output set for that model/configuration; this also removes stale token files
from a prior ablation run. It never implicitly combines phases from separate
invocations.

#### Batch scripts

```bash
# Gold open-model stage: five models, two GPUs, and all eight configurations
DATASET_DIR=/path/to/frozen_tsvs bash benchmark_scripts/run_all_benchmarks.sh

# Non-gold exploratory run against current HuggingFace datasets
ALLOW_LIVE_DATASETS=1 bash benchmark_scripts/run_all_benchmarks.sh

# Quick smoke test: Qwen2.5-0.5B-Instruct only, 10 samples per benchmark
bash benchmark_scripts/simple_smoke_test.sh
```

The exact frozen TSV snapshots are distributed separately from this source
repository and are identified by the filenames and SHA-256 values in
`results/README.md`. `download_all.sh` fetches the upstream datasets for
exploratory runs; it does not claim to reproduce those byte-identical audit
snapshots.

### Step 2: Normalize and consolidate per-model TSVs across models

Align each model file to the shared manifest, materialize availability flags,
then concatenate the `{model}_segment.tsv[.gz]` and
`{model}_tokens.tsv[.gz]` files into tables with a `model` column:

```bash
python -m benchmark_scripts.normalize_segment_outputs results/boolq/sentence
python -m benchmark_scripts.consolidate_results --benchmark boolq
python -m benchmark_scripts.normalize_segment_outputs results/boolq/word
python -m benchmark_scripts.consolidate_results --benchmark boolq --pregrouper word
```

Outputs:
- `results/{benchmark}_{pregrouper}_segments.tsv`
- `results/{benchmark}_{pregrouper}_tokens.tsv` (omitted if no model produced one)

This is also the integration point for offline hosted-model runs. Import
portable JSON with `benchmark_scripts.import_hosted_results`, or normalize a
legacy TSV only with an explicit segment-identity attestation. No hosted-service
client is included in this repository. Legacy TSVs remain usable for exploratory
postprocessing, but corrected gold validation requires raw-JSON imports so
per-call request success cannot be conflated with top-k label censoring. A
producer may mark a missing classification response as `content_filter`.
After the configured retry audit, it may preserve a still-missing response as
`transient_exhausted` only when the import is explicitly marked
`complete_with_terminal_failures`. A successful response with a requested label
absent from top-k is represented as `-inf`; a request with no valid response is
represented as unavailable and becomes `NaN` in derived scores. Gold validation
rejects unclassified failures and any mismatch between the declared availability
status and the per-coordinate statuses. The generated coverage table reports
provider-filtered and terminal-failure counts separately from finite label
coverage. Gold imports also require sanitized producer-audit receipts. The
classification receipt covers the exact seven-configuration by seven-model
inventory. The completion receipt covers all seven hosted LAMBADA models, the
entire fixed sampled-manifest population used by four supported endpoints
(4,387 prompts and 10,000 coordinates, not all 5,153 source prompts or 422,134
possible coordinates), and the fixed 10-prompt/823-segment canary used by three
unsupported endpoints.
Each receipt contains only public identifiers, hashes, protocol fields, and
aggregate counts; every imported run is bound to its unique entry and normalized
response projection. Opaque component digests are finalization-time snapshots
of selected producer/audit source files, not execution-time software
attestations. Prompt/dialog digests come from exhaustive post-run comparisons
of the producer and public builders. Provider-specific per-attempt retry
transcripts were not retained.

```bash
python -m benchmark_scripts.import_hosted_results \
    --input /path/to/model.json \
    --manifest results/boolq/sentence/segments.tsv.gz \
    --output-dir results/boolq/sentence \
    --model hosted-model-name \
    --benchmark boolq \
    --pregrouper sentence \
    --identity-attestation "Hosted prompt and ablation coordinates were verified against the canonical public segment manifest." \
    --producer-revision 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef \
    --generated-at 2026-09-08T00:00:00Z \
    --served-model hosted-model-name \
    --request-parameters-json '{"max_tokens":1,"top_logprobs":19,"echo":false,"scoring":"first_generated_token_label_logprobs","initial_max_transient_attempts":5,"final_audit_attempts_per_failed_coordinate_per_round":5,"final_audit_max_rounds":3,"final_audit_rechecks_content_filters":true,"content_filter_policy":"explicit_missing_after_confirmation","terminal_failure_policy":"explicit_nan_after_final_audit"}' \
    --producer-audit-receipt results/hosted_classification_audit_receipt.json
```

Gold validation requires this fixed public attestation, a public model ID in
`served_model` equal to `model`, a lowercase SHA-256 producer revision, and an
RFC3339 generation time. Provider routing names and local paths must not be
placed in release metadata.

For a LAMBADA endpoint that fails the audited completion canary, pass the same
canary JSON as `--input` and `--canary-file`, add
`--availability-status unsupported_after_canary --canary-seed 42`, bind
`--producer-audit-receipt results/hosted_completion_audit_receipt.json`, and record
the exact protocol as
`{"echo":true,"max_tokens":0,"scoring":"teacher_forced_echo_target_logprob_sum","top_logprobs":20,"max_transient_attempts":5}`.
The importer writes a canonical all-missing placeholder; it never promotes the
sparse canary observations to corrected benchmark measurements. Historical
sparse hosted measurements are reproduced only through the pinned archive.

### Step 3: Compute log-odds from token logprobs (postprocess)

For each (model, prompt, seg, kind), `compute_logodds` aggregates token logprobs into per-label logprobs (`logsumexp` over the benchmark's `label_tokens[label]` set) and per-pair log-odds (`label_lp[a] - label_lp[b]`):

```bash
python -m benchmark_scripts.compute_logodds --benchmark boolq
```

Output: `results/{benchmark}_{pregrouper}_logodds.tsv` with columns `model, prompt_idx, seg_idx, kind, label_lp_<label>…, logodds, logodds_<a>_<b>…`. The canonical `logodds` column uses the first two labels in benchmark-configuration order, matching the paper-era scalar scoring pipeline, except that RACE uses the prompt's correct label versus `logsumexp` of the other three. Binary benchmarks (BoolQ, WinoGrande) also emit two directed `logodds_*_*` columns; ternary ANLI emits six for optional post-hoc contrasts.

### Step 4: Generate the F-table

Reads the consolidated files and writes one row per benchmark configuration,
message scope, contrast, model pair, metric, and statistic. Segment-level
intervals use a prompt-cluster bootstrap. The default cohort is the paper's
five open and six hosted models; the duplicate API-served Llama-8B diagnostic
is excluded unless `--all-models` is passed.

```bash
# Default: seven scalar benchmark/granularity configurations (RACE is separate)
python -m benchmark_scripts.f_table

# Sensitivity using the finite-extreme policy described in the paper prose
python -m benchmark_scripts.f_table \
    --api-infinity-policy finite_extreme \
    --output results/f_table_finite_extreme_sensitivity.tsv

# Later notebook-source prompt-equal transfer statistic (not the PDF cells)
python -m benchmark_scripts.f_table \
    --transfer-aggregation prompt_equal_mean_r2 \
    --output results/f_table_prompt_equal_transfer.tsv

# Recompute all/system/user strata from the same raw full-dialog artifact
python -m benchmark_scripts.f_table \
    --scopes all system user \
    --output results/f_table_message_scopes.tsv

# Both ANLI contrasts, without model reruns
python -m benchmark_scripts.f_table \
    --benchmarks anli_r1 anli_r2 anli_r3 \
    --scopes all system user \
    --contrasts entailment_neutral entailment_contradiction \
    --output results/f_table_anli_contrasts.tsv

# Two explicitly unselected manuscript-revision candidates. Both use user
# segment coordinates and entailment-minus-contradiction for ANLI; they differ
# only in hosted top-k infinity handling.
python -m benchmark_scripts.f_table \
    --scopes user \
    --anli-contrast entailment_contradiction \
    --revision-candidate declared_method_pairwise_drop \
    --output results/f_table_revision_candidate_pairwise_drop.tsv
python -m benchmark_scripts.f_table \
    --scopes user \
    --anli-contrast entailment_contradiction \
    --api-infinity-policy finite_extreme \
    --revision-candidate declared_method_finite_extreme \
    --output results/f_table_revision_candidate_finite_extreme.tsv
```

Prediction and ablation fidelity, plus attribution-based transfer metrics, can
be evaluated for either ANLI contrast from the stored data. `F_attn` and
`F_mag` do not depend on the selected label contrast and therefore repeat
across contrast outputs. `F_align` and `F_align_to_attr` are emitted as explicit
unavailable rows for entailment–contradiction because the stored linear-readout
projection was computed for entailment–neutral; relabeling that projection
would be incorrect. Neither revision candidate is designated primary until the
authors choose a missingness policy on methodological rather than outcome
grounds.

Output is written to `results/f_table.tsv` with columns:

```
artifact_role  candidate_id  selection_status  cohort  pair_population  benchmark  pregrouper  scope  requested_scope  resolved_scope  contrast  requested_contrast  resolved_source_contrast  resolved_target_contrast  readout_contrast  api_infinity_policy  aggregation  model_s  model_t  metric  statistic  availability_status  unavailable_reason  n_observations  n_prompts  f_point  f_lo  f_hi
```

The default `drop` policy reproduces the executed paper-table scalar pipeline
by removing non-finite hosted rows pairwise. `finite_extreme` is an explicit sensitivity:
it replaces infinities in each derived model signal (including the
original-minus-ablated attribution signal) just beyond that signal's finite
range.

Symmetric metrics (correlation of `model_s` vs `model_t` signal): `F_pred`,
`F_attr`, `F_attn_rollout`, `F_attn_mean`, `F_attn_max`, `F_mag`, and
`F_align`. The published transfer cells correlate a source signal with
target-model ablation over the pooled segment rows. The separately published
`f_table_prompt_equal_transfer.tsv` instead averages within-prompt Pearson
`r²` values, matching the later notebook source rather than the cached PDF
table values.

RACE's genuine multivariate result is generated separately. Hosted rows use
the paper Figure 13 pair-specific complete-case rule and report coverage. The
expanded multiclass analysis uses a shared orthonormal CLR basis, adds signed
Frobenius correlation, direction cosine, linear CKA, and three geometric
magnitude correlations, and reports both row-pooled and prompt-equal results:

```bash
python -m benchmark_scripts.race_rv --cohort open \
    --output results/race_rv_open.tsv
python -m benchmark_scripts.race_rv --cohort paper \
    --output results/race_rv_paper_complete_case.tsv
python -m benchmark_scripts.race_multiclass --cohort open \
    --output results/race_multiclass_open.tsv
python -m benchmark_scripts.race_multiclass --cohort paper \
    --output results/race_multiclass_paper_global_complete_case.tsv
```

### Step 5: Validate the artifact

```bash
python -m benchmark_scripts.validate_results --results-dir results \
    --cohort paper --require-derived --skip-manifest
```

This checks the exact frozen segment-grid hashes, rejects duplicate or
misaligned identities and incomplete open payloads, verifies the explicit
11-model cohort and the raw and derived provenance seals without prematurely
writing the release manifest. It also reports hosted top-k/completion
missingness in `coverage.tsv`.

Open-model run records distinguish the four execution-control sources recorded
at run completion (`source_sha256`) from the broader generation and analysis
source set hashed when the release is sealed (`release_source_sha256`). The
portable open-execution receipt preserves the pre-seal run-metadata hash, binds
each numerical verdict to the exact segment/token bytes, and records hashes and
pre-queue modification times for all ten public modules imported by the scoring
path. This corroborates the executed snapshot but is not described as a
start-time cryptographic measurement. Complete selected
model/tokenizer hashes are likewise computed at sealing time; the smaller
identity-file set captured by each run must match them exactly. The records
also pin the canonical public repository and immutable upstream revision.
`execution_model_source` preserves the locator actually configured by the run;
`model_source` names the canonical repository whose pinned bytes were sealed.

### Step 6: Reconcile with the archived paper tables

```bash
python -m benchmark_scripts.reconcile_results \
    --archive-dir /path/to/legacy-wide-tsvs \
    --corrected-results-dir results \
    --output-dir results/reconciliation
```

The archive directory must contain the eight historical
`*_consolidated.tsv` files, the eleven raw RACE four-label JSON files, and an
uncommitted `reconciliation_models.json`.
That JSON provides `archive_to_public_model`, `open_models`, and
`hosted_models`; it keeps historical routing aliases outside the public source
tree while fixing the exact 5-open/6-hosted cohort. Their pinned SHA-256 values
identify the raw archived snapshot bytes, not a normalized Drive export (whose
serialization and hash can differ). `--allow-unpinned-archive` exists only for
hermetic miniature tests and must not be used for a gold reconciliation.
Corrected tables are used
only when their `.provenance.json` sidecars verify against the table and all
bound inputs. The release validator independently owns the top-level artifact
manifest, avoiding a provenance hash cycle.

The command writes `reconciliation.tsv`, `historical_claims.tsv`,
`reconciliation_manifest.json`, and `reconciliation_checks.json`. The main
ledger records declared and executed scope,
contrast, aggregation, effective model pairs, and model-version status. It
never computes a delta across different estimand IDs; corrected hosted
endpoints without immutable snapshot IDs are labeled only nominally comparable.
The historical BoolQ-word Qwen attention cells are labeled as sentence-attention
contamination and are never treated as comparable to corrected word-level
attention results.
The historical-claims ledger independently recomputes exact numerical audit
claims outside Tables 1 and 2 from pinned source bytes, including the raw
four-label RACE responses needed for Figure 13.
The corrected table's `aggregation` column is mandatory and must be uniform
within each summarized cell: default `row_pooled` rows match the PDF method,
while `prompt_equal_mean_r2` rows remain a distinct sensitivity estimand. The
command exits nonzero whenever any required reconciliation check fails.

### Step 7: Seal and re-verify the complete release

```bash
python -m benchmark_scripts.validate_results --results-dir results \
    --cohort paper --require-derived --require-reconciliation
python -m benchmark_scripts.validate_results --results-dir results \
    --cohort paper --require-derived --require-reconciliation --verify-manifest
```

The first command writes the top-level manifest only after the reconciliation
bundle passes; the second independently verifies that immutable file inventory.

## License

Surrogate is MIT licensed, as found in the LICENSE file.
