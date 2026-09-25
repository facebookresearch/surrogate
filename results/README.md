# Audited result artifact

Raw per-model files are the canonical inputs to the checked-in result tables.
Every derived table has a provenance sidecar binding it to those inputs, the
analysis source, parameters, and numerical-library versions.

## Frozen configurations

| Benchmark / granularity | Prompts represented | Segment rows | Frozen TSV SHA-256 |
|---|---:|---:|---|
| BoolQ / sentence | 3,270 | 27,516 | `80040aa10f18e5b01082386dae3bdde48931a0311e807f6cb10f7173995f346a` |
| ANLI R1 / sentence | 1,000 | 8,181 | `b87837e172a9a98c70217677fb4897fd6dd3a32497dface9ad3621ea2c85839d` |
| ANLI R2 / sentence | 1,000 | 8,163 | `10ebd3593714d7fe106a5e2ed7f22ea47a34100aad713d8fde4d190d896479ef` |
| ANLI R3 / sentence | 1,200 | 10,028 | `4b08122b488ad0ed1e3d18ce50ef67ba0d686aaa29a140c621151b8326a36e8f` |
| WinoGrande / sentence | 1,267 | 9,135 | `cd158f8e0699aecbc0c2090d89004514ec515ec7f2a2238dd93d4b4457505ca4` |
| RACE / sentence | 4,934 | 145,544 | `ce70066c14b0f4da95bd5d577a8f0d0208228c04fcff1b11d2e362371a9c6289` |
| BoolQ / word sample | 3,028 | 10,000 | same BoolQ snapshot |
| LAMBADA / word sample | 4,387 | 10,000 | `7bb96e6a12ea76dac59896d2acd9d29599e0b488afc9caedf7140800d8e665b1` |

Sentence configurations cover their complete datasets. The two word
configurations are deterministic seed-42 samples of 10,000
`(prompt_idx, seg_idx)` coordinates. RACE uses one punctuation-normalized
prompt template for every model and phase.

## Layout

```text
results/
├── README.md
├── artifact_manifest.json
├── hosted_classification_audit_receipt.json
├── hosted_completion_audit_receipt.json
├── coverage.tsv
├── f_table.tsv
├── f_table.tsv.provenance.json
├── f_table_finite_extreme_sensitivity.tsv
├── f_table_finite_extreme_sensitivity.tsv.provenance.json
├── race_rv.tsv
├── race_rv.tsv.provenance.json
├── anli_rv.tsv
├── anli_rv.tsv.provenance.json
├── multiclass_floor_sensitivity.tsv
├── multiclass_floor_sensitivity.tsv.provenance.json
├── layerwise_fidelity.tsv
├── layerwise_fidelity.tsv.provenance.json
└── {benchmark}/{pregrouper}/
    ├── segments.tsv.gz
    ├── {model}_segment.tsv.gz
    ├── {model}_tokens.tsv.gz
    ├── {model}_run.json
    ├── {open_model}_layers.tsv.gz
    ├── {open_model}_layers_run.json
    └── {model}_canary.json        # selected hosted LAMBADA endpoints
```

Consolidated `*_segments.tsv`, `*_tokens.tsv`, and `*_logodds.tsv` files are
deterministic scratch products and are not committed.

## Coordinate and value semantics

`segments.tsv.gz` defines the shared coordinate system. Each row contains
`prompt_idx`, `seg_idx`, `message_idx`, `message_role`, `message_seg_idx`,
exact `segment_text`, and `n_segments`. Per-model segment files reproduce
those identities exactly.

Open-model segment tables contain attention and representation signals.
Classification token tables contain one row per
`(prompt, segment, kind, label, token variant)`, where `kind` is `orig` or
`ablated`. LAMBADA stores original and ablated completion scores in its segment
tables instead.

Hosted files distinguish:

- successful finite label scores;
- successful responses where a requested label was absent from top-k (`-inf`);
- provider-filtered or retry-exhausted requests (`NaN` after derivation).

Request-status fields in the segment tables preserve these distinctions.
`coverage.tsv` reports original-prompt, ablated-segment, paired-attribution,
label-cell, provider-filter, and terminal-failure coverage by model and
configuration.

## Models

The headline cohort contains five open and six hosted models:

- Open: Qwen-2.5 0.5B, 3B, 7B, and 14B Instruct; Llama-3.1-8B Instruct.
- Hosted: Llama-3.1-70B, Llama-3.3-70B, Llama-4 Maverick, GPT-4o, GPT-4.1,
  and Gemini 2.5 Flash Lite.

An API-served Llama-3.1-8B result is retained as a serving-path diagnostic but
excluded from the headline cohort.

Open run records pin the public model repository, immutable revision, selected
weight and tokenizer hashes, dataset snapshot, segment manifest, phase
backends, and output hashes. Hosted receipts contain only portable public
identifiers, hashes, protocol fields, and aggregate status counts; no service
implementation or routing identifiers are included.

Open-model run records distinguish the literal tokenizer argument from the
verified effective tokenization. Qwen's rendered token IDs are verified
identical with `add_special_tokens` enabled or disabled, and the pinned Qwen
tokenizer inserts zero BOS tokens. Llama's template contains one BOS, so Llama
artifacts require `add_special_tokens=false` after chat rendering to avoid
adding a second one. Sealing checks these properties against each pinned
tokenizer.

## Canonical analyses

`f_table.tsv` evaluates every system- and user-message coordinate in the
complete dialog. Scalar ANLI prediction and attribution diagnostics use
entailment-minus-contradiction. Non-finite values are removed separately for
each model pair, so every row reports both observed and expected observation
and prompt counts plus coverage fractions. The estimand is therefore
association conditional on both signals being observable; hosted top-k
censoring is not assumed to be random.

Point estimates include signed Spearman and Pearson correlations and Pearson
`r²`. Segment-level intervals use a prompt-cluster bootstrap. Row pooling
weights prompts in proportion to their number of jointly observed segments.

ANLI E-C `F_align` and `F_align_to_attr` use the requested sum-unembedding
direction from the final decoder block in the cryptographically bound layer
artifacts. This is not a relabeling of the ordinary E-N projection. Contrasts
whose ordinary readout already matches retain the ordinary segment-table
implementation. Final-layer `F_pred` and `F_attr` are diagnostically compared
with the ordinary BF16-token reconstruction during F-table generation. Release
validation requires exact coverage, constrains each raw label contrast within
`1e-4`, constrains Pearson endpoints within `1e-4`, and constrains Spearman
endpoints within `1e-3` to allow rank swaps among BF16-near-ties. Canonical
layer runs record BF16, SDPA, automatic device placement, and batch size 32.

`f_table_finite_extreme_sensitivity.tsv` uses the same cohort, coordinates,
contrasts, and aggregation. It differs only by replacing hosted infinities just
outside each signal's finite range when such a range exists. Missing requests
remain missing. This is a sensitivity analysis, not the primary estimate.

The paper-facing cross-benchmark table omits LAMBADA because GPT-4o, GPT-4.1,
and Gemini do not reliably expose the teacher-forced target-token log
probabilities required by this task. The partial public measurements remain in
`f_table.tsv` with explicit availability and coverage metadata.

`race_rv.tsv` treats RACE as a multivariate four-label problem. Prediction and
attribution use centered RV over the six pairwise A--D margins. The scalar
attention, perturbation-magnitude, and answer-conditioned alignment signals use
centered RV on the canonical correct-vs-rest contrast; in one dimension this is
exactly Pearson r-squared. Mechanistic-to-attribution rows likewise compare the
scalar mechanistic signal with the target model's correct-vs-rest ablation
(absolute-valued for the unsigned magnitude metric). Hosted comparisons use
model-pair-specific finite complete cases with explicit observation and prompt
coverage.

`anli_rv.tsv` is the canonical ANLI black-box analysis. It uses centered RV
over the three pairwise entailment--neutral--contradiction margins, with the
same full-dialog coordinates and pair-specific complete-case policy.

`multiclass_floor_sensitivity.tsv` repeats the ANLI and RACE multivariate
analyses while moving a finite floor below each model and benchmark's lowest
observed label log-probability. It is an appendix sensitivity analysis; the
complete-case RV tables remain canonical.

`layerwise_fidelity.tsv` reports open-model `F_pred` and signed `F_attr` at
matched relative decoder depth for BoolQ and ANLI R1-R3. Its compact raw layer
artifacts retain scores for every label and all full-dialog segment coordinates,
so ANLI entailment-minus-contradiction and entailment-minus-neutral, along with
all/system/user scopes, remain available post hoc. No hidden-state vectors are
stored. The included unembedding projections use a uniform sum over accepted
single-token label aliases and are identified as a diagnostic approximation to
the grouped-logsumexp attribution direction. The canonical table uses linear
interpolation across native decoder blocks; `--depth-alignment nearest_native`
provides an unsmoothed post-hoc sensitivity without another model run.
The canonical table uses all full-dialog coordinates.

## Reproduction

From the repository root, consolidate each raw configuration and derive label
scores. For example:

```bash
python -m benchmark_scripts.normalize_segment_outputs results/boolq/sentence
python -m benchmark_scripts.consolidate_results --benchmark boolq
python -m benchmark_scripts.compute_logodds --benchmark boolq
```

Repeat consolidation for all configurations, then generate and validate the
release tables:

```bash
python -m benchmark_scripts.f_table
python -m benchmark_scripts.f_table \
    --api-infinity-policy finite_extreme \
    --output results/f_table_finite_extreme_sensitivity.tsv
python -m benchmark_scripts.race_rv
python -m benchmark_scripts.anli_rv
python -m benchmark_scripts.layerwise_fidelity

python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --skip-manifest
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --verify-manifest
```

The final command verifies every manifest entry without rewriting the release.
