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

## Reproduce the paper figures

[`notebooks/paper_figures.ipynb`](notebooks/paper_figures.ipynb) regenerates
the revised paper's public-data tables and figures from the committed result
artifacts. It uses the canonical user-segment, pairwise-complete analysis,
including entailment-minus-contradiction for ANLI and multivariate RV for
RACE. The per-layer figure reports unnormalized `F_pred`, signed `F_attr`,
and the matched grouped readout-compatible control. Notebook-only appendix
outputs include the prediction-attribution gap and a tuned-lens sensitivity
panel reading
`layer_controls/tuned_lens_boolq_fidelity.tsv`; it is exploratory because the
lenses use one seed, sparse depth grids, and model-specific training budgets.

```bash
pip install -e ".[paper]"
jupyter lab notebooks/paper_figures.ipynb
```

Outputs are written under `paper_outputs/` by default. Set
`SURROGATE_PAPER_OUTPUT_DIR` to write into a paper source tree. Analyses that
require unreleased hidden states, such as CKA, are intentionally excluded.
If the notebook kernel was started outside the checkout and the package is not
installed in editable mode, set `SURROGATE_REPO_ROOT=/path/to/surrogate`.

## Method

The release uses these primary choices:

- User-message segment coordinates, with the full dialog retained as context.
- Entailment-minus-contradiction log odds for ANLI.
- No duplicated BOS or tokenizer-added special tokens: chat templates produce
  the control-token prefix, and current runs subsequently tokenize rendered
  prompts with `add_special_tokens=False`.
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
run_layerwise       -> results/{benchmark}/{pregrouper}/{model}_layers.tsv.gz
layerwise_fidelity  -> results/layerwise_fidelity.tsv
build_layer_control_spec -> layer_controls/layer_control_spec.json
run_layer_controls  -> external per-model projection tensors
analyze_layer_controls -> layer_controls/layer_control_fidelity.tsv
plot_layer_controls -> paper_outputs/figures/fig5_per_layer_fidelity.pdf
```

### 1. Generate open-model outputs

`run_benchmark` supports BoolQ, ANLI R1-R3, WinoGrande, RACE, and LAMBADA.
For the audited coordinate and dataset snapshot, provide the frozen source TSV
named in [`results/README.md`](results/README.md). Exact model-output bytes can
also depend on the pinned model, numerical libraries, hardware, and backend;
live Hugging Face datasets are intended only for exploration.

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

Rendered chat prompts are complete model inputs. The runner therefore disables
tokenizer-level special-token insertion after applying the chat template. This
prevents Llama tokenizers from prepending a second BOS token.

The batch script runs the five open models across all eight ordinary
configurations and the complete BoolQ/ANLI layer matrix on two GPUs:

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

# Compact per-layer scores for one model/configuration
python -m benchmark_scripts.run_layerwise \
    --benchmark boolq \
    --model-set Llama3.1-Instruct \
    --models llama-3.1-8b-instruct \
    --dataset-file /path/to/google_boolq_validation.tsv

# Matched-relative-depth F_pred and F_attr curves
python -m benchmark_scripts.layerwise_fidelity

# Unsmoothed nearest-native-layer sensitivity (no model rerun)
python -m benchmark_scripts.layerwise_fidelity \
    --depth-alignment nearest_native \
    --output results/layerwise_fidelity_nearest_sensitivity.tsv
```

The BoolQ layer-control analysis uses 256 deterministic directions shared by
all five open-model tokenizers, plus independently seeded isotropic directions.
The grouped 9-vs-8 pseudo-label controls match the cardinality and tokenizer
acceptance profile of the BoolQ label groups. Selection depends only on the
tokenizers—not the corpus, activations, or model scores. Cross-model shuffled
assignments break direction identity while preserving each model's marginal
control distribution. A separately derived observation-pair permutation null
breaks `(prompt_idx, seg_idx)` correspondence while preserving each model's
attribution distribution.

The raw projection tensors total roughly 15 GB and are intentionally kept
outside Git. Generate them into a separate directory, then commit only the
compact summary, compressed pair-level draw table, specification, and
provenance:

```bash
python -m benchmark_scripts.build_layer_control_spec \
    --model-root /path/to/models \
    --output layer_controls/layer_control_spec.json

python -m benchmark_scripts.run_layer_controls \
    --control-spec layer_controls/layer_control_spec.json \
    --source-results-dir results \
    --output-dir /path/to/layer-controls \
    --model-set Qwen2.5-Instruct \
    --dataset-file /path/to/google_boolq_validation.tsv

python -m benchmark_scripts.run_layer_controls \
    --control-spec layer_controls/layer_control_spec.json \
    --source-results-dir results \
    --output-dir /path/to/layer-controls \
    --model-set Llama3.1-Instruct \
    --dataset-file /path/to/google_boolq_validation.tsv

python -m benchmark_scripts.analyze_layer_controls \
    --source-results-dir results \
    --control-results-dir /path/to/layer-controls \
    --control-spec layer_controls/layer_control_spec.json \
    --output layer_controls/layer_control_fidelity.tsv \
    --draw-output layer_controls/layer_control_fidelity_draws.tsv.gz

pip install -e ".[plots]"
python -m benchmark_scripts.plot_layer_controls \
    layer_controls/layer_control_fidelity.tsv \
    paper_outputs/figures/fig5_per_layer_fidelity.pdf
```

The headline plot reports per-layer grouped-logsumexp `F_pred`, signed
`F_attr`, and the grouped 9-vs-8 readout-compatible control. Target ribbons are
95% prompt-cluster bootstrap intervals; the control ribbon is the empirical
2.5--97.5% range across directions, not a confidence interval. Pairwise r²
values are averaged over the ten open-model pairs and must not be interpreted
additively. Relative depth excludes the embedding slot by default;
`analyze_layer_controls --include-embedding` provides an explicit all-slot
sensitivity.

`F_pred` is prompt-level and therefore uses the full dialog. Other metrics use
the requested segment coordinates. For ANLI E-C, `F_align` and
`F_align_to_attr` use the requested sum-unembedding direction from the final
decoder block in the sidecar-bound layer artifacts. Contrasts matching the
ordinary run readout continue to use the ordinary segment tables. The F-table
validator requires every raw final-layer contrast and Pearson endpoint to match
the ordinary BF16-token reconstruction within `1e-4`. Spearman endpoints use a
`1e-3` tolerance because rank correlation is discontinuous when independent
BF16 batches swap nearly tied values.

Additional scopes and ANLI contrasts remain reproducible from the raw files:

```bash
python -m benchmark_scripts.f_table --scopes all system user
python -m benchmark_scripts.f_table \
    --benchmarks anli_r1 anli_r2 anli_r3 \
    --scopes all system user \
    --contrasts entailment_neutral entailment_contradiction \
    --output results/anli_sensitivity.tsv
```

Layer artifacts likewise retain every configured label and every system/user
segment. The canonical layerwise table uses user coordinates and ANLI
entailment-minus-contradiction, while entailment-minus-neutral and the `all`,
`system`, and `user` scopes can be selected without rerunning a model. Layer
files contain label-level scalars only—never hidden-state vectors. Slot zero is
the embedding output; later slots are decoder-block outputs with the model's
final normalization applied. Relative-depth comparisons exclude the embedding
slot and linearly interpolate decoder blocks. An unsmoothed nearest-native
sensitivity is available through `--depth-alignment nearest_native`. The stored sum-unembedding
projection is a diagnostic approximation to the grouped-logsumexp attribution
contrast; `F_attr` itself uses the exact grouped label scores. Canonical layer
runs use BF16, SDPA, automatic device placement, and batch size 32. Use
`run_all_benchmarks.sh` to generate the required five-model,
four-configuration matrix.

RACE is evaluated as a multivariate four-label signal using centered RV. The
`all_pairs` representation is the six pairwise A-D margin system;
`anchor_a` is included as a three-dimensional sensitivity. Hosted comparisons
use model-pair-specific finite complete cases and report their coverage.

### 5. Run the Muse Glimmer robustness extension

[Muse Glimmer](robustness/muse_glimmer/README.md) is kept separate from the
paper's fixed model cohort. The public runner pins and content-verifies the
exact Hugging Face revision, uses the model's direct-response routing prefix,
and requires an exact match to the committed BoolQ sentence coordinates.

```bash
pip install -e ".[glimmer]"

CUDA_VISIBLE_DEVICES=0 python -m benchmark_scripts.run_glimmer_boolq \
    --dataset-file /path/to/google_boolq_validation.tsv \
    --results-dir /path/to/glimmer-attention-run \
    --phases attention

CUDA_VISIBLE_DEVICES=1 python -m benchmark_scripts.run_glimmer_boolq \
    --dataset-file /path/to/google_boolq_validation.tsv \
    --results-dir /path/to/glimmer-ablation-run \
    --phases ablation --batch-size 2

python -m benchmark_scripts.merge_phase_results \
    --attention-results-dir /path/to/glimmer-attention-run \
    --ablation-results-dir /path/to/glimmer-ablation-run \
    --output-results-dir robustness/muse_glimmer \
    --benchmark boolq --pregrouper sentence \
    --model muse-glimmer-30b

python -m benchmark_scripts.glimmer_robustness \
    --extension-results-dir robustness/muse_glimmer
```

The committed artifact runs the two scoring phases concurrently on separate
GPUs and preserves both execution receipts in the merged receipt. A direct
single-process `--phases attention,ablation` run computes the same raw scores,
but does not reproduce that split-run receipt; the public robustness analyzer
therefore expects the merge workflow shown above. The derived extension
reports only `F_pred` and signed `F_attr`: Glimmer's nonlinear output softcap
makes the stored fixed-unembedding projection a useful representation
diagnostic, but not an exact decomposition of output log odds.

### 6. Validate and seal

```bash
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --skip-manifest
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived
python -m benchmark_scripts.validate_results \
    --results-dir results --cohort paper --require-derived --verify-manifest
python -m benchmark_scripts.glimmer_robustness --verify-existing
```

Validation checks segment identity, complete model grids, raw artifact hashes,
hosted request-status consistency, coverage, derived-table provenance, and the
top-level release manifest. The final command independently verifies the
optional Glimmer raw and derived artifacts under `robustness/muse_glimmer`.

## License

Surrogate is MIT licensed; see [LICENSE](LICENSE).
