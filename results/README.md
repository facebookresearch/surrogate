# `results/` — benchmark outputs

This directory holds the data feeding the F-table in the paper. Two flavors of file:

- **Per-model TSVs** (gzipped, in `{benchmark}/{pregrouper}/`): the raw outputs of one model on one (benchmark, pregrouper) configuration.
- **`f_table.tsv`** (uncompressed): the headline cross-model F-table, one row per `(benchmark, model_s, model_t, metric, statistic)` tuple with bootstrap CIs.

Consolidated and intermediate TSVs (`{benchmark}_{pregrouper}_segments.tsv`, `_tokens.tsv`, `_logodds.tsv`) are gitignored — regeneratable from the per-model files via the postprocess scripts (see [reproducing](#reproducing-the-pipeline)).

## Layout

```
results/
├── README.md                                    (this file)
├── f_table.tsv                                  headline F-table (small)
└── {benchmark}/{pregrouper}/                    per-model TSVs
    ├── {model}_segment.tsv.gz                   per (prompt, seg)
    └── {model}_tokens.tsv.gz                    per (prompt, seg, kind, label, token)
```

`benchmark` ∈ `{boolq, anli_r1, anli_r2, anli_r3, winogrande, lambada}`.
`pregrouper` ∈ `{sentence, word}` (which segmentation level for ablation).

## Per-model `_segment.tsv.gz`

One row per (prompt, segment). Columns vary by model class:

### Open models (full schema)

22 columns: `prompt_idx, answer, seg_idx, n_segments` plus

- **Attention** (3): `attention_mean`, `attention_max`, `attention_rollout`
- **w-norm** (1): `w_norm`
- **Δz norms** (2): `delta_norm_prenorm`, `delta_norm_postnorm`
- **cos(z_orig, z_pert)** (2): `cossim_prenorm`, `cossim_postnorm`
- **w · Δz** (2): `w_dot_delta_z_prenorm`, `w_dot_delta_z_postnorm`
- **Per-vector norms** (4): `z_orig_norm_{pre,post}norm`, `z_pert_norm_{pre,post}norm`
- **Per-vector w-projections** (4): `w_dot_z_orig_{pre,post}norm`, `w_dot_z_pert_{pre,post}norm`

LAMBADA segment TSVs add two more columns: `orig_completion_logprob`, `ablated_completion_logprob` (used in place of label-token logprobs since the target is per-prompt).

### Closed models (MetaGen-API only)

4 columns: `prompt_idx, answer, seg_idx, n_segments`. Closed models don't expose hidden states, so attention and representation columns are absent by construction.

## Per-model `_tokens.tsv.gz`

One row per (prompt, segment, kind, label, token):

| Column | Notes |
|---|---|
| `prompt_idx` | Dataset row index |
| `seg_idx` | Segment index for ablation; empty for `kind=orig` rows |
| `kind` | `orig` (unablated prompt) or `ablated` |
| `label` | Benchmark label name (e.g., `true`, `false`, `entailment`) |
| `token` | Per-label report-token alias (e.g., `true`, `True`, `sp_True`, `us_True`) |
| `logprob` | log P(token \| context) at the next-token position |

For closed (MG) models, only tokens that landed in the model's top-K are recorded — partial coverage is the expected behavior; downstream `compute_logodds` aggregates via `logsumexp` so missing variants contribute zero mass.

LAMBADA does not produce a tokens TSV (the per-prompt target is scored end-to-end via teacher-forced echo, not per-label-token).

## `f_table.tsv`

Long-format. Columns:

```
benchmark  model_s  model_t  metric  statistic  f_point  f_lo  f_hi
```

- `metric` ∈ `{F_pred, F_attr, F_attn_rollout, F_attn_mean, F_attn_max, F_mag, F_align}` (symmetric correlations) ∪ `{F_align_to_attr, F_mag_to_attr, F_attn_rollout_to_attr, F_attn_mean_to_attr, F_attn_max_to_attr}` (transfer)
- `statistic` ∈ `{spearman, pearson_r2}`
- `f_point` is the point estimate; `f_lo`/`f_hi` are 2.5%/97.5% bootstrap percentiles (1000 resamples by default).

Symmetric metrics emit one row per unordered model pair `(a, b)` with `a < b`. Transfer metrics emit ordered pairs `(a, b)` with `a ≠ b` (source signal `a` predicts target ablation `b`).

## Models

The repo includes 12 model entries:

| Name in TSV | Provenance | Rep cols? |
|---|---|---|
| `qwen2.5-{0.5,3,7,14}b-instruct` | local HuggingFace runner (GPU) | ✓ |
| `llama-3.1-8b-instruct` | local HuggingFace runner (GPU) | ✓ |
| `llama3.1-8b-instruct` | MetaGen API | ✗ |
| `llama3.1-70b-instruct` | MetaGen API | ✗ |
| `llama3.3-70b-instruct` | MetaGen API | ✗ |
| `llama4-maverick-17b-128e-instruct` | MetaGen API | ✗ |
| `gpt-4o` | MetaGen API | ✗ |
| `gpt-4-1` | MetaGen API | ✗ |
| `gemini-2-5-flash-lite-vertex` | MetaGen API | ✗ |

The two `llama-3.1-8b` entries (with hyphen between `3.1` and `8b` for the local HF run, no hyphen for the MG run) are intentional: they're the same Llama-3.1-8B-Instruct weights served two different ways. The diff between their per-prompt `F_pred` is a same-model-different-pipeline diagnostic — measures top-K-truncation effects in the closed-API path against the full-vocab open-source path.

## Coverage at submission

|  | open (5 models) | closed (7 MG models) |
|---|---|---|
| `boolq` sentence | ✓ all | ✓ all |
| `boolq` word | ✓ all | ✓ all except `gpt-4-1` |
| `anli_r1`/`r2`/`r3` sentence | ✓ all | ✓ all |
| `winogrande` sentence | ✓ all | ✓ all |
| `lambada` word | ✓ all | ✓ 5/7 (missing `gpt-4o`, `gpt-4-1`) |

The two missing MG cells (`gpt-4o lambada.word`, `gpt-4-1 boolq.word + lambada.word`) were too slow to finish under the deadline — they may be folded in via a Figshare update post-submission.

## Reproducing the pipeline

From per-model TSVs (committed) → consolidated → log-odds → F-table:

```bash
# Stage 1 — concat per-model TSVs into per-benchmark consolidated TSVs
python -m benchmark_scripts.consolidate_results --benchmark boolq                 # → results/boolq_sentence_segments.tsv, _tokens.tsv
python -m benchmark_scripts.consolidate_results --benchmark boolq --pregrouper word
python -m benchmark_scripts.consolidate_results --benchmark anli_r1
# ... and so on for anli_r2, anli_r3, winogrande, lambada

# Stage 2 — derive log-odds from per-token logprobs (skip lambada — completion-logprob mode)
python -m benchmark_scripts.compute_logodds --benchmark boolq
# ... etc

# Stage 3 — F-table with bootstrap CIs (~20 min single-CPU at 1000 resamples)
python -m benchmark_scripts.f_table
```

The whole regen is ~25 min on a laptop with the per-model TSVs already on disk.

## Conventions

- All TSVs are tab-separated, UTF-8, with a header row. Gzipped files are auto-detected by pandas via the extension (`pd.read_csv(path, sep='\t')` works on `.tsv` and `.tsv.gz` alike).
- `prompt_idx` is the row index into the original dataset (HuggingFace dataset row order, not the manifold-cached TSV row order).
- All log-probs are natural logarithms.
- Bootstrap seed is fixed (`--seed 42`); CIs are reproducible byte-for-byte from the same per-model TSVs.
