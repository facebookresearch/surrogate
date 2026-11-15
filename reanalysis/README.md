# Rebuttal reanalyses

These are reanalyses of the published result artifact, run for the rebuttal.
They make no model or API calls and write nothing under `results/`. Results
and notes on the paper's claims are in [`summary.md`](summary.md).

```bash
OMP_NUM_THREADS=1 python -m benchmark_scripts.reanalysis_rebuttal \
    --results-dir results --output-dir reanalysis
```

The run takes about five minutes on one CPU. It needs the consolidated
`results/boolq_sentence_{segments,logodds}.tsv` (README step 3). Signals come
from the `f_table` extractors, so every quantity means exactly what it means in
`f_table.tsv`. Before writing anything, each analysis checks that its baseline
rows reproduce the published Pearson r² for every pair, and it stops if any
differs by more than 1e-9.

## Files

Each analysis `N_name` writes:

- `N_name.csv`: one row per model pair and variant, with `n_prompts`,
  `n_observations` and `value`.
- `N_name_summary.csv`: the median, minimum and maximum over the pair sets
  Open→Open, Open→Closed and All.
- `*.provenance.json`: binds each CSV to its inputs, source and parameters.

The analyses also write these supporting files:

- `1_label_conditioning_model_accuracy.csv`: per-model accuracy.
- `2_length_control_fits.csv`: per-model length regressions, over all segments
  and within passages.

## Data and definitions

The main setting is BoolQ at sentence level. Analysis 6 re-aggregates the
published Table 2 inputs: `f_table.tsv`, `anli_rv.tsv` and `race_rv.tsv`.

### Fields

| Quantity | Source |
|---|---|
| ℓ | canonical `logodds` = logp(True) − logp(False) |
| Δℓ | ℓ(x) − ℓ(x without segment i) |
| Gold label | `answer` |
| ‖Δz‖ | `delta_norm_postnorm` |
| cos(Δz, v) | F-table alignment |
| Attention | `attention_{mean,max,rollout}` |

### Segment parts

- **System:** system-message segments.
- **Question:** the user segment starting with `Question:` plus any later user
  segments. That is one segment per prompt, except one prompt where it splits
  into two.
- **Passage:** every other user segment.

### Conventions

- **F:** Pearson r² over the coordinates both models observe with finite
  values, pooled over prompts. This is the published pairwise-complete policy.
- **Pairs:**
  - `F_pred` and `F_attr`: 55 unordered pairs.
  - Representation metrics: the 10 open pairs.
  - Cross-level metrics: 50 ordered pairs, from each open source to every
    other model. "All" therefore means Open→All for these.
- **Correctness:** sign(ℓ) equals gold. ℓ = ±∞ counts by its sign, but those
  prompts drop out of the correlations as they do in the baseline. NaN never
  counts as correct.
- **Length:** whitespace word count of `segment_text`, minimum 1.
  - Each open model's signal is residualized on log length by OLS, fitted over
    that model's segments.
  - Passage-only variants fit within passages.
  - The attribution target is never residualized.
- **Within-prompt:**
  - (a) Subtract per-prompt means on the pair's shared coordinates, then pool.
  - (b) Average the per-prompt Spearman ρ over prompts with at least 3 shared
    segments and non-constant signals. This is identical to the F-table's
    prompt-equal sensitivity, as the tests check.
- **Decisive segments:**
  - Each model's top segment is ranked by |Δℓ|, or by attention for the source
    in attention rows. Ranking uses the pair's shared finite segments, for
    prompts with at least 2 of them.
  - Ties are broken at random with a fixed seed.
  - Top-3 overlap is |A ∩ B| / 3, over prompts with more than 3 segments.
  - Chance is the mean of 1/N for top-1 and 3/N for top-3.
  - Q-top is the share of prompts whose target top segment is the question,
    i.e. the agreement of always picking the question.
  - Confidence quintiles are the per-pair quintiles of the mean |ℓ| of the two
    models.
- **Families (analysis 6):**
  - Qwen: the four open Qwen models.
  - Llama: Llama-3.1-8B (open) plus Llama-3.1-70B, Llama-3.3-70B and Llama-4
    Maverick (hosted).
  - GPT: GPT-4o and GPT-4.1.
  - Gemini: Gemini 2.5 Flash Lite.

The variants beyond the requested ones (passage-only rows in analyses 3 and 7,
and within-passage length control in analysis 5) are diagnostics. They were
added because segment role (system, passage, question) turned out to drive
several of the pooled results.
