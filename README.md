# Surrogate Fidelity: When Can Open LLMs Explain Closed Ones?

Benchmarking scripts and results from the paper. 

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

The pipeline is four stages, each writing TSVs into a checked-in `results/` directory (override with `--results-dir`):

```
run_benchmark         → results/{benchmark}/{pregrouper}/{model}_segment.tsv
                       results/{benchmark}/{pregrouper}/{model}_tokens.tsv
consolidate_results   → results/{benchmark}_{pregrouper}_segments.tsv
                       results/{benchmark}_{pregrouper}_tokens.tsv
compute_logodds       → results/{benchmark}_{pregrouper}_logodds.tsv
f_table               → results/f_table.tsv
```

### Step 1: Generate per-model raw outputs

Use `run_benchmark` to score attention and ablation across supported benchmarks (BoolQ, ANLI R1–R3, WinoGrande, LAMBADA). Models and datasets are downloaded from HuggingFace Hub automatically. The script writes only the raw signals from each forward pass — per-token logprobs (chosen from the benchmark's `report_tokens` set), per-segment attention scores, and per-segment representation metrics. Log-odds aggregations are computed downstream.

```bash
# Sentence-level BoolQ with all Qwen2.5 instruct models (default)
python -m benchmark_scripts.run_benchmark --benchmark boolq

# Word-level ablation (requires --max-forward-passes to subsample)
python -m benchmark_scripts.run_benchmark --benchmark boolq --pregrouper word --max-forward-passes 10000

# Run only ablation phase with base models
python -m benchmark_scripts.run_benchmark --benchmark boolq --model-set Qwen2.5-Base --phases ablation

# LAMBADA uses completion-logprob scoring and always requires --max-forward-passes
python -m benchmark_scripts.run_benchmark --benchmark lambada --max-forward-passes 10000

# Filter to specific models within a set
BENCHMARK_MODELS=qwen2.5-0.5b-instruct python -m benchmark_scripts.run_benchmark --benchmark boolq
```

Available benchmarks: `boolq`, `anli_r1`, `anli_r2`, `anli_r3`, `winogrande`, `lambada`

Available model sets: `Qwen2.5-Instruct`, `Qwen2.5-Base`, `Llama3.1-Instruct`

Per-model TSVs are written to `results/{benchmark}/{pregrouper}/`:

- `{model}_segment.tsv` — one row per (prompt, seg). Columns: `prompt_idx, seg_idx, n_segments, answer, attention_{mean,max,rollout}, w_norm, dn_*, cs_*, wdz_*, zno_*, znp_*, wzo_*, wzp_*` (LAMBADA also gets `orig_completion_logprob`, `ablated_completion_logprob`).
- `{model}_tokens.tsv` — one row per (prompt, seg, kind, label, token). Columns: `prompt_idx, seg_idx, kind, label, token, logprob`. `kind` is `orig` (with `seg_idx` empty) or `ablated`. Not produced for LAMBADA.

#### Batch scripts

```bash
# Full suite for all models across all 6 benchmarks
bash benchmark_scripts/run_all_benchmarks.sh

# Quick smoke test: Qwen2.5-0.5B-Instruct only, 10 samples per benchmark
bash benchmark_scripts/simple_smoke_test.sh
```

### Step 2: Consolidate per-model TSVs across models

Concat all `{model}_segment.tsv` and `{model}_tokens.tsv` files for one benchmark into single tables with a `model` column:

```bash
python -m benchmark_scripts.consolidate_results --benchmark boolq
python -m benchmark_scripts.consolidate_results --benchmark boolq --pregrouper word
```

Outputs:
- `results/{benchmark}_{pregrouper}_segments.tsv`
- `results/{benchmark}_{pregrouper}_tokens.tsv` (omitted if no model produced one)

This is also the integration point for offline closed-model runs: drop `{model}_segment.tsv` / `{model}_tokens.tsv` files into `results/{benchmark}/{pregrouper}/` and they'll be picked up.

### Step 3: Compute log-odds from token logprobs (postprocess)

For each (model, prompt, seg, kind), `compute_logodds` aggregates token logprobs into per-label logprobs (`logsumexp` over the benchmark's `label_tokens[label]` set) and per-pair log-odds (`label_lp[a] - label_lp[b]`):

```bash
python -m benchmark_scripts.compute_logodds --benchmark boolq
```

Output: `results/{benchmark}_{pregrouper}_logodds.tsv` with columns `model, prompt_idx, seg_idx, kind, label_lp_<label>…, logodds_<a>_<b>…`. Binary benchmarks (BoolQ, WinoGrande) emit one `logodds_*_*` column; ternary (ANLI) emits six.

### Step 4: Generate the F-table

Reads `results/{benchmark}_{pregrouper}_segments.tsv` and `..._logodds.tsv` from Steps 2–3 and writes a long-format TSV with one row per `(benchmark, model_s, model_t, metric, statistic)` tuple. Each row carries a bootstrap confidence interval `[f_lo, f_hi]` alongside the point estimate `f_point`.

```bash
# Default: all 6 benchmarks, 1000 bootstrap resamples, 95% CI
python -m benchmark_scripts.f_table

# Custom benchmarks / resample count / CI width
python -m benchmark_scripts.f_table \
    --benchmarks boolq anli_r1 \
    --bootstrap-resamples 2000 \
    --confidence-level 0.9
```

Output is written to `results/f_table.tsv` with columns:

```
benchmark  model_s  model_t  metric  statistic  f_point  f_lo  f_hi
```

Symmetric metrics (correlation of `model_s` vs `model_t` signal): `F_pred`, `F_attr`, `F_attn_rollout`, `F_attn_mean`, `F_attn_max`, `F_mag`, `F_align`. Transfer metrics (`model_s` signal predicts `model_t` ablation, per-prompt then averaged): `F_align_to_attr`, `F_mag_to_attr`, `F_attn_rollout_to_attr`, `F_attn_mean_to_attr`, `F_attn_max_to_attr`. Statistic is `spearman` or `pearson_r2`. Models eligible for each metric are auto-discovered from the TSV columns.

## License

Surrogate is MIT licensed, as found in the LICENSE file.
