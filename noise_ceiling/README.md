# BoolQ/sentence run-to-run noise ceiling

Every paper-cohort model (five open, six hosted) was re-run five times on the
full BoolQ/sentence configuration (3,270 prompts, 27,516 segment coordinates)
with the published protocol, and every F-table signal was correlated with
itself across runs.

## Protocol

- **Open models:** the pinned, hash-verified weights of the published runs;
  BF16, batch size 32, eager attention for the attention phase and SDPA for the
  ablation phase; `benchmark_scripts.run_benchmark` unchanged.
- **Hosted models:** request dialogs exported by
  `benchmark_scripts.export_hosted_requests`, whose prompt and ablated-dialog
  identity digests equal those of the gold hosted-classification receipt. The
  request protocol matches the receipt's `request_parameters` (one generated
  token, top-19 log-probabilities, five transient retries, up to three
  final-audit rounds that also re-check content filters). Outputs were imported
  with `benchmark_scripts.import_hosted_results`.
- Each repetition was normalized, consolidated, and converted to log-odds
  exactly as in the main pipeline, then analyzed with
  `benchmark_scripts.noise_ceiling` (README step 6).

## Files

- `noise_ceiling.tsv`: same-model autocorrelations. Columns follow
  `results/f_table.tsv`, with `model`, `run_s`, `run_t`, and `comparison`
  (`rep_vs_rep` or `rep_vs_published`) identifying the run pair. Symmetric
  signals report unordered pairs; `*_to_attr` rows pair the mechanistic signal of
  `run_s` with the attribution of `run_t` for every ordered pair.
- `noise_ceiling_spread.tsv`: every canonical BoolQ/sentence F-table point
  estimate recomputed on each repetition, beside the published value, with the
  repetition mean, sample SD, range, and minimum joint observation count.
- `*.provenance.json`: bind both tables to the published artifact, the SHA-256
  digest of every repetition raw file, the generating source, and all analysis
  parameters. The repetition raw files (about 175 MB) are archived outside Git
  under the identifier recorded as `rep_archive_id`.

## Summary

Mean over the ten repetition pairs (Spearman; Pearson in parentheses):

| Model | `F_pred` | `F_attr` |
|---|---:|---:|
| Open models (all five) | 1.0000 | 1.0000 |
| Llama-3.1-70B | 0.9999 | 0.9917 (0.9988) |
| Llama-3.3-70B | 0.9998 | 0.9930 (0.9995) |
| Llama-4 Maverick | 0.9993 | 0.9834 (0.9960) |
| GPT-4o | 0.9896 | 0.8695 (0.9527) |
| GPT-4.1 | 0.9638 | 0.8575 (0.9580) |
| Gemini 2.5 Flash Lite | 1.0000 | 1.0000 |

Open-model repetitions reproduce the published artifact bit for bit for every
signal (attention rollout differs only by floating-point noise below `1e-19`),
so their noise ceiling is exactly one. Attention, perturbation magnitude, and
alignment are therefore also perfectly repeatable.

Hosted noise concentrates in segment-level attribution for the GPT models. A
cross-model `F_attr` involving GPT-4o or GPT-4.1 is bounded by these
autocorrelations rather than by one.

The cross-model estimates themselves are stable: across repetitions, the
sample SD of every canonical BoolQ/sentence `F_pred` and `F_attr` point
estimate is at most 0.007 (`noise_ceiling_spread.tsv`).

## Caveats

- **Gemini:** all five repetitions returned byte-identical outputs, and a probe
  with a different request (`top_logprobs=20`) reproduced the same scores, so
  the endpoint was deterministic during the repetition window rather than
  serving cached responses. Its outputs nonetheless differ from the published
  run (repetition-versus-published `F_attr` Spearman 0.78), indicating a change
  in the served model between the two collection dates.
- **Repetition versus published:** for hosted models these comparisons combine
  run-to-run noise with drift over the roughly two months between collections;
  `rep_vs_rep` rows isolate run-to-run noise.
- **Coverage:** hosted top-k censoring varies between runs, so joint
  observation coverage for `F_attr` falls to 0.74 (GPT-4.1) and 0.92 (GPT-4o)
  under the canonical pair-specific complete-case policy; every row reports it.
- **Content filters:** each GPT repetition had 124-129 content-filtered
  coordinates (published: 136 for GPT-4o, 142 for GPT-4.1) and no terminal
  request failures.
- Open-model repetitions used a different build of the same PyTorch release
  than the published runs; this did not change any output.
