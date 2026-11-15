# Worked example plan: when can Qwen2.5-14B explain GPT-4o?

This responds to the weak-reject review: "never shows an open model explaining
a closed one", "reframe as an API-compatible screen, and add a worked case
study on the best-matching pair". Qwen2.5-14B is GPT-4o's best open surrogate
on every BoolQ metric. The aim is a catchy, rigorous case study, not an
exhaustive one. There are no model or API calls.

## Phase 0: deterministic tables (done)

`python -m benchmark_scripts.worked_example build` writes the per-prompt,
per-sentence and per-word tables, the stratified summaries (`strata.csv`,
`gate.csv`) and the candidate pools. `inspect <prompt_idx>` prints any
prompt's sentence table.

## Phase 1: one agent per angle, in parallel

Every agent follows `AGENT_BRIEF.md` and writes `candidates/<angle>.md` plus a
reproducible `scratch/<angle>/analysis.py`.

| Angle | Question |
|---|---|
| A. shared_failure | When GPT-4o is wrong, does Qwen's attribution find the sentence that misleads it? |
| B. trust_gate | Does one API call (the ℓ gap) tell you when Qwen's attributions can be trusted? |
| C. different_reasons | Both models right for different reasons: how often, and does the method catch it? |
| D. inside_qwen | Do Qwen's internals (alignment, attention) point at the sentence GPT-4o relies on? |
| E. word_level | High risk: does anything catchy survive at word level, given the sparse sample and API noise? |

## Phase 2: independent verification

A fresh agent re-runs every `analysis.py`, spot-checks every quoted number
against `inspect`, and flags unsupported claims, hidden cherry-picking and
overclaiming.

## Phase 3: scoring and synthesis

Each candidate is scored 1–5 on interestingness, rigor (after verification)
and relevance to the review. The best one or two become `REPORT.md`, with a
ranked shortlist, a draft rebuttal paragraph and a figure for the top case.

## Status

All phases are done. Results, scoring and the draft rebuttal paragraph are in
`REPORT.md`. In synthesis, the trust gate (B) and the Qwen-support check from
(C) were combined into a two-check screen and re-validated out of sample in
`benchmark_scripts/worked_example.py`. The checker's review of that synthesis
is in `verification/report.md`.
