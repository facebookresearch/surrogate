# Muse Glimmer BoolQ robustness extension

This directory evaluates `meta-models/Muse-Glimmer-30B` against the eleven
models in the paper cohort on the complete BoolQ sentence configuration. It is
a post-paper robustness result, not a retroactive change to the fixed headline
cohort.

The directory contains the shared segment manifest, the merged per-segment and
per-token model outputs, a run receipt embedding the independently executed
attention and ablation receipts, and the derived fidelity table with its
provenance sidecar.

The model run is pinned to revision
`a4e59da52a7bc87ae7251dd5545c0dd437c44b68`. The runner verifies SHA-256
digests for both weight shards and all scoring-critical tokenizer and config
files before loading. It also verifies exact equality with the committed BoolQ
segment manifest.

Muse Glimmer's chat template ends at an assistant routing boundary. Scoring
there measures routing tokens rather than answer content, so the adapter uses
the model's direct-response route, ` to=user<|message|>`, before looking up the
next-token label probabilities. This protocol is recorded in the run receipt.

`glimmer_boolq_fidelity.tsv` reports prompt-cluster bootstrap estimates for
`F_pred` and signed `F_attr` on user-message segments. Glimmer's output head
applies a nonlinear softcap, so fixed-unembedding representation projections
are retained in the raw segment artifact as diagnostics but are deliberately
excluded from this headline extension table.

Across the eleven reference models, the median Pearson r-squared is 0.735 for
`F_pred` and 0.557 for `F_attr`. Prediction fidelity exceeds attribution
fidelity for every pair (median gap 0.176). The table contains the per-pair
estimates, confidence intervals, and complete-case coverage.

The raw attention columns likewise expose the decoder's pre-gate QK attention;
Glimmer subsequently gates each layer's attention output. They are useful
architecture-aware diagnostics, not effective contribution weights, and are
not included in the extension table.

From the repository root, rerun
`python -m benchmark_scripts.glimmer_robustness --verify-existing` to verify the
raw coordinate and label grids, execution and output hashes, embedded phase
receipts, derived estimates, and derived provenance without modifying files.
