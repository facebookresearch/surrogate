# Rebuttal reanalyses: summary

Reanalysis of existing BoolQ/sentence results (analysis 6 uses every
Table 2 benchmark). No model or API calls; regenerate with
`python -m benchmark_scripts.reanalysis_rebuttal`. Values are pair-set
medians of Pearson r² unless the statistic says otherwise; per-pair rows
are in the CSVs. Pair sets follow the paper: `F_pred` / `F_attr` use the
55 unordered pairs, representation metrics the 10 open pairs, and
cross-level metrics the 50 ordered open-source pairs (All = Open→All).
Every baseline equals the published `f_table.tsv` value for the same
pair (checked to 1e-9).

## Notes on the paper's claims

1. **Label conditioning (changes the F_pred > F_attr reading).** Within one gold label the ordering mostly holds (All: gold-True F_pred 0.498 vs F_attr 0.428; gold-False 0.407 vs 0.429), but among prompts both models answer correctly, within one label, median F_pred falls to 0.214 / 0.197 while F_attr is 0.559 / 0.508. Conditioning removes the between-class variance (range restriction), and F_pred loses most of its r² while F_attr is unchanged or higher in the same subsets: the headline F_pred mainly reflects agreement on the answer, not on confidence within a class.
2. **Length control (no change).** Pooled over all segments, log word count explains at most 0.11 of any open-model signal's variance, and residualizing moves no pair-set median by more than 0.021. Within passages length explains 0.05–0.30; controlling it there moves passage-only medians by at most 0.028 (analysis 5).
3. **Unsigned attribution (qualifies the attention claim).** Attention is nearly uninformative about the sign of attribution (signed medians ≤ 0.005) but predicts its magnitude: unsigned medians reach 0.233 (mean attention, Open→Closed) and survive length control (0.232). Part of this is segment role (the question segment has the highest mean attention and the largest |Δℓ|), but not all: on passage segments alone, mean and max attention still explain 0.12–0.14 of |Δℓ| (0.09–0.11 after within-passage length control); rollout explains 0.022.
4. **Within-prompt (no change for r²).** Removing per-prompt means changes every pooled-r² median by less than 0.05. Per-prompt Spearman moves `F_mag_to_attr` Open→Open 0.470→0.364; `F_mag_to_attr` Open→Closed 0.495→0.391; `F_mag_to_attr` All 0.489→0.385; `F_attn_mean_to_attr` Open→Open 0.042→-0.036; `F_attn_max_to_attr` Open→Open 0.019→-0.053.
5. **Passage-only (changes the magnitude claim).** On passage segments F_mag→|attr| rises from 0.108 to 0.425 (All; 0.398 with within-passage length control). The published value is depressed by system segments, which have large ‖Δz‖ but small |Δℓ|. F_align→attr falls from 0.257 to 0.183; F_attr changes from 0.451 to 0.404.
6. **Table 2 by pair type (no change).** BoolQ F_pred is similar within and across families (0.710 vs 0.726). F_attr is higher within family on BoolQ (0.489 vs 0.439) and every other benchmark except WinoGrande. Open↔closed medians are below the all-pairs median on every benchmark for both metrics. Only three same-family open↔closed pairs exist (Llama-3.1-8B with each hosted Llama).
7. **Decisive segments (structural, qualify before citing).** Top-1 agreement is 0.618 vs 1/N chance 0.138 (All), rising from 0.340 (q1) to 0.776 (q5). But the question segment is the target's top segment in 61% of prompts (rising with confidence), so 'always pick the question' matches about as often and accounts for the confidence trend. On passage segments agreement is 0.560 vs chance 0.271, nearly flat across quintiles (0.516 → 0.606). Qwen-14B → GPT-4o: 0.667 (passage-only 0.585, chance 0.278). Qwen-14B's top mean-attention segment matches GPT-4o's top |Δℓ| segment in 0.616 of prompts, essentially the share in which GPT-4o's top is the question (0.617), because Qwen-14B's mean attention almost always peaks on the question; on passages it is 0.426 vs chance 0.278.

## 1. Label conditioning

Subsets restrict prompts; correctness is sign(ℓ) = gold for both models.

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_pred` | Open→Open | 0.650 | gold_true | 0.460 [0.104, 0.598] | -0.190 |
| `F_pred` | Open→Open | 0.650 | gold_false | 0.354 [0.025, 0.460] | -0.295 |
| `F_pred` | Open→Open | 0.650 | both_correct | 0.875 [0.545, 0.917] | +0.225 |
| `F_pred` | Open→Open | 0.650 | both_correct_gold_true | 0.177 [0.070, 0.288] | -0.473 |
| `F_pred` | Open→Open | 0.650 | both_correct_gold_false | 0.182 [0.002, 0.283] | -0.468 |
| `F_pred` | Open→Closed | 0.669 | gold_true | 0.429 [0.048, 0.678] | -0.240 |
| `F_pred` | Open→Closed | 0.669 | gold_false | 0.333 [0.004, 0.505] | -0.336 |
| `F_pred` | Open→Closed | 0.669 | both_correct | 0.881 [0.379, 0.933] | +0.212 |
| `F_pred` | Open→Closed | 0.669 | both_correct_gold_true | 0.188 [0.047, 0.368] | -0.481 |
| `F_pred` | Open→Closed | 0.669 | both_correct_gold_false | 0.150 [0.000, 0.366] | -0.519 |
| `F_pred` | All | 0.718 | gold_true | 0.498 [0.048, 0.914] | -0.220 |
| `F_pred` | All | 0.718 | gold_false | 0.407 [0.004, 0.822] | -0.312 |
| `F_pred` | All | 0.718 | both_correct | 0.883 [0.379, 0.964] | +0.165 |
| `F_pred` | All | 0.718 | both_correct_gold_true | 0.214 [0.047, 0.799] | -0.504 |
| `F_pred` | All | 0.718 | both_correct_gold_false | 0.197 [0.000, 0.747] | -0.521 |
| `F_attr` | Open→Open | 0.399 | gold_true | 0.396 [0.083, 0.497] | -0.002 |
| `F_attr` | Open→Open | 0.399 | gold_false | 0.254 [0.070, 0.628] | -0.144 |
| `F_attr` | Open→Open | 0.399 | both_correct | 0.517 [0.185, 0.674] | +0.118 |
| `F_attr` | Open→Open | 0.399 | both_correct_gold_true | 0.552 [0.107, 0.618] | +0.153 |
| `F_attr` | Open→Open | 0.399 | both_correct_gold_false | 0.431 [0.252, 0.728] | +0.032 |
| `F_attr` | Open→Closed | 0.404 | gold_true | 0.388 [0.018, 0.521] | -0.015 |
| `F_attr` | Open→Closed | 0.404 | gold_false | 0.320 [0.038, 0.593] | -0.084 |
| `F_attr` | Open→Closed | 0.404 | both_correct | 0.531 [0.077, 0.669] | +0.127 |
| `F_attr` | Open→Closed | 0.404 | both_correct_gold_true | 0.533 [0.029, 0.632] | +0.129 |
| `F_attr` | Open→Closed | 0.404 | both_correct_gold_false | 0.489 [0.173, 0.662] | +0.085 |
| `F_attr` | All | 0.451 | gold_true | 0.428 [0.018, 0.838] | -0.024 |
| `F_attr` | All | 0.451 | gold_false | 0.429 [0.038, 0.790] | -0.023 |
| `F_attr` | All | 0.451 | both_correct | 0.586 [0.077, 0.839] | +0.135 |
| `F_attr` | All | 0.451 | both_correct_gold_true | 0.559 [0.029, 0.849] | +0.108 |
| `F_attr` | All | 0.451 | both_correct_gold_false | 0.508 [0.173, 0.804] | +0.056 |

Median prompts per subset (All pairs): unconditioned 3267, gold_true 2033, gold_false 1236, both_correct 2356, both_correct_gold_true 1486, both_correct_gold_false 1016.

Model accuracy (sign of ℓ vs. gold): qwen2.5-0.5b-instruct 0.664, qwen2.5-3b-instruct 0.767, llama-3.1-8b-instruct 0.748, qwen2.5-7b-instruct 0.842, qwen2.5-14b-instruct 0.838, llama3.1-70b-instruct 0.895, llama3.3-70b-instruct 0.891, llama4-maverick-17b-128e-instruct 0.831, gpt-4o 0.884, gpt-4-1 0.900, gemini-2-5-flash-lite-vertex 0.881.

**Flag (median F_pred ≤ F_attr):** gold_false / All (0.407 ≤ 0.429); both_correct_gold_true / Open→Open (0.177 ≤ 0.552); both_correct_gold_true / Open→Closed (0.188 ≤ 0.533); both_correct_gold_true / All (0.214 ≤ 0.559); both_correct_gold_false / Open→Open (0.182 ≤ 0.431); both_correct_gold_false / Open→Closed (0.150 ≤ 0.489); both_correct_gold_false / All (0.197 ≤ 0.508)

## 2. Segment-length control

Each open model's ‖Δz‖ and attention scores are residualized on
log(whitespace word count) by OLS over that model's segments; the
attribution target is not residualized.

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_attn_mean` | Open→Open | 0.904 | length_residualized | 0.904 [0.748, 0.986] | +0.000 |
| `F_attn_mean` | All | 0.904 | length_residualized | 0.904 [0.748, 0.986] | +0.000 |
| `F_attn_max` | Open→Open | 0.859 | length_residualized | 0.865 [0.658, 0.919] | +0.006 |
| `F_attn_max` | All | 0.859 | length_residualized | 0.865 [0.658, 0.919] | +0.006 |
| `F_attn_rollout` | Open→Open | 0.861 | length_residualized | 0.882 [0.575, 0.935] | +0.021 |
| `F_attn_rollout` | All | 0.861 | length_residualized | 0.882 [0.575, 0.935] | +0.021 |
| `F_mag` | Open→Open | 0.843 | length_residualized | 0.845 [0.639, 0.904] | +0.001 |
| `F_mag` | All | 0.843 | length_residualized | 0.845 [0.639, 0.904] | +0.001 |
| `F_mag_to_attr` | Open→Open | 0.065 | length_residualized | 0.065 [0.000, 0.231] | +0.000 |
| `F_mag_to_attr` | Open→Closed | 0.112 | length_residualized | 0.113 [0.000, 0.284] | +0.001 |
| `F_mag_to_attr` | All | 0.108 | length_residualized | 0.109 [0.000, 0.284] | +0.001 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | length_residualized | 0.005 [0.000, 0.043] | -0.000 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | length_residualized | 0.001 [0.000, 0.024] | -0.000 |
| `F_attn_mean_to_attr` | All | 0.002 | length_residualized | 0.002 [0.000, 0.043] | +0.000 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | length_residualized | 0.003 [0.000, 0.032] | -0.000 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | length_residualized | 0.002 [0.000, 0.020] | +0.000 |
| `F_attn_max_to_attr` | All | 0.002 | length_residualized | 0.002 [0.000, 0.032] | +0.000 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | length_residualized | 0.001 [0.000, 0.063] | -0.002 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | length_residualized | 0.004 [0.000, 0.014] | -0.000 |
| `F_attn_rollout_to_attr` | All | 0.003 | length_residualized | 0.002 [0.000, 0.063] | -0.001 |

Variance explained by log length (r², range over open models), all
segments / passage segments: `F_attn_mean` 0.00–0.00 / 0.11–0.23, `F_attn_max` 0.00–0.03 / 0.08–0.30, `F_attn_rollout` 0.00–0.11 / 0.05–0.22, `F_mag` 0.00–0.00 / 0.06–0.09.

## 3. Unsigned attribution

F_attn→|attr| for each attention variant. Passage-only rows (diagnostic)
drop system and question segments; their length control is fitted
within passages.

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_attn_mean_to_attr` | Open→Open | 0.005 | unsigned_raw | 0.162 [0.022, 0.290] | +0.157 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | unsigned_length_residualized | 0.161 [0.022, 0.290] | +0.156 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | signed_passage_only | 0.055 [0.012, 0.278] | +0.050 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | unsigned_passage_only | 0.142 [0.083, 0.291] | +0.137 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | unsigned_passage_only_length_residualized | 0.118 [0.062, 0.262] | +0.112 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | unsigned_raw | 0.233 [0.055, 0.400] | +0.232 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | unsigned_length_residualized | 0.232 [0.055, 0.399] | +0.231 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | signed_passage_only | 0.019 [0.004, 0.045] | +0.018 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | unsigned_passage_only | 0.130 [0.070, 0.208] | +0.129 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | unsigned_passage_only_length_residualized | 0.104 [0.051, 0.173] | +0.103 |
| `F_attn_mean_to_attr` | All | 0.002 | unsigned_raw | 0.199 [0.022, 0.400] | +0.197 |
| `F_attn_mean_to_attr` | All | 0.002 | unsigned_length_residualized | 0.199 [0.022, 0.399] | +0.197 |
| `F_attn_mean_to_attr` | All | 0.002 | signed_passage_only | 0.029 [0.004, 0.278] | +0.026 |
| `F_attn_mean_to_attr` | All | 0.002 | unsigned_passage_only | 0.136 [0.070, 0.291] | +0.134 |
| `F_attn_mean_to_attr` | All | 0.002 | unsigned_passage_only_length_residualized | 0.114 [0.051, 0.262] | +0.112 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | unsigned_raw | 0.101 [0.027, 0.200] | +0.098 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | unsigned_length_residualized | 0.095 [0.027, 0.205] | +0.092 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | signed_passage_only | 0.048 [0.002, 0.268] | +0.044 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | unsigned_passage_only | 0.132 [0.036, 0.274] | +0.129 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | unsigned_passage_only_length_residualized | 0.110 [0.023, 0.252] | +0.106 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | unsigned_raw | 0.140 [0.040, 0.285] | +0.139 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | unsigned_length_residualized | 0.146 [0.044, 0.293] | +0.144 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | signed_passage_only | 0.023 [0.001, 0.066] | +0.021 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | unsigned_passage_only | 0.121 [0.029, 0.230] | +0.119 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | unsigned_passage_only_length_residualized | 0.090 [0.017, 0.204] | +0.089 |
| `F_attn_max_to_attr` | All | 0.002 | unsigned_raw | 0.118 [0.027, 0.285] | +0.116 |
| `F_attn_max_to_attr` | All | 0.002 | unsigned_length_residualized | 0.119 [0.027, 0.293] | +0.117 |
| `F_attn_max_to_attr` | All | 0.002 | signed_passage_only | 0.028 [0.001, 0.268] | +0.026 |
| `F_attn_max_to_attr` | All | 0.002 | unsigned_passage_only | 0.121 [0.029, 0.274] | +0.119 |
| `F_attn_max_to_attr` | All | 0.002 | unsigned_passage_only_length_residualized | 0.090 [0.017, 0.252] | +0.088 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | unsigned_raw | 0.041 [0.000, 0.061] | +0.037 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | unsigned_length_residualized | 0.045 [0.000, 0.065] | +0.041 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | signed_passage_only | 0.008 [0.001, 0.055] | +0.005 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | unsigned_passage_only | 0.024 [0.001, 0.074] | +0.021 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | unsigned_passage_only_length_residualized | 0.011 [0.000, 0.051] | +0.008 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | unsigned_raw | 0.040 [0.009, 0.057] | +0.036 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | unsigned_length_residualized | 0.047 [0.009, 0.062] | +0.043 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | signed_passage_only | 0.003 [0.000, 0.019] | -0.001 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | unsigned_passage_only | 0.022 [0.002, 0.083] | +0.018 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | unsigned_passage_only_length_residualized | 0.010 [0.000, 0.053] | +0.006 |
| `F_attn_rollout_to_attr` | All | 0.003 | unsigned_raw | 0.040 [0.000, 0.061] | +0.037 |
| `F_attn_rollout_to_attr` | All | 0.003 | unsigned_length_residualized | 0.046 [0.000, 0.065] | +0.043 |
| `F_attn_rollout_to_attr` | All | 0.003 | signed_passage_only | 0.003 [0.000, 0.055] | -0.000 |
| `F_attn_rollout_to_attr` | All | 0.003 | unsigned_passage_only | 0.022 [0.001, 0.083] | +0.019 |
| `F_attn_rollout_to_attr` | All | 0.003 | unsigned_passage_only_length_residualized | 0.010 [0.000, 0.053] | +0.007 |

## 4. Within-prompt

(a) Pearson r² after subtracting per-prompt means (baseline: pooled r²).

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_attr` | Open→Open | 0.399 | prompt_demeaned | 0.384 [0.078, 0.574] | -0.014 |
| `F_attr` | Open→Closed | 0.404 | prompt_demeaned | 0.419 [0.031, 0.583] | +0.015 |
| `F_attr` | All | 0.451 | prompt_demeaned | 0.473 [0.031, 0.834] | +0.022 |
| `F_mag` | Open→Open | 0.843 | prompt_demeaned | 0.850 [0.634, 0.910] | +0.007 |
| `F_mag` | All | 0.843 | prompt_demeaned | 0.850 [0.634, 0.910] | +0.007 |
| `F_align` | Open→Open | 0.201 | prompt_demeaned | 0.207 [0.054, 0.329] | +0.005 |
| `F_align` | All | 0.201 | prompt_demeaned | 0.207 [0.054, 0.329] | +0.005 |
| `F_attn_mean` | Open→Open | 0.904 | prompt_demeaned | 0.895 [0.724, 0.983] | -0.009 |
| `F_attn_mean` | All | 0.904 | prompt_demeaned | 0.895 [0.724, 0.983] | -0.009 |
| `F_attn_max` | Open→Open | 0.859 | prompt_demeaned | 0.851 [0.606, 0.900] | -0.008 |
| `F_attn_max` | All | 0.859 | prompt_demeaned | 0.851 [0.606, 0.900] | -0.008 |
| `F_attn_rollout` | Open→Open | 0.861 | prompt_demeaned | 0.880 [0.538, 0.961] | +0.019 |
| `F_attn_rollout` | All | 0.861 | prompt_demeaned | 0.880 [0.538, 0.961] | +0.019 |
| `F_mag_to_attr` | Open→Open | 0.065 | prompt_demeaned | 0.047 [0.001, 0.215] | -0.018 |
| `F_mag_to_attr` | Open→Closed | 0.112 | prompt_demeaned | 0.096 [0.000, 0.276] | -0.015 |
| `F_mag_to_attr` | All | 0.108 | prompt_demeaned | 0.092 [0.000, 0.276] | -0.016 |
| `F_align_to_attr` | Open→Open | 0.254 | prompt_demeaned | 0.246 [0.042, 0.404] | -0.008 |
| `F_align_to_attr` | Open→Closed | 0.267 | prompt_demeaned | 0.262 [0.039, 0.431] | -0.004 |
| `F_align_to_attr` | All | 0.257 | prompt_demeaned | 0.259 [0.039, 0.431] | +0.002 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | prompt_demeaned | 0.007 [0.001, 0.094] | +0.002 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | prompt_demeaned | 0.002 [0.000, 0.029] | +0.001 |
| `F_attn_mean_to_attr` | All | 0.002 | prompt_demeaned | 0.004 [0.000, 0.094] | +0.002 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | prompt_demeaned | 0.004 [0.000, 0.085] | +0.001 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | prompt_demeaned | 0.002 [0.000, 0.025] | +0.000 |
| `F_attn_max_to_attr` | All | 0.002 | prompt_demeaned | 0.003 [0.000, 0.085] | +0.001 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | prompt_demeaned | 0.004 [0.000, 0.073] | +0.001 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | prompt_demeaned | 0.005 [0.000, 0.019] | +0.001 |
| `F_attn_rollout_to_attr` | All | 0.003 | prompt_demeaned | 0.005 [0.000, 0.073] | +0.001 |

(b) Mean per-prompt Spearman ρ over prompts with ≥ 3 shared segments
(baseline: pooled Spearman ρ).

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_attr` | Open→Open | 0.372 | mean_per_prompt | 0.415 [0.098, 0.466] | +0.043 |
| `F_attr` | Open→Closed | 0.384 | mean_per_prompt | 0.395 [0.027, 0.528] | +0.011 |
| `F_attr` | All | 0.392 | mean_per_prompt | 0.406 [0.027, 0.768] | +0.014 |
| `F_mag` | Open→Open | 0.890 | mean_per_prompt | 0.858 [0.733, 0.896] | -0.032 |
| `F_mag` | All | 0.890 | mean_per_prompt | 0.858 [0.733, 0.896] | -0.032 |
| `F_align` | Open→Open | 0.355 | mean_per_prompt | 0.383 [0.161, 0.480] | +0.028 |
| `F_align` | All | 0.355 | mean_per_prompt | 0.383 [0.161, 0.480] | +0.028 |
| `F_attn_mean` | Open→Open | 0.951 | mean_per_prompt | 0.915 [0.886, 0.965] | -0.036 |
| `F_attn_mean` | All | 0.951 | mean_per_prompt | 0.915 [0.886, 0.965] | -0.036 |
| `F_attn_max` | Open→Open | 0.919 | mean_per_prompt | 0.885 [0.745, 0.921] | -0.034 |
| `F_attn_max` | All | 0.919 | mean_per_prompt | 0.885 [0.745, 0.921] | -0.034 |
| `F_attn_rollout` | Open→Open | 0.954 | mean_per_prompt | 0.950 [0.895, 0.985] | -0.004 |
| `F_attn_rollout` | All | 0.954 | mean_per_prompt | 0.950 [0.895, 0.985] | -0.004 |
| `F_mag_to_attr` | Open→Open | 0.470 | mean_per_prompt | 0.364 [0.180, 0.508] | -0.107 |
| `F_mag_to_attr` | Open→Closed | 0.495 | mean_per_prompt | 0.391 [0.134, 0.619] | -0.105 |
| `F_mag_to_attr` | All | 0.489 | mean_per_prompt | 0.385 [0.134, 0.619] | -0.104 |
| `F_align_to_attr` | Open→Open | 0.357 | mean_per_prompt | 0.390 [0.130, 0.467] | +0.034 |
| `F_align_to_attr` | Open→Closed | 0.361 | mean_per_prompt | 0.377 [0.055, 0.498] | +0.016 |
| `F_align_to_attr` | All | 0.359 | mean_per_prompt | 0.381 [0.055, 0.498] | +0.021 |
| `F_attn_mean_to_attr` | Open→Open | 0.042 | mean_per_prompt | -0.036 [-0.103, 0.104] | -0.078 |
| `F_attn_mean_to_attr` | Open→Closed | -0.026 | mean_per_prompt | -0.039 [-0.132, 0.066] | -0.013 |
| `F_attn_mean_to_attr` | All | -0.015 | mean_per_prompt | -0.039 [-0.132, 0.104] | -0.024 |
| `F_attn_max_to_attr` | Open→Open | 0.019 | mean_per_prompt | -0.053 [-0.127, 0.152] | -0.072 |
| `F_attn_max_to_attr` | Open→Closed | -0.032 | mean_per_prompt | -0.048 [-0.125, 0.111] | -0.016 |
| `F_attn_max_to_attr` | All | -0.021 | mean_per_prompt | -0.053 [-0.127, 0.152] | -0.031 |
| `F_attn_rollout_to_attr` | Open→Open | -0.033 | mean_per_prompt | -0.002 [-0.121, 0.223] | +0.031 |
| `F_attn_rollout_to_attr` | Open→Closed | -0.060 | mean_per_prompt | -0.034 [-0.286, 0.048] | +0.026 |
| `F_attn_rollout_to_attr` | All | -0.044 | mean_per_prompt | -0.023 [-0.286, 0.223] | +0.022 |

## 5. Passage-only

System-prompt and question segments are dropped. For magnitude and
attention sources, `passage_only_length_residualized` also residualizes
the source on log length within passages.

| Metric | Pair set | Baseline median | Variant | Variant median [min, max] | Change |
|---|---|---:|---|---:|---:|
| `F_attr` | Open→Open | 0.399 | passage_only | 0.394 [0.144, 0.529] | -0.005 |
| `F_attr` | Open→Closed | 0.404 | passage_only | 0.365 [0.076, 0.540] | -0.039 |
| `F_attr` | All | 0.451 | passage_only | 0.404 [0.076, 0.853] | -0.047 |
| `F_mag_to_attr` | Open→Open | 0.065 | passage_only | 0.417 [0.191, 0.506] | +0.353 |
| `F_mag_to_attr` | Open→Open | 0.065 | passage_only_length_residualized | 0.393 [0.167, 0.482] | +0.328 |
| `F_mag_to_attr` | Open→Closed | 0.112 | passage_only | 0.434 [0.124, 0.536] | +0.322 |
| `F_mag_to_attr` | Open→Closed | 0.112 | passage_only_length_residualized | 0.406 [0.107, 0.509] | +0.294 |
| `F_mag_to_attr` | All | 0.108 | passage_only | 0.425 [0.124, 0.536] | +0.317 |
| `F_mag_to_attr` | All | 0.108 | passage_only_length_residualized | 0.398 [0.107, 0.509] | +0.290 |
| `F_align_to_attr` | Open→Open | 0.254 | passage_only | 0.178 [0.050, 0.284] | -0.076 |
| `F_align_to_attr` | Open→Closed | 0.267 | passage_only | 0.187 [0.046, 0.276] | -0.079 |
| `F_align_to_attr` | All | 0.257 | passage_only | 0.183 [0.046, 0.284] | -0.074 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | passage_only | 0.055 [0.012, 0.278] | +0.050 |
| `F_attn_mean_to_attr` | Open→Open | 0.005 | passage_only_length_residualized | 0.046 [0.008, 0.267] | +0.041 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | passage_only | 0.019 [0.004, 0.045] | +0.018 |
| `F_attn_mean_to_attr` | Open→Closed | 0.001 | passage_only_length_residualized | 0.014 [0.002, 0.039] | +0.013 |
| `F_attn_mean_to_attr` | All | 0.002 | passage_only | 0.029 [0.004, 0.278] | +0.026 |
| `F_attn_mean_to_attr` | All | 0.002 | passage_only_length_residualized | 0.023 [0.002, 0.267] | +0.021 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | passage_only | 0.048 [0.002, 0.268] | +0.044 |
| `F_attn_max_to_attr` | Open→Open | 0.003 | passage_only_length_residualized | 0.038 [0.001, 0.268] | +0.034 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | passage_only | 0.023 [0.001, 0.066] | +0.021 |
| `F_attn_max_to_attr` | Open→Closed | 0.002 | passage_only_length_residualized | 0.017 [0.000, 0.062] | +0.015 |
| `F_attn_max_to_attr` | All | 0.002 | passage_only | 0.028 [0.001, 0.268] | +0.026 |
| `F_attn_max_to_attr` | All | 0.002 | passage_only_length_residualized | 0.023 [0.000, 0.268] | +0.021 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | passage_only | 0.008 [0.001, 0.055] | +0.005 |
| `F_attn_rollout_to_attr` | Open→Open | 0.003 | passage_only_length_residualized | 0.004 [0.000, 0.035] | +0.001 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | passage_only | 0.003 [0.000, 0.019] | -0.001 |
| `F_attn_rollout_to_attr` | Open→Closed | 0.004 | passage_only_length_residualized | 0.001 [0.000, 0.013] | -0.003 |
| `F_attn_rollout_to_attr` | All | 0.003 | passage_only | 0.003 [0.000, 0.055] | -0.000 |
| `F_attn_rollout_to_attr` | All | 0.003 | passage_only_length_residualized | 0.001 [0.000, 0.035] | -0.002 |

## 6. Table 2 by pair type

Published per-pair values (BoolQ / WinoGrande / BoolQ word: Pearson r²;
ANLI / RACE: centered RV, as in Table 2), median (n pairs). Families:
Qwen (4 open), Llama (Llama-3.1-8B open; Llama-3.1-70B, Llama-3.3-70B,
Llama-4 Maverick hosted), GPT (GPT-4o, GPT-4.1), Gemini. Same/cross
family span all 55 pairs; open↔closed pairs are further split by family.

| Metric | Pair type | BoolQ | ANLI R1 | ANLI R2 | ANLI R3 | WinoGrande | BoolQ word | RACE RV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `F_pred` | all | 0.718 (55) | 0.531 (55) | 0.443 (55) | 0.520 (55) | 0.224 (55) | 0.719 (55) | 0.791 (55) |
| `F_pred` | same_family | 0.710 (13) | 0.477 (13) | 0.429 (13) | 0.492 (13) | 0.150 (13) | 0.710 (13) | 0.780 (13) |
| `F_pred` | cross_family | 0.726 (42) | 0.534 (42) | 0.456 (42) | 0.530 (42) | 0.237 (42) | 0.732 (42) | 0.795 (42) |
| `F_pred` | open_closed | 0.669 (30) | 0.457 (30) | 0.410 (30) | 0.475 (30) | 0.198 (30) | 0.672 (30) | 0.751 (30) |
| `F_pred` | same_family_open_closed | 0.710 (3) | 0.476 (3) | 0.429 (3) | 0.492 (3) | 0.039 (3) | 0.710 (3) | 0.752 (3) |
| `F_pred` | cross_family_open_closed | 0.656 (27) | 0.445 (27) | 0.367 (27) | 0.443 (27) | 0.203 (27) | 0.665 (27) | 0.751 (27) |
| `F_attr` | all | 0.451 (55) | 0.308 (55) | 0.265 (55) | 0.291 (55) | 0.152 (55) | 0.168 (55) | 0.334 (55) |
| `F_attr` | same_family | 0.489 (13) | 0.371 (13) | 0.304 (13) | 0.326 (13) | 0.077 (13) | 0.258 (13) | 0.394 (13) |
| `F_attr` | cross_family | 0.439 (42) | 0.307 (42) | 0.247 (42) | 0.259 (42) | 0.167 (42) | 0.161 (42) | 0.320 (42) |
| `F_attr` | open_closed | 0.404 (30) | 0.239 (30) | 0.203 (30) | 0.223 (30) | 0.128 (30) | 0.130 (30) | 0.314 (30) |
| `F_attr` | same_family_open_closed | 0.450 (3) | 0.276 (3) | 0.265 (3) | 0.291 (3) | 0.025 (3) | 0.280 (3) | 0.499 (3) |
| `F_attr` | cross_family_open_closed | 0.393 (27) | 0.236 (27) | 0.189 (27) | 0.207 (27) | 0.142 (27) | 0.128 (27) | 0.308 (27) |

## 7. Decisive-segment agreement

Per prompt, each model's top segment by |Δℓ| (or by attention for the
source, in attention rows) over the pair's shared finite segments
(prompts with ≥ 2; top-3 overlap uses prompts with > 3); ties are broken
at random with a fixed seed. Chance is the mean of 1/N (top-1) and 3/N
(top-3 overlap fraction). 'Q-top' is the share of prompts whose target
top segment is the question, i.e. the top-1 agreement of always picking
the question. 'S-top' is the same share for the system prompt.
'Overlap' (passage rows only) is the top-1 agreement of a no-model
baseline that picks the passage sentence sharing the most content words
with the question (stopwords removed, crude plural strip; k tied
sentences share credit 1/k). Confidence quintiles use the mean |ℓ| of the two models
(q5 most confident).

| Scope | Pair set | Stratum | Top-1 | Chance | Q-top | S-top | Overlap | Top-3 | Chance |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| all_segments | Open→Open | all | 0.511 | 0.129 | 0.577 | 0.030 | — | 0.596 | 0.386 |
| all_segments | Open→Open | q1 | 0.330 | 0.127 | 0.239 | 0.076 | — | 0.523 | 0.382 |
| all_segments | Open→Open | q3 | 0.486 | 0.128 | 0.688 | 0.011 | — | 0.609 | 0.384 |
| all_segments | Open→Open | q5 | 0.744 | 0.131 | 0.778 | 0.009 | — | 0.597 | 0.393 |
| all_segments | Open→Closed | all | 0.573 | 0.138 | 0.612 | 0.083 | — | 0.607 | 0.413 |
| all_segments | Open→Closed | q1 | 0.335 | 0.132 | 0.250 | 0.197 | — | 0.525 | 0.397 |
| all_segments | Open→Closed | q3 | 0.608 | 0.135 | 0.698 | 0.032 | — | 0.618 | 0.401 |
| all_segments | Open→Closed | q5 | 0.758 | 0.142 | 0.818 | 0.002 | — | 0.658 | 0.419 |
| all_segments | All | all | 0.618 | 0.138 | 0.607 | 0.062 | — | 0.612 | 0.414 |
| all_segments | All | q1 | 0.340 | 0.132 | 0.241 | 0.168 | — | 0.529 | 0.397 |
| all_segments | All | q3 | 0.656 | 0.135 | 0.677 | 0.037 | — | 0.624 | 0.402 |
| all_segments | All | q5 | 0.776 | 0.142 | 0.773 | 0.006 | — | 0.662 | 0.421 |
| passage_only | Open→Open | all | 0.542 | 0.271 | 0.000 | 0.000 | 0.506 | 0.649 | 0.562 |
| passage_only | Open→Open | q1 | 0.539 | 0.265 | 0.000 | 0.000 | 0.475 | 0.639 | 0.542 |
| passage_only | Open→Open | q3 | 0.533 | 0.269 | 0.000 | 0.000 | 0.489 | 0.649 | 0.562 |
| passage_only | Open→Open | q5 | 0.535 | 0.282 | 0.000 | 0.000 | 0.546 | 0.664 | 0.587 |
| passage_only | Open→Closed | all | 0.555 | 0.271 | 0.000 | 0.000 | 0.510 | 0.633 | 0.562 |
| passage_only | Open→Closed | q1 | 0.510 | 0.272 | 0.000 | 0.000 | 0.456 | 0.625 | 0.555 |
| passage_only | Open→Closed | q3 | 0.562 | 0.269 | 0.000 | 0.000 | 0.488 | 0.630 | 0.558 |
| passage_only | Open→Closed | q5 | 0.605 | 0.279 | 0.000 | 0.000 | 0.582 | 0.650 | 0.584 |
| passage_only | All | all | 0.560 | 0.271 | 0.000 | 0.000 | 0.504 | 0.641 | 0.562 |
| passage_only | All | q1 | 0.516 | 0.272 | 0.000 | 0.000 | 0.458 | 0.632 | 0.555 |
| passage_only | All | q3 | 0.566 | 0.269 | 0.000 | 0.000 | 0.487 | 0.634 | 0.558 |
| passage_only | All | q5 | 0.606 | 0.280 | 0.000 | 0.000 | 0.578 | 0.656 | 0.584 |

Qwen2.5-14B → GPT-4o. Rows `F_attn_*` rank Qwen-14B's segments by
attention and GPT-4o's by |Δℓ|.

| Comparison | Scope | Stratum | Top-1 | Chance | Q-top | S-top | Overlap | Top-3 | Chance | n |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs_attr_vs_abs_attr | all_segments | all | 0.667 | 0.140 | 0.617 | 0.104 | — | 0.612 | 0.414 | 3096 |
| abs_attr_vs_abs_attr | all_segments | q1 | 0.381 | 0.132 | 0.202 | 0.277 | — | 0.531 | 0.397 | 620 |
| abs_attr_vs_abs_attr | all_segments | q2 | 0.477 | 0.133 | 0.447 | 0.171 | — | 0.598 | 0.398 | 619 |
| abs_attr_vs_abs_attr | all_segments | q3 | 0.703 | 0.133 | 0.706 | 0.060 | — | 0.628 | 0.399 | 619 |
| abs_attr_vs_abs_attr | all_segments | q4 | 0.848 | 0.136 | 0.819 | 0.008 | — | 0.647 | 0.407 | 619 |
| abs_attr_vs_abs_attr | all_segments | q5 | 0.926 | 0.165 | 0.911 | 0.002 | — | 0.657 | 0.471 | 619 |
| F_attn_mean_vs_abs_attr | all_segments | all | 0.616 | 0.140 | 0.617 | 0.104 | — | 0.566 | 0.414 | 3096 |
| F_attn_mean_vs_abs_attr | all_segments | q1 | 0.202 | 0.132 | 0.202 | 0.277 | — | 0.515 | 0.397 | 620 |
| F_attn_mean_vs_abs_attr | all_segments | q5 | 0.911 | 0.165 | 0.911 | 0.002 | — | 0.631 | 0.471 | 619 |
| F_attn_max_vs_abs_attr | all_segments | all | 0.059 | 0.140 | 0.617 | 0.104 | — | 0.575 | 0.414 | 3096 |
| F_attn_max_vs_abs_attr | all_segments | q1 | 0.069 | 0.132 | 0.202 | 0.277 | — | 0.530 | 0.397 | 620 |
| F_attn_max_vs_abs_attr | all_segments | q5 | 0.040 | 0.165 | 0.911 | 0.002 | — | 0.626 | 0.471 | 619 |
| F_attn_rollout_vs_abs_attr | all_segments | all | 0.005 | 0.140 | 0.617 | 0.104 | — | 0.327 | 0.414 | 3096 |
| F_attn_rollout_vs_abs_attr | all_segments | q1 | 0.010 | 0.132 | 0.202 | 0.277 | — | 0.357 | 0.397 | 620 |
| F_attn_rollout_vs_abs_attr | all_segments | q5 | 0.002 | 0.165 | 0.911 | 0.002 | — | 0.374 | 0.471 | 619 |
| abs_attr_vs_abs_attr | passage_only | all | 0.585 | 0.278 | 0.000 | 0.000 | 0.481 | 0.622 | 0.569 | 2854 |
| abs_attr_vs_abs_attr | passage_only | q1 | 0.576 | 0.272 | 0.000 | 0.000 | 0.470 | 0.610 | 0.555 | 571 |
| abs_attr_vs_abs_attr | passage_only | q2 | 0.587 | 0.275 | 0.000 | 0.000 | 0.475 | 0.622 | 0.553 | 571 |
| abs_attr_vs_abs_attr | passage_only | q3 | 0.560 | 0.274 | 0.000 | 0.000 | 0.480 | 0.630 | 0.560 | 570 |
| abs_attr_vs_abs_attr | passage_only | q4 | 0.608 | 0.267 | 0.000 | 0.000 | 0.467 | 0.614 | 0.573 | 571 |
| abs_attr_vs_abs_attr | passage_only | q5 | 0.595 | 0.301 | 0.000 | 0.000 | 0.511 | 0.639 | 0.608 | 571 |
| F_attn_mean_vs_abs_attr | passage_only | all | 0.426 | 0.278 | 0.000 | 0.000 | 0.481 | 0.611 | 0.569 | 2854 |
| F_attn_mean_vs_abs_attr | passage_only | q1 | 0.441 | 0.272 | 0.000 | 0.000 | 0.470 | 0.618 | 0.555 | 571 |
| F_attn_mean_vs_abs_attr | passage_only | q5 | 0.431 | 0.301 | 0.000 | 0.000 | 0.511 | 0.638 | 0.608 | 571 |
| F_attn_max_vs_abs_attr | passage_only | all | 0.480 | 0.278 | 0.000 | 0.000 | 0.481 | 0.622 | 0.569 | 2854 |
| F_attn_max_vs_abs_attr | passage_only | q1 | 0.499 | 0.272 | 0.000 | 0.000 | 0.470 | 0.634 | 0.555 | 571 |
| F_attn_max_vs_abs_attr | passage_only | q5 | 0.469 | 0.301 | 0.000 | 0.000 | 0.511 | 0.647 | 0.608 | 571 |
| F_attn_rollout_vs_abs_attr | passage_only | all | 0.409 | 0.278 | 0.000 | 0.000 | 0.481 | 0.592 | 0.569 | 2854 |
| F_attn_rollout_vs_abs_attr | passage_only | q1 | 0.378 | 0.272 | 0.000 | 0.000 | 0.470 | 0.557 | 0.555 | 571 |
| F_attn_rollout_vs_abs_attr | passage_only | q5 | 0.462 | 0.301 | 0.000 | 0.000 | 0.511 | 0.642 | 0.608 | 571 |

