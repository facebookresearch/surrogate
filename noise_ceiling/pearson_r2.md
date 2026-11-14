# Noise ceiling: Pearson r²

These are the Pearson r² values from `noise_ceiling.tsv` (rows with
`statistic == pearson_r2`). The setup is BoolQ/sentence, canonical contrast,
all prompts, pooled rows. r² is the square of the Pearson correlation between
two runs of the same model. For `F_pred` that correlation is over prompt-level
prediction log-odds; for every other signal it is over segment-level values.

Each cell is the mean over run pairs. A range (min–max) follows the mean only
when the pairs differ by 5e-5 or more. Coverage is the smallest joint
observation coverage among the repetition pairs, using the pair-specific
complete-case policy. The 95% prompt-cluster bootstrap intervals for each pair
are in `noise_ceiling.tsv` (`f_lo`, `f_hi`).

## `F_pred` and `F_attr`: repetition versus repetition

This compares the ten pairs drawn from the five repetitions. It isolates
run-to-run noise.

| Model | `F_pred` | `F_attr` | `F_attr` coverage |
|---|---:|---:|---:|
| Open models (all five) | 1.0000 | 1.0000 | 1.00 |
| Llama-3.1-70B | 0.9998 | 0.9976 (0.9975–0.9977) | 0.95 |
| Llama-3.3-70B | 0.9999 | 0.9990 | 0.95 |
| Llama-4 Maverick | 0.9988 (0.9983–0.9994) | 0.9920 (0.9900–0.9939) | 0.97 |
| GPT-4o | 0.9843 (0.9804–0.9879) | 0.9077 (0.8898–0.9232) | 0.91 |
| GPT-4.1 | 0.9919 (0.9914–0.9929) | 0.9177 (0.9133–0.9235) | 0.73 |
| Gemini 2.5 Flash Lite | 1.0000 | 1.0000 | 1.00 |

## `F_pred` and `F_attr`: repetition versus published run

This compares each of the five repetitions with the published artifact. For
hosted models, the value mixes run-to-run noise with endpoint drift over the
roughly two months between the two collections.

| Model | `F_pred` | `F_attr` |
|---|---:|---:|
| Open models (all five) | 1.0000 | 1.0000 |
| Llama-3.1-70B | 0.9998 | 0.9976 (0.9975–0.9976) |
| Llama-3.3-70B | 0.9999 | 0.9990 |
| Llama-4 Maverick | 0.9989 (0.9986–0.9994) | 0.9920 (0.9912–0.9936) |
| GPT-4o | 0.9653 (0.9634–0.9676) | 0.7940 (0.7910–0.7976) |
| GPT-4.1 | 0.9856 (0.9847–0.9864) | 0.8748 (0.8721–0.8782) |
| Gemini 2.5 Flash Lite | 0.9828 | 0.8551 |

Gemini's five repetitions are identical to one another but differ from the
published run, which points to a change in the served model (see `README.md`).

## Representation-level signals

These signals exist only for open models. The table compares repetitions with
each other. The comparisons with the published run give the same values.

| Signal | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-14B | Llama-3.1-8B |
|---|---:|---:|---:|---:|---:|
| `F_attn_mean` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `F_attn_max` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `F_attn_rollout` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `F_mag` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `F_align` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Every value equals one to machine precision.

## Same-model mechanism-to-attribution (reference)

The `*_to_attr` rows in `noise_ceiling.tsv` correlate a model's mechanistic
signal from one run with its own segment attribution from another run. Open
models reproduce bit for bit, so each value is identical across all twenty
ordered pairs, with no run-to-run variation. These are within-model
quantities. They are not noise ceilings, and they do not appear in
`results/f_table.tsv`, which reports only cross-model pairs.

| Signal | Qwen2.5-0.5B | Qwen2.5-3B | Qwen2.5-7B | Qwen2.5-14B | Llama-3.1-8B |
|---|---:|---:|---:|---:|---:|
| `F_attn_mean_to_attr` | 0.0296 | 0.0007 | 0.0012 | 0.0032 | 0.0003 |
| `F_attn_max_to_attr` | 0.0000 | 0.0008 | 0.0076 | 0.0002 | 0.0003 |
| `F_attn_rollout_to_attr` | 0.0266 | 0.0000 | 0.0030 | 0.0011 | 0.0183 |
| `F_mag_to_attr` | 0.0126 | 0.1457 | 0.3022 | 0.1815 | 0.1552 |
| `F_align_to_attr` | 0.7496 | 0.7945 | 0.7583 | 0.7606 | 0.7212 |
