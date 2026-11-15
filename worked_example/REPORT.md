# Worked example: when can Qwen2.5-14B explain GPT-4o?

This responds to the weak-reject review: "never shows an open model explaining
a closed one ... reframe as an API-compatible screen, and add a worked case
study on the best-matching pair". Qwen2.5-14B is GPT-4o's best open surrogate
on BoolQ (F_pred 0.80, F_attr 0.53). Everything here re-analyses existing
results, with no new model or API calls. The GPT-4o noise ceiling also uses
the existing noise-ceiling repeat runs (see "Open decisions").

## Headline

**Cheap checks identify the prompts where Qwen's sentence attributions match
GPT-4o's nearly as well as GPT-4o matches itself, and clearly better than a
no-model word-overlap baseline.**

- **Most of the signal comes from Qwen alone.** When Qwen's top passage
  sentence strongly supports its own answer, that sentence is usually GPT-4o's
  top sentence too.
- **One GPT-4o call sharpens it slightly.** That call gives GPT-4o's log-odds,
  so you can check whether the two models' answers and confidence agree.
- **The right comparator is word overlap, not chance.** Picking the passage
  sentence that shares the most content words with the question already finds
  GPT-4o's top sentence on 0.70 of the screened prompts. Qwen reaches 0.84.
  Qwen's advantage is larger on graded attributions, which overlap can't
  supply: within-prompt r² against GPT-4o's |Δℓ| is 0.65 for Qwen and 0.30 for
  overlap.

The screen has two checks:

1. **Check 1 (one GPT-4o call):** |ℓ_Qwen − ℓ_GPT-4o| < 10 nats.
2. **Check 2 (Qwen only):** Qwen's top passage sentence supports its own
   answer by ≥ 13 nats. This needs Qwen's leave-one-out passes, which run
   locally on the open model.

Both thresholds were fit on even-indexed BoolQ prompts. Results on the 1,419
held-out odd prompts follow.

**Graded agreement** (the stronger evidence):

| Held-out prompts | Share | Passage r², Qwen vs GPT-4o [95% CI] | Passage r², GPT-4o vs its re-runs | \|Δℓ\| r² within prompt: Qwen | \|Δℓ\| r² within prompt: word overlap |
|---|---|---|---|---|---|
| **Both checks pass** | 27% | **0.62** [0.59, 0.66] | 0.80 | **0.65** | 0.30 |
| Check 2 only, matched 27% coverage (no GPT-4o call) | 27% | 0.59 [0.54, 0.63] | 0.81 | 0.62 | 0.29 |
| Check 2 only, its own cut (11.5 nats) | 35% | 0.56 [0.52, 0.59] | 0.79 | 0.59 | 0.27 |
| Check 1 passes, check 2 fails | 43% | 0.21 [0.17, 0.24] | 0.50 | 0.19 | 0.07 |
| Check 1 fails | 30% | 0.18 [0.13, 0.23] | 0.61 | 0.18 | 0.08 |
| Everything not passing both | 73% | 0.18 [0.15, 0.21] | 0.55 | 0.17 | 0.07 |
| No screen | 100% | 0.40 [0.37, 0.43] | 0.67 | 0.44 | 0.14 |

**Top sentence:**

| Held-out prompts | Same top-\|Δℓ\| passage sentence: Qwen | Word overlap (no model) | GPT-4o vs its re-runs | Same decisive sentence (signed) | Chance |
|---|---|---|---|---|---|
| **Both checks pass** | **0.84** | 0.70 | 0.89 | **0.85** | 0.29 |
| Check 2 only, matched coverage | 0.82 | 0.68 | 0.87 | 0.84 | 0.30 |
| Check 2 only, its own cut | 0.78 | 0.66 | 0.86 | 0.80 | 0.29 |
| Check 1 passes, check 2 fails | 0.51 | 0.41 | 0.61 | 0.50 | 0.26 |
| Check 1 fails | 0.46 | 0.41 | 0.63 | 0.41 | 0.28 |
| Everything not passing both | 0.49 | 0.41 | 0.62 | 0.46 | 0.27 |
| No screen | 0.58 | 0.49 | 0.69 | 0.57 | 0.28 |

Definitions:
- "Word overlap" picks the passage sentence sharing the most content words
  with the question (stopwords removed, plurals stripped). It uses no model.
  When k sentences tie for the most overlap, a match counts 1/k, the expected
  value of a random tie-break. The same baseline is now in reanalysis 7.
- "\|Δℓ\| r² within prompt" centres each prompt's passage sentences before
  pooling. For the overlap column, the predictor is the overlap count. All
  top-1 numbers are over passage sentences only; system and question segments
  are excluded.

What the tables show:

- **Where both checks pass,** Qwen reaches 78% of GPT-4o's agreement with
  itself in passage r² (0.62 vs 0.80), and 95% of it on the top sentence
  (0.84 vs 0.89).
- **On the other 73%,** Qwen reaches only 32% (0.18 vs 0.55). So the screen
  isn't just finding prompts where GPT-4o happens to be stable.
- **Against word overlap,** Qwen gains 0.14 on top-1 where both checks pass
  (0.84 vs 0.70). On those 379 prompts, Qwen is right and overlap wrong in
  about 82 (counting ties fractionally), and the reverse in about 27. On graded agreement Qwen's r² is about twice
  overlap's.
- **Check 1 adds little on top of check 2:** about +0.04 r² and +0.02 top-1
  at matched coverage.
- **Check 1 on its own still sorts prompts:** r² 0.48 for the 70% that pass,
  against 0.18 for those that fail.

Figure: `figures/fig_trust_gate.pdf`.
- (a) Passage r² by tenths of the gap. It stays near 0.5 up to about 8 nats,
  then falls to 0.13. The decline starts inside the accepted range (8–10
  nats).
- (b) The screen: Qwen vs GPT-4o top-sentence match, the word-overlap
  baseline, GPT-4o vs its re-runs, and chance.
- (c) Poison ivy, prompt 3228. Both checks pass and both models hinge on the
  topical-ointment sentence. Word overlap picks the injection sentence, which
  shares more words with the question and which Qwen reads as evidence
  against (−18). This shows what Qwen adds over overlap; 22% of passing
  prompts are like this.
- (d) Pork ribs, prompt 582. It passes check 1 (gap 4.7 nats) but fails
  check 2 (Qwen's top support is 4.5 nats). GPT-4o's most influential passage
  sentence is the synonym sentence, "Baby back ribs (also back ribs or loin
  ribs)", at Δℓ +12.5. Qwen pushes the same sentence toward False (−8.75), and
  gets its True from the question instead. Such opposite readings are rare:
  2.9% of prompts have a passage sentence with |Δℓ| ≥ 5 for both models and
  opposite signs.

A typical passing prompt, if a plainer example is wanted: prompt 3257 ("is
jaguar and land rover owned by tata"). "…purchased Jaguar Land Rover from Ford
in 2008" scores Qwen +24.5 and GPT-4o +12.8, and every other passage sentence
is ≤ 2.3 for both. Word overlap gets this one too.

**Draft rebuttal paragraph.**

> *Worked example: when can an open model explain GPT-4o?* For the
> best-matching pair, Qwen2.5-14B → GPT-4o on BoolQ, we turned our fidelity
> metrics into a per-prompt screen. GPT-4o's own leave-one-out attribution
> needs a median of 9 API calls (5 for passage sentences only). The screen
> uses Qwen's own attributions and at most one GPT-4o call. Trust Qwen's
> sentence attribution when (1) Qwen's most influential passage sentence
> supports its answer by at least 13 nats, and (2) the two models' log-odds
> differ by less than 10 nats. We fit both thresholds on even-indexed
> prompts. On held-out prompts, 27% pass. For those, passage attribution r²
> between Qwen and GPT-4o is 0.62, against 0.80 for GPT-4o vs its own re-runs;
> on the remaining 73% it is 0.18, against 0.55. Qwen names GPT-4o's most
> influential passage sentence 84% of the time, against 89% for GPT-4o vs
> itself and 70% for a no-model baseline that picks the sentence sharing the
> most words with the question. On graded attributions the baseline falls
> further behind (within-prompt r² 0.30 vs Qwen's 0.65). Qwen's own check
> does most of the work (r² 0.59 at the same coverage with no GPT-4o call).
> Passing prompts are mostly ones where a single passage sentence settles the
> question, which in BoolQ mostly means "True" answers. Surrogate fidelity
> thus becomes a measured, per-prompt property, not an assumption.

**Claim limits.**
- The screen certifies agreement in input saliency, not in mechanism.
- It selects prompts with one clearly decisive passage sentence, which are
  the easier prompts. Much of the top-1 agreement there is also reachable by
  word overlap; Qwen's added value is +0.14 top-1 and roughly double the
  graded r². The overlap baseline is crude, and a dense retriever or
  extractive-QA baseline might close more of the top-1 gap. It cannot
  produce signed, graded attributions.
- The screen mostly passes prompts Qwen answers True: 45% of those pass,
  against 7% of False answers, and 88% of passing prompts are True. It still
  sorts prompts within each answer (top-1 0.87 vs 0.54 for True, 0.64 vs 0.46
  for False; r² 0.62 vs 0.23 and 0.36 vs 0.11). The likely reason is that True
  answers are usually stated in one sentence, while False answers rest on
  what is missing or on the question itself.
- Only the thresholds are held out. The form of the rule (gap, then Qwen
  support) was chosen after exploring the full data.
- The thresholds are loosely identified: reversing the even/odd roles gives
  t = 8 and c = 9, with the same qualitative picture.
- Check 1 also works with a fresh GPT-4o call instead of the published one:
  r² 0.61–0.62 and top-1 0.83–0.84 for prompts passing both checks.
- On top-1 alone, check 1's gain is modest (0.64 vs 0.58), and part of it
  tracks GPT-4o's own instability. Lead with r² and the two-check result.
- The re-run columns drop 520 of 13,156 passage coordinates that are
  non-finite in some repeat run.
- In seven scored prompts (385, 718, 1668, 2620, 2961, 3141, 3154), GPT-4o's
  question-deleted call has no finite log-odds, so the question segment is
  missing from the shared table. None passes check 2, so they only enter the
  "not passing" rows.

## Transfer: other targets and a second surrogate

The screen above was explored on one pair. Two follow-ups test whether it is
specific to that pair. Both are reanalyses with no model or API calls
(`benchmark_scripts/screen_transfer.py`, outputs in `transfer/`).

**Setup.** The rule is check 2 alone: the surrogate's top passage support is
at least c. For each surrogate, c is the only fitted number, fit on even
prompts against GPT-4o with the same χ² criterion. All rows are odd prompts;
every other target uses the frozen cut.

- Qwen2.5-14B: c = 11.5 nats (the Qwen-only cut above), passing 35%.
- Llama-3.1-8B: c = 4.5 nats, passing 29%. The reversed split also gives 4.5.
  The cuts are not comparable in nats: Llama's log-odds are about three times
  smaller (median |ℓ| 6.2 against 20.0).

**Result.** Passing against failing prompts, per target:

| Surrogate | Target | Passage r² | Top-1 | Word overlap (pass) |
|---|---|---|---|---|
| Qwen2.5-14B | GPT-4o (fitted) | 0.56 / 0.15 | 0.78 / 0.47 | 0.65 |
| | GPT-4.1 | 0.42 / 0.11 | 0.75 / 0.41 | 0.58 |
| | Gemini 2.5 Flash Lite | 0.60 / 0.15 | 0.81 / 0.47 | 0.63 |
| | Llama-3.1-70B | 0.61 / 0.22 | 0.84 / 0.54 | 0.64 |
| | Llama-3.3-70B | 0.64 / 0.23 | 0.84 / 0.53 | 0.64 |
| | Llama-4 Maverick | 0.59 / 0.17 | 0.79 / 0.50 | 0.62 |
| Llama-3.1-8B | GPT-4o (fitted) | 0.48 / 0.12 | 0.82 / 0.43 | 0.64 |
| | GPT-4.1 | 0.39 / 0.08 | 0.71 / 0.42 | 0.60 |
| | Gemini 2.5 Flash Lite | 0.58 / 0.13 | 0.85 / 0.43 | 0.67 |
| | Llama-3.1-70B | 0.58 / 0.16 | 0.83 / 0.50 | 0.65 |
| | Llama-3.3-70B | 0.60 / 0.16 | 0.86 / 0.48 | 0.66 |
| | Llama-4 Maverick | 0.57 / 0.15 | 0.82 / 0.47 | 0.65 |

Top-1 chance is 0.26–0.29. The Qwen → GPT-4o row equals `gate_rule.csv`
(`qwen_only_supported`); `build` stops if it does not.

Figure: `figures/fig_screen.pdf`, the surrogate-only counterpart of
`fig_trust_gate`.
- (a) Passage r² against GPT-4o by decile of Qwen's own top passage support
  (`transfer/support_bins.csv`). It rises from 0.06 in the lowest decile to
  0.68 in the highest; top-1 match rises from 0.37 to 0.92.
- (b) Top-sentence match on passing and failing prompts for each hosted
  target, with word overlap and chance.
- (c) Prompt 3228, which passes. (d) Prompt 525, a failing prompt chosen by
  rule: the held-out prompt with the median Qwen support among failing
  prompts that both models answer correctly, with three to five passage
  sentences, differing top sentences, and neither top sentence first in the
  passage. Both answers are carried by the question.

**What it shows.**

- The prompts a surrogate flags on its own are the ones where it matches
  every hosted target, not only the one the cut was fit on.
- The rule form carries to a second surrogate from another family with one
  re-fit number. The form was never tuned on Llama-3.1-8B.
- On flagged prompts the surrogate is about as good a stand-in as a second
  hosted model. Hosted–hosted pairs on Qwen's flagged prompts have median
  top-1 0.76 and r² 0.58, against 0.48 and 0.26 on the rest
  (`hosted_pairs.csv`).

**Claim limits.**

- The screen detects something partly about the prompt and partly about the
  surrogate. Hosted–hosted pairs split on the surrogate's flags (above), but
  the two surrogates' flags only partly coincide: Jaccard 0.45 on held-out
  prompts (20% flagged by both, 15% by Qwen only, 9% by Llama only). Where
  both flag, they name the same top sentence 93% of the time and each matches
  the targets at 0.78–0.92 (`flag_overlap.csv`).
- It is a screen for True answers, for both surrogates and every target. When
  the surrogate answers True, top-1 is 0.74–0.90 on passing prompts against
  0.33–0.57 on failing ones. When it answers False (12–18% of passing
  prompts) it is 0.49–0.69 against 0.42–0.53, and Qwen → Gemini shows almost
  no effect.
- No family advantage: on the three Llama targets Llama-3.1-8B reaches
  0.82–0.86 top-1 on its flagged prompts and Qwen2.5-14B 0.79–0.84 on its
  own. Without a screen Qwen2.5-14B is the better surrogate for them.
- The cut must be re-fit per surrogate. Qwen's 11.5 nats passes only 7–8% of
  Llama's prompts.
- No per-target noise ceiling is reported here: it needs each target's
  repeat runs, which are outside the repo. GPT-4.1 is scored on about 80% of
  prompts because of missing labels.
- The rule's form still comes from exploring Qwen → GPT-4o.

## Competing explanations

After the first draft, two more readers went through the prompts. A skeptic
looked for trivial explanations, and a second reader looked for a qualitative
pattern. Both read packets of even prompts and tested what they found on odd
prompts. Their files are in `scratch/qualitative/`. I re-ran the skeptic's
script and recomputed the overlap baseline, the True/False split and the
opposite-direction rate with my own code.

**Ruled out:**
- **Position, including the `Context:` header sentence.** The first passage
  sentence matches GPT-4o's top on 0.48 of passing prompts. Where GPT-4o's
  top isn't the first sentence, Qwen still matches 0.82 (chance 0.25).
- **Sentence length.** The longest sentence gets 0.40 on passing prompts.
- **Fewer sentences per passage.** The effect holds in every passage-length
  stratum, and passing prompts have a similar mix (4.4 vs 4.6 sentences).
- **Repeated passages.** De-duplicated by passage, top-1 is 0.84.
- **A trivial screen replacing check 2.** Screens built from overlap or
  length at 27% coverage reach 0.66–0.71 mostly by selecting short passages.
  Within passage-length strata they top out near 0.65.

**Partly explains it:** question-word overlap, as above. A conditional logit
on position, length and overlap fit on even prompts does no better than the
best single overlap rule (0.68 on passing prompts). When the two pick
different sentences on passing prompts, Qwen is right 70% of the time and the
baseline 22%. Qwen wins on answer sentences that don't reuse the question's
words: "did smeagol die" is answered by "fell into the fires of the volcano".

**The qualitative pattern:**
- **Agreement means a settling sentence.** In a blind hand-labelled sample of
  50 odd prompts, Qwen's top sentence matched GPT-4o's in 31 of 42 cases
  where it states, defines, qualifies or implies the answer. It matched in 1
  of 8 where it is an introduction or filler sentence. This has n = 50 and
  one labeller, and it is partly circular with check 2.
- **Half of all top-1 mismatches are prompts where Qwen answers from the
  question.** Qwen's best passage sentence is under 5 nats while GPT-4o
  spreads small same-signed credit over on-topic sentences. The two still
  give the same answer about 94% of the time.
- **Mismatches among passing prompts are mostly near-ties.** GPT-4o's pick is
  usually Qwen's runner-up, with the same sign.
- **Opposite readings are rare and have no lexical signature.** Negation and
  alias words are no more common in opposite-sign sentences than in other
  large ones. Definition and alias sentences usually match (9 of 11 in the
  labelled sample), so the pork-ribs case is not typical.

## How the candidates scored

Five agents each took one angle. An independent checker re-ran every script,
recomputed each candidate's key number with its own code, and read every
showcased prompt.
- **Numbers:** every quoted number reproduced.
- **Substantive problems:** two code artefacts in D, a misdefined rate and a
  dependency on data outside the repo in C, an asymmetric stability filter
  driving A's negative conclusion, and an undisclosed noise-model assumption
  in E.
- **Overclaims:** in every candidate.

Rigor below is the checker's score. Interestingness and relevance to the
review are mine.

| Angle | Interest | Rigor | Relevance | Verdict |
|---|---|---|---|---|
| B. Trust gate (one-call screen) | 4 | 4 | 5 | **Lead,** combined with C's Qwen-support check into the two-check screen. |
| C. Right answer, different reasons | 4 | 4 | 4 | **Second.** Source of check 2 and of the ribs panel. |
| D. Inside Qwen (internals) | 3 | 3 | 4 | One sentence, optionally the panthers vignette. |
| A. Shared failure | 3 | 3 | 3 | Optional sidebar on label errors. |
| E. Word level (high risk) | 3 | 4 | 3 | One sentence: question words transfer, passage words don't. |

### Supporting vignettes (use at most one or two)

- **C. Same answer, different evidence** (prompt 582, as in panel d). This
  is a rare case: opposite readings occur in 2.9% of prompts.
  - GPT-4o's most influential passage sentence is the synonym sentence:
    +12.5, averaging +17.5 over six runs. Qwen pushes it toward False
    (−8.75). Deleting the sentence still leaves GPT-4o at True (+4.7), because
    the question carries +16.5. So this is GPT-4o's most influential sentence,
    not the one its answer rests on.
  - The same pattern appears on prompts 1182 and 2999, which share this
    passage. Count all three as one example.
  - The synonym sentence also carries the `Context:` header.
  - Base rate: take the 2,088 prompts both models answer correctly in all six
    GPT-4o runs. In about 20% (23% against the published run alone), the two
    models' most influential passage sentences clearly differ. GPT-4o
    compared with itself gives 9% (published vs repeats).
  - In most of those prompts no single sentence is decisive, so the models
    differ in supporting evidence, not in what the answer hinges on.
- **A. Shared errors are often label errors** (prompt 1010: "does the world
  cup trophy have a name?", gold False).
  - Both models answer True, and both hinge on "called the 'FIFA World Cup
    Trophy'". Deleting that sentence flips both. The gold label is very
    likely wrong.
  - Shared errors where one sentence robustly drives GPT-4o's wrong answer:
    - In all 45, Qwen's decisive sentence is that sentence 73% of the time
      (chance 33%).
    - After also requiring the flip not to be fragile (a symmetric 1-nat
      stability filter), 22 remain, and the rate is 0.82 (chance 0.31).
  - On reading the agent's 17-prompt subset: 6 conflict with the gold label
    (3 clearly; 3 possibly because facts changed over time), and only 2 are
    clean model errors.
  - When Qwen is right and GPT-4o wrong, Qwen still points at GPT-4o's
    misleading sentence: 0.63 vs chance 0.37 (n = 19, p ≈ 0.01). The sample
    is small.
  - Nothing here is specific to errors: Qwen's hit rate is higher on prompts
    both get right (0.93). This uses gold labels and GPT-4o's ablations, not
    the screen.
  - A human should re-read the prompts before quoting.
- **D. Qwen's internals** (prompt 923: "are panthers and mountain lions the
  same thing?").
  - Qwen's Δℓ, its alignment, and ‖Δz‖ (how far deleting a sentence moves
    Qwen's final residual stream) all pick GPT-4o's only flip sentence
    ("...mountain lion, puma, panther..."). Qwen's most-attended sentence is
    the "Pantherinae" one, which GPT-4o ranks last. The attention margin is a
    near-tie.
  - Across prompts, top-1 match with GPT-4o (chance 0.28):

    | Qwen signal | Top-1 match |
    |---|---|
    | ‖Δz‖ | 0.60 |
    | Qwen's Δℓ | 0.58 |
    | Alignment | 0.48 |
    | Max attention (one pass) | 0.48 |
    | Mean attention | 0.43 |
    | First passage sentence | 0.40 |

  - ‖Δz‖'s edge comes from prompts whose top sentence is the first one, which
    carries the `Context:` header.
  - Internals point at GPT-4o's key sentence well above chance, about as well
    as Qwen's own Δℓ. The cosine normalisation in alignment costs about 10
    points.
- **E. Word level.**
  - For question words, Qwen predicts GPT-4o's word-deletion effects with
    r² 0.55, and when deleting a question word flips GPT-4o's answer, it also
    flips Qwen's in 40 of 78 cases.
  - For passage words, r² is 0.03, and Qwen flips along with only 7 of
    GPT-4o's 122 passage-word flips.
  - API noise is unlikely to explain this. All words in a prompt share one
    unablated GPT-4o call, which gives a word-level estimate of per-call
    noise from repo data alone. This assumes the ablated calls are as noisy
    as the shared baseline call. On that estimate, Qwen explains about 5%
    [3%, 8%] of GPT-4o's reproducible passage-word signal, against roughly
    60% for question words. The agent's own 10%, which carried
    sentence-level repeat noise over to words, is very sensitive to the
    assumed noise; don't quote it.
  - Discard the agent's showcase prompt 2284. In it, GPT-4o moves −7 when a
    single "the" is deleted, which is most likely API noise.
  - The word run is a sparse sample, about 3 words per prompt.

## Open decisions for you

1. **The GPT-4o repeats table** (`gpt4o_repeats.tsv.gz`, 1.8 MB, untracked).
   - It holds GPT-4o's rows from the five noise-ceiling repeat runs, the only
     data behind the "GPT-4o vs its re-runs" columns and the noise-corrected
     numbers in C and E.
   - Committing it makes those numbers reproducible from the public repo.
     Without it, `build` writes NaN in those columns and the figure omits
     the diamonds. Everything else reproduces from `results/`.
2. **Agent working files.**
   - `candidates/`, `scratch/` and `verification/` contain local paths: the
     venv, the checkout, and the repeats directory.
   - I would commit only `REPORT.md`, `PLAN.md`, `AGENT_BRIEF.md`, the
     figure, the derived tables and the two scripts, and keep the rest local
     unless it's scrubbed.
   - `AGENT_BRIEF.md` itself names the local venv and needs one line edited.
3. **The candidates are uncorrected.** Each `verification/<angle>.md` lists
   the required corrections. They are applied in this report, not in the
   candidates.
4. **The word-level scratch tables** under `results/` (`boolq_word_*`) are
   untracked and rebuildable (README step 3). They can be deleted.

## Reproduce

```bash
python -m benchmark_scripts.consolidate_results --benchmark boolq --pregrouper word   # word level only
python -m benchmark_scripts.compute_logodds --benchmark boolq --pregrouper word       # word level only
# optional, GPT-4o noise ceiling:
python -m benchmark_scripts.worked_example extract-repeats --reps-dir <noise-ceiling reps>
python -m benchmark_scripts.worked_example build
python -m benchmark_scripts.plot_worked_example
python -m benchmark_scripts.worked_example inspect 3228 582 2403 1010 923
python -m benchmark_scripts.screen_transfer   # transfer/ (other targets, Llama-3.1-8B)
python -m benchmark_scripts.plot_screen_transfer   # figures/fig_screen (surrogate-only screen)
```

Screen and figure numbers: `gate_rule.csv` and `gate_bins.csv`. Strata:
`strata.csv` and `gate.csv`. Each agent's numbers: `scratch/<angle>/`.
Checks: `verification/<angle>.md` and `verification/report.md`. Competing
explanations: `scratch/qualitative/skeptic/findings.md` and
`scratch/qualitative/qualitative/findings.md`.
