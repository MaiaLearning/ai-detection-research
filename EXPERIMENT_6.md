# Experiment 6: Do the Tier-1 features carry essay quality above word count?

Read `CLAUDE.md` and `RESEARCH_PLAN.md` first. This is the last experiment in the
study. Experiment 5 is shelved (see note at the end).

## Why this exists

The AI detection panel is dead — gate 2 failed on the composite three times, and
TPR@1%FPR peaked at 41.3%, meaning the modal AI essay scored clean.

But every composite fitted so far was fitted to detect **AI**, with quality
checked only as a safety gate. Nobody has fitted a composite to predict
**quality**. PERSUADE ships holistic scores. The data is already on disk.

If the features carry quality, they can condition the essay review feedback. If
they do not, this direction closes and only the verification use (below) remains.

## The only number that matters

Raw rho(word_count, holistic_score) = 0.76. **Length already explains over half
the variance in essay quality.** Any quality model built on these features will
look impressive and be a length proxy unless the test is nested.

So the deliverable is **incremental predictive power over word count alone**, not
R² of the full model.

Fit three nested models, same folds, same seed:

- **M0** — word count only
- **M1** — word count + the 9 Tier-1 features
- **M2** — the 9 features only, no word count (diagnostic: how much of the
  features' apparent signal *is* length?)

Report out-of-fold R² and Spearman rho for each, with bootstrap CIs, plus the
M1 − M0 delta with a CI on the **delta itself**, not on the two endpoints
separately. A delta CI that includes zero means the features add nothing.

Pre-register this expectation before running: the best single feature
correlations with quality were +0.147 and −0.145, roughly 2% of variance each.
A fitted composite should beat any single feature, but the honest prior is
incremental rho of about 0.2–0.3 over length. If the delta comes back far above
that, suspect leakage before celebrating.

## Method

- **Data.** PERSUADE human essays only. Same cleaned sample and filters as
  experiment 2 (24,695 essays). No AI text anywhere in this experiment.
- **Target.** PERSUADE holistic score.
- **Features.** The same 9 Tier-1 features. Use MTLD, not raw TTR — experiment 2
  showed raw TTR carries residual length confound that rank-partial correlation
  does not fully scrub.
- **Model.** Start with ridge regression — interpretable coefficients matter more
  than squeezing out performance, because the coefficient signs are what would
  condition the feedback prompt. Report gradient boosting as a secondary check
  only to bound how much non-linearity is being left behind. If GBM massively
  outperforms ridge, say so; do not silently switch.
- **Validation.** The dev/held-out split established earlier. Fit and select on
  dev, report held-out figures. 5-fold CV within dev for model selection. All
  reported scores out-of-fold.
- **Subgroups.** Report the M1 − M0 delta separately for ELL and non-ELL, and by
  grade band. A quality model that works only for native speakers is not usable
  in this product, and the ELL flag is already in the corpus.

## Report the coefficient signs explicitly

This is not a formality — it is the operationally load-bearing output.

Experiment 2 found that **weaker essays have more erratic sentence lengths;
stronger essays are more consistent.** That inverts the conventional advice to
"vary your sentence lengths," which is what MaiaLearning's own retired
medium-tier copy told students.

For every feature that lands in the model, state the direction in plain language:
which way it moves with quality, and therefore what advice it implies. That table
is the thing that would go into a feedback prompt. If it is wrong or vague, the
model will fall back on folk theories about writing and give students advice that
makes their essays worse — with our own measurements cited as justification.

## Do not build percentiles

PERSUADE is grades 6–12 argumentative essays with source material, averaging ~418
words. Production input is 300–650 word college personal statements. A percentile
computed against PERSUADE is not a percentile for an admissions essay.

No percentile, band, or score derived from this corpus goes in front of a student.
That is the same unvalidated-number failure the study just spent itself retiring.
The output here is for internal conditioning only.

## Deliverables

1. `results/experiment6_nested.csv` — M0/M1/M2 metrics, held-out, with CIs.
2. The M1 − M0 delta with its own CI, overall and by subgroup.
3. A plain-language table of feature direction versus quality.
4. A recommendation: do the features carry quality above length, yes or no, with
   the number that decides it.

## Second deliverable, independent of the result above

Regardless of how the nested test comes out, produce a **feedback verification
harness**: given an essay and generated review feedback, check the feedback's
factual claims about the writing against the measured features.

Example: feedback asserts "your sentences are monotonous" but sentence-length SD
sits at the 70th percentile of the corpus — flag the claim as unsupported.

This addresses the failure mode that started the whole investigation — confident
assertions with nothing behind them — and it does not depend on the features
predicting quality. It only requires that they measure what they claim to measure,
which is established.

Start with the claims that map cleanly onto a measured feature: sentence variety,
transition density, lexical repetition, paragraph consistency. Claims about
specificity or personal detail are not measurable this way; leave them alone.

## Interpretation

- **Delta clears ~0.2 incremental rho, holds across ELL and non-ELL** → features
  can condition the review prompt. Next step is a design for passing them as
  internal context, never as displayed numbers.
- **Delta is small or subgroup-unstable** → close this direction. Ship the
  verification harness alone; it stands on its own merits.

Either outcome is a result. Report it plainly. The study has already produced
three real findings — do not stretch this one to make a fourth.

## Note on experiment 5

Shelved, not cancelled. L2 was found not to exist (DAIGT contributors fed the raw
PERSUADE assignment straight through, with no persona or style scaffolding), which
dissolved the vendor-versus-effort confound the experiment was designed to
untangle. With the shipping decision closed, it no longer bears on any product
decision. Revisit only if the findings are written up for external publication.
