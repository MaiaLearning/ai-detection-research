# Amendments log

`RESEARCH_PLAN.md` was written on 2026-08-13, before any data was
downloaded or any code was written (see the pre-registration note at its
top). This log records every consequential decision made or changed after
that point, in chronological order, with the reasoning behind each — so
that reconstructing the study doesn't require inferring intent from script
diffs or commit messages. It exists because several of these decisions are
not otherwise recorded anywhere in the repository.

Implementation-level bugs caught and fixed during execution (a bootstrap
resampling bug in an early correlation CI, an OpenAI reasoning-token
truncation issue) are documented in their own scripts and result manifests,
not repeated here — this log is about changes to what was being tested or
how a gate was judged, not code fixes.

## 1. Gate 2 corrected from per-feature to composite-level testing

**What the plan said:** "Spearman correlation between each feature and
PERSUADE's holistic quality score" (Experiment 2) — a per-feature test.

**What was wrong with that:** two defects, identified during analysis of
the per-feature results. First, a per-feature test checks the raw sign of
corr(feature, quality), not the sign relative to that feature's actual
direction of "AI-likeness." `transition_phrase_rate` passed the per-feature
test (+0.147, positive) despite high transition-phrase rate itself being a
production-flagged AI-like signal — meaning on its real AI-suspicion axis
it fails the same way `sentence_length_std` does with the opposite raw
sign. Second, the panel emits one composite score, not nine independent
ones: individual features can lean in different directions and cancel, or
the composite can fail even when every individual feature passes.

**What changed:** Gate 2 was reapplied to the fitted composite's
out-of-fold P(AI) score directly (Experiment 3), instead of to individual
Tier 1 features. This is the number the study treats as the gate-2 verdict:
partial ρ = +0.135 (CI excludes 0), replicated at +0.134 with the OpenAI
arm added, and again (+0.249) on ELLIPSE's near-genre subset.

**Status:** the original per-feature results are still reported — they're
informative (they show, for instance, why raw TTR needed to be dropped in
favor of MTLD) — but they are not the basis for the gate-2 verdict.

## 2. Gate 2's "meaningful correlation" threshold was never numerically specified

Unlike Gate 1, which fixed AUC > 0.65 in the original plan, Gate 2's
language ("a meaningful negative correlation... is disqualifying") was
never given a number. During analysis, a threshold of |ρ| ≥ 0.1 (Cohen's
floor for a "small" effect) combined with a bootstrap CI excluding zero was
applied post hoc to individual features in `experiment2_quality_gate.py`.
This was identified as a threshold the analysis chose, not one the plan
set — particularly since at n = 24,695, CI-excludes-zero alone is close to
uninformative (a ρ of 0.02 is "significant" at that sample size; the
question was always magnitude). The composite-level gate-2 verdict (item 1
above) does not depend on this per-feature threshold. The 0.1 cutoff
remains in the per-feature code and its output, labeled for what it is
rather than removed.

## 3. ELLIPSE near-genre and ELL × genre compounding analyses added mid-study

Not in the original plan or either `EXPERIMENT_*.md` design doc. Added
after Experiment 4 (RAID) showed severe far-genre FPR degradation (36.9% vs
a 1% target), to bracket that number against a smaller "near genre" shift
(`scripts/analyze_ellipse_neargenre.py`) and to test whether the ELL
fairness gap and genre shift compound multiplicatively or are independent
(`scripts/analyze_ell_genre_compounding.py`). Method detail is in each
script's own docstring. No pre-registered prediction was recorded for
either analysis before running it — flagged here explicitly, since every
other gate/experiment in this study did record one in advance.

Headline results: FPR degrades from 1% (PERSUADE) to 2.4-3.7% on ELLIPSE
(near genre) to 36.9% on RAID abstracts (far genre) — a bracket, not a
cliff. The compounding test found no evidence of compounding: within
PERSUADE's own genre, held-out prompts do not raise FPR for either ELL or
non-ELL writers, meaning ELLIPSE's higher ELL rate is better explained as a
corpus-identity effect (a different writing instrument/population) than as
ELL status and topic novelty multiplying together.

## 4. Experiment 5 shelved: the L2 condition was found not to exist

`EXPERIMENT_5.md` (2026-08-14) specified four prompting-effort conditions —
L0 (naive), L1 (verbatim task), L2 ("DAIGT-replicated": persona + word-count
target + student-voice scaffolding), L3 (adversarial paraphrase) — designed
to test whether Claude's high detectability in Experiment 3 was a vendor
property or a prompting-effort artifact (DAIGT's GPT-family prompts were
assumed to have been more deliberately constructed than this project's own
generic Claude prompt).

Before generating anything, the plan's own sourcing constraint ("do not
compose the prompts yourself... if a level cannot be sourced, log that and
stop") was applied to L2. Tracing DAIGT-v2's actual documented generation
methodology across five of its source datasets — darraghdog's Claude set
(via its linked discussion thread and raw CSV), `chat_gpt_moth`, `radek1`,
nbroad's Llama/Falcon set, and kingki19's PaLM set (including its linked
Colab notebook with the actual prompt-construction code) — found that no
contributor used anything beyond an L1-equivalent prompt: the raw or
lightly-wrapped assignment text, with no persona, word-count target, or
student-voice instruction anywhere. L2 as specified does not exist in the
source data.

This is itself a finding, not just a blocker: the premise that DAIGT
contributors "deliberately constructed student-like text" while this
project's own prompting was generic does not hold up — which weakens the
vendor-vs-effort confound the experiment existed to untangle in the first
place.

**Decision:** collapse to three levels (L0, L1, L3), sourcing L3 from a
peer-reviewed paper (Lu et al. 2024, *TMLR*, "Large Language Models can be
Guided to Evade AI-Generated Text Detection," Table 11) rather than
inventing the vary-sentence-length/add-anecdote/avoid-transitions recipe
originally guessed at.

**Later decision (see `EXPERIMENT_6.md`'s closing note):** shelve the
experiment entirely rather than complete the reduced 3-level design. With
Experiment 6 showing the underlying scoring-panel question already closed
(gate 2 fails at the composite level regardless of vendor), the
vendor-vs-effort question no longer bore on any product decision, and
completing the reduced design was judged not worth its cost. A pilot run
and one full-scale generation run were executed and are recorded in
`results/experiment5_*`, but the design was not carried to a reported
conclusion.

## 5. Experiment 4 (RAID): initially not planned, then run; its headline finding was not the one it was run to measure

Experiment 4's original framing (`RESEARCH_PLAN.md`) was paraphrasing
robustness — "does it survive paraphrasing?" It was initially set aside
while Experiment 5's live-generation design was being pursued for testing
robustness against one specific, sourced paraphrase prompt on Claude
specifically — until it became clear that Experiment 5, even completed,
would have no equivalent for testing robustness *across* attack types and
generating models, only one prompt on one vendor. Experiment 4 was then run
using RAID's pre-existing adversarial-attack releases
(`scripts/experiment4_raid_robustness.py`), at effectively zero marginal
cost since RAID ships the attacked text already generated — no live model
calls were needed.

Its headline result was not the one it was designed to measure. The
paraphrase attack did drop TPR as expected (~21 points, on the one
model/domain tested). But the larger and more consequential number was a
domain-shift finding that fell out of the same run: human-written RAID
abstracts, scored against the composite's PERSUADE-calibrated threshold,
came back at 36.9% FPR against a 1% target. That number motivated the
ELLIPSE near-genre follow-up (item 3 above) more directly than the
paraphrasing result did.

## 6. Experiment 7 (dispersion mechanism test) added post hoc

Not in the original plan or pre-registration. Added after Experiment 3
produced an unexpected capability gradient — frontier proprietary models
(Claude Sonnet 5, GPT-5.6 Terra) were the *most* detectable of 17 sources,
with open-weight and older models less detectable — to test a mechanism
hypothesis raised during analysis of that result, rather than to inform any
product decision. Gate 2 had already failed at the composite level
(item 1) and TPR@1%FPR (41.3%) was already known before this experiment
ran; nothing here reopens that verdict. The writeup and this entry present
it as exploratory, per `EXPERIMENT_7.md`'s own closing instruction.

**What it tested:** whether the capability gradient is better explained by
H1 (pretraining register — detectability tracks each essay's own polish,
regardless of source) or H2 (post-training homogenization — detectability
tracks how tightly a source's outputs cluster around their own centroid,
relative to human writing). Predictions were pre-registered in
`results/experiment7_manifest.json` before computing anything: an ordering
of across-document dispersion (frontier proprietary < older proprietary <
open-weight < human) and a primary correlation (dispersion vs TPR@1%FPR:
negative, |ρ| > 0.5).

**Results, reported plainly because they are mixed:**

- The pre-registered **ordering held exactly**: frontier proprietary
  (mean dispersion 1.845) < older proprietary (2.060) < open-weight
  (2.421) < human (2.422).
- The pre-registered **primary correlation did not clear its own bar**:
  dispersion vs TPR, ρ = −0.321 (bootstrap CI −0.738 to 0.253). Right
  direction, but the CI crosses zero and the magnitude falls short of the
  |ρ| > 0.5 threshold set in advance. At n = 17 sources this is not strong
  evidence either way.
- **Centroid distance from the human mean was the stronger predictor**:
  ρ = 0.775 (CI 0.348 to 0.958) against TPR — sources farther from the
  human centroid are more detectable, more confidently than the dispersion
  result above. A joint regression of TPR on both terms gets R² = 0.815
  (coef_dispersion = −0.107, coef_centroid_distance = +0.130).
- The **H1-specific direction check failed to support H1**: projecting
  each source's centroid onto the human-mean-to-high-quality-quartile
  vector correlated with TPR at ρ = −0.174 (CI −0.614 to 0.288) — weak and
  wrong-signed relative to what H1 predicts (frontier models displaced
  *toward* the high-quality region).
- A secondary, independent route to Experiment 2's finding: human top
  quality quartile dispersion (2.166) is lower than bottom quartile
  (2.772), as predicted — essays rated as stronger writing are also more
  internally regular. ELL essays are more dispersed than non-ELL (2.843 vs
  2.370).
- A stricter robustness check, holding the prompt set to the only 2 of 15
  PERSUADE prompts common to all 18 sources (several sources drop to
  n = 12-18 there), gives a materially different correlation (ρ = 0.199) —
  reported in `results/experiment7_strict_common_prompt_check.csv` as an
  exploratory, small-n check, not as evidence against the primary result.

**Verdict:** the data leans mildly toward H2 over H1 — the ordering
prediction held, and centroid distance (a homogenization-flavored measure)
predicts TPR more strongly than dispersion does — but this experiment does
not confirm H2 outright, since its own pre-registered dispersion-TPR test
came in short of the threshold set for it, and n = 17 is too small to treat
any of these correlations as decisive. The direction check argues against
H1 as stated. Neither hypothesis should be reported as confirmed; this is
suggestive, correlational evidence about model outputs, not a demonstration
about what any lab actually trained on.

## On this repository's single commit

This repository was pushed as a single commit imported from the working
directory where the study was conducted, not an incremental commit
history — so git history alone cannot independently corroborate the
pre-registration and amendment timeline described above. This is disclosed
rather than concealed: the original plan and the two later design docs
carried filesystem modification timestamps on the machine where the work
was done consistent with the sequence above (`RESEARCH_PLAN.md`/`CLAUDE.md`
on 2026-08-13; `EXPERIMENT_5.md`/`EXPERIMENT_6.md` on 2026-08-14), but git
and GitHub do not preserve filesystem timestamps through a commit, so this
is not independently checkable by a reader from the repository alone — it
is the authors' own record, not cryptographic proof. Treat this document
as self-reported process documentation, not git-verified fact.
