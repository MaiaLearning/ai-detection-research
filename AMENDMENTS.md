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

**The same failure mode recurred in item 6.** Experiment 7's pre-registered
correlation threshold (dispersion vs. TPR: negative, |ρ| > 0.5) was set
without a power calculation at n = 17 sources — a threshold chosen because
it "sounded like a strong correlation," not because a power analysis said
it was the right bar for 17 data points to plausibly clear. At n = 17, a
bootstrap CI wide enough to include ρ = −0.5 comfortably is close to
guaranteed unless the true relationship is very strong and very clean; the
threshold made a genuine effect look like a failed prediction (dispersion
vs. TPR came in at ρ = −0.321, right direction, short of the bar) when a
properly powered threshold might have called it differently. This is
disclosed as the analyst's own design error, made the same way as the item
above: a number set post hoc by feel rather than by calculation, presented
as if it were a pre-registered bar with statistical teeth.

## 3. ELLIPSE near-genre and ELL × genre compounding analyses added mid-study

Not in the original plan or any `EXPERIMENT_*.md` design doc. Added
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
Experiment 3 having already shown the underlying scoring-panel question
closed (gate 2 fails at the composite level regardless of vendor — item 1
above), the vendor-vs-effort question no longer bore on any product
decision, and completing the reduced design was judged not worth its
cost. A pilot run and one full-scale generation run were executed and are
recorded in `results/experiment5_*`, but the design was not carried to a
reported conclusion.

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

- The pre-registered **ordering held in point-estimate order**: frontier
  proprietary (mean dispersion 1.845) < older proprietary (2.060) <
  open-weight (2.421) < human (2.422) — though the last gap (0.0008) is
  more than two orders of magnitude smaller than the other two and not
  meaningfully non-zero; 3-of-3 clean plus one negligible tie is the more
  accurate description than "held exactly" (see `TECHNICAL_REPORT.md`
  §5.10, whose wording this entry has been aligned to).
- The pre-registered **primary correlation did not clear its own bar**:
  dispersion vs TPR, ρ = −0.321 (bootstrap CI −0.738 to 0.253). Right
  direction, but the CI crosses zero and the magnitude falls short of the
  |ρ| > 0.5 threshold set in advance. At n = 17 sources this is not strong
  evidence either way.
- **Centroid distance from the human mean was the stronger predictor**:
  ρ = 0.775 (CI 0.348 to 0.957) against TPR — sources farther from the
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
- A follow-up, added after the above was first written up: rather than
  inferring the discriminant-vs-quality relationship from a 17-source
  correlation, compute it directly as one number. The frozen composite's
  own P(AI) discriminant coefficient vector and the human-mean-to-top-
  quality-quartile vector were compared by cosine similarity, both
  expressed in the composite's own `StandardScaler` space (the only space
  the discriminant vector is meaningful in) — **cosine = 0.067, angle =
  86.2°**. This is small and positive exactly as predicted, but "small and
  positive" here means *close to orthogonal*: for two random vectors in
  9 dimensions, the expected cosine similarity is 0 with a standard
  deviation of roughly 1/√9 ≈ 0.33, so 0.067 sits well inside the range
  produced by chance alone. It does not support reading the composite's
  AI-detection axis as pointed toward the same direction as human-rated
  quality. This is not a contradiction of Experiment 2's positive partial
  correlation (+0.135, partialling out length) between the composite's
  P(AI) score and quality — that correlation is real and reproducible —
  but it means the *coarse* quality direction used here (top quartile
  centroid minus overall centroid) is not the same direction the
  discriminant weights, and the two experiments' positive-correlation and
  near-orthogonal-vector results should not be described as the same
  finding measured two ways. (No CI is reported on the angle itself: it is
  a fixed geometric quantity between two vectors, not a sampled statistic;
  human n = 24,695 makes the quality-quartile centroid a stable estimate.)
- **Two conditioning checks on that cosine result** (`scripts/analyze_discriminant_conditioning.py`,
  `results/experiment7_conditioning_checks.json`), run because the 0.067/86.2°
  result above compares two vectors that were NOT conditioned on word count
  the same way (the discriminant was fit on raw features; Gate 2's +0.135
  explicitly removes word count), and because +0.135 is a Spearman
  (rank) statistic being compared, in spirit, against a linear-geometry
  one (cosine). Both results are logged here **alongside**, not in place
  of, the original 0.067/86.2° figure — the discrepancy between
  conditionings is itself part of the finding.
    - *Check 1 (consistent conditioning):* residualizing both vectors on
      log word count (regressions fit on human data only, applied
      uniformly to AI) in the same standardized space, then refitting the
      discriminant in that residualized space, moves the result from
      **cosine 0.067 (86.2°) to cosine 0.533 (57.8°)** — well outside the
      ~0.33 (1 SD) noise band for two random 9-D vectors. The original
      near-orthogonal reading does not survive consistent conditioning:
      **the original claim is retracted.** Once both vectors are freed of
      word-count confounding, the composite's discriminant direction and
      the human quality direction show a real, moderate alignment — not
      full alignment (57.8° is still closer to orthogonal than to
      parallel), but clearly more overlap than chance.
    - *Check 2 (rank vs. linear):* the Pearson partial correlation between
      the composite's OOF P(AI) score and quality, controlling word count,
      on the identical sample and scores Gate 2 used
      (`results/experiment3_human_scores.csv`), is **−0.027 (CI −0.046 to
      0.002)** — not merely smaller than Gate 2's Spearman partial of
      **+0.135 (CI 0.123 to 0.148)**, but on the opposite side of zero,
      with non-overlapping CIs.
    - **Correction to how Check 2 was first read here:** an earlier version
      of this entry called the Pearson/Spearman gap an "artifact" alongside
      Check 1's conditioning artifact. On review that framing was wrong.
      Both estimates are tight (n = 24,695) and disagree in sign — that
      combination is not what a broken measurement looks like; it is what
      a relationship concentrated in *part* of the quality range looks
      like, seen through a rank statistic that is sensitive to it and a
      linear one that averages it away. Spearman was the statistic
      specified in the original plan and is the one to keep. The sign
      reversal is diagnostic, not disqualifying, of Gate 2's finding.
    - **Where the effect is concentrated — first attempt was wrong, corrected
      below.** `scripts/analyze_quality_bin_profile.py` binned human essays
      by holistic score (1-6) and plotted the *arithmetic mean* of
      word-count-residualized P(AI) per bin. That plot showed score 1
      distinctly elevated (+0.063) and scores 2-6 flat-to-declining
      (score 6 numerically lowest at −0.012), which was reported here as
      "the effect is concentrated at the bottom of the range, the opposite
      of a penalty on the strongest essays." **That reported conclusion was
      wrong, and has been retracted** — not because Gate 2 has a bug, but
      because the mean-per-bin visualization was misleading. See the
      resolution below.
    - **Resolution** (`scripts/analyze_gate2_sign_resolution.py`,
      `results/experiment7_gate2_sign_resolution.json`,
      `results/experiment7_quality_bin_rank_profile.png`), run because the
      bin-mean plot appeared to contradict Gate 2's sign directly — the
      paper's central, three-times-replicated claim — which made this the
      first thing to resolve before anything else. Four checks:
        1. *Sign convention:* scored 10 known-human and 10 known-AI (DAIGT)
           essays directly through the frozen composite. Mean P(AI):
           human 0.132, AI 0.845 — correct direction, consistent with
           Experiment 3's full-corpus AUC = 0.945. Not the cause.
        2. *Same sample:* `results/experiment3_human_scores.csv` is written
           from the exact `human` dataframe and `human_scores` array Gate 2
           uses, inside the same script run — same rows by construction,
           not coincidence.
        3. *Recompute Spearman on the bin plot's own residualized values:*
           `partial_spearman` recomputed from that CSV reproduces Gate 2's
           +0.135 exactly. A plain Spearman correlation between the bin
           plot's own residualized P(AI) values and quality gives **+0.136**
           — matching Gate 2, not contradicting it. **The bin plot's own
           underlying rank relationship was positive all along; only its
           arithmetic-mean visualization looked negative.**
        4. *Look at the scatter, not the means:* residualized P(AI) is
           heavily right-skewed (skew 1.2 to 2.5 across bins, a long tail of
           essays scored much more AI-like than word count predicts), with
           bin sizes ranging n=842 to n=7,965 — arithmetic means are not
           robust to that combination and are dominated by a handful of
           extreme values, especially in smaller bins. The **mean rank
           percentile per bin** (what Spearman actually reflects) tells a
           different, much more monotonic story: 0.440 (score 2, lowest) →
           0.481 (3) → 0.536 (4) → 0.557 (5) → 0.587 (score 6, highest).
           Scores 2 through 6 increase cleanly. Score 1 (0.521) sits mildly
           out of order — between scores 3 and 4, not at either extreme —
           rather than at the bottom the mean plot suggested.
    - **On the score-1 "structural degeneracy" citation:** score-1 essays
      are genuinely shorter (median word count 249 vs. 824 for score 6), a
      partial match for a length-related hypothesis. But no "§5.7" or
      comparable section on degenerate feature values under broken
      tokenization was found anywhere in this repository (checked
      `README.md`, this file, `PRACTITIONER_BRIEF.md`, every
      `EXPERIMENT_*.md`, and `scripts/`) — the closest existing finding
      here is RAID's homoglyph/zero-width-space *adversarial* attacks
      breaking tokenization (item 5 / experiment 4), a different mechanism
      (deliberately injected unicode in AI-generated text, not naturally
      short human writing). If that citation refers to a document outside
      this repository it cannot be verified from what's here and is not
      repeated as fact. For what it's worth, score 1's own residual
      distribution has the *lowest* skew of the six bins (1.20, vs. 2.46 for
      score 6) — its elevated mean looks more like ordinary outlier
      leverage in a smaller bin (n=1,024, the second-smallest) than a
      distinct degeneracy regime, though this reading is not conclusive
      either way.
    - **Stated outcome:** Check 1 (discriminant/quality cosine, both
      conditioned consistently) — retraction stands: real, moderate
      alignment (cosine 0.533). Check 2 (Pearson vs. Spearman on Gate 2) —
      not an artifact: Spearman is the correct, pre-specified read, and its
      sign reversal under Pearson reflects a real concentration of the
      effect in part of the quality range. The bin-profile follow-up that
      was supposed to locate that concentration **was itself wrong on first
      report** — its mean-based visualization suggested a bottom-of-range
      effect that does not survive a skew-robust (rank/median) view. The
      corrected view shows the effect rising strictly monotonically from
      score 2 through score 6 (matching `TECHNICAL_REPORT.md` §5.2's
      wording), i.e. closer to a top-of-range effect than a bottom-of-range
      one, with score 1 a mild, unresolved exception. This
      episode is logged in full, including the wrong first conclusion,
      because the error was in this study's own analysis, not in Gate 2 —
      exactly the kind of mistake this log exists to surface rather than
      quietly fix.

**Verdict:** the data leans mildly toward H2 over H1 — the ordering
prediction held, and centroid distance (a homogenization-flavored measure)
predicts TPR more strongly than dispersion does — but this experiment does
not confirm H2 outright, since its own pre-registered dispersion-TPR test
came in short of the threshold set for it, and n = 17 is too small to treat
any of these correlations as decisive. The direction check argues against
H1 as stated (weak, wrong-signed correlation with TPR across sources); that
test was already computed in a length-residualized space for both the
direction vector and the projected source centroids (unlike the
pre-correction cosine test), so it is not subject to the same retraction —
but its outcome variable, TPR, is itself computed from the composite's
original, non-length-residualized threshold, which the dispersion- and
centroid-distance-vs-TPR correlations above share. **Logged as a known
limitation and deliberately not pursued further**: fixing it would mean
re-deriving a threshold from a residualized-feature discriminant and
rescoring all 17 sources, a larger undertaking than the checks logged
here, for a correction to an already-exploratory (n=17) section the
writeup will report as unresolved either way. The
discriminant/quality cosine, once both vectors are conditioned consistently,
is the cleanest single mechanism number this experiment produced (cosine
0.533, 57.8°): a real, moderate — not full — alignment between what the
composite detects and what makes human writing rate as high quality.
Neither
hypothesis should be reported as confirmed; this is suggestive,
correlational evidence about model outputs, not a demonstration about what
any lab actually trained on.

**Note on framing across documents:** this log's "leans mildly toward H2"
describes the direction of two secondary, non-adjudicating measures
(ordering, centroid distance). `TECHNICAL_REPORT.md` §5.10 deliberately
states the more conservative "we cannot adjudicate H1 against H2... the
mechanism is open," since neither of those two measures actually
distinguishes H1 from H2 (both hypotheses predict them equally), and the
one test that does distinguish them — the H1 direction check — came back
null. Read this log's "leans mildly" as background colour on which way the
non-discriminating evidence points, not as a claim this study confirmed;
the paper's "open" is the more defensible framing of the finding as a
whole, and `README.md`'s summary has been aligned to it.

## 7. A misattributed citation, traced and corrected

Both `TECHNICAL_REPORT.md` and `PRACTITIONER_BRIEF.md` originally stated
that Turnitin "disputed the methodology" of Liang et al. (2023) — citing
short texts, small samples, and noting Turnitin's own detector was not
among those tested. This was flagged for tracing before either document
shipped, since it is a specific attribution claim about who made a
methodological argument, not just a missing citation detail.

**What was found.** Turnitin did publish a real, dated response (2023):
a company blog post reporting that its own detector, tested on a much
larger sample of authentic student essays, showed no statistically
significant ELL bias. Secondary sources gave conflicting accounts of
whether that post also critiques Liang et al.'s methodology directly; the
primary source itself could not be fetched to settle it (blocked by the
site returning HTTP 403; an archive.org mirror was also inaccessible from
this environment). One secondary source explicitly stated the post does
*not* critique Liang et al.'s sample size or essay length, and only
acknowledges that Turnitin's own system "lacks enough linguistic
information" to score very short documents — a statement about Turnitin's
product, not an argument about the Stanford study's validity.

The specific critique originally attributed to Turnitin — small sample
(91 essays), other unspecified "methodological flaws," and the note that
a named detector was excluded from testing — was traced instead to a blog
post by **Pangram**, a different AI-detection vendor, about **Pangram's**
own product being excluded from Liang et al.'s evaluation. This was
confirmed by fetching the Pangram blog post directly, which contains that
exact language.

**Why this matters beyond getting a name right.** Even correctly
attributed to Pangram, this critique is a competing detection vendor's
marketing content about its own product's exclusion from a study that
found bias in vendor detectors generally — precisely the category CLAUDE.md
already excludes as evidence ("Treat detector-vendor and 'humanizer' site
numbers as marketing... Peer-reviewed sources... only"). The fix was not
to re-attribute the critique correctly; it was to drop the critique
entirely from both documents and report only what is independently
verifiable: Liang et al.'s own stated sample (their paper, not a critic's
gloss on it) and Turnitin's own confirmed finding.

**Process note.** This was caught only because the specific pairing of
claims — "disputed the methodology... small sample... short texts...
detector not tested" — was checked against a fetched primary or secondary
source rather than repeated from an earlier draft. The earlier draft's
version was itself apparently assembled from conversational recall rather
than a source read at the time, which is the same failure mode as the
RAID attack-list hallucination caught earlier in this project (see the
Errors and Fixes record for that incident) — a plausible-sounding claim
that was never actually checked against a primary source. No claim about
who-argued-what should ship in either document without this check.

## 8. Two pre-registered arms, dropped without disclosure, then found, disclosed, and run

While preparing this document for a completeness review, two components of
`RESEARCH_PLAN.md`'s original pre-registration were found to have never been
run, and never been logged anywhere as dropped — a gap in this log's own
stated discipline ("This log records every consequential decision made or
changed... it exists because several of these decisions are not otherwise
recorded anywhere in the repository"). Both are now run; this entry records
the omission and the results.

**What was missed.** (a) Experiment 1's Tier-2 arm: the plan specifies
computing Gate 1 (ELL predictability) with a model-based zero-shot detector,
Binoculars, in addition to the Tier-1 deterministic features that shipped as
the gate's sole basis. (b) Experiment 4's manual light-edit condition: the
plan specifies running RAID's adversarial subsets "plus a manual light-edit
condition (vary sentence lengths, add contractions) simulating a student who
has been told what detectors look for" — only the RAID subsets were run and
reported.

**Why they were skipped without disclosure.** Best reconstruction: (a) was
skipped because it required a GPU-based model pipeline that did not exist
yet when Experiment 1 was first run, and no fallback plan (Fast-DetectGPT,
per the plan's own "as a lighter alternative") was substituted or logged as
a substitution. (b) was skipped because RAID's official attack list already
gave twelve conditions, and the manual edit condition was, in retrospect,
conflated with that coverage rather than recognised as a distinct
pre-registered item. Neither is a defensible reason to omit a pre-registered
test silently; both are logged here as process failures, not as considered
scope decisions.

**Given the choice to run them or leave them disclosed-but-dropped**, a
probability estimate was made first, before committing GPU/analyst time:
both were assessed as low-probability (under 10%) to change any of this
study's existing conclusions, since neither tests the mechanism the shipped
conclusions rest on — Gate 2's composite-level quality inversion (item 1)
and the ELL penalty (§5.5) do not depend on Gate 1's Tier-1 arm being
Binoculars-confirmed, and the light-edit condition tests evasion-under-
editing, a question orthogonal to the quality-inversion and fairness
findings that already close the scoring-panel question. Both were then run
anyway, since both were pre-registered and GPU time was available locally
(an RTX 3060 Laptop GPU, 6GB VRAM) — reporting a study as complete when two
of its own pre-registered components were never run is a worse outcome than
spending the (modest, ~25 minutes combined) compute to close the gap.

**(a) Tier-2 Binoculars result** (`scripts/experiment1b_tier2_ell_gate.py`,
`results/experiment1b_tier2_*`). The plan's default Falcon-7B observer/
performer pair does not fit this machine's 6GB VRAM budget; substituted
Qwen2.5-0.5B (observer, base) / Qwen2.5-0.5B-Instruct (performer,
instruction-tuned) — a smaller pair from the same base/instruct family,
which the plan explicitly permits ("Substitute smaller models if VRAM is
tight and record which"). Scored on the exact same caliper-matched sample
Tier-1's Gate 1 used (n = 1,810 pairs; caliper 0.15; see
`results/experiment1_manifest.json`), with a single forward pass per essay
(batching was attempted first at batch sizes up to 8 and reliably exhausted
the card's VRAM — Qwen's ~152k-token vocabulary makes each position's
logits and softmax buffers large relative to the models' own 0.5B-parameter
size; single-essay scoring peaks at ~2.8GB, comfortably within budget).
Result: **AUC against ELL status 0.533, 95% CI [0.515, 0.552]** — passes
the 0.65 gate with room to spare, and is in fact *weaker* than the Tier-1
composite's own 0.601 [0.583, 0.619]. The model-based arm does not
overturn Gate 1; if anything it reinforces it.

**(b) Manual light-edit result** (`scripts/experiment4b_light_edit.py`,
`src/light_edit.py`, `results/experiment4b_light_edit*`). The edit itself is
deterministic pure Python — merge every third sentence into the next with
", and ", plus a fixed ~30-entry contraction map — not a paraphrase-tool
rewrite. Applied to the same llama-chat/human RAID baseline pair Experiment
4 already used (n = 107 AI, n = 493 human), scored with the same frozen
composite and threshold. Result, reported plainly because it is asymmetric:
AI TPR moves from 52.3% to 49.5% (Δ = −2.8 pts, 95% CI [−8.4, +3.7] — not
distinguishable from no change), while human FPR drops from 36.9% to 20.1%
(Δ = −16.8 pts, 95% CI [−20.5, −13.6] — large and clearly non-zero). The
edit is not a reliable evasion technique for AI-generated text, but it is a
substantial de-flagging technique for genuinely human text that happens to
read uniformly — which is exactly the failure mode the plan's own framing
warned about ("the panel predominantly catches honest students who happen
to write uniformly"), now measured rather than merely anticipated.

**Disposition.** Both results are now integrated into `TECHNICAL_REPORT.md`
(§5.1 for the Tier-2 arm, §5.7 for the light-edit condition, and a
qualifying paragraph added to §7.3 noting the asymmetry) and this log.
Neither changes any existing gate verdict or headline conclusion, consistent
with the pre-run probability estimate above — which is itself worth noting:
the estimate was right, but it was made honestly before the results were
known, not fitted to them afterward.

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
