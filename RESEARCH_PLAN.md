# Calibration study: statistical texture measures on student essays

> **Pre-registration note.** This plan was written on 2026-08-13, before any
> data was downloaded or any code was written. The AUC > 0.65 threshold in
> Experiment 1 and the ≤1% target FPR in Experiment 3 were fixed at that
> time, prior to seeing any result they would later be judged against.
> Experiments 5 and 6 do not appear below — they were designed later, in
> response to findings from Experiments 1-4, and are documented separately
> in `EXPERIMENT_5.md` and `EXPERIMENT_6.md`. See `AMENDMENTS.md` for a full
> log of decisions made or changed during the study, including the one gate
> threshold (Experiment 2's "meaningful correlation") that this plan never
> numerically specified.

## Background

MaiaLearning's essay review feature currently asks Claude to output a probability
that an essay would be flagged by AI detection tools. Three problems with it:

1. The number was never validated against any real detector.
2. Observed output rated a short, underdeveloped essay as low-risk and described
   it as showing "a natural, personal writing style" — i.e. the measure appears
   anti-correlated with essay quality.
3. The prompt explicitly instructed the model to score as a detector would while
   acknowledging that ELL students trigger the same signals innocently. Published
   research finds perplexity-based detectors misclassify a majority of TOEFL
   essays by non-native writers as AI-generated.

This study establishes whether a deterministic texture measure avoids both
failure modes well enough to support a user-facing panel.

## Datasets

All public. No production data.

**PERSUADE 2.0** — primary. ~25,996 argumentative essays, US students grades
6–12, collected pre-ChatGPT (before Nov 2022), so human labels are reliable.
Includes holistic quality scores and demographic metadata: ELL flag (~1,330
train / ~914 test), identified disability, economically disadvantaged, grade,
race. The ELL flag plus quality scores are what make this corpus irreplaceable
for our two gate experiments.
- Via LEAR Lab: https://learlab.org/data/
- Kaggle mirror: `nbroad/persaude-corpus-2`

**ELLIPSE** — ~7,000 essays by English language learners with individual
difference metadata (economic status, gender, grade 8–12, race). Use as a denser
ELL sample if PERSUADE's ELL subset is too small for tight intervals.
- https://learlab.org/data/

**DAIGT-v2** — 44,840 essays: PERSUADE humans plus generations from ChatGPT,
GPT-4, Llama-70B, Falcon-180B, Mistral, Claude, Cohere, PaLM. Note it includes
~2,000 Claude-generated essays, which matters because MaiaLearning generates
with Claude.
- `thedrcat/daigt-v2-train-dataset` on Kaggle

**AIDE** — built by Vanderbilt and The Learning Agency Lab specifically for AI
detection in education. Student argumentative essays plus multi-LLM generations
on matched topics.

**RAID** — 10M+ documents, 11 LLMs, 11 genres, 4 decoding strategies, 12
adversarial attacks, with a public leaderboard. Use the adversarial subsets for
experiment 4. Do not download the whole thing.
- `liamdugan/raid` on Hugging Face
- Paper: https://aclanthology.org/2024.acl-long.674/

## Features to compute

**Tier 1 — deterministic, cheap, no model.** Compute these first; they may
answer the gate questions on their own.
- Sentence-length standard deviation (this is "burstiness")
- Mean sentence length
- Type-token ratio, and a length-normalized variant (MTLD or moving-average TTR
  — raw TTR is length-confounded and PERSUADE essays vary widely in length)
- Transition-phrase frequency against a fixed lexicon
- Paragraph-length variance
- Punctuation variety; contraction and colloquialism rate
- Function-word distribution

**Tier 2 — model-based, needs GPU.**
- **Binoculars** (ICML 2024, Apache-2.0, https://github.com/ahans30/Binoculars).
  Ratio of log perplexity under an observer model to cross-perplexity under a
  performer model. Reported >90% TPR at 0.01% FPR on ChatGPT text without
  training on ChatGPT data. Chosen specifically because the cross-perplexity
  normalization is designed to cancel baseline perplexity effects — the mechanism
  that makes naive perplexity misfire on ELL writing. Whether it actually does
  that for ELL writers is experiment 1.
  - Note: ships with a fixed global threshold. Authors describe it as academic,
    not a consumer product, and caution against unsupervised use. We will fit our
    own threshold.
  - Paper config uses Falcon-7B observer/performer. Substitute smaller models if
    VRAM is tight and record which.
- **Fast-DetectGPT** as a lighter alternative if Binoculars is too heavy.

## Experiments

Run in order. 1 and 2 are gates.

### 1. Does the score predict ELL status? (GATE)

Human PERSUADE essays only — no AI text in this experiment.

Match ELL and non-ELL samples on grade level, prompt ID, word count, and
holistic score. Without matching you will confound language background with
grade level and essay quality and the result will be meaningless.

Compute AUC of each feature, and of a combined model, against the ELL flag.

- AUC ≈ 0.5 → the feature does not track language background. Proceed.
- AUC > 0.65 → the feature is substantially a language-background detector.
- Report per-feature AUC, not just the combined figure. If one feature carries
  the bias, it can be dropped.

**This is the crisp form of the central concern: if the score predicts whether a
student is an English language learner, it is measuring the wrong thing.**

### 2. Is the score anti-correlated with essay quality? (GATE)

Same human-only sample. Spearman correlation between each feature and PERSUADE's
holistic quality score.

A meaningful negative correlation reproduces the production failure — weak
essays reading as "more human" — and is disqualifying for a product whose
purpose is helping students write better essays. Control for word count;
PERSUADE holistic scores correlate with length.

### 3. Does it separate human from AI at a usable false-positive rate?

Add DAIGT-v2 AI essays. Also generate a matched set via Bedrock with the same
Claude model production uses, on PERSUADE prompts — no other corpus covers our
own generator.

Fit the threshold at a target FPR of ≤1% (stakes justify a conservative
threshold) and report TPR there.

**Report TPR and FPR separately for ELL and non-ELL subgroups.** The aggregate
figure hides the thing we care about. A 1% overall FPR that is 0.3% for native
speakers and 4% for ELL writers is a fairness failure wearing a good headline.

Break TPR down by generating model. Expect Claude to be harder to detect than
GPT-family output.

### 4. Does it survive paraphrasing?

Run the RAID adversarial subsets, plus a manual light-edit condition
(vary sentence lengths, add contractions) simulating a student who has been told
what detectors look for.

If detection collapses under light editing, the panel predominantly catches
honest students who happen to write uniformly. That is the worst possible
selectivity and must be stated plainly to students if we ship anyway.

## Decision gates

- Fail 1 or 2 → no scoring panel. Report and stop.
- Pass 1 and 2, fail 3 → ship a narrower claim, or descriptive statistics with no
  conclusion.
- Pass 1–3, fail 4 → ship with explicit scope: detects unedited AI output only.
- Pass all four → proceed to a production design, with periodic recalibration
  budgeted. RAID's central finding is that detectors degrade against unseen
  models, so any threshold fitted now has a shelf life.

## Deliverables

1. `results/` — CSVs and plots per experiment.
2. A writeup stating, for each gate, the number and the decision.
3. If viable: recommended feature set, threshold, abstain band, and the
   subgroup error rates a student-facing claim would have to disclose.
4. Reference implementation of the Tier 1 features suitable for porting into the
   Lambda — pure Python, no model dependency, no network call.

## Known limitations to state in the writeup

- PERSUADE is grades 6–12 argumentative essays responding to source material.
  Production input is 300–650 word college personal statements. The domain shift
  is real and thresholds will move. PERSUADE is used anyway because no public
  corpus of admissions essays carries ELL labels.
- Any threshold degrades as new generative models ship.
- This study cannot produce a Turnitin predictor. Turnitin's classifier is not
  public and its labels are not obtainable without institutional access. No
  product claim of the form "X% likely to be flagged by Turnitin" is supportable
  and that framing is retired permanently. GPTZero's API can supply one real
  commercial anchor at modest cost if a comparison point is wanted.
