# Experiment 7: Is high detectability driven by *within-document* polish or *across-document* homogenization?

Read `CLAUDE.md`, `RESEARCH_PLAN.md`, and `AMENDMENTS.md` first. This uses only
data already on disk — no generation, no API spend, no new corpora.

## Why this exists

Experiment 3 found current frontier models to be the *most* detectable of 17
sources (GPT-5.6 Terra 78.1%, Claude Sonnet 5 73.6%) while open-weight and older
models sat at the bottom. Experiment 2 found the composite score positively
correlated with human-rated essay quality (partial rho = +0.135).

Two mechanisms could produce both results, and they make different predictions:

**H1 — pretraining register.** Pretraining mixtures skew toward professionally
edited prose relative to writing by students in grades 6–12. The detector is
therefore partly learning "does this read like edited adult writing," which
strong student essays also do. Under H1 the driver is *within-document* polish:
each generated essay is individually regular, and detectability should track the
polish features regardless of generator family.

**H2 — post-training homogenization.** All these models share broadly similar
pretraining mixtures, so pretraining alone cannot explain why frontier models
separate from open-weight ones. RLHF and related post-training compress output
toward a modal preferred register — polished *and* homogeneous. Under H2 the
driver is *across-document* variance: frontier outputs should cluster tightly
around their own centroid relative to human essays and to older models.

These are not exclusive. The question is how much of the gradient each explains.

**Pre-register before running** (write into the manifest):
Predicted ordering of across-document dispersion, tightest to widest —
frontier proprietary < older proprietary < open-weight < human. Predicted
correlation between per-source dispersion and per-source TPR: negative,
|rho| > 0.5. Record these before computing anything.

## Data

Use the existing scored feature matrix from experiment 3. Every essay already
has its nine Tier-1 features and a source label across 17 generator sources plus
PERSUADE humans. Nothing new is needed.

If per-essay features were not persisted, recompute them with the existing
`src/` feature code — deterministic, no cost.

## Method

### 1. Per-source dispersion

For each of the 18 sources (17 generators + human), compute across-document
dispersion in the nine-dimensional feature space.

- **Standardize first.** z-score each feature using the *human* corpus mean and
  SD, so dispersion is expressed in human-variation units and features are
  commensurable. State this in the writeup; it is the step that makes the
  numbers interpretable.
- **Primary statistic:** mean Euclidean distance from each document to its own
  source centroid, in standardized space.
- **Secondary:** total variance (trace of the covariance matrix) and generalized
  variance (determinant). Report all three; if they disagree, say so.
- **Per-feature:** SD of each individual feature within each source. This shows
  *which* dimensions are compressed, which is more informative than a single
  scalar.

### 2. Control for sample size and length

Dispersion estimates are sensitive to both.

- **Subsample every source to a common n** (the smallest source's n, or 500,
  whichever is lower). Bootstrap the subsample 1,000 times and report the
  dispersion CI. Without this, sources with more documents will appear more
  dispersed for purely statistical reasons.
- **Match on word count.** Several features are length-sensitive. Either
  rejection-sample each source to a common word-count distribution, or report
  dispersion on length-residualized features. Do whichever is cleaner and state
  which.
- **Hold the prompt set fixed.** Compare only documents responding to prompts
  present across sources; otherwise prompt diversity is confounded with source.

### 3. The key correlation

Across the 17 generator sources, correlate per-source **dispersion** against
per-source **TPR@1%FPR** (already computed in experiment 3).

- **Strong negative correlation → H2 supported.** Tighter clustering predicts
  higher detectability, and homogenization is doing the work.
- **Weak or absent → H1 favoured.** Detectability is about per-document polish,
  not fleet-level uniformity.

n = 17 is small. Report the CI, use Spearman, and do not over-read a point
estimate. This is suggestive evidence, not a decisive test, and the writeup must
say so.

### 4. Centroid distance

Compute each source's centroid distance from the human centroid in standardized
space, and correlate that with TPR.

This separates two further things: a source could be detectable because it sits
far from human writing (centroid distance) or because it is internally
consistent (dispersion). Report both, and their relative contribution in a
simple regression of TPR on both terms.

**Direction check that bears on H1:** if H1 holds, frontier centroids should sit
displaced from the human student centroid *in the direction of high measured
quality* — i.e. toward the region occupied by high-scoring PERSUADE essays.
Compute the centroid of the top quality quartile of human essays and report
where each source sits relative to the human-mean-to-high-quality vector. If
frontier models project far along that vector, H1 has direct support.

### 5. Human comparison sets

Human dispersion is the reference. Where the data allows, report it separately
for the full human corpus, the top quality quartile, the bottom quality quartile,
and the ELL subset.

**A prediction worth testing explicitly:** if the detector measures register
regularity, the top human quality quartile should show *lower* dispersion than
the bottom. That would be a second, independent route to the experiment 2 result
and would tie the two experiments together.

## What this does and does not settle

It does not reopen the shipping decision. Gate 2 failed on the composite and
TPR@1%FPR is 41.3%. This is a mechanism question for the writeup.

It also cannot establish *why* the models behave this way. Corpus composition and
post-training recipes are not published by any lab. The strongest available claim
is a mechanism hypothesis consistent with the observed dispersion structure — not
a demonstration about training data. The writeup must be explicit that this is
correlational evidence about model outputs, not evidence about model training.

Avoid the overstatement that LLMs are "trained on peer-reviewed and professionally
edited work." Web-scale crawl dominates most pretraining mixtures by volume;
quality filtering and upweighting are standard but partial. The defensible framing
is that the mixture skews toward edited prose *relative to writing by students in
grades 6–12*, which is the comparison class in this study.

## Deliverables

1. `results/experiment7_dispersion.csv` — per-source dispersion (three
   statistics), centroid distance, TPR, n, bootstrap CIs.
2. `results/experiment7_per_feature_sd.csv` — per-source SD for each of the nine
   features, standardized.
3. Plot: dispersion vs TPR, sources labelled, human marked as reference.
4. Plot: per-feature SD heatmap, sources × features, ordered by TPR.
5. `results/experiment7_manifest.json` — pre-registered predictions recorded
   before the run, standardization choice, matching method, seeds, and whether
   each prediction held.
6. A short writeup: which mechanism the dispersion structure favours, with the
   correlation, its CI, and an explicit statement of how much weight n = 17
   supports.

## Amendments

Log this experiment in `AMENDMENTS.md`: added post hoc, after experiment 3
produced an unexpected capability gradient, to test a mechanism hypothesis raised
during analysis. It was not part of the original pre-registration and the writeup
should present it as exploratory.
