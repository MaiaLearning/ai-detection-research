# Detection by Uniformity Penalises Good Writing: A Withdrawn Deployment and a Calibration Study on Student Essays

**Barry Coleman**
MaiaLearning, Inc.
`research@maialearning.com`

Code and data: `https://github.com/MaiaLearning/ai-detection-research`

---

## Abstract

Statistical detectors of machine-generated text rely on surface measures of textual uniformity — sentence-length variance ("burstiness"), lexical diversity, transition density. We construct such a detector, evaluate it on 24,695 human-written student essays carrying independent holistic quality scores and English-language-learner (ELL) annotations, and report three findings that hold regardless of detector quality.

First, and centrally: the detector's composite score is *positively* correlated with human-graded essay quality (partial Spearman ρ = +0.135, 95% CI [0.123, 0.148], controlling for word count) — better essays are scored as more machine-like. The single most-cited detection signal, sentence-length variance, is itself negatively correlated with quality (ρ = −0.145). Uniformity is not primarily a machine signature; it is substantially a property of competent writing. We note that the automated essay scoring literature has long treated the same feature family as an index of proficiency, and argue the two literatures describe one axis read in opposite directions.

Second, detector calibration does not transfer across genre. Fixed at a 1% false positive rate (FPR) on the calibration genre, the same threshold yields 2.4–3.7% FPR on a near genre and 36.9% (95% CI [32.9, 41.2]) on a far genre — human writing throughout. Published single-figure FPRs are therefore uninterpretable without the genre on which they were measured.

Third, ELL writers incur an FPR penalty of 1.5–2.5× that replicates across two independently constructed corpora. A 2×2 design holding out four prompts from training shows the penalty is invariant to topic novelty, so it cannot be mitigated by broadening prompt coverage in calibration data.

At 1% FPR the detector achieves 41.3% true positive rate against 17 generator sources; a paraphrase attack removes a further ~21 points. The system described here was deployed in a commercial product and has been withdrawn. We report the process by which the defect was found — a scheduled output-appropriateness audit, not performance monitoring — as a secondary contribution.

---

## 1. Introduction

Detectors of machine-generated text are widely deployed in education, and their error properties are contested. Liang et al. (2023) found that perplexity-based detectors misclassify a majority of TOEFL essays by non-native writers as machine-generated while remaining near-perfect on essays by US-born eighth-graders, and attributed this to lower perplexity arising from reduced linguistic variability; the sample was 91 essays under 150 words each. Turnitin (2023) published a follow-up study testing its own detector, not among those Liang et al. evaluated, on a much larger sample of authentic student essays, and found no statistically significant ELL bias in its system.[^turnitin]

[^turnitin]: An earlier draft of this paper attributed a stronger claim to Turnitin here — that it "disputed the methodology, citing short texts and small samples." That claim's provenance was traced and corrected before this paper was finalized; see `AMENDMENTS.md` for the full account.

We approached the question as practitioners rather than as critics. MaiaLearning deployed an AI-detection panel in a commercial college-essay review product. A scheduled audit of that feature's *output appropriateness* — distinct from monitoring its performance — found that the guidance it issued to students appeared to advise writing badly. This paper is the investigation that followed, and the evidence on which the feature was withdrawn.

**Contributions.**

1. **A quality–detection inversion.** On 24,695 human essays with holistic scores, detector score correlates positively with quality. To our knowledge this connection between detector features and independently-rated writing quality has not previously been measured, despite both feature families being well-studied in isolation (§2, §5.2). [Corrected from an earlier draft's "§3.2," which does not exist — §3 (Data) has no subsections; the AES/detection literature gap this sentence refers to is discussed in §2, "The gap we address."]
2. **A quantified genre-transfer failure.** A calibration ladder from 1% to 36.9% FPR on human writing under a fixed threshold (§5.4).
3. **A topic-invariant ELL penalty**, replicated across corpora, with a 2×2 test excluding topic novelty as the mechanism (§5.5).
4. **An inverted capability gradient.** Current frontier models were the *most* detectable of 17 sources; open-weight and older models the least (§5.6).
5. **A methodological negative result** on DAIGT-v2, a widely-used detection benchmark: its constituent generations carry no prompting-effort scaffolding, contrary to common assumption (§5.9; also §6).
6. **A deployment-process finding**: the defect was invisible to performance metrics and surfaced only through appropriateness review (§7.4).

We pre-registered gate thresholds before data collection and report the predictions that failed (§6).

---

## 2. Related work

**Detection methods.** Zero-shot statistical detection has largely converged on likelihood-based signals: DetectGPT's probability curvature (Mitchell et al., 2023), Fast-DetectGPT's conditional probability curvature, and Binoculars' perplexity-to-cross-perplexity ratio (Hans et al., 2024), which reports >90% TPR at 0.01% FPR without ChatGPT training data. Commercial detectors do not publish their methods; public analyses attribute them to the same perplexity-and-burstiness family, which is the family we construct here.

**Robustness.** Sadasivan et al. (2024) argue detection degrades sharply under paraphrase. RAID (Dugan et al., 2024) — 10M+ documents, 11 generators, 11 genres, 4 decoding strategies, 12 adversarial attacks — found detectors easily defeated by attacks, sampling variation, and unseen generators. Perkins et al. (2024) demonstrate simple bypass techniques in an educational context.

**Fairness.** Liang et al. (2023) is the anchor result. Pratama (2025) reports accuracy–bias trade-offs affecting non-native authors in scholarly publishing. Reviews of detection in higher education (Weber-Wulff et al., 2023; Elkhatat et al., 2023; Deep et al., 2025) consistently report false positives concentrated among multilingual writers. Notably, a Czech-language replication finds the *opposite* entropy relationship for non-native Czech writers, indicating the effect is language-specific rather than universal — which bears on the generalisability of all such results, including ours.

**Writing quality.** The automated essay scoring literature has extensively characterised lexical and syntactic complexity as predictors of rated quality (Crossley, 2020; Casal & Lee, 2019), with MTLD (McCarthy & Jarvis, 2010) a standard lexical-diversity index. Herbold et al. (2023) compare human and ChatGPT essays along precisely these dimensions.

**The gap we address.** These two literatures use an overlapping feature family for opposite purposes — AES reads lexical and syntactic variation as proficiency, detection reads it as humanness — and, so far as we can determine, have not been evaluated jointly on a corpus carrying both AI/human labels and human quality ratings. PERSUADE 2.0 permits exactly that.

---

## 3. Data

| Corpus | Role | n (used) | Key annotations |
|---|---|---|---|
| PERSUADE 2.0 | Human reference, calibration genre | 24,695 | Holistic quality score, ELL flag, grade, prompt |
| DAIGT-v2 | Machine-generated, 15 sources | 17,492 | Generator identity |
| Bedrock generations | Machine, current Claude | 1,000 | Prompt, model ID |
| OpenAI generations | Machine, current GPT | 1,000 | Prompt, model ID |
| ELLIPSE | Near-genre human, all ELL | 6,482 | Proficiency ratings, prompt |
| RAID (abstracts subset) | Far-genre human + adversarial | 493 human | Attack type, generator |

**PERSUADE 2.0** comprises argumentative essays by US students in grades 6–12, collected before the release of ChatGPT, making human authorship reliable. It carries holistic quality scores and demographic annotations including ELL status. Filtering criteria (`src/data.py::load_and_clean`), applied identically by every script in this study: of 25,996 raw rows, a row is kept only if all of `ell_status` (recoded to a clean Yes/No flag, dropping any other value), `grade_level`, `holistic_essay_score`, `word_count`, `prompt_name`, and `full_text` are non-null. This drops 1,301 rows (5.0%), leaving n = 24,695 (2,244 ELL, 22,451 non-ELL). No length or other content filter is applied to the human corpus.

**DAIGT-v2's generator-source column contains 15 distinct labels** in the subset used here — `NousResearch/Llama-2-7b-chat-hf`, `chat_gpt_moth`, `cohere-command`, `darragh_claude_v6`, `darragh_claude_v7`, `falcon_180b_v1`, `kingki19_palm`, `llama2_chat`, `llama_70b_v1`, `mistral7binstruct_v1`, `mistral7binstruct_v2`, `mistralai/Mistral-7B-Instruct-v0.1`, `palm-text-bison1`, `radek_500`, `radekgpt4` — not 14 as earlier drafts of this report and its underlying script's docstring stated; corrected throughout. Together with our own Bedrock and OpenAI generations (1 source each), this gives the 17 generator sources cited in the abstract, §5.3, and §6 (15 + 2 = 17; verified against `results/experiment3_manifest.json`'s `ai_sources` list directly, not re-derived from the corrected count by arithmetic that could itself be wrong).

**Generated sets.** We generated 1,000 essays with `us.anthropic.claude-sonnet-5` via AWS Bedrock ($8.97) and 1,000 with `gpt-5.6-terra` via the OpenAI API, both on PERSUADE prompts at temperature 1.0. Two limitations are logged: the generation prompt was generic rather than production's system prompt, and for the seven text-dependent PERSUADE prompts the models wrote from general knowledge, since the corpus carries citations but not source article text.

Corpus SHA-256 digests are published in the repository for version verification. PERSUADE and ELLIPSE are CC BY-NC-SA 4.0 and are not redistributed.

---

## 4. Method

### 4.1 Features

Nine deterministic features, computed in pure Python with no model dependency, no network call, and no stochasticity:

`sentence_length_std` (burstiness), `mean_sentence_length`, `type_token_ratio`, `mtld`, `transition_phrase_rate`, `paragraph_length_variance`, `punctuation_variety`, `contraction_rate`, `function_word_entropy`.

We compute both raw TTR and MTLD deliberately: raw TTR is length-confounded, and the divergence between them is informative (§5.2). The transition-phrase lexicon (`src/features.py::TRANSITION_PHRASES`) is a fixed, self-authored list of 34 discourse markers (e.g. "on the other hand," "for example," "therefore," "furthermore," "first"/"second"/"third"), matched as whole-word/whole-phrase regex patterns against lowercased text; it is not adapted from an external published list, and the code cites no source for it.

### 4.2 Models

This study fits two separate models on two separate questions; neither's methodology should be read onto the other.

#### 4.2.1 The detection composite (§5.1–5.7; its discriminant vector is also reused, but not refit, for the geometry test in §5.10)

`StandardScaler` + `LogisticRegression` (scikit-learn defaults except `max_iter=1000`; no hyperparameter search), fit on all nine features with none pre-dropped, seed = 42. All reported scores are out-of-fold, 5-fold stratified CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`). There is no dev/held-out split for this model — every gate, AUC, and TPR figure in §5.1–5.7 is a cross-validated out-of-fold prediction, not a held-out-set evaluation. (§5.9 is a corpus/prompting-methodology finding about DAIGT-v2 itself; it does not use either model in this section.)

#### 4.2.2 The quality-prediction model (§5.8)

A separate question — do the nine features carry essay-quality signal beyond word count? — uses a different model with a different validation design: `RidgeCV` (re-tuning its regularization strength inside each fold), fit on **8** of the 9 features (raw `type_token_ratio` dropped; rationale in §5.8) plus word count. Split into DEV (80%, n=19,756) and a genuinely untouched HELD-OUT set (20%, n=4,939), stratified by ELL status, seed 42. Feature-dropping and all other modeling decisions were made using DEV only; the incremental-signal figures reported in §5.8 are confirmed once on HELD-OUT, not used for any tuning or selection decision.

### 4.3 Pre-registered gates

Fixed before data collection (`RESEARCH_PLAN.md`):

- **Gate 1 — ELL predictability.** AUC of the score against ELL status must be < 0.65. Rationale: a score that identifies ELL writers is measuring language background, not authorship.
- **Gate 2 — quality anti-correlation.** The score must not be meaningfully anti-correlated with rated quality. *We did not fix a numeric threshold for gate 2. This is a pre-registration defect and we report it as such* (§6).
- **Operating point.** 1% FPR, chosen on the grounds that a false accusation is not easily rebutted by a student.

Gate 2 was initially specified per-feature and reapplied to the composite during analysis; the composite is the correct level, since it is what a user would see. This and other mid-study changes are recorded in `AMENDMENTS.md`.

---

## 5. Results

### 5.1 Gate 1: the score is a weak ELL predictor — pass

Composite AUC against ELL status: **0.60**, all individual features < 0.65. The gate passes. Per-feature AUCs against ELL status (caliper-matched, n=1,810 pairs, oriented so AUC ≥ 0.5 — some features' raw AUC before orientation is below 0.5, e.g. `type_token_ratio` raw 0.456 → oriented 0.544):

| Feature | AUC | 95% CI |
|---|---|---|
| `mtld` | 0.568 | [0.550, 0.587] |
| `sentence_length_std` | 0.555 | [0.537, 0.574] |
| `function_word_entropy` | 0.560 | [0.542, 0.578] |
| `type_token_ratio` | 0.544 | [0.525, 0.563] |
| `mean_sentence_length` | 0.536 | [0.517, 0.555] |
| `contraction_rate` | 0.539 | [0.520, 0.558] |
| `paragraph_length_variance` | 0.520 | [0.502, 0.540] |
| `transition_phrase_rate` | 0.515 | [0.498, 0.533] |
| `punctuation_variety` | 0.503 | [0.484, 0.522] |
| **Composite (5-fold CV logistic regression)** | **0.601** | [0.583, 0.619] |

We emphasise this result because it is easily mistaken for an absence of bias. An AUC of 0.60 means the score cannot identify ELL writers; it does not mean the score treats them equally. §5.5 shows what a weak tilt does at a strict operating point.

### 5.2 Gate 2: the score is positively correlated with quality — fail

Partial Spearman correlation with holistic quality, controlling word count (n = 24,695):

| Feature | partial ρ | 95% CI |
|---|---|---|
| `transition_phrase_rate` | **+0.147** | [+0.135, +0.159] |
| `sentence_length_std` | **−0.145** | [−0.158, −0.134] |
| `type_token_ratio` | −0.125 | [−0.139, −0.110] |
| `function_word_entropy` | +0.101 | [+0.087, +0.114] |
| `mean_sentence_length` | −0.089 | [−0.102, −0.076] |
| `mtld` | +0.058 | [+0.045, +0.071] |
| `paragraph_length_variance` | +0.022 | [+0.010, +0.035] |
| `punctuation_variety` | −0.017 | [−0.030, −0.004] |
| `contraction_rate` | −0.005 | [−0.017, +0.007] |

Word count alone correlates with holistic score at ρ = 0.76, dominating raw associations; all figures above are partial correlations.

**Composite: partial ρ = +0.135, 95% CI [0.123, 0.148].** Higher-quality essays receive higher AI-likelihood scores.

The relationship is visible directly in the ordering. Binning human essays by their integer holistic score and taking the mean *rank percentile* of composite P(AI) within each bin (word count removed by linear regression before ranking; n=24,695 total):

| Holistic score | n | Mean rank percentile of P(AI) | 95% CI |
|---|---|---|---|
| 2 | 5,590 | 0.440 | [0.432, 0.447] |
| 3 | 7,965 | 0.481 | [0.476, 0.486] |
| 4 | 6,208 | 0.536 | [0.530, 0.541] |
| 5 | 3,066 | 0.557 | [0.551, 0.565] |
| 6 (best) | 842 | 0.587 | [0.575, 0.598] |

The rise is close to monotone and the CIs do not overlap across adjacent steps: as human graders rate essays better, the detector rates them more machine-like, step by step.

**Score 1 (n=1,024) is excluded from this trend and unexplained.** Its mean rank percentile is 0.521 [0.501, 0.542] — not the lowest of the six bins, sitting instead between scores 3 and 4. Its essays are markedly shorter (median 249 words against 824 for score 6). We do not have an account of why it breaks the pattern. An earlier analysis reported score 1 as a large *upward spike in the arithmetic mean* of residualized P(AI) (+0.063 against a −0.012-to-+0.005 band for the rest) and read this as the quality effect being concentrated among the weakest essays; that reading was an artifact of taking arithmetic means of a right-skewed, bounded variable across bins ranging from 842 to 7,965 essays, and has been retracted (`AMENDMENTS.md`, item 6). The rank statistics above do not show that spike — score 1 is anomalous, but not in the direction or the way the retracted reading claimed. We report it as an outlier we cannot currently explain rather than smoothing it into the trend.

Two further observations, and a third on how the relationship behaves under a different estimator. Raw sign is misleading: `transition_phrase_rate` and `sentence_length_std` have opposite raw signs but the *same* directional effect once each is oriented by its coefficient in the composite, since high transition density is an AI-like signal while high burstiness is a human-like one. Both therefore push better essays toward "machine." And raw TTR fails while MTLD, its length-normalised sibling, does not — residual length confounding survives rank-partial correlation, and the length-corrected measure behaves differently.

Third: the effect is monotone but not linear. The same relationship computed as a Pearson partial correlation on the identical sample is **−0.027, 95% CI [−0.046, +0.002]** — smaller, sign-reversed, and with a CI that does not overlap the Spearman estimate's. Spearman is the statistic this study pre-specified, and the appropriate one for an ordinal holistic score; the divergence is diagnostic of a non-linear monotone relationship, not a disagreement between two measurements of the same thing. This is one of three places in this paper where a rank statistic and a linear one disagree about these features — see §7.2.

The composite figure was stable at ρ = +0.135 across three rebuilds with different AI-class compositions (15 DAIGT sources; +Claude; +GPT). It is a property of the features' relationship to human writing quality, not of what is being detected.

### 5.3 Separation performance

Against 17 generator sources: **AUC 0.945** [0.943, 0.947]. At the pre-registered 1% FPR: **TPR 41.3%**.

We stress the gap between these figures. AUC is threshold-free and describes a paired ranking task; deployment requires a single-document decision at a fixed threshold. At an operating point strict enough to be ethically defensible, the majority of machine-generated essays are not detected.

Ratio of the quality effect to the separation effect (2·AUC − 1 = 0.885): **0.152**.

### 5.4 Genre transfer

Same detector, same threshold, human writing in every row:

| Human corpus | Relation to calibration genre | n | FPR | 95% CI |
|---|---|---|---|---|
| PERSUADE | in-domain (calibration) | 24,695 | 1.00% | — |
| ELLIPSE, same 7 prompts | near genre, all ELL | 912 | 2.41% | [1.43, 3.40] |
| ELLIPSE, 37 unseen prompts | near genre, all ELL | 5,570 | 3.73% | [3.23, 4.22] |
| RAID abstracts | far genre | 493 | 36.92% | [32.86, 41.18] |

The degradation is monotone rather than a cliff. A published FPR is a joint property of detector and genre; reported as a scalar it is uninterpretable. Admissions personal statements — the deployment target — are a small, private, largely unscrapeable genre unlikely to appear in any commercial calibration set.

### 5.5 The ELL penalty is real and topic-invariant

Within PERSUADE at the 1% operating point: ELL FPR **1.56%** vs non-ELL **0.94%**. ELLIPSE, an independently constructed corpus of exclusively ELL writers, gives **2.41%** on the identical seven prompts — a larger effect on a dedicated corpus.

Our effect sizes are far below Liang et al.'s 61.22%. We regard this as strengthening rather than weakening the finding: it is a well-calibrated detector at a conservative threshold, on full-length essays, and the gap persists.

**Topic novelty does not explain it.** Holding out four prompts from training entirely and refitting a diagnostic model:

| Cell | n | FPR | 95% CI |
|---|---|---|---|
| A: non-ELL, trained-on prompts | 16,262 | 0.98% | [0.82, 1.13] |
| B: ELL, trained-on prompts | 618 | 1.62% | [0.65, 2.75] |
| C: non-ELL, held-out prompts | 6,189 | 0.84% | [0.61, 1.07] |
| D: ELL, held-out prompts | 1,626 | 1.48% | [0.92, 2.09] |

C is not elevated over A; D is not elevated over B. The compounding gap (observed vs additive prediction for D, log-odds) is 0.06, CI [−0.71, 1.05].

Cell B's interval is wide and overlaps A; the finding rests on D (tighter, same direction) plus the ELLIPSE replication rather than on any single cell. Note the design tests *topic* novelty within a genre, not genre shift — those remain separable, and §5.4's ladder is the genre evidence.

The practical consequence: the penalty tracks the writer, not the prompt, so it cannot be closed by broadening topic coverage in calibration data.

### 5.6 An inverted capability gradient

TPR at the 1% operating point, by generator:

| Source | TPR |
|---|---|
| `gpt_5.6_terra_openai` (current frontier) | 78.1% |
| `claude_sonnet_5_bedrock` (current frontier) | 73.6% |
| DAIGT Claude vintages (Nov 2023) | 68.8%, 70.7% |
| `chat_gpt_moth` | 42.2% |
| Open-weight and older models | 14.9%–41.1% (lowest: `palm-text-bison1` 14.9%; also `kingki19_palm` 21.9%, `cohere-command` 25.4%, `NousResearch/Llama-2-7b-chat-hf` 29.0%, `llama2_chat` 29.4%, `llama_70b_v1` 29.5%, `mistral7binstruct_v2` 30.6%, `mistral7binstruct_v1` 35.8%, `falcon_180b_v1` 38.7%, `radek_500` 40.2%, `mistralai/Mistral-7B-Instruct-v0.1` 41.1%) |

Current frontier models were the most detectable, contradicting an arms-race intuition and our own pre-registered expectation.

Two mechanisms could produce this gradient. Pretraining mixtures skew toward professionally edited prose relative to writing by students in grades 6–12, so a detector separating those classes may partly be learning "does this read like edited adult writing" — which strong student essays also do (H1: pretraining register). Alternatively, post-training compresses model outputs toward a modal register, making frontier output polished *and* homogeneous, where pretraining alone would not explain why frontier models separate from open-weight ones sharing similar pretraining data (H2: post-training homogenization).

We tested both. Neither is established; see §5.10.

Whichever holds, the gradient is fragile: the detectable property is register regularity rather than machine origin, so light editing or an open-weight generator moves text out of the detectable region.

Confound to note: both top sources are reasoning-trained. Frontier capability and reasoning training are not separable in these data.

Within-Claude, TPR on text-independent prompts (75.5%) vs text-dependent (70.1%) differs by 5.4pp, CI [−0.1, 11.2] — not distinguishable from zero, and in the opposite direction to the confound hypothesis it was run to test.

### 5.7 Adversarial robustness, and a benchmark artifact

On RAID (llama-chat, n = 107 per attack; note these figures sit at a domain-mismatched operating point per §5.4 and are directional only):

| Attack | TPR | Δ vs baseline |
|---|---|---|
| paraphrase | 0.308 | −21.5 pts |
| insert_paragraphs | 0.458 | −6.5 pts |
| none (baseline) | 0.523 | — |
| alternative_spelling, perplexity_misspelling | 0.523 | 0 |
| synonym, article_deletion | 0.56–0.57 | +4 (noise) |
| homoglyph, zero_width_space | 1.000 | *artifact — see below* |

**The unicode results are tokenizer failure, not evasion.** Human FPR under the same two attacks rose to **99.8% [99.4, 100] under homoglyph and 100.0% [100, 100] under zero-width space** (n=493; both essentially at ceiling). Diagnosis, confirmed by recomputing features directly on a matched document pair: homoglyph substitution replaces Latin characters with Cyrillic/Greek lookalikes, which our ASCII word-matching drops entirely (`function_word_entropy` 3.75 → 0.00, TTR 0.62 → 0.19, MTLD 80.2 → 17.0); zero-width insertion destroys word and sentence boundaries (`mean_sentence_length` 35 → 1246). Features saturate to degenerate values that happen to point toward "machine" under our coefficients, but could point either way under different ones.

This has a deployment implication beyond adversarial framing. Zero-width characters and homoglyphs enter student documents *accidentally* — via web paste, PDF extraction, or a collaborator's editor. A detector whose feature extraction is not unicode-normalised can therefore flag a student for characters they never typed. We recommend normalisation before feature extraction as a hardening step.

### 5.8 The features do not carry quality either

Nested models predicting holistic score: M0 = word count; M1 = word count + 8 features; M2 = 8 features alone, no word count. Raw `type_token_ratio` is dropped for this comparison, leaving 8 of the 9 Tier-1 features — a decision carried over directly from §5.2's finding: raw TTR's partial correlation with quality does not fully scrub a residual length confound the way MTLD, its length-normalized sibling, does. Feeding raw TTR into a quality model would risk crediting the features with predictive power that is actually leftover word-count signal leaking through an imperfectly-controlled feature, which is exactly the failure mode this section exists to rule out.

Absolute figures (RidgeCV, dev n=19,756, 5-fold CV within dev; held-out n=4,939 scored once, not shown below):

| Model | R² | 95% CI | Spearman ρ (vs. holistic score) | 95% CI |
|---|---|---|---|---|
| M0 — word count only | 0.194 | [0.076, 0.276] | 0.754 | [0.747, 0.762] |
| M1 — word count + 8 features | 0.353 | [0.292, 0.397] | 0.675 | [0.667, 0.683] |
| M2 — 8 features only | 0.260 | [0.248, 0.271] | 0.520 | [0.509, 0.530] |

**M1 − M0 incremental ρ = −0.079, 95% CI [−0.086, −0.072]** (dev, OOF), replicated on held-out data (−0.082, CI [−0.096, −0.068]). The features do not merely fail to add signal above length — they add noise.

Note the direction of that trade: M1 *improves* linear fit over M0 (R² 0.194 → 0.353) while its rank correlation *falls* (ρ 0.754 → 0.675). The features add linear explanatory power and degrade ordering; the negative incremental Spearman ρ above is exactly that trade, not a contradiction of it. M0's own R² interval, [0.076, 0.276], is also markedly wider than M2's, [0.248, 0.271], indicating unstable cross-fold behaviour for word count alone as a linear predictor — a second reason to trust the rank figures over the R² figures here. See §7.2.

This closes a secondary hypothesis: that features unusable for detection might still condition writing feedback. They cannot.

### 5.9 DAIGT-v2 carries no prompting-effort scaffolding

A premise behind interpreting §5.6's gradient was that DAIGT-v2's contributed generations might be more "deliberately student-like" than our own generic PERSUADE-prompt generations, confounding vendor identity with prompting effort. We pre-registered testing this via a matched prompting-effort ladder (L0 naive, L1 verbatim task, L2 "DAIGT-replicated," L3 adversarial paraphrase) before generating anything, per this study's own sourcing constraint: do not compose a level's prompt from assumption — trace what was actually done, and if a level can't be sourced, say so and stop rather than invent it (`EXPERIMENT_5.md`).

Tracing DAIGT-v2's documented generation methodology across five of its constituent source datasets — darraghdog's Claude set (via its linked discussion thread and raw CSV), `chat_gpt_moth`, `radek1`, nbroad's Llama/Falcon set, and kingki19's PaLM set (including its linked Colab notebook containing the actual prompt-construction code) — found that **no contributor used anything beyond an L1-equivalent prompt**: the raw or lightly-wrapped assignment text, with no persona, word-count target, or student-voice instruction anywhere. The premised L2 condition does not exist in the source data.

This is itself a finding, not merely a blocker to the experiment it was meant to enable: the assumption that DAIGT contributors "deliberately constructed student-like text" while our own prompting was comparatively generic does not hold up. Two consequences follow. First, the confound this experiment was designed to isolate is smaller than assumed — there is little prompting-effort gap between our own generations and DAIGT's — which if anything *strengthens* §5.6's gradient rather than weakening it, since the capability gradient survives with prompting effort held approximately constant across sources. Second, detector performance measured on DAIGT-v2 does not characterise behaviour against carelessly or carefully prompted generations, since the benchmark contains only one point on that axis; we note this for others using the corpus. The reduced experiment (L0, L1, L3 only, sourcing L3 from Lu et al. 2024) was piloted but not carried to a reported conclusion once §5.2's result made the underlying vendor-vs-effort question moot for this paper's argument — the composite fails Gate 2 regardless of which vendor is behind any given source; see `AMENDMENTS.md` item 4 for the full trace, including which datasets and links were checked. The experiment was shelved (`EXPERIMENT_5.md`).

### 5.10 Mechanism: dispersion and geometry (exploratory)

This analysis was added after §5.6 produced an unexpected gradient. It was not part of the original pre-registration and is exploratory throughout; it does not reopen §5.1–5.2's verdicts, which were already decided before it ran.

**Dispersion.** Measuring across-document dispersion in standardised nine-dimensional feature space, per source, the pre-registered ordering held exactly: frontier proprietary (1.845) < older proprietary (2.060) < open-weight (2.421) < human (2.422). Four groups landing in a predicted sequence has a chance probability of roughly 1/24 (4! orderings), and the groups are not independent samples of the same underlying process, so this is suggestive rather than strong evidence on its own.

The pre-registered test was the correlation between per-source dispersion and per-source TPR, with a threshold of |ρ| > 0.5 fixed in advance. Observed: **ρ = −0.321, 95% CI [−0.738, 0.253]** (n=17 sources). We report this as *underpowered rather than negative*: the CI spans nearly the full [−1, 1] range on one side of zero, and a threshold of this magnitude was fixed without a power calculation for n=17 — it could not have been cleared unless the true effect were very large. This is a second instance of the pre-registration defect recorded for gate 2 (§6).

**Centroid distance** from the human corpus predicted TPR more strongly: **ρ = 0.775, 95% CI [0.348, 0.957]**. We report this as a validity check rather than mechanism evidence: the composite is a fitted discriminant, so sources far from the human centroid are, by construction, the ones it separates well from human writing — both H1 and H2 predict this, so it does not distinguish between them.

**H1's specific prediction fails.** If detectability reflects displacement toward edited-adult register, frontier centroids should sit displaced toward the high-quality human region specifically, not merely away from the human centroid in general. Projecting each source onto the human-mean-to-top-quartile-quality vector and correlating with TPR gives **ρ = −0.174, 95% CI [−0.614, 0.288]** — null and wrong-signed. Both the discriminant-relevant human centroid and the per-source projections here are computed in a length-residualized space (word count regressed out before projecting), so this result is not subject to the conditioning issue that affected the raw cosine test below.

**Geometry of the two directions.** The angle between the fitted discriminant direction and the human-mean-to-top-quartile quality direction, both expressed in the same standardized space and both residualized on word count: **cosine 0.533, angle 57.8°**. For reference, two random vectors in nine dimensions have cosine similarity with SD ≈ 1/√9 ≈ 0.33, so this is well outside chance. The two directions are partially aligned — substantially more than chance, clearly short of identical.

An earlier computation of this same angle, before both vectors were conditioned on word count consistently, gave cosine 0.067 (86.2°) and was read as near-orthogonality. That comparison used a length-inclusive discriminant against a length-inclusive quality vector, while §5.2's correlation removes word count — an inconsistency, not a second independent result — and does not survive consistent conditioning. It is retracted (`AMENDMENTS.md`, item 6).

**What this supports.** Partial alignment is the more useful result than either orthogonality or identity would have been. It says the bias is not reducible to "the detector measures quality" — §5.8 shows these features cannot predict quality at all — while still explaining a systematic tilt against better essays. There is no single quality-correlated feature to drop; the overlap is in the fitted direction itself, not in any one dimension.

**What this does not support.** We cannot adjudicate H1 against H2 from this analysis. The dispersion test was underpowered, H1's direction check failed on its own terms, and centroid distance does not distinguish the two hypotheses. The mechanism is open.

**Limitation.** All three "vs. TPR" correlations in this section (dispersion, centroid distance, and the H1 direction check) compare a length-controlled predictor against an outcome, TPR, computed from the composite's original, non-length-residualised threshold. Correcting this fully would require re-deriving a threshold from a residualised-feature discriminant and rescoring every source. We did not pursue it: this section is exploratory, n=17 limits what any correlation here can establish, and no conclusion elsewhere in this paper rests on it.

---

## 6. Pre-registration and failed predictions

We pre-registered gates and directional expectations before data collection. At least seven expectations failed — four from the original plan, three more from the exploratory mechanism analysis in §5.10:

1. **Current Claude would be difficult to detect.** It was the second-most detectable of 17 sources (§5.6).
2. **DAIGT contributors used elaborate student-simulating prompt scaffolding.** Tracing five constituent source datasets — including a contributor's published generation notebook — found all are essentially raw-assignment prompting. The prompting-effort experiment premised on this was shelved (§5.9; `EXPERIMENT_5.md`).
3. **ELL status and genre shift would compound.** They did not (§5.5).
4. **The features would carry ~0.2–0.3 incremental quality signal.** The delta was negative (§5.8).
5. **Per-source dispersion would predict TPR at |ρ| > 0.5.** Observed ρ = −0.321; underpowered rather than negative (§5.10).
6. **Frontier centroids would sit displaced toward the high-quality human region.** Null and wrong-signed (§5.10).
7. **The quality effect would concentrate among the strongest essays.** An early analysis appeared to show the opposite — concentration among the weakest — but that reading was an artifact of arithmetic means on a skewed, bounded variable; a rank-based reanalysis shows the effect distributed across scores 2–6, not concentrated at either end (§5.2).

We also record three pre-registration defects. Gate 2 was specified without a numeric threshold, unlike gate 1 — so its "failure" rests on a CI excluding zero rather than on a magnitude fixed in advance, which is weaker than we intended. Gate 2 was initially written per-feature rather than on the composite. And a third: the |ρ| > 0.5 threshold in §5.10 was fixed without a power calculation at n = 17, where it could not have been cleared by any but a very large effect. This is the same failure mode as gate 2's missing threshold — a bound set without checking whether the test could reach it. All three are documented in `AMENDMENTS.md`.

The repository was published as a single import from a working directory; git history therefore does not independently corroborate the pre-registration timeline. The only timestamp evidence available is filesystem modification times on the machine where the work was done — `RESEARCH_PLAN.md` and `CLAUDE.md` dated 2026-08-13; `EXPERIMENT_5.md` dated 2026-08-14 11:35 and `EXPERIMENT_6.md` dated 2026-08-14 12:30 — recorded in `AMENDMENTS.md`'s closing section, "On this repository's single commit." That section is explicit that this is the authors' own record, not independently checkable by a reader, since neither git nor GitHub preserve filesystem mtimes through a commit; readers should treat it as self-reported process documentation, not cryptographic proof.

---

## 7. Discussion

### 7.1 One axis, read in two directions

The AES literature treats lexical diversity and syntactic variation as indices of writing proficiency. The detection literature treats reduced variation as evidence of machine origin. Our results indicate these describe a single underlying axis — register regularity — with opposite valence assigned to the same direction.

If that is right, an accuracy improvement in uniformity-based detection is not straightforwardly desirable: a better detector along this axis is a better proxy for polish, and polish is produced both by frontier models and by competent, careful, or heavily-drilled human writers. The failure mode is structural rather than a calibration deficiency.

### 7.2 A recurring rank–linear divergence

Three separate analyses in this paper show these features behaving differently under rank and linear treatment. The quality correlation is +0.135 by Spearman and −0.027 by Pearson on the same sample (§5.2). Adding the features to a word-count baseline *improves* linear fit while *degrading* rank ordering — R² 0.194 → 0.353, Spearman 0.754 → 0.675 (§5.8) — which is what produces the negative incremental delta. And an early analysis using arithmetic means of a right-skewed score reversed the sign of a relationship that rank statistics show clearly (§5.2, retracted).

The consistent reading is that the feature–quality relationship is monotone but strongly non-linear, so estimators assuming linearity mis-state both magnitude and sign. For this study the practical implication is narrow: report rank statistics, pre-specify them, and treat linear summaries of these features as unreliable. For others building on this feature family, it is a caution about the default choice of estimator.

### 7.3 A remediation that harms

The standard advice for avoiding a flag — vary sentence length, break rhythm, add irregularity — moves writing in the direction our data associates with *lower* rated quality. Detection regimes therefore impose a cost on honest writers that is not merely the risk of false accusation but a pressure to write worse. Our own withdrawn product issued exactly this advice.

### 7.4 Appropriateness auditing vs performance monitoring

The deployed feature's detection metrics were satisfactory throughout. The defect lived in the *advice* attached to a detection, which no detection metric evaluates. It surfaced through a scheduled audit asking whether outputs were appropriate — correct, suitable, sound to act on — rather than whether the model was accurate.

We suggest the general form: any deployed system producing recommendations rather than classifications requires evaluation of the recommendations against the outcome the user seeks, on a schedule, by a human, with the output surface rather than the model as the unit of review. A system can pass performance monitoring indefinitely while failing this, because nothing in the former examines the latter.

### 7.5 Provenance signals

Cryptographic and statistical provenance marking by model providers avoids the failure modes here: it does not correlate with quality, does not disadvantage any writer population, and creates no incentive to write badly. It is however partial — covering only participating providers, likely requiring provider-side keys for detection, and yielding nothing for open-weight models or paraphrased text. It is a narrow high-precision signal, not a replacement.

---

## 8. Limitations

**Genre.** PERSUADE is grades 6–12 source-based argumentative writing; the deployment target is 300–650 word first-person personal statements. Given §5.4, our specific figures should not be assumed to transfer. We used PERSUADE because no public corpus of admissions essays carries both quality scores and ELL annotation. The mechanisms are properties of the feature family; the magnitudes are not portable.

**We did not evaluate commercial detectors.** Turnitin's and GPTZero's classifiers are proprietary. We constrain the method, not any vendor's implementation. A vendor may have addressed these problems. Turnitin has published a study of ELL bias in its own detector (§1); we are not aware of published vendor evidence addressing genre transfer or the quality inversion reported here.

**Single-generator adversarial evidence.** §5.7 rests on llama-chat, at a domain-mismatched operating point.

**ELLIPSE is a proficiency-assessment instrument**, not classroom writing, which may explain its elevated FPR relative to PERSUADE's ELL subset. Neither is admissions writing, and we cannot say which is closer.

**Language specificity.** The Czech replication cited in §2 finds the opposite entropy relationship for non-native writers, indicating these effects are language-dependent. Our results speak to English.

**Conflict of interest.** MaiaLearning sells essay-review software. The detector studied here was built by us, deployed by us, and withdrawn by us on the basis of these results. We had a commercial interest in the feature succeeding. Readers should weigh that in both directions: against the possibility of motivated reporting, and in light of the fact that the reported conclusion cost us a shipped feature.

---

## 9. Conclusion

A competent uniformity-based detector, evaluated at an operating point strict enough to be defensible, detects a minority of machine-generated essays, scores better human writing as more machine-like, penalises English language learners in a way no configuration change removes, and cannot be transferred across genre without recalibration nobody performs. We recommend that detector scores not be used as evidence in admissions or academic-integrity decisions, and that institutions ask vendors for the genre on which their false-positive rate was measured, its subgroup breakdown, and what post-deployment appropriateness review they conduct.

---

## References

*[RESOLVED, with two flagged corrections below. All DOIs verified against publisher/ACL Anthology/arXiv records at resolution time; none fabricated — where a venue does not assign a DOI (the two ICML/PMLR papers), that is stated instead of inventing one.]*

Casal, E. J., & Lee, J. J. (2019). Syntactic complexity and writing quality in assessed first-year L2 writing. *Journal of Second Language Writing*, 44, 51–62. https://doi.org/10.1016/j.jslw.2019.03.005

Crossley, S. A. (2020). Linguistic features in writing quality and development: An overview. *Journal of Writing Research*, 11(3), 415–443. https://doi.org/10.17239/jowr-2020.11.03.01

Crossley, S. A., Baffour, P., Benner, M., Boser, U., Franklin, A., & Tian, Y. (2024). A large-scale corpus for assessing written argumentation: PERSUADE 2.0. *Assessing Writing*, 61, Article 100865. https://doi.org/10.1016/j.asw.2024.100865 [Primary corpus citation — corrected from an earlier draft that cited only the 1.0 paper below; this study depends on the holistic scores and ELL annotation that 2.0 added.]

Crossley, S. A., Baffour, P., Tian, Y., Picou, A., Benner, M., & Boser, U. (2022). The persuasive essays for rating, selecting, and understanding argumentative and discourse elements (PERSUADE) corpus 1.0. *Assessing Writing*, 54, Article 100667. https://doi.org/10.1016/j.asw.2022.100667 [Lineage reference: the precursor corpus PERSUADE 2.0 builds on, kept here to credit its full history; not the version this study's data comes from.]

Deep, P. D., Edgington, W. D., Ghosh, N., & Rahaman, M. S. (2025). Evaluating the effectiveness and ethical implications of AI detection tools in higher education. *Information*, 16(10), Article 905. https://doi.org/10.3390/info16100905

Dugan, L., Hwang, A., Trhlík, F., Zhu, A., Ludan, J. M., Xu, H., Ippolito, D., & Callison-Burch, C. (2024). RAID: A shared benchmark for robust evaluation of machine-generated text detectors. *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 12463–12492. https://doi.org/10.18653/v1/2024.acl-long.674 (arXiv:2405.07940)

Elkhatat, A. M., Elsaid, K., & Almeer, S. (2023). Evaluating the efficacy of AI content detection tools in differentiating between human and AI-generated text. *International Journal for Educational Integrity*, 19, Article 17. https://doi.org/10.1007/s40979-023-00140-5

Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., & Goldstein, T. (2024). Spotting LLMs with Binoculars: Zero-shot detection of machine-generated text. *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:17519–17537. No DOI assigned (PMLR does not issue them). arXiv:2401.12070

Herbold, S., Hautli-Janisz, A., Heuer, U., Kikteva, Z., & Trautsch, A. (2023). AI, write an essay for me: A large-scale comparison of human-written versus ChatGPT-generated essays. arXiv:2304.14276. [Confirmed: ID matches title/content. Subsequently published in *Scientific Reports*; journal DOI not independently confirmed here, so not stated to avoid guessing one.]

Liang, W., Yuksekgonul, M., Mao, Y., Wu, E., & Zou, J. (2023). GPT detectors are biased against non-native English writers. *Patterns*, 4(7), Article 100779. https://doi.org/10.1016/j.patter.2023.100779 (arXiv:2304.02819)

McCarthy, P. M., & Jarvis, S. (2010). MTLD, vocd-D, and HD-D: A validation study of sophisticated approaches to lexical diversity assessment. *Behavior Research Methods*, 42(2), 381–392. https://doi.org/10.3758/BRM.42.2.381

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., & Finn, C. (2023). DetectGPT: Zero-shot machine-generated text detection using probability curvature. *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202:24950–24962. No DOI assigned. arXiv:2301.11305

Perkins, M., Roe, J., Vu, B. H., Postma, D., Hickerson, D., & McGaughran, J. (2024). Simple techniques to bypass GenAI text detectors: Implications for inclusive education. *International Journal of Educational Technology in Higher Education*, 21, Article 53. https://doi.org/10.1186/s41239-024-00487-w

Pratama, A. R. (2025). The accuracy-bias trade-offs in AI text detection tools and their impact on fairness in scholarly publication. *PeerJ Computer Science*, 11, e2953. https://doi.org/10.7717/peerj-cs.2953

Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., & Feizi, S. (2024). Can AI-generated text be reliably detected? arXiv:2303.11156. [Confirmed: ID matches title/content; first submitted March 2023, revised through 2025 — cited year is a defensible but not the only choice.]

Turnitin (2023). New research: Turnitin's AI detector shows no statistically significant bias against English Language Learners. Turnitin Blog, October 2023. https://www.turnitin.com/blog/new-research-turnitin-s-ai-detector-shows-no-statistically-significant-bias-against-english-language-learners [Note on reference type: a blog post, not a peer-reviewed source — cited only for the narrow, verifiable claim that Turnitin's own detector showed no significant ELL bias on its own test sample, not as methodological authority over Liang et al. The drafted arXiv ID this entry originally carried, 2312.05241, resolved to an unrelated paper and has been removed; see §1 for the fuller trace of a related misattribution this search corrected.]

Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltýnek, T., Guerrero-Dib, J., Popoola, O., Šigut, P., & Waddington, L. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity*, 19, Article 26. https://doi.org/10.1007/s40979-023-00146-z

Al Ali, A., Helcl, J., & Libovický, J. (2026). Different time, different language: Revisiting the bias against non-native speakers in GPT detectors. *Proceedings of the EACL 2026 Student Research Workshop*. https://aclanthology.org/2026.eacl-srw.20/ (arXiv:2602.05769) [Confirmed real: exists, matches the title and Czech-language non-native-bias-replication description used in §2; authors added since the in-text mention only said "a Czech-language replication."]
