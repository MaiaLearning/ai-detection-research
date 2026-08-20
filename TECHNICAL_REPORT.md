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

Detectors of machine-generated text are widely deployed in education, and their error properties are contested. Liang et al. (2023) found that perplexity-based detectors misclassify a majority of TOEFL essays by non-native writers as machine-generated while remaining near-perfect on essays by US-born eighth-graders, and attributed this to lower perplexity arising from reduced linguistic variability. Turnitin disputed the methodology, citing short texts and small samples, and noting their own detector was not among those tested (Turnitin, 2023).

We approached the question as practitioners rather than as critics. MaiaLearning deployed an AI-detection panel in a commercial college-essay review product. A scheduled audit of that feature's *output appropriateness* — distinct from monitoring its performance — found that the guidance it issued to students appeared to advise writing badly. This paper is the investigation that followed, and the evidence on which the feature was withdrawn.

**Contributions.**

1. **A quality–detection inversion.** On 24,695 human essays with holistic scores, detector score correlates positively with quality. To our knowledge this connection between detector features and independently-rated writing quality has not previously been measured, despite both feature families being well-studied in isolation (§2, §5.2) [FLAGGED: cited "§3.2" in the current text, but §3 (Data) has no subsections — corrected here to §2, which is where the AES/detection literature gap is actually discussed ("The gap we address"); §5.2 is correct as-is.].
2. **A quantified genre-transfer failure.** A calibration ladder from 1% to 36.9% FPR on human writing under a fixed threshold (§5.4).
3. **A topic-invariant ELL penalty**, replicated across corpora, with a 2×2 test excluding topic novelty as the mechanism (§5.5).
4. **An inverted capability gradient.** Current frontier models were the *most* detectable of 17 sources; open-weight and older models the least (§5.6).
5. **A methodological negative result** on DAIGT-v2, a widely-used detection benchmark: its constituent generations carry no prompting-effort scaffolding, contrary to common assumption (§6 [FLAGGED: cited here and at its other occurrence, in §6 item 2, as "§5.7" — but §5.7's actual heading is "Adversarial robustness, and a benchmark artifact" (RAID/unicode), an unrelated finding. The DAIGT-scaffolding result has no numbered §5 subsection of its own; it appears only in §6 and `AMENDMENTS.md` item 4. Both citations need correcting to §6, or the finding needs its own §5 subsection.]).
6. **A deployment-process finding**: the defect was invisible to performance metrics and surfaced only through appropriateness review (§7.3).

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
| DAIGT-v2 | Machine-generated, 15 sources [FLAGGED: table and §5.2/§3.2 text say "14"; the manifest lists 15 distinct DAIGT source labels used here (excludes our own Bedrock/OpenAI generations) — see note below table] | 17,492 [RESOLVED] | Generator identity |
| Bedrock generations | Machine, current Claude | 1,000 | Prompt, model ID |
| OpenAI generations | Machine, current GPT | 1,000 | Prompt, model ID |
| ELLIPSE | Near-genre human, all ELL | 6,482 | Proficiency ratings, prompt |
| RAID (abstracts subset) | Far-genre human + adversarial | 493 human | Attack type, generator |

**PERSUADE 2.0** comprises argumentative essays by US students in grades 6–12, collected before the release of ChatGPT, making human authorship reliable. It carries holistic quality scores and demographic annotations including ELL status. [RESOLVED] Filtering criteria (`src/data.py::load_and_clean`), applied identically by every script in this study: of 25,996 raw rows, a row is kept only if all of `ell_status` (recoded to a clean Yes/No flag, dropping any other value), `grade_level`, `holistic_essay_score`, `word_count`, `prompt_name`, and `full_text` are non-null. This drops 1,301 rows (5.0%), leaving n = 24,695 (2,244 ELL, 22,451 non-ELL). No length or other content filter is applied to the human corpus.

[FLAGGED — not a [VERIFY] item, found while resolving the one above: DAIGT-v2's own generator-source column contains 15 distinct labels in the corpus subset used here (`NousResearch/Llama-2-7b-chat-hf`, `chat_gpt_moth`, `cohere-command`, `darragh_claude_v6`, `darragh_claude_v7`, `falcon_180b_v1`, `kingki19_palm`, `llama2_chat`, `llama_70b_v1`, `mistral7binstruct_v1`, `mistral7binstruct_v2`, `mistralai/Mistral-7B-Instruct-v0.1`, `palm-text-bison1`, `radek_500`, `radekgpt4`), not 14 as stated in the Data table above, the Contributions list ("14 DAIGT sources; +Claude; +GPT," §... near the abstract), and `scripts/experiment3_separation.py`'s own docstring. This is a genuine miscount that predates this report — it isn't a transcription slip introduced here — and appears in at least three places. Needs an authorial decision: correct all three instances to 15, or, if "14" was intended to mean distinct model *families* rather than distinct dataset source-labels (e.g. treating `darragh_claude_v6`/`darragh_claude_v7` as one "Claude" family), state that grouping explicitly, since it doesn't match the corpus's own source cardinality.]

**Generated sets.** We generated 1,000 essays with `us.anthropic.claude-sonnet-5` via AWS Bedrock ($8.97) and 1,000 with `gpt-5.6-terra` via the OpenAI API, both on PERSUADE prompts at temperature 1.0. Two limitations are logged: the generation prompt was generic rather than production's system prompt, and for the seven text-dependent PERSUADE prompts the models wrote from general knowledge, since the corpus carries citations but not source article text.

Corpus SHA-256 digests are published in the repository for version verification. PERSUADE and ELLIPSE are CC BY-NC-SA 4.0 and are not redistributed.

---

## 4. Method

### 4.1 Features

Nine deterministic features, computed in pure Python with no model dependency, no network call, and no stochasticity:

`sentence_length_std` (burstiness), `mean_sentence_length`, `type_token_ratio`, `mtld`, `transition_phrase_rate`, `paragraph_length_variance`, `punctuation_variety`, `contraction_rate`, `function_word_entropy`.

We compute both raw TTR and MTLD deliberately: raw TTR is length-confounded, and the divergence between them is informative (§5.2). [RESOLVED] The transition-phrase lexicon (`src/features.py::TRANSITION_PHRASES`) is a fixed, self-authored list of 34 discourse markers (e.g. "on the other hand," "for example," "therefore," "furthermore," "first"/"second"/"third"), matched as whole-word/whole-phrase regex patterns against lowercased text; it is not adapted from an external published list, and the code cites no source for it.

### 4.2 Composite

[RESOLVED] `StandardScaler` + `LogisticRegression` (scikit-learn defaults except `max_iter=1000`; no hyperparameter search), fit on all nine features with none pre-dropped, seed = 42. All reported scores are out-of-fold, 5-fold stratified CV (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

[FLAGGED — the second sentence does not describe this model.] "Feature-dropping decisions were made on dev; reported figures are held-out" is not true of the detection composite above: it uses all 9 features (none dropped) and a pure 5-fold CV design with no dev/held-out split at all. That sentence describes a *different* model — the RidgeCV quality-prediction model in §5.8 (`scripts/experiment6_quality_composite.py`), which does drop raw `type_token_ratio` (using 8 of the 9 features) and does use an 80/20 dev/held-out split (dev n=19,756, held-out n=4,939, stratified by ELL status, seed 42). §4.2 appears to have merged the two models' methodology sections. Needs a rewrite that either (a) states §4.2 is about the detection composite only and moves the dropped-sentence to §5.8, or (b) splits §4.2 into two subsections, one per model.

### 4.3 Pre-registered gates

Fixed before data collection (`RESEARCH_PLAN.md`):

- **Gate 1 — ELL predictability.** AUC of the score against ELL status must be < 0.65. Rationale: a score that identifies ELL writers is measuring language background, not authorship.
- **Gate 2 — quality anti-correlation.** The score must not be meaningfully anti-correlated with rated quality. *We did not fix a numeric threshold for gate 2. This is a pre-registration defect and we report it as such* (§6).
- **Operating point.** 1% FPR, chosen on the grounds that a false accusation is not easily rebutted by a student.

Gate 2 was initially specified per-feature and reapplied to the composite during analysis; the composite is the correct level, since it is what a user would see. This and other mid-study changes are recorded in `AMENDMENTS.md`.

---

## 5. Results

### 5.1 Gate 1: the score is a weak ELL predictor — pass

Composite AUC against ELL status: **0.60**, all individual features < 0.65. The gate passes. [RESOLVED] Per-feature AUCs against ELL status (caliper-matched, n=1,810 pairs, oriented so AUC ≥ 0.5 — some features' raw AUC before orientation is below 0.5, e.g. `type_token_ratio` raw 0.456 → oriented 0.544):

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

Two observations. Raw sign is misleading: `transition_phrase_rate` and `sentence_length_std` have opposite raw signs but the *same* directional effect once each is oriented by its coefficient in the composite, since high transition density is an AI-like signal while high burstiness is a human-like one. Both therefore push better essays toward "machine." And raw TTR fails while MTLD, its length-normalised sibling, does not — residual length confounding survives rank-partial correlation, and the length-corrected measure behaves differently.

The composite figure was stable at ρ = +0.135 across three rebuilds with different AI-class compositions (15 DAIGT sources [FLAGGED: see the source-count note in §3]; +Claude; +GPT). It is a property of the features' relationship to human writing quality, not of what is being detected.

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
| Open-weight and older models | 14.9%–41.1% [RESOLVED] (lowest: `palm-text-bison1` 14.9%; also `kingki19_palm` 21.9%, `cohere-command` 25.4%, `NousResearch/Llama-2-7b-chat-hf` 29.0%, `llama2_chat` 29.4%, `llama_70b_v1` 29.5%, `mistral7binstruct_v2` 30.6%, `mistral7binstruct_v1` 35.8%, `falcon_180b_v1` 38.7%, `radek_500` 40.2%, `mistralai/Mistral-7B-Instruct-v0.1` 41.1%) |

Current frontier models were the most detectable, contradicting an arms-race intuition and our own pre-registered expectation.

We propose **convergence on polish** as the mechanism, consistent with §5.2: the features measure uniformity; heavily post-trained frontier models produce highly regular prose; so does competent human writing. The detector is not measuring machine origin but register regularity, and frontier models occupy the same region of that axis as strong human writers. This makes the gradient fragile — light editing or an open-weight generator moves text out of the detectable region.

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

**The unicode results are tokenizer failure, not evasion.** Human FPR under the same two attacks also rose to ~100%. Diagnosis: homoglyph substitution replaces Latin characters with Cyrillic/Greek lookalikes, which our ASCII word-matching drops entirely (`function_word_entropy` → 0.0, TTR 0.62 → 0.19, MTLD 80 → 17); zero-width insertion destroys word and sentence boundaries (`mean_sentence_length` → 1246). Features saturate to degenerate values that happen to point toward "machine" under our coefficients, but could point either way under different ones.

This has a deployment implication beyond adversarial framing. Zero-width characters and homoglyphs enter student documents *accidentally* — via web paste, PDF extraction, or a collaborator's editor. A detector whose feature extraction is not unicode-normalised can therefore flag a student for characters they never typed. We recommend normalisation before feature extraction as a hardening step and note this as a false-positive mechanism absent from the literature.

### 5.8 The features do not carry quality either

Nested models predicting holistic score: M0 = word count; M1 = word count + 8 features [FLAGGED: this and the next figure both say "9 features" in the current text, but `scripts/experiment6_quality_composite.py` drops raw `type_token_ratio` for this comparison — "residual length confound per Experiment 2's partial-correlation gate" — leaving 8 of the 9 Tier-1 features. Corrected to 8 here; the M0/M1/M2 figures below are computed on that 8-feature set, not 9.]; M2 = 8 features alone, no word count.

[RESOLVED] Absolute figures (RidgeCV, dev n=19,756, 5-fold CV within dev; held-out n=4,939 scored once, not shown below):

| Model | R² | 95% CI | Spearman ρ (vs. holistic score) | 95% CI |
|---|---|---|---|---|
| M0 — word count only | 0.194 | [0.076, 0.276] | 0.754 | [0.747, 0.762] |
| M1 — word count + 8 features | 0.353 | [0.292, 0.397] | 0.675 | [0.667, 0.683] |
| M2 — 8 features only | 0.260 | [0.248, 0.271] | 0.520 | [0.509, 0.530] |

**M1 − M0 incremental ρ = −0.079, 95% CI [−0.086, −0.072]** (dev, OOF), replicated on held-out data (−0.082, CI [−0.096, −0.068]). The features do not merely fail to add signal above length — they add noise.

This closes a secondary hypothesis: that features unusable for detection might still condition writing feedback. They cannot.

---

## 6. Pre-registration and failed predictions

We pre-registered gates and directional expectations before data collection. Four expectations failed:

1. **Current Claude would be difficult to detect.** It was the second-most detectable of 17 sources (§5.6).
2. **DAIGT contributors used elaborate student-simulating prompt scaffolding.** Tracing five constituent source datasets — including a contributor's published generation notebook — found all are essentially raw-assignment prompting. The prompting-effort experiment premised on this was shelved (see the flagged cross-reference at this finding's other citation, in the Contributions list; `EXPERIMENT_5.md`).
3. **ELL status and genre shift would compound.** They did not (§5.5).
4. **The features would carry ~0.2–0.3 incremental quality signal.** The delta was negative (§5.8).

We also record two pre-registration defects. Gate 2 was specified without a numeric threshold, unlike gate 1 — so its "failure" rests on a CI excluding zero rather than on a magnitude fixed in advance, which is weaker than we intended. And gate 2 was initially written per-feature rather than on the composite. Both are documented in `AMENDMENTS.md`.

The repository was published as a single import from a working directory; git history therefore does not independently corroborate the pre-registration timeline. [RESOLVED] The only timestamp evidence available is filesystem modification times on the machine where the work was done — `RESEARCH_PLAN.md` and `CLAUDE.md` dated 2026-08-13; `EXPERIMENT_5.md` dated 2026-08-14 11:35 and `EXPERIMENT_6.md` dated 2026-08-14 12:30 — recorded in `AMENDMENTS.md`'s closing section, "On this repository's single commit." That section is explicit that this is the authors' own record, not independently checkable by a reader, since neither git nor GitHub preserve filesystem mtimes through a commit; readers should treat it as self-reported process documentation, not cryptographic proof.

---

## 7. Discussion

### 7.1 One axis, read in two directions

The AES literature treats lexical diversity and syntactic variation as indices of writing proficiency. The detection literature treats reduced variation as evidence of machine origin. Our results indicate these describe a single underlying axis — register regularity — with opposite valence assigned to the same direction.

If that is right, an accuracy improvement in uniformity-based detection is not straightforwardly desirable: a better detector along this axis is a better proxy for polish, and polish is produced both by frontier models and by competent, careful, or heavily-drilled human writers. The failure mode is structural rather than a calibration deficiency.

### 7.2 A remediation that harms

The standard advice for avoiding a flag — vary sentence length, break rhythm, add irregularity — moves writing in the direction our data associates with *lower* rated quality. Detection regimes therefore impose a cost on honest writers that is not merely the risk of false accusation but a pressure to write worse. Our own withdrawn product issued exactly this advice.

### 7.3 Appropriateness auditing vs performance monitoring

The deployed feature's detection metrics were satisfactory throughout. The defect lived in the *advice* attached to a detection, which no detection metric evaluates. It surfaced through a scheduled audit asking whether outputs were appropriate — correct, suitable, sound to act on — rather than whether the model was accurate.

We suggest the general form: any deployed system producing recommendations rather than classifications requires evaluation of the recommendations against the outcome the user seeks, on a schedule, by a human, with the output surface rather than the model as the unit of review. A system can pass performance monitoring indefinitely while failing this, because nothing in the former examines the latter.

### 7.4 Provenance signals

Cryptographic and statistical provenance marking by model providers avoids the failure modes here: it does not correlate with quality, does not disadvantage any writer population, and creates no incentive to write badly. It is however partial — covering only participating providers, likely requiring provider-side keys for detection, and yielding nothing for open-weight models or paraphrased text. It is a narrow high-precision signal, not a replacement.

---

## 8. Limitations

**Genre.** PERSUADE is grades 6–12 source-based argumentative writing; the deployment target is 300–650 word first-person personal statements. Given §5.4, our specific figures should not be assumed to transfer. We used PERSUADE because no public corpus of admissions essays carries both quality scores and ELL annotation. The mechanisms are properties of the feature family; the magnitudes are not portable.

**We did not evaluate commercial detectors.** Turnitin's and GPTZero's classifiers are proprietary. We constrain the method, not any vendor's implementation. A vendor may have addressed these problems; none has published evidence of doing so.

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

Crossley, S. A., Baffour, P., Benner, M., Boser, U., Franklin, A., & Tian, Y. (2024). A large-scale corpus for assessing written argumentation: PERSUADE 2.0. *Assessing Writing*, 61, Article 100865. https://doi.org/10.1016/j.asw.2024.100865 [FLAGGED — corrected from what was drafted: the entry read "Crossley, S. A., et al. (2022). The PERSUADE corpus," which is PERSUADE **1.0** (Crossley, Baffour, Tian, Picou, Benner, & Boser, 2022, *Assessing Writing* 54, Article 100667, https://doi.org/10.1016/j.asw.2022.100667) — a real, correctly-dated paper, but not the corpus version this study uses. This study downloads Kaggle's `persaude-corpus-2` and depends on holistic scores and ELL annotation, which PERSUADE 2.0 (2024) added; 1.0 (2022) is the precursor without them. Cite the 2.0 (2024) entry above as primary; keep 1.0 only as a secondary reference if the intent is to credit the corpus's full lineage.]

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

Turnitin (2023). New research: Turnitin's AI detector shows no statistically significant bias against English Language Learners. Turnitin Blog, October 2023. https://www.turnitin.com/blog/new-research-turnitin-s-ai-detector-shows-no-statistically-significant-bias-against-english-language-learners [FLAGGED — corrected: the drafted arXiv ID, 2312.05241, does not resolve to a Turnitin response; that ID belongs to an unrelated paper, "Contra generative AI detection in higher education assessments." Turnitin's actual response to Liang et al. was published as a company blog post, not an arXiv preprint — the citation above is a blog post, which is a different reference type than every other entry in this list, so flagging in case a more formal citation is preferred, or in case a genuine Turnitin arXiv/preprint response exists that this search did not surface.]

Weber-Wulff, D., Anohina-Naumeca, A., Bjelobaba, S., Foltýnek, T., Guerrero-Dib, J., Popoola, O., Šigut, P., & Waddington, L. (2023). Testing of detection tools for AI-generated text. *International Journal for Educational Integrity*, 19, Article 26. https://doi.org/10.1007/s40979-023-00146-z

Al Ali, A., Helcl, J., & Libovický, J. (2026). Different time, different language: Revisiting the bias against non-native speakers in GPT detectors. *Proceedings of the EACL 2026 Student Research Workshop*. https://aclanthology.org/2026.eacl-srw.20/ (arXiv:2602.05769) [Confirmed real: exists, matches the title and Czech-language non-native-bias-replication description used in §2; authors added since the in-text mention only said "a Czech-language replication."]
