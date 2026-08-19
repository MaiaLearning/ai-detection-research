# Experiment 5: Does detectability measure prompting effort or model identity?

Read `CLAUDE.md` and `RESEARCH_PLAN.md` first. This extends experiment 3.

## Why this exists

Experiment 3 found `claude_sonnet_5_bedrock` at 71.9% TPR — the most-detected of
all 16 sources — versus `chat_gpt_moth` at 42.2%. That was read as a vendor
difference, but it is confounded: our Claude generations used a generic prompt,
while DAIGT's GPT generations came from contributors deliberately constructing
student-like text. Prompting effort and vendor vary together, so neither is
identified.

This experiment separates them by varying prompt specification while holding
vendor constant, across both vendors.

**Pre-registered prediction — record this in the manifest before running:**
monotone decline in TPR from L0 to L3, with the largest single drop between L2
and L3. If that holds for both vendors, detectability tracks prompting effort
rather than model identity. If Claude stays high across all levels while GPT
declines, it is a model property and the writeup says something different.

## Do not compose the prompts yourself

This is the central methodological constraint. If the prompt text is authored by
whoever builds this experiment, the result measures that person's beliefs about
what looks AI-generated rather than what students actually do. Every prompt level
must derive from a documented external artifact:

- **L1** text comes verbatim from PERSUADE's own task descriptions — the corpus
  ships the actual instructions given to students.
- **L2** replicates DAIGT contributors' documented generation templates. Find
  them in the Kaggle dataset descriptions and discussion threads for
  `thedrcat/daigt-v2-train-dataset` and its listed source datasets.
- **L3** style guidance must be sourced from publicly circulating student advice
  — Reddit, TikTok transcripts, essay-help blogs. Quote it; cite where each piece
  came from in the manifest.

If a level cannot be sourced, log that and stop rather than inventing text.

## Design

Four prompt levels x two vendors, 150 essays per cell, 1,200 total.

- **L0 — naive.** One line, no scaffolding, no length, no persona.
  Example shape: `write me an essay about whether cars should be limited in cities`
- **L1 — task description.** PERSUADE's full multi-paragraph instruction verbatim.
  Nothing added.
- **L2 — DAIGT replication.** L1 plus grade-level persona, word-count target, and
  an instruction to write as a student would, following DAIGT's documented
  templates.
- **L3 — adversarial.** L2 plus circulating student style guidance (vary sentence
  lengths, add a personal anecdote, avoid formulaic transitions). This doubles as
  experiment 4's adversarial arm — if TPR craters here, adversarial robustness is
  answered without a separate run.

**Vendors:** current production Claude via Bedrock (same model ID as experiment
3, recorded from `list_foundation_models`), and the current ChatGPT free-tier
default via the OpenAI API. Query `/v1/models` at run time; do not hardcode a
model name. Record exact IDs in the manifest.

## Run this first — it may be free

Before generating anything, split the existing experiment-3
`claude_sonnet_5_bedrock` TPR by text-dependent versus text-independent PERSUADE
prompts. Seven of the fifteen prompts reference source articles the corpus does
not carry, so Claude wrote those from general knowledge — producing more generic
essays by construction. If the TPR gap between the two groups is large, a
meaningful share of the 71.9% is already explained and this experiment's scope
may narrow.

Report that number before proceeding.

## Controls — hold everything else fixed

- **Restrict to the 8 text-independent PERSUADE prompts.** Removes the confound
  above. Same 8 prompts across all cells.
- **Length matching is mandatory.** Word count dominates the quality signal
  (raw rho = 0.76) and drives sentence-length SD and TTR. Match the PERSUADE
  word-count distribution per prompt. Generate unconstrained, record true length,
  then rejection-sample into the target distribution and report the discard rate
  per cell. The discard rate is itself a finding — L0 will likely overshoot.
  Plot generated vs PERSUADE length distributions per cell and verify overlap
  before scoring. If they do not overlap, the cell is uninterpretable.
- **Decoding: defaults.** Temperature 1.0, no top-p tuning, for both vendors.
  Students get defaults, and RAID found decoding strategy materially affects
  detectability. Record parameters.
- **Frozen classifier.** Load experiment-3 weights and score. No refitting.
- **Blind scoring.** Compute scores before joining condition labels.
- **Separate OpenAI research key.** Not production credentials, not production
  config.
- **Contamination check.** n-gram overlap against PERSUADE source text; verify
  generations do not restate corpus material.

## Metadata per essay

Exact model ID, vendor, prompt level, prompt name, prompt text hash, temperature,
timestamp, raw word count, post-truncation word count, rejection-sampled flag,
composite score.

## Analysis

1. TPR@1%FPR per cell. Plot vendor x level. Test the monotone trend.
2. TPR by level, pooled across vendors — the effort effect.
3. TPR by vendor, pooled across levels — the identity effect.
4. Compare effect sizes. Which explains more variance?
5. **Gate 2 on every cell.** Composite P(AI) vs PERSUADE holistic quality,
   partial Spearman controlling word count. This has been invariant at
   rho = +0.134 across every data change so far; expect it to stay invariant,
   and flag it loudly if it does not.
6. ELL/non-ELL FPR per cell.

## Budget

1,200 essays x ~600 output tokens, plus rejection-sampled discards. Under $25
across both vendors. Experiment 3's Claude arm cost $8.97 for 1,000, so scale
from that.

## What this does not do

This does not reopen the shipping decision. Gate 2 failed on the composite and
39.2% TPR at 1% FPR means the modal AI essay is scored clean. A strong L0 result
would support a narrow, explicitly-scoped claim at most — never a general panel.

The deliverable is a finding for the writeup, not a path to shipping.

## Report

Append to the experiment writeup: the text-dependent split result, the four-cell
table with CIs, whether the pre-registered prediction held, and the gate 2 figure
per cell. State plainly if the prediction failed.
