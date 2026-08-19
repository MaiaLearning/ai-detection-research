# Essay texture analysis — calibration study

## What this project is

An offline study to determine whether a statistical "texture" measure of student
essays can support a user-facing panel in MaiaLearning's essay review feature.

The existing production feature asks Claude to estimate the probability that an
essay would be flagged by AI detectors. That estimate has never been validated
against any real detector. It is being removed. This study determines what, if
anything, can honestly replace it.

Full experimental design is in `RESEARCH_PLAN.md`. Read it before writing code.

## The question

Not "can we detect AI-written essays." The question is whether a texture measure
can separate AI from human writing **without** separating English language
learners from native speakers, and **without** rating weak essays as more
authentic than strong ones.

Both of those failure modes were observed in the production feature. They are
the reason this study exists.

## Hard constraints

**No MaiaLearning student data in this repository, ever.** Public research
corpora only. Production essays are covered by district Student Data Protection
Agreements that almost certainly do not permit model-development use, and B2B
and B2C records are not currently distinguishable in storage. If a task seems to
need real student essays, stop and flag it rather than working around it.

**No LLM in the scoring path.** Every feature must be deterministic and
reproducible — same input, same output, no temperature, no API call. Binoculars
and Fast-DetectGPT use local models for perplexity computation, which is fine;
they are deterministic. Asking a language model to estimate a score is the thing
this study replaces.

**Do not build an "I am an English language learner" input.** It was considered
and rejected: self-reported, trivially gamed by exactly the students it would
fail to protect, and it creates a stored protected-class proxy sitting next to an
AI-suspicion score. If a residual subgroup gap survives calibration, the remedy
is an abstain band over the region where distributions overlap — not a
per-group threshold.

**Treat detector-vendor and "humanizer" site numbers as marketing.** Sites
selling detection-evasion tools publish benchmark tables with unaudited figures
and a commercial interest in the results. Peer-reviewed sources and published
technical reports only. Named exception: GPTZero's API returns real labels and
can be used as one commercial anchor.

## Conventions

- Python, `uv` or `venv`, pinned dependencies.
- Every experiment is a script that writes results to `results/` as CSV plus a
  plot. No notebook-only analysis — results must be regenerable from a clean
  checkout with one command.
- Set and record random seeds. Log the exact corpus version and filter criteria
  used for every run.
- Report confidence intervals, not point estimates. Sample sizes here are large
  enough that there is no excuse not to.
- When a result is ambiguous, say so in the writeup rather than picking the
  reading that favors shipping.

## Working style

Ask before pulling multi-gigabyte datasets — some of these are large and the
study only needs subsets.

Experiments 1 and 2 gate the rest. If the score predicts ELL status or
anti-correlates with essay quality, the panel does not ship in scoring form and
the remaining experiments are academic. Run them in order and report the gate
results before moving on.

The honest outcome of this study may be "this cannot be done defensibly." That
is a useful result and should be reported plainly if the data says it.
