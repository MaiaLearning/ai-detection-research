# Essay Texture Calibration Study

An offline research study asking whether a deterministic statistical "texture"
measure of student essays can honestly support a user-facing AI-detection or
quality panel in MaiaLearning's essay review feature. Full background,
methodology, and hard constraints are in [`CLAUDE.md`](CLAUDE.md); the
experimental design is in [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md) and the
`EXPERIMENT_5.md` / `EXPERIMENT_6.md` design docs. The practitioner-facing
write-up is [`PRACTITIONER_BRIEF.md`](PRACTITIONER_BRIEF.md)
([PDF](PRACTITIONER_BRIEF.pdf)) — that's the primary document; this
repository exists so every number in it can be independently reproduced.

`RESEARCH_PLAN.md` was pre-registered before any data was collected (see
the note at its top); [`AMENDMENTS.md`](AMENDMENTS.md) logs every
consequential decision made or changed after that point, including the two
analyses below that have no design doc of their own and the one gate
threshold the original plan never numerically specified. Read it before
treating any of the mid-study additions as having been planned in advance —
they weren't, and that document says so plainly.

## Findings summary

| Gate / experiment | Result |
|---|---|
| 1 — predicts ELL status? | PASS (composite AUC 0.60, all features < 0.65) |
| 2 — anti-correlated with quality? | **FAIL** at the composite level (partial ρ = +0.135, CI excludes 0 — better essays score more AI-like) |
| 3 — separates human/AI? | AUC 0.945, but only 41.3% TPR at a 1% FPR target; ELL FPR (1.56%) > non-ELL (0.94%) |
| 4 — survives editing? | Paraphrase attack drops TPR ~21 points; two RAID attacks (homoglyph, zero-width space) break tokenization rather than demonstrating evasion |
| Near/far genre transfer | FPR: 1% (PERSUADE) → 2.4–3.7% (ELLIPSE, near genre) → 36.9% (RAID abstracts, far genre) |
| ELL × genre compounding | No compounding — the ELL FPR penalty is intrinsic to ELL status, not explained by topic novelty |
| 6 — features carry quality above word count? | **NO** — M1−M0 delta ρ = **−0.079** (CI −0.086 to −0.072), replicated on a held-out set |
| 5 — vendor vs. prompting-effort confound | Shelved: DAIGT's real generation prompts turned out to be unscaffolded, dissolving the premise |

Net: this study does not support shipping a scoring panel of this design —
the panel was shipped, evaluated, and is being withdrawn. See
[`PRACTITIONER_BRIEF.md`](PRACTITIONER_BRIEF.md) for the full argument,
recommendations, and caveats.

## Repository layout

```
src/                  Pure, unit-tested library code (features, matching, stats, verification)
scripts/              One script per experiment/analysis, each regenerates its own results/ outputs
tests/                pytest suite for everything in src/ (72 tests, ~3s)
data/                 Corpora (see Data below — most are NOT committed here)
results/              CSVs, plots, and JSON manifests written by scripts/ (committed)
CLAUDE.md             Project constraints and conventions
RESEARCH_PLAN.md       Original experimental design (experiments 1-4), pre-registered 2026-08-13
EXPERIMENT_5.md        Design doc for the prompting-effort experiment (shelved)
EXPERIMENT_6.md        Design doc for the quality-composite + verification-harness experiment
AMENDMENTS.md          Log of every consequential decision made or changed mid-study, with reasoning
PRACTITIONER_BRIEF.md  The practitioner-facing write-up (also available as PRACTITIONER_BRIEF.pdf)
```

## A note on this repository's history

This repository was pushed as a single commit imported from the working
directory where the study was conducted — it does not preserve an
incremental commit history, so git alone cannot corroborate the
pre-registration and amendment timeline this README and `AMENDMENTS.md`
describe. That limitation is disclosed rather than papered over: see the
closing section of `AMENDMENTS.md` for exactly what evidence does and
doesn't exist for that timeline, and why we didn't try to construct commit
history that would misrepresent when the work actually happened.

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest       # 72 tests, ~3s, no data required
```

## Reproducing the study

### 1. Get the raw corpora

PERSUADE 2.0, DAIGT-v2, and ELLIPSE are released under non-commercial
licenses (see **Data licensing** below) and are **not redistributed in this
repository**. Download them yourself with a free
[Kaggle account + API token](https://www.kaggle.com/docs/api):

```bash
uv run kaggle datasets download -d nbroad/persaude-corpus-2 \
  -f persuade_2.0_human_scores_demo_id_github.csv -p data/ --unzip

uv run kaggle datasets download -d thedrcat/daigt-v2-train-dataset \
  -f train_v2_drcat_02.csv -p data/ --unzip

uv run kaggle datasets download -d matthewjansen/ellipse-corpus -p data/ --unzip
mv data/train.csv data/ellipse_corpus.csv
```

Verify you have byte-identical corpus versions:

```bash
sha256sum data/persuade_2.0_human_scores_demo_id_github.csv
# a28abb40eae9075191ae58627132172a576bc21fed7595c76d8318d927b5024d
sha256sum data/train_v2_drcat_02.csv
# 737a355ef9c43ef2e4c2f5556c03ad52a84d7da0e7e7ac311ec59375bc667bc7
sha256sum data/ellipse_corpus.csv
# 27652147d6c10c46b63b1d561cf835467b56db507fd6a051028aa254f6a3a7b9
```

### 2. Already included — no regeneration needed

These were generated or derived by this project itself (no external
redistribution restriction) and are committed so you don't need to repeat
paid API calls or re-stream RAID:

- `data/bedrock_claude_essays.csv` — 1000 essays via `us.anthropic.claude-sonnet-5` (Bedrock), ~$9
- `data/openai_gpt56terra_essays.csv` — 1000 essays via `gpt-5.6-terra` (OpenAI), ~$9
- `data/raid_abstracts_subset.csv` — filtered subset of [RAID](https://huggingface.co/datasets/liamdugan/raid) (MIT-licensed), streamed at no cost
- `results/` — every experiment's output CSVs, plots, and manifests

If you want to regenerate the essay sets yourself: `scripts/generate_bedrock_claude_essays.py`
needs AWS Bedrock model access (edit `AWS_PROFILE`/`MODEL_ID` for your own
account) and `scripts/generate_openai_essays.py` needs an OpenAI API key at
`~/.openai/api_key`. Both cost real money and sample at `temperature=1.0`,
so a fresh run will not exactly reproduce the committed files.

**Fitted model artifacts (`*.joblib`) are excluded and gitignored** — they're
derived from CC-BY-NC-SA training data, and the licensing status of derived
model weights is unresolved, so we chose not to distribute them rather than
resolve that question. Regenerate them locally (deterministic, seeded):

```bash
uv run python scripts/experiment3_separation.py           # -> results/experiment3_frozen_composite.joblib
uv run python scripts/analyze_ell_genre_compounding.py    # -> results/ell_genre_diagnostic_model.joblib
```

### 3. Run the pipeline

```bash
uv run python scripts/experiment1_ell_gate.py
uv run python scripts/experiment2_quality_gate.py
uv run python scripts/experiment3_separation.py            # also writes the frozen composite (see above)
uv run python scripts/experiment4_raid_robustness.py       # streams RAID directly; ~5-8 min
uv run python scripts/analyze_ellipse_neargenre.py
uv run python scripts/analyze_ell_genre_compounding.py     # also writes the diagnostic model (see above)
uv run python scripts/experiment6_quality_composite.py
uv run python scripts/experiment6_verification_demo.py
```

`scripts/experiment5_prompt_effort.py` exists but is **shelved** — see the
closing note in `EXPERIMENT_6.md`. Running it requires AWS Bedrock access
and produces real API spend; it does not bear on the study's conclusions.

### Cost & time (measured, not estimated)

| step | cost | time |
|---|---|---|
| Kaggle downloads | free | a few minutes |
| Experiments 1-2 | free | ~1 minute |
| Experiment 3 (fit + score) | free | ~1.5 minutes |
| Bedrock + OpenAI generation (optional; already included) | ~$18 total | ~20-25 min each |
| Experiment 4 (RAID streaming) | free | ~5-8 minutes |
| ELLIPSE / compounding / experiment 6 analyses | free | a few minutes each |

## Data licensing

| corpus | license | redistributed here? | source |
|---|---|---|---|
| PERSUADE 2.0 | CC BY-NC-SA 4.0 | No | `kaggle:nbroad/persaude-corpus-2` |
| DAIGT-v2 | mixed, non-commercial-leaning per source | No | `kaggle:thedrcat/daigt-v2-train-dataset` |
| ELLIPSE | CC BY-NC-SA 4.0 | No | `kaggle:matthewjansen/ellipse-corpus` |
| RAID | MIT | Yes (filtered subset) | `huggingface:liamdugan/raid` |
| Bedrock/OpenAI generations | ours | Yes | generated by this project |

## License

Source code (`src/`, `scripts/`, `tests/`) is MIT-licensed — see
[`LICENSE`](LICENSE). Written content (`PRACTITIONER_BRIEF.md`/`.pdf`, this
README, `CLAUDE.md`, `RESEARCH_PLAN.md`, `AMENDMENTS.md`, the experiment
design docs, and `results/`) is licensed
CC BY 4.0 — see [`LICENSE-CONTENT.md`](LICENSE-CONTENT.md). Neither license
covers the third-party data corpora — see **Data licensing** above.
