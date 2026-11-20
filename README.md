# Partial Audit of Hugging Face Open ASR Leaderboard

A reproduction of the [Hugging Face Open ASR
Leaderboard](https://hf.co/spaces/hf-audio/open_asr_leaderboard) (version
v1.16.1) across three open models, and an analysis of the output relative to
Hugging Face's reporting.

The report is in [`report/REPORT.md`](report/REPORT.md). This file is about
rerunning the work.

## What is measured

| | |
|---|---|
| Board | Hugging Face Open ASR Leaderboard, English short-form |
| Systems run | `Qwen/Qwen3-ASR-1.7B-hf`, `Musci-research/Musci-ASR-qwen-1019-hf`, `CohereLabs/cohere-transcribe-03-2026` |
| Reproduction data | 600 clips: 200 each from AMI, GigaSpeech, VoxPopuli |
| Analysis data | All 7 public datasets, full splits, about 70,000 clips per system |
| Scoring | The board's own normalizer, vendored verbatim, with `kaldialign` compound merging |

## Headline results

- The board's text normalization **changes which system ranks first on three of
  the seven datasets**. On AMI, `cohere-transcribe` is 2.8 points clear of
  `Musci-ASR` scored strictly and 0.5 points behind it after normalization: a
  3.2-point swing from the same transcripts.
- `Qwen3-ASR` writes **3.8%** of the filled pauses its references contain, while
  `cohere-transcribe` writes 98.0%. The board's metric cannot see this, because
  it deletes fillers from both sides before scoring.
- The board's headline average covers nine datasets, **two of them private**, so
  the published number cannot be fully reproduced by anyone outside Hugging Face.

## Rerunning it

Needs Python 3.11. No API keys for the two Qwen-family models; Cohere's weights
are gated behind a click-through agreement, so that one needs a free Hugging
Face account.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 1. Verify the scorer before trusting anything

```bash
.venv/bin/python scripts/validate_scorer.py
```

Re-scores the board's own published outputs and checks that it lands on their
published WER. Every pair should agree within 0.02 points, worst case 0.005,
which is publication rounding. On a fresh clone this reports **9 of 9**, the
committed subset; after step 2 syncs the full bucket it reports **64 of 64**.

This runs first on purpose: it separates scoring error from inference error, so
any later disagreement is attributable to one or the other.

This step needs `data/board_results/`, which is partly included (see
[What is in this repo](#what-is-in-this-repo)) and otherwise fetched by step 2.

### 2. Fetch the board's published outputs

```bash
.venv/bin/python -m pip install "huggingface_hub[cli]>=2.4"   # separate venv advised, see note
hf buckets sync hf://buckets/hf-audio/asr_leaderboard_h200 data/board_results/predictions --include "*.jsonl"
hf buckets sync hf://buckets/hf-audio/asr_leaderboard_h200 data/board_results/scores --include "*.score.json"
```

> **Note.** The `hf` CLI requires `huggingface_hub >= 2.4`, which is
> incompatible with the pinned `transformers`. Install it into a throwaway
> virtualenv rather than this project's.

### 3. Build the subset and fetch its audio

```bash
.venv/bin/python scripts/build_subset.py            # -> data/manifest.csv
.venv/bin/python scripts/build_row_index_parquet.py # -> data/row_index/
.venv/bin/python scripts/fetch_audio.py             # -> data/audio/
```

`build_row_index_parquet.py` is not optional. The board's prediction manifests
are written in processing order, not dataset order, and contain fewer rows than
the dataset, so their line numbers cannot be used to locate audio. This step
reads each dataset's own `id` column from its parquet files to get the true row
offsets. Without it, audio is paired with the wrong clip's reference and the
result looks like a bad model rather than a broken pipeline.

### 4. Transcribe and compare

```bash
.venv/bin/python scripts/smoke_test_models.py   # 2 clips per model, fails fast
.venv/bin/python scripts/transcribe.py          # -> data/predictions.csv
.venv/bin/python scripts/compare_to_board.py    # -> results/reproduction.json
.venv/bin/python scripts/check_duration_metadata.py  # audio vs the board's metadata
```

Run the smoke test first. Each model in this set has failed at least once in a
different way during development, and two minutes per model is cheaper than
discovering it three hours into a run.

`transcribe.py` is resumable: it skips `(clip, model)` pairs already present in
`data/predictions.csv`.

### 5. The analyses, which need no inference

```bash
.venv/bin/python scripts/power_analysis.py       # derives the 200-clip size
.venv/bin/python scripts/score_ladder.py         # normalization ladder
.venv/bin/python scripts/disfluency_fidelity.py  # verbatim fidelity metric
.venv/bin/python scripts/characterize.py         # per-clip metadata + breakdowns
.venv/bin/python scripts/characterize_report.py  # section 4's tables from that file
```

These run on the board's published per-clip outputs, so they cover the full
public splits rather than the 600-clip sample.

## What is in this repo

Committed: all code, the 600-clip manifest, the predictions, the computed
results, and the board's published outputs **for the three datasets and three
models used in the reproduction** (about 28 MB), so the clip-level comparison
works out of the box.

Not committed: evaluation **audio** (not mine to redistribute, and regenerated
by step 3) and the board's outputs for the other datasets and models (fetched by
step 2).

```
scripts/
  validate_scorer.py          checks the scoring against the board's published WER
  build_subset.py             seeded 200-clips-per-dataset sample
  build_row_index_parquet.py  clip id -> true dataset row offset
  fetch_audio.py              fetches only the sampled clips
  smoke_test_models.py        2 clips per model, before any long run
  transcribe.py               runs the models, resumable
  compare_to_board.py         aggregate and per-clip comparison
  normalize_ladder.py         the board's normalizer with steps switched off
  score_ladder.py             scores every rung on the full public splits
  disfluency_fidelity.py      the added metric
  characterize.py             per-clip metadata and WER
  characterize_report.py      section 4's tables, from per_clip.csv.gz
  check_duration_metadata.py  the board's duration field against decoded audio
  power_analysis.py           derives the 200-clip sample size
  verify_models.py            checks downloaded weights against Hub file sizes
  measure_runtime.py          dtype timing, used to justify the float32 deviation
  test_batching.py            checks batch size does not change transcripts
vendor/board_normalizer/      verbatim copy of the board's normalizer (Apache-2.0)
report/REPORT.md              the write-up, including every choice and its reason
LICENSE                       MIT, except the vendored normalizer
```

## Configuration, and where it deviates from the board

Read from the board's own submit scripts so that a difference in results is
attributable to inference rather than to a decoding choice made by accident.

| | Board | Here | Why |
|---|---|---|---|
| `max_new_tokens` | 512 | 512 | same |
| attention | sdpa | sdpa | same |
| dtype | bfloat16 | **float32** | bfloat16 is emulated on this CPU and 1.6x slower; transcripts measured identical on 6/6 clips |
| batch size | 256 | **8** | memory; transcripts measured identical to batch 1 on 8/8 clips |
| device | GPU | **CPU** | hardware available |

Each deviation was measured before being adopted, not assumed to be harmless.

## Known limits

- Three systems of the 69 the board ranks, all open-weight. Two of the three
  share a lineage by design, as a controlled fine-tune-versus-base comparison.
  Every section of the report compares the same three.
- Three of seven public datasets in the reproduction. The other four were set
  aside for reasons recorded in section 0 of the report, and still appear in the
  analyses. None was unverifiable in principle; each was dropped for time, and
  the report records what was not tried.
- One of them, URGENT24, matched the board's published outputs at only 46%. It
  was dropped rather than diagnosed, so the cause is unknown.
- Everything is seeded (`SEED = 42` for the subset, `42` for bootstraps), so a
  rerun on the same dataset revision reproduces the same clips and numbers.
