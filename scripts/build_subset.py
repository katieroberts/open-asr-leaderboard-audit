"""Select the evaluation subset, from the board's own manifests.

Why this needs no network
-------------------------
The board publishes its per-clip outputs for every model, and those
manifests list every clip id in every split. So the sample can be drawn
offline, and only the selected clips' audio has to be fetched later.
That matters: the seven public splits total ~154 h, roughly 20-30 GB,
while the sample is ~4.5 h.

Sampling
--------
200 clips per dataset, seeded. Equal allocation rather than proportional,
because the leaderboard takes a macro-average over datasets ("For each
task, we take a macro-average over all datasets", constants.py) and so
weights a 628-clip split exactly as heavily as a 39,341-clip one.
Sampling proportionally would silently reweight the benchmark.

200 comes from a power analysis on the board's own per-clip outputs, in
scripts/power_analysis.py: over 500 seeded draws it recovers the full
three-model ranking in 100% of trials, against 99.8% at 150 and 98.6% at
100. Rerun that script to regenerate the figures rather than trusting
this comment.

Earnings22-Cleaned-AA-chunked is a special case twice over. Its scoring
unit is the parent recording, not the chunk, and it contains only SIX
parent recordings in total (341 chunks). There is nothing to sample, so
all six are taken whole. Worth stating in the report: one of the seven
datasets behind the headline average rests on six recordings, and the
macro-average gives it the same weight as SPGISpeech.

Output: data/manifest.csv

Run:  .venv/bin/python scripts/build_subset.py
"""

from __future__ import annotations

import csv
import json
import os
import random

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(BASE, "data", "board_results", "predictions")

# Any model's manifest lists the same clip ids; this one covers all seven.
SOURCE_MODEL = "Musci-research-Musci-ASR-qwen-1019-hf"

SEED = 42
CLIPS_PER_DATASET = 200

P = "hf-audio-open-asr-leaderboard_"
# report label -> (manifest suffix, hub dataset, config, split)
DATASETS = {
    "ami_cleaned": (P + "ami_cleaned_test", "hf-audio/open-asr-leaderboard", "ami_cleaned", "test"),
    "earnings22_cleaned_aa_chunked": (
        "ArtificialAnalysis-Earnings22-Cleaned-AA-chunked__test",
        "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", None, "test"),
    "gigaspeech_cleaned": (P + "gigaspeech_cleaned_test", "hf-audio/open-asr-leaderboard", "gigaspeech_cleaned", "test"),
    "spgispeech": (P + "spgispeech_test", "hf-audio/open-asr-leaderboard", "spgispeech", "test"),
    "urgent2024": (P + "urgent2024_test", "hf-audio/open-asr-leaderboard", "urgent2024", "test"),
    # Renamed upstream: VoiceArena/Monsoon_en_IN_test now 307-redirects, and
    # the datasets-server 404s on the old path.
    "voice_arena_monsoon": ("VoiceArena-Monsoon_en_IN_test__test", "VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN", None, "test"),
    "voxpopuli_cleaned_aa": (P + "voxpopuli_cleaned_aa_test", "hf-audio/open-asr-leaderboard", "voxpopuli_cleaned_aa", "test"),
}

FIELDNAMES = [
    "clip_id", "row_index", "dataset", "hub_dataset", "hub_config", "hub_split",
    "duration_s", "parent_id", "chunk_index", "ref_text",
]


def read_manifest(suffix: str) -> list[dict]:
    path = os.path.join(
        PREDICTIONS_DIR, SOURCE_MODEL,
        f"MODEL_{SOURCE_MODEL}_DATASET_{suffix}.jsonl",
    )
    return [json.loads(line) for line in open(path)]


def sample_dataset(label, records, hub, config, split, rng) -> list[dict]:
    """Draw the sample for one dataset, respecting its scoring unit.

    Each record carries its line number in the board's manifest as
    `row_index`. That is the identifier used to fetch audio later, because
    the clip id is not always unique: SPGISpeech's 39,341 rows carry only
    421 distinct `audio_filepath` values (the board stores the basename and
    drops the directory that disambiguates them), so an id alone cannot say
    which row was sampled.
    """
    for index, record in enumerate(records):
        record["_row_index"] = index
    is_chunked = bool(records) and "parent_id" in records[0]

    if is_chunked:
        # The scoring unit is the recording. Take every parent whole, so a
        # recording is never scored against a partial transcript.
        parents = sorted({r["parent_id"] for r in records})
        chosen = [r for r in records if r["parent_id"] in set(parents)]
        print(f"  {label:32s} {len(chosen):5d} chunks from ALL "
              f"{len(parents)} parent recordings (nothing to sample)")
    else:
        records_sorted = sorted(records, key=lambda r: r["audio_filepath"])
        chosen = rng.sample(records_sorted, min(CLIPS_PER_DATASET, len(records_sorted)))
        print(f"  {label:32s} {len(chosen):5d} of {len(records_sorted)} clips")

    return [
        {
            "clip_id": r["audio_filepath"],
            "row_index": r["_row_index"],
            "dataset": label,
            "hub_dataset": hub,
            "hub_config": config or "",
            "hub_split": split,
            "duration_s": round(r.get("duration", 0.0), 4),
            "parent_id": r.get("parent_id", ""),
            "chunk_index": r.get("chunk_index", ""),
            "ref_text": r["text"],
        }
        for r in chosen
    ]


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    for label, (suffix, hub, config, split) in DATASETS.items():
        rows.extend(sample_dataset(label, read_manifest(suffix), hub, config, split, rng))

    out_path = os.path.join(BASE, "data", "manifest.csv")
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    total_hours = sum(float(r["duration_s"]) for r in rows) / 3600
    print(f"\nWrote {len(rows)} clips to {out_path}")
    print(f"Audio to fetch: {total_hours:.2f} h per model")


if __name__ == "__main__":
    main()
