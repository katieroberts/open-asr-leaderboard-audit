"""Per-clip metadata for the evaluation set, and per-clip WER to break down by.

Scope
-----
Runs on the board's published per-clip outputs, so the breakdowns cover the
FULL public splits (~70,000 clips per model) rather than the inference
subset. Metadata that can be derived from the manifests costs nothing:
duration ships with every record, and everything else is computed from the
reference text.

Fields
------
  duration_s          from the board's manifest
  ref_words           words in the board-normalized reference
  speech_rate_wps     ref_words / duration_s
  duration_bucket     short / medium / long, fixed thresholds
  has_numbers         reference contains digits after normalization
  n_fillers           filler tokens the board's rule would delete
  filler_density      n_fillers / ref_words
  wer                 per-clip WER under the board's normalization

Thresholds are fixed before looking at any WER, so buckets cannot be drawn
to flatter a result.

Output: results/per_clip.csv.gz
        (scripts/characterize_report.py turns that into the report's tables
        and results/characterization.json)

Run:  .venv/bin/python scripts/characterize.py
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from normalize_ladder import LADDER  # noqa: E402

BOARD = LADDER["L6_board"]
NO_FILLER_RULE = LADDER["L0_strict"]
WITH_FILLER_RULE = LADDER["L1_fillers"]

PREDICTIONS_DIR = os.path.join(BASE, "data", "board_results", "predictions")

# The three systems this evaluation reports on. The results bucket holds
# published outputs for more models, but every section of the report compares
# the same three, so a reader is never shown a system that was analyzed from
# published output but never run here.
SYSTEMS = (
    "Qwen-Qwen3-ASR-1.7B-hf",
    "Musci-research-Musci-ASR-qwen-1019-hf",
    "CohereLabs-cohere-transcribe-03-2026",
)

SHORT_MAX_S = 5.0
MEDIUM_MAX_S = 15.0

P = "hf-audio-open-asr-leaderboard_"
PUBLIC_DATASETS = {
    "ami_cleaned": P + "ami_cleaned_test",
    "earnings22_cleaned_aa_chunked": "ArtificialAnalysis-Earnings22-Cleaned-AA-chunked",
    "gigaspeech_cleaned": P + "gigaspeech_cleaned_test",
    "spgispeech": P + "spgispeech_test",
    "urgent2024": P + "urgent2024_test",
    "voice_arena_monsoon": "VoiceArena-Monsoon_en_IN_test__test",
    "voxpopuli_cleaned_aa": P + "voxpopuli_cleaned_aa_test",
}

HAS_DIGIT = re.compile(r"\d")

FIELDNAMES = [
    "model", "dataset", "clip_id", "duration_s", "ref_words",
    "speech_rate_wps", "duration_bucket", "has_numbers", "n_fillers",
    "errors", "wer",
]


def duration_bucket(duration_s: float) -> str:
    if duration_s <= 0:
        return "unknown"
    if duration_s < SHORT_MAX_S:
        return "short"
    if duration_s < MEDIUM_MAX_S:
        return "medium"
    return "long"


def read_records(path: str) -> list[dict]:
    records = [json.loads(line) for line in open(path)]
    if records and "parent_id" in records[0]:
        by_parent: dict[str, list[dict]] = {}
        for record in records:
            by_parent.setdefault(record["parent_id"], []).append(record)
        joined = []
        for parent_id, chunks in by_parent.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            joined.append({
                "audio_filepath": parent_id,
                "text": chunks[0]["text"],
                "pred_text": " ".join(c.get("pred_text") or "" for c in chunks),
                "duration": sum(c.get("duration", 0.0) for c in chunks),
            })
        return joined
    return records


def find_manifest(model_dir: str, suffix: str):
    directory = os.path.join(PREDICTIONS_DIR, model_dir)
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jsonl") and suffix in filename:
            return os.path.join(directory, filename)
    return None


def main() -> None:
    from kaldialign import edit_distance

    models = [
        d for d in SYSTEMS
        if os.path.isdir(os.path.join(PREDICTIONS_DIR, d))
    ]
    missing = [d for d in SYSTEMS if d not in models]
    if missing:
        raise SystemExit(f"no published outputs for {missing} in {PREDICTIONS_DIR}")
    out_path = os.path.join(BASE, "results", "per_clip.csv.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Reference-derived fields depend only on the text, so cache them: the
    # same reference appears once per model.
    reference_cache: dict[str, dict] = {}

    written = 0
    with gzip.open(out_path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()

        for dataset_label, suffix in PUBLIC_DATASETS.items():
            for model_dir in models:
                path = find_manifest(model_dir, suffix)
                if path is None:
                    continue
                for record in read_records(path):
                    raw_reference = record.get("text") or ""
                    if raw_reference not in reference_cache:
                        normalized = BOARD(raw_reference)
                        words = normalized.split()
                        reference_cache[raw_reference] = {
                            "normalized": normalized,
                            "ref_words": len(words),
                            "has_numbers": bool(HAS_DIGIT.search(normalized)),
                            "n_fillers": (
                                len(NO_FILLER_RULE(raw_reference).split())
                                - len(WITH_FILLER_RULE(raw_reference).split())
                            ),
                        }
                    info = reference_cache[raw_reference]
                    if not info["normalized"].strip():
                        continue

                    prediction = BOARD(record.get("pred_text") or "")
                    reference = info["normalized"]
                    # See the note in validate_scorer.py: the board applies
                    # normalize_compound_pairs only on its multilingual path.
                    stats = edit_distance(
                        tuple(reference.split()), tuple(prediction.split()),
                        merge_compounds=True,
                    )
                    errors = stats["ins"] + stats["del"] + stats["sub"]
                    reference_words = len(reference.split())
                    duration = float(record.get("duration") or 0.0)

                    writer.writerow({
                        "model": model_dir,
                        "dataset": dataset_label,
                        "clip_id": record.get("audio_filepath", ""),
                        "duration_s": round(duration, 3),
                        "ref_words": reference_words,
                        "speech_rate_wps": (
                            round(reference_words / duration, 3) if duration > 0 else ""
                        ),
                        "duration_bucket": duration_bucket(duration),
                        "has_numbers": int(info["has_numbers"]),
                        "n_fillers": info["n_fillers"],
                        "errors": errors,
                        "wer": round(errors / reference_words, 6) if reference_words else "",
                    })
                    written += 1
                print(f"  {dataset_label:32s} {model_dir[:36]:36s} done "
                      f"({written} rows so far)", flush=True)

    print(f"\nWrote {written} rows to {out_path}")


if __name__ == "__main__":
    main()
