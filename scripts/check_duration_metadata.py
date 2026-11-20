"""Is the board's `duration` field the actual length of the audio?

Section 4 breaks results down by clip duration and by speech rate, both of which
come from the board's own metadata rather than from the audio. A breakdown is
only worth reading if the field it slices on is correct, so this decodes the
sampled clips and compares.

Reports the maximum absolute error, not the mean: a metadata field that is right
on average and wrong on a subset would still invalidate a bucket boundary.

Needs the audio from scripts/fetch_audio.py (step 3 in the README), which is not
committed. Clips are matched by id, so a partial download is fine: whatever is
present on disk is checked and the count is reported.

Output: results/duration_check.json

Run:  .venv/bin/python scripts/check_duration_metadata.py
"""

from __future__ import annotations

import csv
import json
import os

import soundfile

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(BASE, "data", "manifest.csv")
AUDIO_DIR = os.path.join(BASE, "data", "audio")
OUT = os.path.join(BASE, "results", "duration_check.json")

# Chunked datasets carry the parent recording's duration on every chunk, so the
# field is not comparable to the length of one chunk's audio.
SKIP_DATASETS = ("earnings22_cleaned_aa_chunked",)


def safe_name(dataset: str, clip_id: str) -> str:
    """The filename scheme fetch_audio.py writes, kept in sync with it."""
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
    return f"{dataset}__{cleaned}.wav"


def decoded_seconds(path: str) -> float:
    """Length from the file header. fetch_audio.py writes float32 WAVs, which the
    standard library's `wave` module cannot parse, so soundfile is used."""
    info = soundfile.info(path)
    return info.frames / info.samplerate


def main() -> None:
    rows = list(csv.DictReader(open(MANIFEST)))
    checked, missing, worst = [], 0, None

    for row in rows:
        if row["dataset"] in SKIP_DATASETS or not row["duration_s"]:
            continue
        path = os.path.join(AUDIO_DIR, safe_name(row["dataset"], row["clip_id"]))
        if not os.path.exists(path):
            missing += 1
            continue
        claimed = float(row["duration_s"])
        actual = decoded_seconds(path)
        error = abs(actual - claimed)
        checked.append(error)
        if worst is None or error > worst["error_s"]:
            worst = {
                "clip_id": row["clip_id"],
                "dataset": row["dataset"],
                "claimed_s": round(claimed, 3),
                "decoded_s": round(actual, 3),
                "error_s": round(error, 6),
            }

    if not checked:
        raise SystemExit(
            f"no audio found in {AUDIO_DIR}; run scripts/fetch_audio.py first"
        )

    results = {
        "n_checked": len(checked),
        "n_missing_audio": missing,
        "max_abs_error_s": round(max(checked), 6),
        "mean_abs_error_s": round(sum(checked) / len(checked), 8),
        "n_exact": sum(1 for e in checked if e == 0.0),
        "worst_clip": worst,
    }
    with open(OUT, "w") as handle:
        json.dump(results, handle, indent=1)

    print(f"checked {results['n_checked']} clips ({missing} without local audio)")
    print(f"max absolute error  {results['max_abs_error_s']:.6f}s")
    print(f"mean absolute error {results['mean_abs_error_s']:.8f}s")
    print(f"exact to the sample {results['n_exact']} of {results['n_checked']}")
    print(f"worst: {worst}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
