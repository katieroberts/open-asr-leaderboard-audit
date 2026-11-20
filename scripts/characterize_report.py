"""Section 4's tables, regenerated from results/per_clip.csv.gz.

characterize.py does the expensive part: it reads the board's published per-clip
outputs, derives metadata and scores every clip. This reads only its output, so
every breakdown in section 4 of the report is re-derivable in seconds without
touching the predictions or re-running inference.

Corpus WER throughout (sum of errors over sum of reference words), not the mean
of per-clip WERs. The two differ, and the board reports the former.

Output: results/characterization.json

Run:  .venv/bin/python scripts/characterize_report.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_CLIP = os.path.join(BASE, "results", "per_clip.csv.gz")
OUT = os.path.join(BASE, "results", "characterization.json")


def corpus_wer(frame: pd.DataFrame) -> float:
    """Corpus WER as a percentage, the board's aggregation."""
    words = frame["ref_words"].sum()
    return float("nan") if words == 0 else round(100.0 * frame["errors"].sum() / words, 2)


def breakdown(frame: pd.DataFrame, column: str) -> dict:
    """Corpus WER per (model, column value), as {model: {value: wer}}."""
    out: dict = {}
    for (model, value), group in frame.groupby(["model", column], observed=True):
        out.setdefault(model, {})[str(value)] = corpus_wer(group)
    return out


def spread(table: dict) -> dict:
    """Min and max across models of each value's WER, which is what the report quotes."""
    values = {v for per_model in table.values() for v in per_model}
    return {
        v: {
            "min": min(per_model[v] for per_model in table.values()),
            "max": max(per_model[v] for per_model in table.values()),
        }
        for v in sorted(values)
    }


def main() -> None:
    frame = pd.read_csv(PER_CLIP)
    frame = frame[frame["ref_words"] > 0]
    frame["has_disfluency"] = frame["n_fillers"] > 0

    duration = breakdown(frame, "duration_bucket")
    disfluency = breakdown(frame, "has_disfluency")
    digits = breakdown(frame, "has_numbers")

    # Speech rate is words/duration, so one reference-word error moves it by
    # 1/duration. On short clips that is large enough to make the metadata
    # unusable, which is why AMI is excluded from speech-rate breakdowns.
    # One row per clip, taken by fixing a single model rather than by
    # deduplicating on clip_id: SPGISpeech's published manifest stores only the
    # basename of each id, so 39,341 rows carry 421 distinct ids and a
    # drop_duplicates would discard 99% of the dataset. See section 0.
    one_model = sorted(frame["model"].unique())[0]
    per_dataset = []
    for dataset, group in frame[frame["model"] == one_model].groupby("dataset"):
        clips = group
        median_duration = float(clips["duration_s"].median())
        median_words = float(clips["ref_words"].median())
        per_dataset.append({
            "dataset": dataset,
            "median_duration_s": round(median_duration, 2),
            "shift_per_one_word_error_pct": round(100.0 / median_words, 1),
            "fillers_per_1000_words": round(
                1000.0 * clips["n_fillers"].sum() / clips["ref_words"].sum(), 2
            ),
        })
    per_dataset.sort(key=lambda r: r["fillers_per_1000_words"], reverse=True)

    results = {
        "n_rows": int(len(frame)),
        "n_models": int(frame["model"].nunique()),
        "duration_bucket": {"per_model": duration, "across_models": spread(duration)},
        "has_disfluency": {"per_model": disfluency, "across_models": spread(disfluency)},
        "has_numbers": {"per_model": digits, "across_models": spread(digits)},
        "per_dataset": per_dataset,
    }
    with open(OUT, "w") as handle:
        json.dump(results, handle, indent=1)

    print(f"{len(frame):,} rows, {results['n_models']} models\n")
    for name in ("duration_bucket", "has_disfluency", "has_numbers"):
        print(name)
        for value, bounds in results[name]["across_models"].items():
            print(f"  {value:8s} {bounds['min']:6.2f} to {bounds['max']:6.2f}")
    print("\nper dataset")
    for row in per_dataset:
        print(f"  {row['dataset']:32s} median {row['median_duration_s']:6.2f}s"
              f"  shift {row['shift_per_one_word_error_pct']:5.1f}%"
              f"  fillers/1k {row['fillers_per_1000_words']:6.2f}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
