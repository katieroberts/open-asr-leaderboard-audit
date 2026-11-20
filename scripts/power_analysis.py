"""Why 200 clips per dataset, derived rather than chosen.

A subset is worth running only if it reaches the same conclusion as the full
corpus. So this asks, for a given sample size, how often a seeded draw recovers
the ranking the full public splits produce.

Method: the board publishes per-clip outputs, so the full-corpus answer is known.
For each trial, draw n clips per dataset without replacement, score every system
on that draw (corpus WER, macro-averaged over datasets), and check whether the
resulting order of systems matches the full-corpus order. The reported figure is
the share of trials that match.

This runs on results/per_clip.csv.gz, so it needs no inference and no audio.

Output: results/power_analysis.json

Run:  .venv/bin/python scripts/power_analysis.py
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PER_CLIP = os.path.join(BASE, "results", "per_clip.csv.gz")
OUT = os.path.join(BASE, "results", "power_analysis.json")

SEED = 42
TRIALS = 500
SAMPLE_SIZES = (100, 150, 200, 300)

# SPGISpeech is excluded: its published manifest stores only the basename of each
# clip id, so a sampled clip cannot be resolved back to a dataset row and it is
# not a candidate for the inference subset in the first place. See section 0.
EXCLUDED_DATASETS = ("spgispeech",)


def macro_wer(frame: pd.DataFrame) -> pd.Series:
    """Per-model macro WER: corpus WER within each dataset, then averaged."""
    per_dataset = frame.groupby(["model", "dataset"], observed=True).apply(
        lambda g: 100.0 * g["errors"].sum() / g["ref_words"].sum(), include_groups=False
    )
    return per_dataset.groupby("model").mean()


def ranking(scores: pd.Series) -> tuple:
    return tuple(scores.sort_values().index)


def main() -> None:
    frame = pd.read_csv(PER_CLIP)
    frame = frame[(frame["ref_words"] > 0) & (~frame["dataset"].isin(EXCLUDED_DATASETS))]

    truth = ranking(macro_wer(frame))
    print("full-corpus ranking:")
    for position, model in enumerate(truth, 1):
        print(f"  {position}. {model}")
    print()

    # Clip ids per dataset, sampled once and shared across models, because the
    # board scores every system on the same clips and so does the reproduction.
    clips_by_dataset = {
        dataset: group["clip_id"].unique()
        for dataset, group in frame.groupby("dataset", observed=True)
    }
    indexed = frame.set_index(["dataset", "clip_id"]).sort_index()

    rng = np.random.default_rng(SEED)
    results = {}
    for size in SAMPLE_SIZES:
        matches = 0
        for _ in range(TRIALS):
            keys = []
            for dataset, clips in clips_by_dataset.items():
                take = min(size, len(clips))
                chosen = rng.choice(clips, size=take, replace=False)
                keys.extend((dataset, clip) for clip in chosen)
            draw = indexed.loc[keys]
            if ranking(macro_wer(draw.reset_index())) == truth:
                matches += 1
        results[size] = round(100.0 * matches / TRIALS, 1)
        print(f"n={size:4d} per dataset   recovers the full ranking {results[size]:5.1f}% of trials")

    with open(OUT, "w") as handle:
        json.dump(
            {
                "seed": SEED,
                "trials": TRIALS,
                "excluded_datasets": list(EXCLUDED_DATASETS),
                "full_corpus_ranking": list(truth),
                "recovery_rate_pct": results,
            },
            handle,
            indent=1,
        )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
