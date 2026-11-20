"""Compare my reproduction to the leaderboard, at two levels.

Level A -- aggregate
    My WER per model per dataset against the board's published figure.
    This is the headline "did it reproduce" number, and it carries sampling
    error: I score 200 clips per dataset against their full splits.

Level C -- per clip
    My transcript against the board's own published transcript for the
    same clip. This is the diagnostic the aggregate cannot give: when the
    numbers disagree, it says WHICH clips differ and how.

Why both, and why Level C matters
---------------------------------
Level B (my scorer re-run on the board's own outputs) already passed on
52 of 53 manifests to within 0.02pp, so the arithmetic is known good. Any
gap at Level A is therefore inference, not scoring. Level C then localises
that gap to individual clips. During development this caught a bug where
audio was paired with the wrong clip's reference: the aggregate WER simply
looked bad, while the per-clip diff showed transcripts of entirely
different speech.

Scoring uses the board's vendored normalizer and kaldialign with
merge_compounds, matching their procedure exactly.

Output: results/reproduction.json

Run:  .venv/bin/python scripts/compare_to_board.py
"""

from __future__ import annotations

import csv
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))

from board_normalizer.normalizer import EnglishTextNormalizer  # noqa: E402

normalizer = EnglishTextNormalizer()

# Hub id -> the name the results bucket uses.
BUCKET_NAME = {
    "Qwen/Qwen3-ASR-1.7B-hf": "Qwen-Qwen3-ASR-1.7B-hf",
    "Musci-research/Musci-ASR-qwen-1019-hf": "Musci-research-Musci-ASR-qwen-1019-hf",
    "CohereLabs/cohere-transcribe-03-2026": "CohereLabs-cohere-transcribe-03-2026",
}
P = "hf-audio-open-asr-leaderboard_"
BOARD_SUFFIX = {
    "ami_cleaned": P + "ami_cleaned_test",
    "gigaspeech_cleaned": P + "gigaspeech_cleaned_test",
    "voxpopuli_cleaned_aa": P + "voxpopuli_cleaned_aa_test",
}

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 42


def errors_and_words(reference: str, prediction: str) -> tuple[int, int]:
    from kaldialign import edit_distance

    # The board applies normalize_compound_pairs ONLY when multilingual=True
    # (eval_utils._align_pair). Every English score file it publishes carries
    # "multilingual": false, so the English path goes straight to kaldialign with
    # merge_compounds=True. Applying it here as well made this scorer lenient by
    # about 0.004 WER points and broke exact agreement on one manifest.
    stats = edit_distance(
        tuple(reference.split()), tuple(prediction.split()), merge_compounds=True
    )
    return stats["ins"] + stats["del"] + stats["sub"], len(reference.split())


def bootstrap_interval(per_clip: list[tuple[int, int]]) -> tuple[float, float]:
    """Percentile bootstrap over clips, so within-clip correlation is kept."""
    import numpy as np

    errors = np.array([e for e, _ in per_clip], dtype=float)
    words = np.array([w for _, w in per_clip], dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(per_clip), size=(BOOTSTRAP_RESAMPLES, len(per_clip)))
    resampled = errors[draws].sum(axis=1) / words[draws].sum(axis=1) * 100
    return float(np.quantile(resampled, 0.025)), float(np.quantile(resampled, 0.975))


def board_published(bucket_name: str, suffix: str) -> float | None:
    path = os.path.join(
        BASE, "data", "board_results", "scores", bucket_name,
        f"MODEL_{bucket_name}_DATASET_{suffix}.score.json",
    )
    return json.load(open(path))["wer"] if os.path.exists(path) else None


def board_predictions(bucket_name: str, suffix: str) -> dict[str, str]:
    path = os.path.join(
        BASE, "data", "board_results", "predictions", bucket_name,
        f"MODEL_{bucket_name}_DATASET_{suffix}.jsonl",
    )
    if not os.path.exists(path):
        return {}
    out = {}
    for line in open(path):
        record = json.loads(line)
        out[record["audio_filepath"]] = record.get("pred_text") or ""
    return out


def main() -> None:
    references = {
        r["clip_id"]: r["ref_text"]
        for r in csv.DictReader(open(os.path.join(BASE, "data", "manifest.csv")))
    }
    ours = list(csv.DictReader(open(os.path.join(BASE, "data", "predictions.csv"))))

    results: dict = {}
    for model_id, bucket_name in BUCKET_NAME.items():
        model_rows = [r for r in ours if r["model"] == model_id]
        if not model_rows:
            continue
        for dataset, suffix in BOARD_SUFFIX.items():
            rows = [r for r in model_rows if r["dataset"] == dataset]
            if not rows:
                continue

            per_clip = []
            theirs = board_predictions(bucket_name, suffix)
            # Two senses of "the same transcript", reported separately because
            # they answer different questions. Raw equality says the decode
            # produced the identical string; equality after normalization says
            # the two would be scored identically, which is the weaker claim.
            agree = agree_raw = comparable = 0
            for row in rows:
                reference = normalizer(references[row["clip_id"]])
                if not reference.strip():
                    continue
                ours_text = normalizer(row["hypothesis"])
                per_clip.append(errors_and_words(reference, ours_text))
                if row["clip_id"] in theirs:
                    comparable += 1
                    agree += normalizer(theirs[row["clip_id"]]) == ours_text
                    agree_raw += theirs[row["clip_id"]].strip() == row["hypothesis"].strip()

            total_errors = sum(e for e, _ in per_clip)
            total_words = sum(w for _, w in per_clip)
            our_wer = 100.0 * total_errors / total_words
            low, high = bootstrap_interval(per_clip)
            published = board_published(bucket_name, suffix)

            results.setdefault(model_id, {})[dataset] = {
                "our_wer": round(our_wer, 3),
                "ci_low": round(low, 3),
                "ci_high": round(high, 3),
                "published_wer": published,
                "difference_pp": round(our_wer - published, 3) if published else None,
                "published_inside_ci": (
                    None if published is None else bool(low <= published <= high)
                ),
                "n_clips": len(per_clip),
                "transcript_agreement": (
                    round(agree / comparable, 4) if comparable else None
                ),
                "transcript_agreement_raw": (
                    round(agree_raw / comparable, 4) if comparable else None
                ),
                "n_comparable": comparable,
            }

    out_path = os.path.join(BASE, "results", "reproduction.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)

    print(f"{'model':26s} {'dataset':22s} {'ours (95% CI)':>22s} "
          f"{'board':>7s} {'diff':>8s} {'in CI':>6s} {'clip agree':>11s}")
    for model_id, datasets in results.items():
        for dataset, r in datasets.items():
            published = f"{r['published_wer']:.2f}" if r["published_wer"] else "  n/a"
            diff = f"{r['difference_pp']:+.2f}" if r["difference_pp"] is not None else "   n/a"
            inside = {True: "yes", False: "NO", None: "n/a"}[r["published_inside_ci"]]
            agree = f"{r['transcript_agreement']:.1%}" if r["transcript_agreement"] else "n/a"
            print(f"{model_id.split('/')[-1]:26s} {dataset:22s} "
                  f"{r['our_wer']:7.2f} [{r['ci_low']:5.2f},{r['ci_high']:5.2f}] "
                  f"{published:>7s} {diff:>8s} {inside:>6s} {agree:>11s}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
