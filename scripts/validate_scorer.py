"""Prove my scorer reproduces the board's published WER, before trusting it.

Why this runs first
-------------------
A reproduction has two independent places it can go wrong: my inference
and my scoring. If both are unverified, a gap between my number and the
board's is uninterpretable. This removes one of them.

The board publishes, per model per dataset, both its per-clip outputs
(`*.jsonl`) and the WER it computed from them (`*.score.json`). So I can
run MY scorer over THEIR predictions and check it lands on THEIR number.
If it does, the scorer is correct by construction, and any later discrepancy
is attributable to inference alone.

This needs no audio, no models and no network.

Run:  .venv/bin/python scripts/validate_scorer.py
"""

from __future__ import annotations

import json
import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))

from board_normalizer.normalizer import EnglishTextNormalizer  # noqa: E402

PREDICTIONS_DIR = os.path.join(BASE, "data", "board_results", "predictions")
SCORES_DIR = os.path.join(BASE, "data", "board_results", "scores")

# How closely the match must be to call the scorer correct. The board rounds its
# published WER to 2dp, so anything inside half a unit in the last place is
# agreement; this demands considerably tighter than that.
TOLERANCE_PP = 0.02

normalizer = EnglishTextNormalizer()


def read_pairs(path: str) -> tuple[list[str], list[str]]:
    """(reference, prediction) pairs for one manifest.

    Chunked datasets need different handling. For those the board sets every
    chunk's `text` to the FULL parent transcript (`data_utils.load_chunked_data`:
    `sample["text"] = parent_text[sample["parent_id"]]`), so scoring a chunk
    against its own `text` compares a fragment to the whole recording and
    yields ~98% WER. The chunks must be concatenated per `parent_id` in
    `chunk_index` order and scored once per recording.
    """
    records = [json.loads(line) for line in open(path)]
    is_chunked = bool(records) and "parent_id" in records[0]

    if is_chunked:
        by_parent: dict[str, list[dict]] = {}
        for record in records:
            by_parent.setdefault(record["parent_id"], []).append(record)
        records = []
        for parent_id, chunks in by_parent.items():
            chunks.sort(key=lambda c: c["chunk_index"])
            records.append(
                {
                    "text": chunks[0]["text"],
                    "pred_text": " ".join(c.get("pred_text") or "" for c in chunks),
                }
            )

    references, predictions = [], []
    for record in records:
        reference = normalizer(record["text"])
        prediction = normalizer(record.get("pred_text") or "")
        # The board drops clips whose normalized reference is empty; WER is
        # undefined with a zero-length reference.
        if not reference.strip():
            continue
        references.append(reference)
        predictions.append(prediction)
    return references, predictions


def score_manifest(path: str) -> tuple[float, int]:
    """WER over one manifest, following the board's own procedure."""
    from kaldialign import edit_distance

    references, predictions = read_pairs(path)

    # The board applies normalize_compound_pairs ONLY when multilingual=True
    # (eval_utils._align_pair). Every English score file it publishes carries
    # "multilingual": false, so the English path goes straight to kaldialign with
    # merge_compounds=True. Applying it here as well made this scorer lenient by
    # about 0.004 WER points and broke exact agreement on one manifest.

    total_errors = 0
    total_words = 0
    for reference, prediction in zip(references, predictions):
        stats = edit_distance(
            tuple(reference.split()), tuple(prediction.split()), merge_compounds=True
        )
        total_errors += stats["ins"] + stats["del"] + stats["sub"]
        total_words += len(reference.split())
    return 100.0 * total_errors / total_words, len(references)


def main() -> None:
    rows = []
    for model_dir in sorted(os.listdir(PREDICTIONS_DIR)):
        model_path = os.path.join(PREDICTIONS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
        for filename in sorted(os.listdir(model_path)):
            if not filename.endswith(".jsonl"):
                continue
            score_path = os.path.join(
                SCORES_DIR, model_dir, filename.replace(".jsonl", ".score.json")
            )
            if not os.path.exists(score_path):
                continue
            published = json.load(open(score_path))
            if published.get("multilingual"):
                continue
            dataset = re.search(r"_DATASET_(.+?)\.jsonl", filename).group(1)
            ours, n_clips = score_manifest(os.path.join(model_path, filename))
            rows.append(
                {
                    "model": model_dir,
                    "dataset": dataset,
                    "ours": ours,
                    "published": published["wer"],
                    "delta": ours - published["wer"],
                    "n_clips": n_clips,
                }
            )
            print(
                f"  {model_dir[:34]:34s} {dataset[-28:]:28s} "
                f"ours {ours:6.2f}  published {published['wer']:6.2f}  "
                f"delta {ours - published['wer']:+6.3f}"
            )

    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)
    with open(os.path.join(BASE, "results", "scorer_validation.json"), "w") as handle:
        json.dump(rows, handle, indent=2)

    failures = [r for r in rows if abs(r["delta"]) > TOLERANCE_PP]
    worst = max((abs(r["delta"]) for r in rows), default=0.0)
    print(f"\n{len(rows) - len(failures)}/{len(rows)} manifests match within "
          f"{TOLERANCE_PP}pp.  worst |delta| = {worst:.3f}pp")
    if failures:
        print("\nWorst mismatches:")
        for r in sorted(failures, key=lambda r: -abs(r["delta"]))[:10]:
            print(f"  {r['model'][:34]:34s} {r['dataset'][-28:]:28s} {r['delta']:+.3f}pp")


if __name__ == "__main__":
    main()
