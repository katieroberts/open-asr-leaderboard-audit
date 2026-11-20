"""Score every model on every public dataset, under each normalization rung.

Runs entirely on the board's published per-clip outputs: no inference, no
audio, no network. That means the attribution table covers the FULL public
splits (tens of thousands of clips per dataset) rather than the 1,541-clip
inference subset, so the normalization findings carry far more weight than
the reproduction does.

Output: results/ladder.json

Run:  .venv/bin/python scripts/score_ladder.py
"""

from __future__ import annotations

import json
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))
sys.path.insert(0, os.path.join(BASE, "scripts"))

from normalize_ladder import LADDER  # noqa: E402

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


def read_records(path: str) -> list[dict]:
    """Raw (text, pred_text) pairs, with chunked datasets joined per recording."""
    records = [json.loads(line) for line in open(path)]
    if records and "parent_id" in records[0]:
        by_parent: dict[str, list[dict]] = {}
        for record in records:
            by_parent.setdefault(record["parent_id"], []).append(record)
        joined = []
        for chunks in by_parent.values():
            chunks.sort(key=lambda c: c["chunk_index"])
            joined.append({
                "text": chunks[0]["text"],
                "pred_text": " ".join(c.get("pred_text") or "" for c in chunks),
            })
        return joined
    return records


def wer(references: list[str], predictions: list[str]) -> float:
    from kaldialign import edit_distance

    # The board applies normalize_compound_pairs ONLY when multilingual=True
    # (eval_utils._align_pair). Every English score file it publishes carries
    # "multilingual": false, so the English path goes straight to kaldialign with
    # merge_compounds=True. Applying it here as well made this scorer lenient by
    # about 0.004 WER points and broke exact agreement on one manifest.
    errors = words = 0
    for reference, prediction in zip(references, predictions):
        stats = edit_distance(
            tuple(reference.split()), tuple(prediction.split()), merge_compounds=True
        )
        errors += stats["ins"] + stats["del"] + stats["sub"]
        words += len(reference.split())
    return 100.0 * errors / words if words else float("nan")


def find_manifest(model_dir: str, suffix: str) -> str | None:
    directory = os.path.join(PREDICTIONS_DIR, model_dir)
    for filename in os.listdir(directory):
        if filename.endswith(".jsonl") and suffix in filename:
            return os.path.join(directory, filename)
    return None


def main() -> None:
    models = [
        d for d in SYSTEMS
        if os.path.isdir(os.path.join(PREDICTIONS_DIR, d))
    ]
    missing = [d for d in SYSTEMS if d not in models]
    if missing:
        raise SystemExit(f"no published outputs for {missing} in {PREDICTIONS_DIR}")
    results: dict = {}
    started = time.perf_counter()

    # Normalized-string cache, keyed on (rung, raw text). Keying on the TEXT is
    # the point: an earlier version keyed references on (dataset, rung, count)
    # and reused them across models, which silently paired one model's
    # predictions with another model's references in that model's order.
    # Record order differs between models (SPGISpeech is not even in dataset
    # order), so this produced plausible-looking but wrong WERs.
    norm_cache: dict[tuple[str, str], str] = {}

    def normalize(rung_name: str, normalizer, text: str) -> str:
        key = (rung_name, text)
        if key not in norm_cache:
            norm_cache[key] = normalizer(text)
        return norm_cache[key]

    for dataset_label, suffix in PUBLIC_DATASETS.items():
        for model_dir in models:
            path = find_manifest(model_dir, suffix)
            if path is None:
                continue
            records = read_records(path)
            keep = [r for r in records if (r.get("text") or "").strip()]

            for rung_name, normalizer in LADDER.items():
                # Pair each record's own reference with its own prediction, so
                # the two can never be drawn from different orderings.
                pairs = []
                for record in keep:
                    reference = normalize(rung_name, normalizer, record["text"])
                    if not reference.strip():
                        continue
                    prediction = normalize(
                        rung_name, normalizer, record.get("pred_text") or ""
                    )
                    pairs.append((reference, prediction))
                value = wer([p[0] for p in pairs], [p[1] for p in pairs])
                results.setdefault(model_dir, {}).setdefault(dataset_label, {})[rung_name] = {
                    "wer": value,
                    "n": len(pairs),
                }
            print(f"  {dataset_label:32s} {model_dir[:38]:38s} "
                  f"L0 {results[model_dir][dataset_label]['L0_strict']['wer']:6.2f}  "
                  f"L6 {results[model_dir][dataset_label]['L6_board']['wer']:6.2f}  "
                  f"({len(keep)} clips)", flush=True)

    out_path = os.path.join(BASE, "results", "ladder.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out_path} in {(time.perf_counter()-started)/60:.1f} min")


if __name__ == "__main__":
    main()
