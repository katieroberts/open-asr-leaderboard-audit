"""Disfluency fidelity: does a model transcribe what the speaker actually said?

The question
------------
The leaderboard's normalizer deletes filler words from BOTH the reference
and the hypothesis before scoring, so a model that writes every "um" and a
model that silently drops them score identically. That is deliberate: it
stops a transcription-style difference from being counted as an accuracy
difference.

But it also means WER is structurally blind to the difference, and the
difference is not cosmetic. A verbatim requirement (legal discovery,
medical records, qualitative research) makes a model that discards
disfluencies unusable regardless of its WER; a summarization pipeline
would rather not strip them itself. The leaderboard cannot distinguish
these two products.

The measurement
---------------
For each (model, dataset): count filler tokens in the reference and in the
hypothesis, where "filler" means exactly what the board's own normalizer
removes (its `ignore_patterns` regex), so this measures precisely the
quantity the board discards.

    retention = filler tokens emitted / filler tokens in the reference

Near 1.0 means verbatim. Near 0 means the model cleans up silently.

Also recorded: the WER the model would be charged WITHOUT the filler rule
(L0) and WITH it (L1), to show how much of its score depends on the rule.

Caveats this reports rather than hides
--------------------------------------
- Retention is a ratio of counts, not an alignment: a model could emit
  fillers in the wrong places and still score 1.0. It is a style
  indicator, not a correctness measure.
- Datasets whose references are already normalized (Monsoon, LibriSpeech)
  carry few or no fillers, so retention there is undefined or noisy. Any
  dataset with too few reference fillers is reported as such rather than
  given a meaningless ratio.

Runs on the board's published outputs: no inference, no audio, no network.

Output: results/disfluency.json

Run:  .venv/bin/python scripts/disfluency_fidelity.py
"""

from __future__ import annotations

import json
import os
import sys

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
WITHOUT_FILLER_RULE = LADDER["L0_strict"]
WITH_FILLER_RULE = LADDER["L1_fillers"]

# Below this many reference fillers a retention ratio is noise, not signal.
MIN_REFERENCE_FILLERS = 50

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


def filler_count(text: str) -> int:
    """Tokens the board's filler rule would delete from this string."""
    return len(WITHOUT_FILLER_RULE(text).split()) - len(WITH_FILLER_RULE(text).split())


def repetition_count(text: str) -> int:
    """Immediately repeated words, counted after the filler rule has run.

    The control for "is this model just cleaning up everything?". Filled pauses
    and stutter-style repetitions are both disfluency, so a model that rewrites
    speech into tidy prose should drop both. One that drops only filled pauses is
    making a targeted choice. Counted on the filler-stripped text so that a
    repeated "uh uh" cannot be mistaken for a repeated word.
    """
    words = WITH_FILLER_RULE(text).split()
    return sum(1 for a, b in zip(words, words[1:]) if a == b)


def read_records(path: str) -> list[dict]:
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


def wer(pairs) -> float:
    from kaldialign import edit_distance

    references = [p[0] for p in pairs]
    predictions = [p[1] for p in pairs]
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


def find_manifest(model_dir: str, suffix: str):
    directory = os.path.join(PREDICTIONS_DIR, model_dir)
    for filename in sorted(os.listdir(directory)):
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

    for dataset_label, suffix in PUBLIC_DATASETS.items():
        for model_dir in models:
            path = find_manifest(model_dir, suffix)
            if path is None:
                continue
            records = [r for r in read_records(path) if (r.get("text") or "").strip()]

            reference_fillers = 0
            hypothesis_fillers = 0
            reference_repetitions = 0
            hypothesis_repetitions = 0
            pairs_without = []
            pairs_with = []
            for record in records:
                reference = record["text"]
                hypothesis = record.get("pred_text") or ""
                reference_fillers += filler_count(reference)
                hypothesis_fillers += filler_count(hypothesis)
                reference_repetitions += repetition_count(reference)
                hypothesis_repetitions += repetition_count(hypothesis)

                stripped = WITHOUT_FILLER_RULE(reference)
                if stripped.strip():
                    pairs_without.append((stripped, WITHOUT_FILLER_RULE(hypothesis)))
                kept = WITH_FILLER_RULE(reference)
                if kept.strip():
                    pairs_with.append((kept, WITH_FILLER_RULE(hypothesis)))

            enough = reference_fillers >= MIN_REFERENCE_FILLERS
            entry = {
                "reference_fillers": reference_fillers,
                "hypothesis_fillers": hypothesis_fillers,
                "retention": (hypothesis_fillers / reference_fillers) if enough else None,
                "measurable": enough,
                "wer_without_filler_rule": wer(pairs_without),
                "wer_with_filler_rule": wer(pairs_with),
                "reference_repetitions": reference_repetitions,
                "hypothesis_repetitions": hypothesis_repetitions,
                "repetition_retention": (
                    (hypothesis_repetitions / reference_repetitions)
                    if reference_repetitions >= MIN_REFERENCE_FILLERS else None
                ),
                "n_records": len(records),
            }
            entry["rule_worth_pp"] = (
                entry["wer_without_filler_rule"] - entry["wer_with_filler_rule"]
            )
            results.setdefault(model_dir, {})[dataset_label] = entry

            retention = f"{entry['retention']:6.1%}" if enough else "   n/a"
            print(f"  {dataset_label:30s} {model_dir[:34]:34s} "
                  f"ref {reference_fillers:6d}  hyp {hypothesis_fillers:6d}  "
                  f"retention {retention}  rule worth {entry['rule_worth_pp']:+6.2f}pp",
                  flush=True)

    out_path = os.path.join(BASE, "results", "disfluency.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
