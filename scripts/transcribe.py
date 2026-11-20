"""Transcribe the evaluation subset, following the board's own decode path.

Configuration, and why
----------------------
Taken from `transformers/submit_jobs_qwen3asr.sh` so that a difference between
my numbers and the board's is attributable to inference rather than to a
decoding choice made by accident. A decode setting that quietly differs from
the reference implementation can account for an entire discrepancy while
looking like a model difference, so these are read from the board's scripts
rather than guessed.

  max_new_tokens   512          board's value, "matches the official
                                qwen-asr package default"
  attention        sdpa         board default; granite overrides to eager
                                in its own submit script, so it is set
                                per-model here
  dtype            float32      DEVIATION. The board runs bfloat16 on H200s.
                                On this CPU bfloat16 is emulated and 1.6x
                                SLOWER, and measured output was identical on
                                6/6 clips, so float32 is a free speedup.
  batch size       8            DEVIATION. The board uses 256 on a GPU. 8 was
                                measured at 1.8x the throughput of batch 1
                                with transcripts identical on 8/8 clips
                                (scripts/test_batching.py).
  device           cpu          DEVIATION, hardware available.

Model input is built with `processor.apply_transcription_request(...)` where
the processor provides it, which is what run_eval.py does for this family.
The generic `pipeline("automatic-speech-recognition")` does NOT work for
Qwen3-ASR: it fails with a mel-shape error, because the model expects a
transcription request (audio plus instruction prompt), not a bare waveform.

Earnings22 is excluded from inference: six recordings of ~20 minutes each,
42% of the audio budget for the least reliable number in the suite, and a
long-form chunked decode unlike the other six. It is still reported, from
the board's published outputs, in the analyses that need no inference.

Output: data/predictions.csv, resumable.

Run:  .venv/bin/python scripts/transcribe.py [model_id ...]
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time

import soundfile
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE, "data", "audio")
PREDICTIONS_PATH = os.path.join(BASE, "data", "predictions.csv")

# Three systems the board ranks. Qwen3-ASR-1.7B and Musci-ASR are a
# controlled pair -- Musci is a GRPO fine-tune of Qwen3-ASR, same
# architecture and size, so the difference between them isolates what a
# leaderboard-targeted fine-tune buys. cohere-transcribe is a separate
# lineage, so the set is not a single family.
MODELS = [
    "Qwen/Qwen3-ASR-1.7B-hf",
    "Musci-research/Musci-ASR-qwen-1019-hf",
    "CohereLabs/cohere-transcribe-03-2026",
]

# granite's submit script passes --attn_implementation=eager explicitly;
# everything else uses run_eval.py's sdpa default.
ATTENTION_OVERRIDE = {"ibm-granite/granite-speech-5.0-2b": "eager"}

# The brief asks for "a meaningful subset of its evaluation data", not all of
# it. These three span the axis that drives ASR error -- scripted oratory
# (VoxPopuli), edited media (GigaSpeech) and spontaneous multi-party meetings
# (AMI) -- and, crucially, each was VERIFIED to align with the board's own
# per-clip outputs before being used (94-99% hypothesis agreement).
#
# The other four public datasets are reported from the board's published
# outputs in the offline analyses, and set aside for inference. None of these
# is a dead end; each is a problem that was not worth solving inside the time
# available, given that three datasets already cover the scripted-to-
# spontaneous axis:
#   spgispeech   the board's manifest stores only the basename of each clip
#                id, so 39,341 rows carry 421 distinct ids; resolving a clip
#                back to a dataset row would need text matching, not tried
#   voice_arena_monsoon  renamed upstream; the dataset now behind that name
#                matched 0/200 clips. The revision the board evaluated may
#                still be locatable, not chased down
#   urgent2024   matched 46%; dropped rather than diagnosed, cause unknown
#   earnings22   six 20-minute recordings, long-form chunked decode; a
#                compute budget call rather than a blocker
INFERENCE_DATASETS = {
    "ami_cleaned",
    "gigaspeech_cleaned",
    "voxpopuli_cleaned_aa",
}
CLIPS_PER_DATASET = 200
MAX_NEW_TOKENS = 512
BATCH_SIZE = 8
DTYPE = torch.float32

FIELDNAMES = ["clip_id", "dataset", "model", "hypothesis", "audio_s", "infer_s"]


def load_manifest() -> list[dict]:
    with open(os.path.join(BASE, "data", "manifest.csv"), newline="") as handle:
        rows = list(csv.DictReader(handle))

    chosen: list[dict] = []
    seen: dict[str, int] = {}
    for row in rows:
        dataset = row["dataset"]
        if dataset not in INFERENCE_DATASETS:
            continue
        if seen.get(dataset, 0) >= CLIPS_PER_DATASET:
            continue
        # By clip id, never by row number: see safe_name() in fetch_audio.py.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", row["clip_id"])
        path = os.path.join(AUDIO_DIR, f"{dataset}__{safe}.wav")
        if not os.path.exists(path):
            continue
        row["audio_path"] = path
        chosen.append(row)
        seen[dataset] = seen.get(dataset, 0) + 1
    return chosen


def load_done() -> set[tuple[str, str]]:
    if not os.path.exists(PREDICTIONS_PATH):
        return set()
    with open(PREDICTIONS_PATH, newline="") as handle:
        return {(r["clip_id"], r["model"]) for r in csv.DictReader(handle)}


def remove_brackets(text: str) -> str:
    """Drop parentheses, as the board does for Cohere output.

    Their normalizer deletes bracketed spans entirely, so a model that emits
    parentheses would have the enclosed words removed and be charged
    deletions for them.
    """
    return re.sub(r"\s+", " ", text.replace("(", " ").replace(")", " ")).strip()


def build_inputs(processor, audios, model_dtype, family):
    """Model input, following run_eval.py's per-family branching."""
    if family == "cohere":
        # Cohere's processor takes the task arguments directly; the board
        # passes punctuation=False so output matches the reference style.
        inputs = processor(
            audios, sampling_rate=16000, return_tensors="pt",
            language="en", punctuation=False,
        )
        prompt_len = 0
    elif hasattr(processor, "apply_transcription_request"):
        inputs = processor.apply_transcription_request(audio=audios, language="en")
        prompt_len = inputs["input_ids"].shape[1]
    else:
        inputs = processor(
            audios, sampling_rate=16000, return_tensors="pt",
            padding="longest", return_attention_mask=True,
        )
        prompt_len = 0
    return inputs.to(dtype=model_dtype), prompt_len


def decode(processor, predicted, prompt_len, family, inputs=None):
    tokens = predicted[:, prompt_len:] if prompt_len else predicted
    if family == "cohere":
        decoded = processor.decode(
            tokens, skip_special_tokens=True,
            audio_chunk_index=inputs.get("audio_chunk_index"), language="en",
        )
        decoded = decoded if isinstance(decoded, list) else [decoded]
        return [remove_brackets(t) for t in decoded]
    if family == "qwen3_asr":
        decoded = processor.decode(tokens, return_format="transcription_only")
    else:
        decoded = processor.decode(tokens, skip_special_tokens=True)
    return decoded if isinstance(decoded, list) else [decoded]


def transcribe_model(model_id: str, manifest, done, writer, handle) -> None:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

    pending = [r for r in manifest if (r["clip_id"], model_id) not in done]
    if not pending:
        print(f"  {model_id}: nothing to do")
        return
    print(f"  {model_id}: {len(pending)} clips", flush=True)

    config = AutoConfig.from_pretrained(model_id)
    family = getattr(config, "model_type", "")
    family = "cohere" if family == "cohere_asr" else family
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=DTYPE,
        attn_implementation=ATTENTION_OVERRIDE.get(model_id, "sdpa"),
    ).eval()
    model_dtype = next(model.parameters()).dtype

    with torch.no_grad():
        for start in range(0, len(pending), BATCH_SIZE):
            group = pending[start : start + BATCH_SIZE]
            audios, durations = [], []
            for row in group:
                samples, sample_rate = soundfile.read(row["audio_path"], dtype="float32")
                audios.append(samples)
                durations.append(len(samples) / sample_rate)

            started = time.perf_counter()
            inputs, prompt_len = build_inputs(processor, audios, model_dtype, family)
            predicted = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
            texts = decode(processor, predicted, prompt_len, family, inputs)
            elapsed = time.perf_counter() - started

            # Per-clip time is the batch time shared out, as the board does
            # ("normalize by minibatch size since we want the per-sample time").
            per_clip = elapsed / len(group)
            for row, text, duration in zip(group, texts, durations):
                writer.writerow({
                    "clip_id": row["clip_id"],
                    "dataset": row["dataset"],
                    "model": model_id,
                    "hypothesis": (text or "").strip(),
                    "audio_s": round(duration, 4),
                    "infer_s": round(per_clip, 4),
                })
            handle.flush()
            if (start // BATCH_SIZE) % 5 == 0:
                print(f"    {start + len(group)}/{len(pending)}", flush=True)
    print(f"  {model_id}: done", flush=True)


def main() -> None:
    models = sys.argv[1:] or MODELS
    manifest = load_manifest()
    if not manifest:
        raise SystemExit("No manifest clips with audio on disk; run fetch_audio.py")

    counts: dict[str, int] = {}
    for row in manifest:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    total_audio = sum(float(r["duration_s"]) for r in manifest)
    print(f"{len(manifest)} clips, {total_audio/3600:.2f}h audio")
    for dataset, count in sorted(counts.items()):
        print(f"   {dataset:32s} {count}")
    print()

    done = load_done()
    is_new = not os.path.exists(PREDICTIONS_PATH)
    with open(PREDICTIONS_PATH, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if is_new or os.path.getsize(PREDICTIONS_PATH) == 0:
            writer.writeheader()
        for model_id in models:
            transcribe_model(model_id, manifest, done, writer, handle)


if __name__ == "__main__":
    main()
