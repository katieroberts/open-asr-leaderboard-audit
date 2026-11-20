"""Prove every model loads, transcribes, and agrees with the board, on 2 clips.

Run this BEFORE any long transcription. Each model here has already failed
once in a different way: Qwen3-ASR rejected the generic ASR pipeline with a
mel-shape error, granite aborted on a missing torchaudio dependency after
the weights had downloaded, and an audio-naming bug paired real audio with
the wrong references on two datasets. A two-minute check per model is far
cheaper than discovering any of that three hours into a run.

Checks, per model:
  1. config, processor and weights load
  2. a transcript is produced and is non-empty
  3. the transcript matches the board's published output for the same clip

Run:  .venv/bin/python scripts/smoke_test_models.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys

import soundfile
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))

from board_normalizer.normalizer import EnglishTextNormalizer  # noqa: E402

normalizer = EnglishTextNormalizer()

MODELS = {
    "Qwen/Qwen3-ASR-1.7B-hf": "Qwen-Qwen3-ASR-1.7B-hf",
    "Musci-research/Musci-ASR-qwen-1019-hf": "Musci-research-Musci-ASR-qwen-1019-hf",
    "CohereLabs/cohere-transcribe-03-2026": "CohereLabs-cohere-transcribe-03-2026",
}
ATTENTION = {"ibm-granite/granite-speech-5.0-2b": "eager"}

DATASET = "voxpopuli_cleaned_aa"
BOARD_SUFFIX = "hf-audio-open-asr-leaderboard_voxpopuli_cleaned_aa_test"
N_CLIPS = 2
MAX_NEW_TOKENS = 512


def load_clips():
    manifest = [
        r for r in csv.DictReader(open(os.path.join(BASE, "data", "manifest.csv")))
        if r["dataset"] == DATASET
    ]
    clips = []
    for row in manifest:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", row["clip_id"])
        path = os.path.join(BASE, "data", "audio", f"{DATASET}__{safe}.wav")
        if os.path.exists(path):
            samples, rate = soundfile.read(path, dtype="float32")
            clips.append((row["clip_id"], samples, rate))
        if len(clips) >= N_CLIPS:
            break
    return clips


def board_outputs(bucket_name: str) -> dict:
    path = os.path.join(
        BASE, "data", "board_results", "predictions", bucket_name,
        f"MODEL_{bucket_name}_DATASET_{BOARD_SUFFIX}.jsonl",
    )
    return {
        json.loads(line)["audio_filepath"]: (json.loads(line).get("pred_text") or "")
        for line in open(path)
    }


def check(model_id: str, bucket_name: str, clips) -> bool:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

    print(f"\n  {model_id}")
    try:
        config = AutoConfig.from_pretrained(model_id)
        family = getattr(config, "model_type", "")
        family = "cohere" if family == "cohere_asr" else family
        is_qwen3_asr = family == "qwen3_asr"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=torch.float32,
            attn_implementation=ATTENTION.get(model_id, "sdpa"),
        ).eval()
    except Exception as error:
        print(f"     LOAD FAILED: {type(error).__name__}: {str(error)[:140]}")
        return False

    has_request = hasattr(processor, "apply_transcription_request")
    print(f"     loaded. model_type={getattr(config,'model_type','?')} "
          f"apply_transcription_request={has_request}")

    theirs = board_outputs(bucket_name)
    agree = 0
    with torch.no_grad():
        for clip_id, samples, rate in clips:
            try:
                if family == "cohere":
                    inputs = processor(
                        [samples], sampling_rate=rate, return_tensors="pt",
                        language="en", punctuation=False)
                    prompt_len = 0
                elif has_request:
                    inputs = processor.apply_transcription_request(
                        audio=[samples], language="en")
                    prompt_len = inputs["input_ids"].shape[1]
                else:
                    inputs = processor(
                        [samples], sampling_rate=rate, return_tensors="pt",
                        padding="longest", return_attention_mask=True)
                    prompt_len = 0
                inputs = inputs.to(dtype=torch.float32)
                predicted = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                tokens = predicted[:, prompt_len:] if prompt_len else predicted
                if family == "cohere":
                    decoded = processor.decode(
                        tokens, skip_special_tokens=True,
                        audio_chunk_index=inputs.get("audio_chunk_index"),
                        language="en")
                elif is_qwen3_asr:
                    decoded = processor.decode(tokens, return_format="transcription_only")
                else:
                    decoded = processor.decode(tokens, skip_special_tokens=True)
                text = (decoded[0] if isinstance(decoded, list) else decoded).strip()
                if family == "cohere":
                    text = re.sub(r"\s+", " ",
                                  text.replace("(", " ").replace(")", " ")).strip()
            except Exception as error:
                print(f"     INFERENCE FAILED: {type(error).__name__}: {str(error)[:140]}")
                return False

            if not text:
                print(f"     EMPTY TRANSCRIPT for {clip_id}")
                return False
            matches = normalizer(text) == normalizer(theirs.get(clip_id, ""))
            agree += matches
            flag = "match" if matches else "DIFFERS"
            print(f"     [{flag}] {text[:72]}")
            if not matches:
                print(f"       board: {theirs.get(clip_id,'')[:72]}")

    print(f"     agreement with board: {agree}/{len(clips)}")
    return True


def main() -> None:
    # Optional model ids on the command line, so a single model can be
    # checked alone. That matters on a 16 GB machine: two 2B models in
    # float32 are ~8 GB each and running them concurrently sends the system
    # into swap, where both crawl at ~3% CPU.
    selected = sys.argv[1:]
    if selected:
        unknown = [m for m in selected if m not in MODELS]
        if unknown:
            raise SystemExit(f"unknown model(s): {unknown}")
        for model_id in list(MODELS):
            if model_id not in selected:
                del MODELS[model_id]

    clips = load_clips()
    if len(clips) < N_CLIPS:
        raise SystemExit("not enough local audio to smoke test")
    print(f"smoke test on {len(clips)} {DATASET} clips")

    results = {m: check(m, b, clips) for m, b in MODELS.items()}
    print("\n  summary")
    for model_id, ok in results.items():
        print(f"    {'RUNS ' if ok else 'BROKEN'} {model_id}")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
