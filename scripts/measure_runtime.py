"""Measure inference speed and dtype sensitivity on locally-cached audio.

Why this replaces the earlier version
-------------------------------------
The first attempt timed `pipeline(...)` construction together with
inference. On a slow connection that meant it was measuring a model
download, and it produced a confident "RTFx under 0.25, ~15 hours per
model" that was pure network latency. Here the model is loaded and warmed
BEFORE the clock starts, and the audio is already on disk, so nothing in
the timed region touches the network.

It also answers the dtype question. The board runs bfloat16 on H200s
(`run_eval.py` argparse default). I run on CPU, where float32 is the
natural choice. Rather than assume the difference is immaterial, both are
timed and their transcripts compared: if they differ, that is a deviation
to report; if they agree, the choice can be made on speed alone.

Run:  .venv/bin/python scripts/measure_runtime.py [model_id ...]
"""

from __future__ import annotations

import glob
import os
import sys
import time

import soundfile
import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE, "data", "audio")

DEFAULT_MODELS = ["Qwen/Qwen3-ASR-1.7B-hf"]
N_CLIPS = 6

# The agreed run: 200 clips from each of 7 public datasets, 4.54 h of audio.
FULL_RUN_AUDIO_SECONDS = 4.54 * 3600

# The board's own setting, from transformers/run_eval.py's argparse default.
BOARD_DTYPE = torch.bfloat16


def load_clips(n: int):
    paths = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))[:n]
    clips = []
    for path in paths:
        samples, sample_rate = soundfile.read(path, dtype="float32")
        clips.append((os.path.basename(path), samples, sample_rate))
    return clips


# The board's own generation limit for the Qwen3-ASR family, from
# transformers/submit_jobs_qwen3asr.sh ("matches the official qwen-asr
# package default").
MAX_NEW_TOKENS = 512


def transcribe_one(model, processor, samples, sample_rate, is_qwen3_asr):
    """One clip, following `transformers/run_eval.py`'s own code path.

    The generic `pipeline("automatic-speech-recognition")` does NOT work for
    these models: it fails with a mel-shape error, because Qwen3-ASR expects
    a transcription *request* (audio plus an instruction prompt), not a bare
    waveform. The board builds that request with
    `processor.apply_transcription_request(...)` and then strips the prompt
    tokens off the generated ids before decoding. Doing anything else here
    would measure the harness rather than the model.
    """
    inputs = processor.apply_transcription_request(audio=samples, language="en")
    # Feature tensors come out float32; cast them to the model's dtype, as
    # run_eval.py does (`inputs.to(args.device, dtype=torch_dtype)`).
    # Without this a bfloat16 model rejects float32 features outright.
    inputs = inputs.to(dtype=dtype_of(model))
    prompt_len = inputs["input_ids"].shape[1]
    predicted_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    if is_qwen3_asr:
        decoded = processor.decode(
            predicted_ids[:, prompt_len:], return_format="transcription_only"
        )
    else:
        decoded = processor.decode(predicted_ids[:, prompt_len:], skip_special_tokens=True)
    # The processor decodes a batch, so a single clip comes back as a
    # one-element list.
    return decoded[0] if isinstance(decoded, list) else decoded


def dtype_of(model):
    return next(model.parameters()).dtype


def time_model(model_id: str, clips, dtype) -> dict:
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

    config = AutoConfig.from_pretrained(model_id)
    is_qwen3_asr = getattr(config, "model_type", "") == "qwen3_asr"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=dtype, attn_implementation="sdpa"
    ).eval()

    # Warm up outside the timed region: the first call pays one-off
    # initialization that would otherwise be charged to the sample.
    with torch.no_grad():
        transcribe_one(model, processor, clips[0][1], clips[0][2], is_qwen3_asr)

    audio_seconds = 0.0
    compute_seconds = 0.0
    transcripts = []
    with torch.no_grad():
        for _, samples, sample_rate in clips:
            started = time.perf_counter()
            text = transcribe_one(model, processor, samples, sample_rate, is_qwen3_asr)
            compute_seconds += time.perf_counter() - started
            audio_seconds += len(samples) / sample_rate
            transcripts.append((text or "").strip())

    return {
        "audio_s": audio_seconds,
        "compute_s": compute_seconds,
        "rtfx": audio_seconds / compute_seconds,
        "transcripts": transcripts,
    }


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    clips = load_clips(N_CLIPS)
    if not clips:
        raise SystemExit(f"No audio in {AUDIO_DIR}; run scripts/fetch_audio.py first")
    total_audio = sum(len(s) / r for _, s, r in clips)
    print(f"{len(clips)} local clips, {total_audio:.1f}s audio "
          f"(no network in the timed region)\n")

    for model_id in models:
        print(f"{model_id}")
        results = {}
        for label, dtype in (("float32", torch.float32), ("bfloat16", BOARD_DTYPE)):
            try:
                results[label] = time_model(model_id, clips, dtype)
                r = results[label]
                projected = FULL_RUN_AUDIO_SECONDS / r["rtfx"] / 3600
                print(f"   {label:9s} {r['compute_s']:7.1f}s for {r['audio_s']:.0f}s audio"
                      f"   RTFx {r['rtfx']:6.2f}   full run {projected:5.2f}h")
            except Exception as error:
                print(f"   {label:9s} FAILED {type(error).__name__}: {str(error)[:90]}")

        if len(results) == 2:
            a = results["float32"]["transcripts"]
            b = results["bfloat16"]["transcripts"]
            identical = sum(1 for x, y in zip(a, b) if x == y)
            print(f"   transcripts identical across dtypes: {identical}/{len(a)}")
            for x, y in zip(a, b):
                if x != y:
                    print(f"      fp32: {x[:90]}")
                    print(f"      bf16: {y[:90]}")
                    break
        print()


if __name__ == "__main__":
    main()
