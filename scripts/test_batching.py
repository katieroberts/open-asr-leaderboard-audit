"""Does batching change the output, and does it pay for itself on CPU?

The board runs at batch 256; this machine cannot, so this measures what the difference
costs. Batching is not automatically output-neutral: padding a batch to its
longest member feeds real signal to the model, and whether that changes the
result depends on how the architecture handles the padded region. Qwen3-ASR is
a seq2seq model with proper attention masks, where it should not matter. That
is a claim about the model, not a guarantee, so it is tested rather than
assumed.

A speedup that changes the output is not a speedup; it is a different
experiment.

Two questions, one run:
  1. speed      - does batching actually help on CPU, where there is no
                  GPU parallelism to exploit?
  2. fidelity   - are the transcripts identical to batch-1 decoding?

A speedup that changes the output is not a speedup; it is a different
experiment.

Run:  .venv/bin/python scripts/test_batching.py
"""

from __future__ import annotations

import glob
import os
import time

import soundfile
import torch
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq, AutoProcessor

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
N_CLIPS = 8
BATCH_SIZES = (1, 4, 8)
MAX_NEW_TOKENS = 512
FULL_RUN_AUDIO_SECONDS = 1.96 * 3600  # option (c): 6 datasets, 150 clips each

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    paths = sorted(glob.glob(os.path.join(BASE, "data", "audio", "*.wav")))[:N_CLIPS]
    clips = [soundfile.read(p, dtype="float32") for p in paths]
    audio_seconds = sum(len(s) / r for s, r in clips)

    config = AutoConfig.from_pretrained(MODEL)
    is_qwen3_asr = getattr(config, "model_type", "") == "qwen3_asr"
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()

    def run(batch_size: int) -> list[str]:
        out: list[str] = []
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                group = clips[start : start + batch_size]
                inputs = processor.apply_transcription_request(
                    audio=[samples for samples, _ in group], language="en"
                )
                prompt_len = inputs["input_ids"].shape[1]
                predicted = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
                decoded = (
                    processor.decode(predicted[:, prompt_len:],
                                     return_format="transcription_only")
                    if is_qwen3_asr
                    else processor.decode(predicted[:, prompt_len:],
                                          skip_special_tokens=True)
                )
                out.extend(decoded if isinstance(decoded, list) else [decoded])
        return [t.strip() for t in out]

    run(1)  # warm up outside the timed region
    print(f"{len(clips)} clips, {audio_seconds:.0f}s audio, float32 CPU\n")

    results = {}
    for batch_size in BATCH_SIZES:
        started = time.perf_counter()
        results[batch_size] = run(batch_size)
        elapsed = time.perf_counter() - started
        rtfx = audio_seconds / elapsed
        print(f"  batch {batch_size:2d}: {elapsed:6.1f}s   RTFx {rtfx:5.2f}   "
              f"3-model run {3 * FULL_RUN_AUDIO_SECONDS / rtfx / 3600:5.2f}h")

    print()
    baseline = results[1]
    for batch_size in BATCH_SIZES[1:]:
        identical = sum(1 for a, b in zip(baseline, results[batch_size]) if a == b)
        verdict = "identical" if identical == len(baseline) else "DIFFERS"
        print(f"  batch {batch_size} vs batch 1: {identical}/{len(baseline)} {verdict}")
        for a, b in zip(baseline, results[batch_size]):
            if a != b:
                print(f"     batch1: {a[:85]}")
                print(f"     batch{batch_size}: {b[:85]}")
                break


if __name__ == "__main__":
    main()
