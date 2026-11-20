"""Map clip id -> dataset row offset, by reading parquet id columns.

Why parquet rather than the rows endpoint
-----------------------------------------
Both give the same ordering (verified: parquet row 4799 and endpoint row
4799 are the same clip). The endpoint is far faster per row, but it returns
502s under sustained scanning, which makes it unusable for an unattended
run. Parquet is plain file downloads over HTTPS: slower, but it does not
rate-limit and it resumes cleanly.

Parquet is columnar, so reading only the `id` column skips the audio
entirely -- a few MB instead of gigabytes.

Why any of this is needed
-------------------------
The board's prediction manifests are NOT in dataset order (they appear to be
written in processing order, likely sorted by length for batching) and
contain fewer rows than the dataset, because unscorable references are
dropped. For AMI: 7,715 manifest rows against 7,805 dataset rows, matching
positionally only 45.8% of the time. Using manifest line numbers as row
offsets silently pairs audio with the wrong reference, which looks like a
bad model rather than a broken pipeline.

Resumable per dataset: completed maps are cached and skipped.

Output: data/row_index/<dataset>.json

Run:  .venv/bin/python scripts/build_row_index_parquet.py
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "data", "row_index")

DATASETS = {
    "ami_cleaned": ("hf-audio/open-asr-leaderboard", "ami_cleaned", "test"),
    "voxpopuli_cleaned_aa": ("hf-audio/open-asr-leaderboard", "voxpopuli_cleaned_aa", "test"),
    "urgent2024": ("hf-audio/open-asr-leaderboard", "urgent2024", "test"),
    "voice_arena_monsoon": ("VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN", "default", "test"),
    "gigaspeech_cleaned": ("hf-audio/open-asr-leaderboard", "gigaspeech_cleaned", "test"),
    "spgispeech": ("hf-audio/open-asr-leaderboard", "spgispeech", "test"),
}

# SPGISpeech's board ids are basenames of the dataset's own ids.
BASENAME_ID_DATASETS = {"spgispeech"}

ID_COLUMN_CANDIDATES = ("id", "audio_filepath", "segment_id", "utt_id")


def parquet_urls(hub: str, config: str, split: str) -> list[str]:
    url = (f"https://huggingface.co/api/datasets/{hub}/parquet/"
           f"{config}/{split}")
    with urllib.request.urlopen(url, timeout=120) as response:
        listing = json.load(response)
    return listing if isinstance(listing, list) else []


def read_ids(url: str) -> list[str]:
    import fsspec
    import pyarrow.parquet as pq

    filesystem = fsspec.filesystem("http")
    with filesystem.open(url) as handle:
        parquet_file = pq.ParquetFile(handle)
        names = parquet_file.schema_arrow.names
        column = next((c for c in ID_COLUMN_CANDIDATES if c in names), None)
        if column is None:
            raise RuntimeError(f"no id column among {names}")
        table = parquet_file.read(columns=[column])
    return [str(v) for v in table.column(column).to_pylist()]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, (hub, config, split) in DATASETS.items():
        out_path = os.path.join(OUT_DIR, f"{label}.json")
        if os.path.exists(out_path):
            print(f"  {label}: cached ({len(json.load(open(out_path)))} ids)", flush=True)
            continue
        print(f"  {label}: reading parquet id columns ...", flush=True)
        started = time.perf_counter()
        try:
            mapping: dict[str, int] = {}
            offset = 0
            for url in parquet_urls(hub, config, split):
                for clip_id in read_ids(url):
                    key = os.path.basename(clip_id) if label in BASENAME_ID_DATASETS else clip_id
                    # First occurrence wins: SPGISpeech basenames repeat.
                    mapping.setdefault(key, offset)
                    offset += 1
                print(f"     {offset} rows", flush=True)
            with open(out_path, "w") as handle:
                json.dump(mapping, handle)
            print(f"  {label}: {len(mapping)} unique ids over {offset} rows "
                  f"({time.perf_counter()-started:.0f}s)", flush=True)
        except Exception as error:
            print(f"  {label}: FAILED {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
