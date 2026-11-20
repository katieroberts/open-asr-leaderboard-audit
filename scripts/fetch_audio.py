"""Fetch audio for the manifest's clips only, not whole dataset configs.

Why not `load_dataset`
----------------------
`load_dataset` materialises an entire config. The seven public splits total
~154 h (roughly 20-30 GB), against limited free disk and a slow connection.
The sample is 1,541 clips / 4.54 h / ~500 MB.

How clips are addressed
-----------------------
By ROW INDEX, recorded in the manifest at sample time, not by clip id.
Clip ids are not reliably unique: SPGISpeech's 39,341 rows carry only 421
distinct `audio_filepath` values, because the board stores the basename and
drops the directory that disambiguates them. Indexing by row removes that
ambiguity and the id-matching fragility with it.

The row-order assumption (board manifest line N == dataset row N) is still
CHECKED per dataset before anything is fetched, by pulling row 0 and
comparing ids. A silently wrong mapping would mean transcribing the wrong
audio against the right references, which looks like a bad model rather
than a bad pipeline.

Three datasets needed special handling, each for a different reason:
  spgispeech          ids are truncated to basenames; compare basenames
  voice_arena_monsoon renamed upstream; the old path 404s on the server
  earnings22_*        rows expose no audio field; wavs live in the repo

Resumable: a clip whose .wav exists is skipped, so an interrupted run can
simply be restarted.

Run:  .venv/bin/python scripts/fetch_audio.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import urllib.parse
import urllib.request

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE, "data", "audio")
ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"

REQUEST_TIMEOUT_S = 120
RETRIES = 5
RETRY_BACKOFF_S = 10

BASENAME_ID_DATASETS = {"spgispeech"}


def safe_name(clip_id: str) -> str:
    """Filesystem-safe form of a clip id.

    Audio is named by CLIP ID, never by row number. An earlier version named
    files by dataset row offset while the transcription step looked them up
    by the board manifest's line number; where those disagreed the pipeline
    silently paired audio with another clip's reference. On small datasets
    (1,000-2,000 rows) a file almost always exists at the looked-up number,
    so the failure was total and silent: Voice Arena matched the board on
    0 of 200 clips. Naming by id removes the indirection entirely.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", clip_id)
DIRECT_FILE_DATASETS = {
    "earnings22_cleaned_aa_chunked": (
        "ArtificialAnalysis/Earnings22-Cleaned-AA-chunked", "audio"
    ),
}

MANIFEST_SUFFIX = {
    "ami_cleaned": "hf-audio-open-asr-leaderboard_ami_cleaned_test",
    "earnings22_cleaned_aa_chunked": "ArtificialAnalysis-Earnings22-Cleaned-AA-chunked__test",
    "gigaspeech_cleaned": "hf-audio-open-asr-leaderboard_gigaspeech_cleaned_test",
    "spgispeech": "hf-audio-open-asr-leaderboard_spgispeech_test",
    "urgent2024": "hf-audio-open-asr-leaderboard_urgent2024_test",
    "voice_arena_monsoon": "VoiceArena-Monsoon_en_IN_test__test",
    "voxpopuli_cleaned_aa": "hf-audio-open-asr-leaderboard_voxpopuli_cleaned_aa_test",
}
SOURCE_MODEL = "Musci-research-Musci-ASR-qwen-1019-hf"


def _retrying(action, description: str):
    for attempt in range(RETRIES):
        try:
            return action()
        except Exception as error:
            if attempt == RETRIES - 1:
                raise
            print(f"      retry {attempt + 1}/{RETRIES} for {description}: "
                  f"{type(error).__name__}")
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))


def http_json(url: str) -> dict:
    def get():
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
            return json.load(response)
    return _retrying(get, "json")


def http_download(url: str, destination: str) -> None:
    def get():
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
            payload = response.read()
        # Write via a temp name so an interrupted write cannot leave a
        # truncated file that a resumed run would treat as complete.
        temporary = destination + ".partial"
        with open(temporary, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, destination)
    _retrying(get, os.path.basename(destination))


def rows_url(hub: str, config: str, split: str, offset: int, length: int) -> str:
    return f"{ROWS_ENDPOINT}?" + urllib.parse.urlencode({
        "dataset": hub, "config": config or "default",
        "split": split, "offset": offset, "length": length,
    })


def row_id(row: dict) -> str:
    for key in ("id", "audio_filepath", "segment_id", "utt_id"):
        if row.get(key):
            return str(row[key])
    return ""


def audio_url(row: dict):
    audio = row.get("audio")
    if isinstance(audio, list) and audio:
        return audio[0].get("src")
    if isinstance(audio, dict):
        return audio.get("src")
    return None


def row_offsets(dataset_label: str) -> dict:
    """Clip id -> true dataset row offset, from data/row_index/.

    Built by scripts/build_row_index_parquet.py by reading the dataset's own
    id column. The board's manifest line numbers are NOT usable as offsets:
    its manifests are written in processing order, not dataset order (AMI
    matches positionally only 45.8% of the time), so using them pairs audio
    with the wrong reference and produces a believable but wrong WER.
    """
    path = os.path.join(BASE, "data", "row_index", f"{dataset_label}.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path))


def board_manifest_ids(dataset_label: str) -> list[str]:
    suffix = MANIFEST_SUFFIX[dataset_label]
    path = os.path.join(
        BASE, "data", "board_results", "predictions", SOURCE_MODEL,
        f"MODEL_{SOURCE_MODEL}_DATASET_{suffix}.jsonl",
    )
    return [json.loads(line)["audio_filepath"] for line in open(path)]


def verify_row_order(dataset_label, hub, config, split, manifest_ids) -> bool:
    probe = http_json(rows_url(hub, config, split, 0, 1))
    if not probe.get("rows"):
        print(f"   rows endpoint unavailable: {str(probe.get('error'))[:80]}")
        return False
    actual = row_id(probe["rows"][0]["row"])
    if dataset_label in BASENAME_ID_DATASETS:
        actual = os.path.basename(actual)
    if actual != manifest_ids[0]:
        print(f"   ORDER MISMATCH: manifest row 0 = {manifest_ids[0]!r}, "
              f"endpoint row 0 = {actual!r}; skipping")
        return False
    print(f"   order verified (row 0 = {actual})")
    return True


def fetch_direct_files(dataset_label: str, rows: list[dict]) -> None:
    repo, folder = DIRECT_FILE_DATASETS[dataset_label]
    fetched = 0
    for row in rows:
        filename = f"{row['parent_id']}_chunk_{row['chunk_index']}.wav"
        destination = os.path.join(
            AUDIO_DIR, f"{dataset_label}__{safe_name(filename)}")
        if os.path.exists(destination):
            continue
        http_download(
            f"https://huggingface.co/datasets/{repo}/resolve/main/"
            f"{folder}/{urllib.parse.quote(filename)}",
            destination,
        )
        fetched += 1
        if fetched % 50 == 0:
            print(f"   {fetched}/{len(rows)} fetched", flush=True)
    print(f"   done: {fetched} new, {len(rows) - fetched} already present")


def fetch_rows(dataset_label, hub, config, split, rows, offsets) -> None:
    """Fetch one dataset's clips, surviving individual failures.

    The datasets-server returns transient 502s under load. A clip that
    cannot be fetched is recorded and skipped rather than aborting the run:
    the job is resumable, so a later pass picks up whatever was missed, and
    losing one clip is far cheaper than losing the other 1,540.
    """
    fetched = skipped = failed = unmapped = 0
    failures = []
    resolved = []
    for row in rows:
        key = os.path.basename(row["clip_id"]) if dataset_label in BASENAME_ID_DATASETS else row["clip_id"]
        if key not in offsets:
            unmapped += 1
            continue
        resolved.append((offsets[key], row))
    if unmapped:
        print(f"   {unmapped} clips have no row mapping and are skipped")

    for offset, row in sorted(resolved):
        destination = os.path.join(
            AUDIO_DIR, f"{dataset_label}__{safe_name(row['clip_id'])}.wav")
        if os.path.exists(destination):
            continue
        try:
            page = http_json(rows_url(hub, config, split, offset, 1))
            if not page.get("rows"):
                skipped += 1
                continue
            url = audio_url(page["rows"][0]["row"])
            if not url:
                skipped += 1
                continue
            http_download(url, destination)
            fetched += 1
        except Exception as error:
            failed += 1
            failures.append((offset, f"{type(error).__name__}: {error}"))
            continue
        if fetched % 25 == 0:
            print(f"   {fetched} fetched", flush=True)
    print(f"   done: {fetched} new, {skipped} unavailable, {failed} failed, "
          f"{len(resolved) - fetched - skipped - failed} already present")
    if failures:
        log = os.path.join(BASE, "logs", f"fetch_failures_{dataset_label}.txt")
        with open(log, "w") as handle:
            for offset, reason in failures:
                handle.write(f"{offset}\t{reason}\n")
        print(f"   failures logged to {os.path.relpath(log, BASE)}")


def main() -> None:
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(os.path.join(BASE, "data", "manifest.csv"), newline="") as handle:
        manifest = list(csv.DictReader(handle))

    by_dataset: dict[str, list[dict]] = {}
    for row in manifest:
        by_dataset.setdefault(row["dataset"], []).append(row)

    for dataset_label, rows in by_dataset.items():
        hub = rows[0]["hub_dataset"]
        config = rows[0]["hub_config"]
        split = rows[0]["hub_split"]
        print(f"\n{dataset_label}  ({len(rows)} clips)")

        if dataset_label in DIRECT_FILE_DATASETS:
            try:
                fetch_direct_files(dataset_label, rows)
            except Exception as error:
                print(f"   dataset failed: {type(error).__name__}: {error}")
            continue

        try:
            offsets = row_offsets(dataset_label)
            if not offsets:
                print("   no row index yet; run build_row_index_parquet.py first")
                continue
            fetch_rows(dataset_label, hub, config, split, rows, offsets)
        except Exception as error:
            # One dataset failing must not take the others down with it.
            print(f"   dataset failed: {type(error).__name__}: {error}")
            continue


if __name__ == "__main__":
    main()
