"""Check every downloaded model is complete, by comparing against the Hub.

Why this is separate from the download
--------------------------------------
`snapshot_download` reported success for a model whose weights were never
fetched: a run killed mid-download left a snapshot symlink pointing at a
blob that did not exist, and the next call saw the symlink and concluded
there was nothing to do. The local directory was 1.1 MB for a model whose
weights are 2.5 GB.

A download that silently half-succeeds is worse than one that fails,
because inference then fails much later with an unrelated-looking error.
So completeness is checked against the Hub's own reported file sizes
rather than trusted.

Run:  .venv/bin/python scripts/verify_models.py
      .venv/bin/python scripts/verify_models.py --fix   # re-download bad ones
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

# The three models this evaluation uses throughout. fluid-speech-pro is
# absent: the board evaluates it through a NeMo runner with 4-beam batched
# beam search from a .nemo checkpoint, not through transformers. Reproducing
# it with a transformers greedy decode would measure my decoding choice
# rather than the model, so it is excluded from this evaluation entirely.
MODELS = [
    "Qwen/Qwen3-ASR-1.7B-hf",
    "Musci-research/Musci-ASR-qwen-1019-hf",
    "CohereLabs/cohere-transcribe-03-2026",
]

# Only these are large enough to matter; configs and tokenizers are cheap
# and their absence surfaces immediately at load time.
WEIGHT_SUFFIXES = (".safetensors",)
TOLERANCE = 0.999  # allow a hair under, for filesystem reporting differences


def hub_weight_sizes(model_id: str) -> dict[str, int]:
    url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = json.load(response)
    return {
        s["rfilename"]: s["size"]
        for s in (data.get("siblings") or [])
        if s["rfilename"].endswith(WEIGHT_SUFFIXES) and s.get("size")
    }


def local_snapshot_dir(model_id: str) -> str | None:
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    repo_dir = os.path.join(cache, "models--" + model_id.replace("/", "--"))
    snapshots = os.path.join(repo_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return None
    entries = [os.path.join(snapshots, d) for d in os.listdir(snapshots)]
    entries = [d for d in entries if os.path.isdir(d)]
    return max(entries, key=os.path.getmtime) if entries else None


def check(model_id: str) -> tuple[bool, str]:
    expected = hub_weight_sizes(model_id)
    if not expected:
        return False, "no weight files listed on the Hub"
    snapshot = local_snapshot_dir(model_id)
    if snapshot is None:
        return False, "not downloaded"

    problems = []
    for filename, size in expected.items():
        path = os.path.join(snapshot, filename)
        if not os.path.exists(path):  # follows symlinks; False if target missing
            problems.append(f"{filename}: missing")
            continue
        actual = os.path.getsize(path)
        if actual < size * TOLERANCE:
            problems.append(
                f"{filename}: {actual/1e9:.2f}GB of {size/1e9:.2f}GB"
            )
    if problems:
        return False, "; ".join(problems)
    total = sum(expected.values()) / 1e9
    return True, f"complete ({total:.2f}GB across {len(expected)} file(s))"


def main() -> None:
    fix = "--fix" in sys.argv
    bad = []
    for model_id in MODELS:
        ok, detail = check(model_id)
        print(f"  {'OK  ' if ok else 'BAD '} {model_id:44s} {detail}")
        if not ok:
            bad.append(model_id)

    if bad and fix:
        from huggingface_hub import snapshot_download

        for model_id in bad:
            print(f"\nre-downloading {model_id} ...")
            snapshot_download(
                model_id,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.py"],
                force_download=True,
            )
        print("\nre-checking:")
        for model_id in bad:
            ok, detail = check(model_id)
            print(f"  {'OK  ' if ok else 'BAD '} {model_id:44s} {detail}")
    elif bad:
        print(f"\n{len(bad)} model(s) incomplete. Re-run with --fix to repair.")
        sys.exit(1)


if __name__ == "__main__":
    main()
