"""A normalization ladder: the board's pipeline with steps switched off.

What this is for
----------------
The leaderboard scores every system after a normalization pass that
erases a great deal: filler words, contractions, title abbreviations,
number formatting, British/American spelling, acronym spacing and
compound-word boundaries. Each of those erasures is a real difference
between what a system emitted and what the reference says, and a buyer
receiving the transcript would have to fix every one of them.

So "what is the board's normalization worth, and to whom" is a real
question, and it is answered by scoring the same predictions under
progressively more of the pipeline.

Faithfulness
------------
The rungs are NOT a reimplementation. `ConfigurableNormalizer` copies the
board's `__call__` verbatim and gates individual steps behind flags, so
step order (which matters: number normalization depends on symbols that a
later step strips) is identical to theirs. With every flag on it must
reproduce the board's own normalizer exactly, and `verify_board_parity`
asserts that on real data before any scoring happens.

The rungs
---------
  L0 strict          lowercase, strip annotations/symbols, collapse space
  L1 +fillers        remove um/uh/hmm...
  L2 +numbers        "twenty five" -> "25"
  L3 +spellings      colour -> color
  L4 +replacers      contractions and titles: "don't" -> "do not", mr -> mister
  L5 +names/acronyms "b b c" -> "bbc", name canonicalization, compounds
  L6 board           identical to the leaderboard's normalizer

L0 is deliberately not "raw": casing and punctuation are still removed,
because keeping them would mean scoring truecasing and punctuation
restoration, which are different tasks. The datasets that DO support a
cased comparison are handled separately (see the transcript-style column
in the board's own dataset table).
"""

from __future__ import annotations

import os
import re
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE, "vendor"))

from board_normalizer.normalizer import (  # noqa: E402
    EnglishTextNormalizer,
    remove_non_speech_annotations,
    remove_symbols_and_diacritics,
)


class ConfigurableNormalizer(EnglishTextNormalizer):
    """The board's normalizer with individually disableable steps.

    Subclasses rather than copies so the filler patterns, replacer table,
    spelling map, name map and compound map all come from the board's own
    definitions and stay in sync with the vendored copy.
    """

    def __init__(
        self,
        fillers: bool = True,
        replacers: bool = True,
        compounds: bool = True,
        numbers: bool = True,
        spellings: bool = True,
        names: bool = True,
        acronyms: bool = True,
    ) -> None:
        super().__init__()
        self.use_fillers = fillers
        self.use_replacers = replacers
        self.use_compounds = compounds
        self.use_numbers = numbers
        self.use_spellings = spellings
        self.use_names = names
        self.use_acronyms = acronyms

    def __call__(self, s: str) -> str:
        s = s.lower()
        s = remove_non_speech_annotations(s)

        if self.use_fillers:
            s = re.sub(self.ignore_patterns, "", s)

        s = re.sub(r"\s+'", "'", s)

        if self.use_replacers:
            for pattern, replacement in self.replacers.items():
                s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")

        if self.use_compounds:
            for pattern, replacement in self.compound_words.items():
                s = re.sub(pattern, replacement, s)

        if self.use_numbers:
            s = self.standardize_numbers(s)
        if self.use_spellings:
            s = self.standardize_spellings(s)
        if self.use_names:
            s = self.standardize_names(s)
        if self.use_acronyms:
            s = self.standardize_acronyms(s)

        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)
        s = re.sub(r"\s+", " ", s)
        return s


OFF = dict(
    fillers=False, replacers=False, compounds=False,
    numbers=False, spellings=False, names=False, acronyms=False,
)


def _with(**overrides):
    flags = dict(OFF)
    flags.update(overrides)
    return ConfigurableNormalizer(**flags)


# Ordered worst-matched to best-matched; each rung adds one group of rules.
LADDER = {
    "L0_strict": _with(),
    "L1_fillers": _with(fillers=True),
    "L2_numbers": _with(fillers=True, numbers=True),
    "L3_spellings": _with(fillers=True, numbers=True, spellings=True),
    "L4_replacers": _with(fillers=True, numbers=True, spellings=True, replacers=True),
    "L5_names_acronyms": _with(
        fillers=True, numbers=True, spellings=True, replacers=True,
        names=True, acronyms=True, compounds=True,
    ),
    "L6_board": EnglishTextNormalizer(),
}

PRIMARY = ("L0_strict", "L6_board")


def verify_board_parity(samples: list[str]) -> None:
    """The top rung must be byte-identical to the board's own normalizer."""
    board = EnglishTextNormalizer()
    full = ConfigurableNormalizer()
    mismatches = [s for s in samples if full(s) != board(s)]
    if mismatches:
        example = mismatches[0]
        raise AssertionError(
            f"ladder top rung diverges from the board normalizer on "
            f"{len(mismatches)}/{len(samples)} samples; e.g. {example!r}\n"
            f"  ladder: {full(example)!r}\n  board : {board(example)!r}"
        )


if __name__ == "__main__":
    import json
    import glob

    texts = []
    for path in sorted(glob.glob(os.path.join(
            BASE, "data", "board_results", "predictions", "*", "*.jsonl")))[:4]:
        for i, line in enumerate(open(path)):
            if i >= 400:
                break
            record = json.loads(line)
            texts.append(record["text"])
            if record.get("pred_text"):
                texts.append(record["pred_text"])

    verify_board_parity(texts)
    print(f"parity OK: top rung matches the board normalizer on {len(texts)} real strings\n")

    demo = "I don't think it's colour-coded, Mr. Smith. Um, the B B C said 3,500 units in 2026."
    for name, normalizer in LADDER.items():
        print(f"  {name:18s} {normalizer(demo)}")
