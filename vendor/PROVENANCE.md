# Vendored from the Open ASR Leaderboard

`board_normalizer/` is a verbatim copy of `normalizer/` from
https://github.com/huggingface/open_asr_leaderboard at commit
`4a113dd90c9ffd011d7d3d105da76980d511d029` (fetched 2026-11-18).

It is vendored rather than reimplemented for one reason: a reproduction is
only comparable to the board's published numbers if it scores with the
board's own normalizer. A previous attempt used Whisper's
`EnglishTextNormalizer` from `transformers` as a stand-in and produced
systematically higher WER on every dataset, because the board's version
adds acronym collapsing, name canonicalization, compound-word mapping,
title expansion and an extended spelling map on top of Whisper's.

## License

The upstream repository, `huggingface/open_asr_leaderboard`, is licensed
**Apache-2.0**, which permits redistribution in source form with attribution.
The full license text as published upstream is kept beside the code in
`vendor/board_normalizer/LICENSE`. This directory is redistributed unmodified
under those terms; it is not covered by this repository's own license.

Nothing here is modified. To re-verify:

    git clone https://github.com/huggingface/open_asr_leaderboard /tmp/check
    diff -r /tmp/check/normalizer vendor/board_normalizer
