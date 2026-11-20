# Partial Audit of Hugging Face Open ASR Leaderboard

Katie Roberts, November 2026

A reproduction of the Hugging Face Open ASR Leaderboard (version v1.16.1) across
three open models, and an analysis of the output relative to Hugging Face's
reporting.

Every figure here is produced by a script in this repository and written to
`results/`, and each section names the script behind its numbers. None is
transcribed by hand.

---

## 0. Introduction and method

The task: take a public voice-AI leaderboard, reproduce part of it, and probe
what its ranking metric conceals. How is the text normalized before scoring?
What can the metric not see? Is the evaluation data what it claims to be?

### Why this leaderboard

The Hugging Face Open ASR Leaderboard publishes its evaluation audio, its
scoring code, its per-dataset results, and its models' per-clip transcripts.

ASR was chosen over text-to-speech because its evaluation is cheaper to
reproduce, which matters on a short task. Transcription has one correct output,
so the metric is an edit distance against a reference, and anyone holding the
transcripts can recompute it exactly. Synthesis has no single correct output, so
its boards rank by human preference or by distributional metrics, neither of
which a third party can re-derive from published artifacts alone.

### Why these three systems

The board's top five entries are commercial APIs (Zoom, Otter, Yoob, Azure,
OpusClip). They cannot be pinned to a version, their decoding cannot be
inspected, and a rerun is not guaranteed to reproduce. Otter's API is
enterprise-only, so a literal top-three reproduction is not available to an
individual at any price. To best fulfill the task, I chose systems I could
reproduce as accurately as possible.

| System | Why |
|---|---|
| `Musci-research/Musci-ASR-qwen-1019-hf` | Highest-ranked open model on the public data |
| `Qwen/Qwen3-ASR-1.7B-hf` | The model Musci is a GRPO fine-tune of. Same architecture, same size, so the pair is a controlled comparison: the difference isolates what a leaderboard-targeted fine-tune buys |
| `CohereLabs/cohere-transcribe-03-2026` | A separate lineage, giving the set some architectural spread |

### Why these evaluation sets

The board's headline average covers **nine** datasets, **two of which are
private**. This report uses only the seven public ones; see section 4 for more
details.

The reproduction re-transcribes a sample from three of them, 200 clips each,
rather than the full splits. Between them the three span the axis that drives
recognition error, namely how scripted the speech is, and each was verified to
align with the board's own per-clip outputs before being used.

| Dataset | Why |
|---|---|
| `AMI` | Spontaneous multi-party meetings: the hardest end of the axis, with published WERs of 6.5 to 8.3 across the three systems. It also carries by far the densest reference disfluency, 36.18 filler words per thousand against 8.39 for GigaSpeech, which is where the section 3 metric has its clearest signal |
| `GigaSpeech` | Edited podcasts and video: the middle of the axis, where speech is spontaneous but the audio and transcripts have been cleaned up. The largest of the three at 18,757 clips in the full split |
| `VoxPopuli` | Parliamentary oratory: the scripted end of the axis, and the opposite extreme from AMI, so the pair bounds the range rather than sampling one register twice |

Four were set aside for the reproduction. Three of them hit an obstacle in
pairing the board's published outputs back to the audio, and in each of those
cases the obstacle looks workable with more time; none was pursued, because
three datasets already cover the axis the reproduction needs. The fourth was a
compute budget call.

| Dataset | What stopped it, and what was not tried |
|---|---|
| SPGISpeech | The board's published manifest stores only the basename of each clip id, so 39,341 rows carry 421 distinct ids. Resolving a sampled clip back to a dataset row would need matching on reference text instead of id, which was not attempted |
| Voice Arena Monsoon | Renamed upstream; the dataset now behind that name matched 0 of 200 clips. The revision the board actually evaluated may still be locatable, which was not chased down |
| URGENT24 | Matched 46%. Dropped rather than diagnosed, for time; the cause is not known |
| Earnings22-Cleaned | Six twenty-minute recordings requiring a long-form chunked decode, 42% of the audio budget for the least reliable number in the suite. A deliberate budget call, not a blocker |

All four still appear in sections 2 to 4, which run on the board's published
outputs and so need no pairing to audio.

### Sample size

200 clips per dataset, 600 in total, chosen by measurement rather than by
formula. `scripts/power_analysis.py` draws 500 seeded samples at each size from
the board's own per-clip outputs and asks how often the three systems come out in
the order the full public splits give:

| Clips per dataset | Trials recovering the full-corpus ranking |
|---|---:|
| 100 | 98.6% |
| 150 | 99.8% |
| **200** | **100%** |
| 300 | 100% |

For these three systems 200 is comfortably more than needed; they are separated
by 0.6 and 0.5 WER points on the macro-average, which a smaller sample also
resolves.

Recovering a *ranking* is a weaker requirement than reproducing an *absolute*
WER, and the distinction matters: section 1 shows a dataset where a 200-clip draw
was still unrepresentative enough to move a score by 1.3 points. 200 is enough to
order these systems and not always enough to match the board's number.

Scoring uses the board's own normalizer, vendored verbatim from their repository
at commit `4a113dd9`, and `kaldialign` with compound merging, the same call
their code makes.

### A check run before anything else

Two things can go wrong in a reproduction: the inference, and the scoring. If
both are unverified, a gap between my number and theirs is uninterpretable.

So before running any model, I ran my scorer over **the board's own published
outputs** to see whether it lands on **their** published WER. It does, on **all
64** dataset-model pairs to within 0.02 points, worst case 0.005
(`results/scorer_validation.json`), which is what rounding a published figure to
two decimals leaves behind. The arithmetic matches theirs, so any later
disagreement is attributable to inference. The check covers every manifest in the
public bucket, not only the three systems compared here, because it tests the
scorer rather than any model.

It also found a bug in my scorer. I was applying the board's
`normalize_compound_pairs` step, which reconciles `white paper` against
`whitepaper`, to every pair. The board applies it only to multilingual sets, and
its English score files all carry `"multilingual": false`. Scoring English
through the extra step was lenient by 0.04 points on one manifest, enough to fail
the tolerance. The fix was a deletion.

The check also surfaced a scoring rule I had missed: chunked datasets must have
their chunks concatenated per recording before scoring, because the board assigns
every chunk the full parent transcript. Scoring per chunk gives 98% WER. The
board's About page documents this, and the check found it independently.

---

## 1. Model output reproduction

Each model was run on the 600-clip sample and compared to the board two ways: in
aggregate against its published WER, and clip by clip against its published
transcript.

| Model | Dataset | Mine (95% CI) | Published | Diff | Consistent? | Transcripts identical |
|---|---|---:|---:|---:|---|---:|
| Qwen3-ASR-1.7B | AMI | 8.76 [7.10, 10.44] | 8.31 | +0.45 | yes | 93.5% |
| Qwen3-ASR-1.7B | GigaSpeech | 5.89 [4.63, 7.20] | 7.22 | -1.33 | **no** | 95.0% |
| Qwen3-ASR-1.7B | VoxPopuli | 2.81 [2.31, 3.27] | 2.85 | -0.04 | yes | 94.0% |
| Musci-ASR | AMI | 6.76 [5.52, 8.16] | 6.53 | +0.23 | yes | 97.5% |
| Musci-ASR | GigaSpeech | 5.30 [4.18, 6.43] | 6.41 | -1.11 | just | 97.0% |
| Musci-ASR | VoxPopuli | 1.53 [1.19, 1.88] | 1.47 | +0.07 | yes | 96.0% |
| cohere-transcribe | AMI | 7.73 [6.17, 9.51] | 7.02 | +0.71 | yes | 98.5% |
| cohere-transcribe | GigaSpeech | 6.45 [5.17, 7.79] | 7.88 | -1.43 | **no** | 96.5% |
| cohere-transcribe | VoxPopuli | 5.10 [4.37, 5.91] | 5.38 | -0.28 | yes | 95.5% |

Confidence intervals are a percentile bootstrap over clips, which keeps
within-clip error correlation rather than assuming independence.

### Why they differ

**Seven of nine comparisons sit inside their interval**, so they are consistent
with the published figures given a 600-clip sample.

**Every GigaSpeech comparison is low, by 1.1 to 1.4 points, and GigaSpeech is
the only dataset where that happens.** Two of the three fall outside their
interval; the third, Musci-ASR, lands inside it by 0.02 points, which is close
enough to the edge that treating it as a different outcome from the other two
would be reading the interval too precisely.

The per-clip comparison explains the pattern. On GigaSpeech my transcripts are
byte-identical to the board's published transcripts on 95 to 97% of clips, and
identical after normalization on 98.5 to 99%. When the transcripts agree almost
perfectly but the scores do not, the cause is **which clips were scored**. I
scored 200; the board scored all 18,757, and my seeded draw is easier than the
full split. Three systems reproducing the board's transcripts while all missing
in the same direction on the same dataset points at the sample rather than the
pipeline.

One qualification: all three models scored the *same* 200 clips, so this is a
single unrepresentative draw observed three times rather than three independent
pieces of evidence. The bootstrap interval is meant to cover sampling variation,
and on this dataset it only just did. It is the clearest argument in this report
for sampling above the power-analysis minimum when a dataset is large and
varied.

That diagnosis is only possible because the board publishes per-clip outputs. An
aggregate gap alone would have been ambiguous between "my pipeline is wrong" and
"my sample is unlucky".

### Transcript agreement

Agreement is reported two ways, since the word carries different meanings.
**Byte-identical** is raw string equality before any processing. **After
normalization** applies the board's normalizer to both sides first, which forgives
casing, punctuation, filled pauses and number formatting.

| | Byte-identical | After normalization |
|---|---:|---:|
| Qwen3-ASR-1.7B | 93.5 to 95.0% | 98.0 to 99.0% |
| Musci-ASR | 96.0 to 97.5% | 99.5 to 100% |
| cohere-transcribe | 95.5 to 98.5% | 97.0 to 99.0% |

Roughly 19 clips in 20 are character-for-character identical to the board's
output. That also validates the configuration deviations below: if dtype or
batch size were changing the decode, byte-level agreement would be far lower.

### Configuration

Read from the board's own submit scripts so that a difference in results is
attributable to inference rather than to a decoding choice made by accident.

| | Board | Here | Why |
|---|---|---|---|
| `max_new_tokens` | 512 | 512 | same |
| attention | sdpa | sdpa | same |
| dtype | bfloat16 | **float32** | bfloat16 is emulated on this CPU and 1.6x slower; transcripts measured identical on 6 of 6 clips |
| batch size | 256 | **8** | memory; transcripts measured identical to batch 1 on 8 of 8 clips |
| device | GPU | **CPU** | hardware available |

Each deviation was measured on real clips before adoption.

---

## 2. Normalization

### What the board does

Before scoring, the board rewrites both the reference and the model output with
its own `EnglishTextNormalizer`
([normalizer/normalizer.py](https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/normalizer.py)),
applied symmetrically to both sides in
[eval_utils.py](https://github.com/huggingface/open_asr_leaderboard/blob/main/normalizer/eval_utils.py):

```python
ref, pred = _normalize(ref, language), _normalize(pred, language)
```

It is a fork of Whisper's normalizer, extended. Fifteen steps run in order; the
consequential ones:

- delete filler words: `uh`, `um`, `hmm`, with elongation patterns
- expand contractions and titles: `don't` to `do not`, `mr` to `mister`
- convert spoken numbers to digits: `twenty five thousand euros` to `EUR25000`
- map British to American spelling, from a 1,740-entry table
- collapse spaced acronyms: `b b c` to `bbc`

Note the board's About page describes this as "closely following that of
Whisper". The acronym, name and compound-word handling are its own additions.

### My rules

My scheme is **strict**: lowercase, strip punctuation, collapse whitespace, and
nothing else. Numbers, spellings, contractions and fillers stay exactly as the
system produced them. Casing and punctuation are still removed, because keeping
them would mean scoring truecasing and punctuation restoration, which are
different tasks with their own references.

The difference between the two schemes is who absorbs the model's formatting
choices. The board's rules forgive them: write "twenty five" for `25`, or drop
every "um", and the normalizer erases the discrepancy before it reaches the
score. Strict scoring charges for them, because a downstream consumer of the
transcript either lives with those choices or pays to clean them up.

### What changes

Between the two I scored a ladder, each rung adding one real board rule.
`scripts/normalize_ladder.py` subclasses the board's own normalizer and gates
individual steps, which preserves step order. The top rung is verified
byte-identical to the board's normalizer on 3,082 real strings before any
scoring happens.

This needs no inference, so it runs on the **complete public evaluation data**,
around 70,000 clips per system, rather than the 600-clip inference subset.

| Model | Strict | +fillers | +numbers | +spelling | +contractions | Board | Benefit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Musci-ASR | 7.29 | 6.03 | 5.49 | 5.40 | 5.15 | **5.15** | 2.14 |
| Qwen3-ASR-1.7B | 7.85 | 6.54 | 6.22 | 6.07 | 5.76 | 5.75 | 2.11 |
| cohere-transcribe | 8.22 | 7.42 | 6.56 | 6.45 | 6.26 | 6.26 | 1.96 |

Normalization removes **24% to 29%** of a system's strict WER depending on the
system, so it does not move every score by the same amount. On the macro-average
the order survives it: Musci-ASR leads under both schemes.

Per dataset it does not. **On three of the seven datasets, normalization changes
which system ranks first.** AMI is the clearest case:

| AMI | Strict | Board |
|---|---:|---:|
| cohere-transcribe | **7.35** | 7.02 |
| Musci-ASR | 10.10 | **6.53** |
| Qwen3-ASR-1.7B | 12.45 | 8.31 |

Scored strictly, cohere-transcribe is 2.8 points clear of Musci-ASR on AMI.
After normalization it is 0.5 points behind. That is a 3.2-point swing in the gap
between two systems, from the same transcripts, decided by the scoring rules
rather than by the audio. Section 3 explains the mechanism.

The other two reorderings are weak and worth marking as such: Voice Arena
Monsoon separates all three systems by 0.07 points under strict scoring, which is
inside the noise, and Earnings22-Cleaned is six recordings. AMI is the case that
carries weight.

Rule by rule, averaged across the three systems:

| Rule | WER removed |
|---|---:|
| **Filler removal** | **1.12 pp** |
| Number normalization | 0.58 pp |
| Contractions and titles | 0.25 pp |
| Spelling map | 0.11 pp |
| Acronyms, names, compounds | 0.01 pp |

Filler removal does roughly double the work of number handling, the reverse of
what a read-speech benchmark would show, reflecting that half this suite is
spontaneous speech. The board's own additions beyond Whisper's normalizer are
worth **0.01 points across 70,000 clips**.

---

## 3. An added metric: verbatim fidelity

### What the leaderboard's metric misses

The filler rule above conceals a difference between products.

On AMI, whose transcripts contain 2,975 filled pauses, **Qwen3-ASR writes 114 of
them (3.8%)**. cohere-transcribe writes 2,915 (98.0%). Both figures are measured
on the board's own published transcripts across the full 7,715-clip split, against
the same references, so the comparison does not depend on my inference run.
Because the board deletes fillers from both sides before scoring, its metric
cannot distinguish the two systems by construction, and no re-aggregation of WER
recovers the difference.

This is also what produces the AMI reordering in section 2. Under strict scoring
cohere-transcribe is credited for the filled pauses it writes and Qwen3-ASR and
Musci-ASR are charged for the ones they omit; the filler rule deletes exactly
that, and the ranking changes.

### The metric

**Verbatim fidelity**: the share of reference disfluencies a model actually writes
down, where "disfluency" means exactly what the board's own filler regex removes.

```
retention = filler tokens emitted / filler tokens in the reference
```

It needs no audio and no extra labeling, so it is computable from any published
transcript, for every model on the board.

### Checking for innocent explanations

A count of missing words could have several, so each was tested.

**Is a prompt instructing the model to clean up?** No. The prompt the model
receives is empty apart from the audio and a language tag:

```
<|im_start|>system<|im_end|><|im_start|>user<|audio_start|>...<|audio_end|><|im_end|>
<|im_start|>assistant\nlanguage English<asr_text>
```

**Is my decoding discarding them?** No. The processor exposes two output formats
(`parsed` and `transcription_only`), so the raw generated tokens were inspected
directly. They are identical to what I extract.

**Is it just producing a cleaner rendition generally?** No, and this is the
sharpest evidence. A model rewriting speech into tidy prose would drop every kind
of disfluency. So alongside filled pauses, `scripts/disfluency_fidelity.py`
counts immediate word repetitions, the stutter-and-restart kind, on the same
AMI references:

| System | Filled pauses kept | Word repetitions kept |
|---|---:|---:|
| Qwen3-ASR-1.7B | **3.8%** | 98.7% |
| Musci-ASR | **14.7%** | 117.0% |
| cohere-transcribe | 98.0% | 97.8% |

Repetition retention is flat across all three, within a few points of the
reference rate. Filled-pause retention varies by a factor of 26 across the same
systems on the same audio. Whatever Qwen3-ASR and Musci-ASR are doing, it is
specific to filled pauses rather than to disfluency in general.

```
reference : So uh um and a special uh button for shifting up uh and uh shifting down uh channel
model     : So and special button for shifting up and shifting down channel.
```

It writes "I, I mean" and "so so yeah" but drops "uh". The behavior is targeted
and it originates in the model rather than anywhere downstream of it. (Musci-ASR
exceeding 100% means it writes slightly more repetitions than the references
contain, which is a reminder that this is a ratio of counts rather than an
alignment.)

### Whose decision it informs

Anyone with a verbatim requirement. For legal discovery, medical records or
qualitative research, a model that omits filled pauses is unusable regardless of
its accuracy score; the omission is silent and the transcript looks clean. For a
summarization or note-taking product it is exactly what you want, and a verbatim
model's output is noise you will pay to strip.

These are two different purchases. The leaderboard's single number cannot tell
them apart, and nothing on the page signals the difference.

### Limit

Retention varies by domain as well as by system. cohere-transcribe keeps 98.0%
of AMI's fillers, 71.0% of GigaSpeech's and 14.7% of Voice Arena Monsoon's, for
reasons this work did not investigate. It therefore has to be measured per dataset
and published alongside WER, not reduced to one number per system.

---

## 4. Data characterization

Per-clip metadata was computed for all seven public datasets and joined to
per-clip WER for all three systems: **208,647 rows** (`results/per_clip.csv.gz`).
Every table below is regenerated from that file by
`scripts/characterize_report.py`, so none of it is re-derived by hand.
Fields are duration, reference word count, speech rate, disfluency count, whether
the reference contains digits, and a duration bucket. Thresholds were fixed
before any WER was inspected, so buckets cannot be drawn to flatter a result.

### What the breakdowns show

| Breakdown | Result |
|---|---|
| Clip duration | Clips under 5s are **2.3 to 2.6x harder** than 5 to 15s clips for every system (7.9-9.5% against 3.1-4.0%) |
| Disfluency | Clips containing disfluencies cost **4.5 to 5.0 points** for every system |
| Digits | Clips whose reference contains digits are no harder (+0.01 to -0.16 points) |

The disfluency result is measured *after* the board deletes the fillers, so the
penalty comes from the surrounding speech, with its restarts and repairs, rather
than from the "um"s.

The digits result is worth keeping as a negative: it argues against a clip-level
number-accuracy metric, the first candidate for section 3. A token-level version
might still show something, but at clip level the effect is absent.

### Is the metadata reliable?

**Duration: yes.** The board's `duration` field matches the decoded audio to a
maximum error of **0.0001s** across 1,000 clips, and is exact on 785 of them
(`scripts/check_duration_metadata.py`). The check is not circular: the duration
comes from the board's published manifest while the audio is decoded from the
Hugging Face dataset, so the two are independent sources.

**Speech rate: only on longer clips.** It is words divided by duration, so a
single reference-word error moves it by 1/duration. On AMI, median clip 2.9
seconds, that is a **12% swing**, so AMI is excluded from speech-rate breakdowns.

| Dataset | Median clip | Shift per 1-word error |
|---|---:|---:|
| AMI | 2.90s | **11.1%** |
| URGENT24 | 6.95s | 5.3% |
| GigaSpeech | 6.51s | 5.0% |
| SPGISpeech | 8.85s | 4.2% |
| Voice Arena Monsoon | 10.41s | 4.2% |
| VoxPopuli | 9.94s | 4.0% |

**The board's own labels: one fails.** SPGISpeech is described in the board's
dataset table as "Oratory, spontaneous". Its references contain **0.01 filler
words per thousand**, against 36.13 for AMI under the same label. Its transcripts
have been cleaned of disfluency while the label says otherwise. It is the largest
of the seven at 39,341 clips and carries a full seventh of the macro-average.

| Dataset | Board's label | Fillers per 1,000 words |
|---|---|---:|
| Earnings22-Cleaned | Oratory, spontaneous | 38.97 |
| AMI | Spontaneous | 36.13 |
| Voice Arena Monsoon | Conversational, spontaneous | 11.17 |
| GigaSpeech | Read, spontaneous | 8.37 |
| VoxPopuli | Oratory | 7.54 |
| URGENT24 | Read, spontaneous | 6.31 |
| **SPGISpeech** | **Oratory, spontaneous** | **0.01** |

### Two structural facts the single number hides

**The headline average cannot be fully reproduced.** It covers nine datasets, two
of them private. Roughly a fifth of the published score cannot be verified by
anyone outside Hugging Face. And the private portion is where nearly all the
**accent diversity** sits: Appen Australian, Canadian, Indian and American,
DataoceanAI US and British, Liva AI Egyptian and Nigerian. The seven public sets
are overwhelmingly US, UK and European, with Voice Arena Monsoon the lone
accented exception.

**One public dataset is six recordings.** Earnings22-Cleaned-AA-chunked contains
341 chunks from six parent recordings. The macro-average weights those six
equally with SPGISpeech's 39,341 clips.

---

## 5. Limitations

- **Three systems of the 69 ranked**, all open-weight. No commercial system is
  covered, and the board's top five are all commercial.
- **Two of the three share a lineage**, by design, as a controlled
  fine-tune-versus-base comparison. This narrows the reproduction: a different
  architecture could behave differently in ways I would not see.
- **Three of seven public datasets** in the reproduction, each exclusion
  documented. None of the four was excluded because it is unverifiable in
  principle; each was set aside for time, and the report says what was not tried.
  URGENT24 is the clearest case: it matched the board's outputs at 46% and was
  dropped without diagnosis, so the cause is unknown rather than ruled out.
- **The GigaSpeech sample is unrepresentative**, by slightly more than the
  bootstrap interval allows, and all three models scored the same draw.
- **Inference ran on a laptop CPU** in float32 at batch 8 against the board's
  datacenter GPUs in bfloat16 at batch 256. Each deviation was measured to leave
  transcripts unchanged, and byte-level agreement of 93.5 to 98.5% supports that,
  though the hardware still differs.
- **Model revisions are pinned by name, not commit hash.** A checkpoint update
  between the board's run and mine would be invisible.
- **Verbatim fidelity is a ratio of counts rather than an alignment.** A model
  emitting fillers in the wrong places would still score well, so it indicates
  style rather than correctness.

---

## 6. How I would automate this

The repository reruns from fixed seeds, but it reruns as seventeen separate
scripts invoked in a particular order, and the order is documented in a README
rather than enforced anywhere. That is the honest state of it. What I would
build:

1. **Collapse the scripts into one pipeline with declared stages.** Seventeen
   entry points grew one at a time as questions came up, and several encode a
   dependency that nothing enforces. `fetch_audio.py` downloads the wrong clips
   if `build_row_index_parquet.py` has not run first, which is the failure
   point 3 below is about. `compare_to_board.py` skips any model or
   dataset missing from the predictions file and reports a smaller table rather
   than an error, so a half-finished transcription run looks like a complete
   result. One entry point with named stages, each declaring its inputs and
   refusing to run without them, would turn both of those from silent wrong
   answers into a stop.

2. **Record provenance at run time.** Model commit hashes, dataset revision,
   library versions and hardware, written into the results JSON. Nothing in the
   current version records any of it, so a rerun six months from now could not
   distinguish a model change from a library change.

3. **Make the alignment check a hard gate.** The most dangerous failure in this
   project was silent: audio paired with the wrong clip's reference, which
   produces believable transcripts and a plausible-looking bad WER. Any pipeline
   doing this should verify id alignment against a known-good source and refuse
   to proceed on mismatch.

4. **Diff against the previous run rather than a fixed expectation.** Alert on
   movement in WER, in verbatim fidelity, and in transcript agreement
   separately, since they fail for different reasons.

5. **Keep per-clip outputs as the artifact.** `results/per_clip.csv.gz` is what
   makes any aggregate re-derivable later under a different normalization or
   threshold without re-running inference, and it is what made the GigaSpeech
   diagnosis possible. Publishing per-clip outputs, as this leaderboard does, is
   among the most useful things a benchmark can do for its users.

6. **Run the normalization ladder on every new system automatically.** A system
   that benefits unusually from the scoring convention is then flagged on
   arrival rather than discovered later.

7. **Treat dataset labels as claims to be tested.** The SPGISpeech
   disfluency-density check is a few lines and it found a label contradicted by
   its own data. That check belongs in dataset intake rather than in a report.

---

## 7. Where I used AI tools, and where I did not

I used Claude (Claude Code) throughout, which is how I normally work.

**AI-assisted:**

- Writing **all** of the code in `scripts/`, from my decisions about what to
  measure and how
- Reading the leaderboard's own source to extract its decode configuration,
  normalizer behavior and chunked-dataset scoring rule
- Debugging: a torch/torchcodec ABI incompatibility, a Whisper long-form decode
  failure, a silent audio-to-reference misalignment, and a reference-caching bug
  that paired one model's predictions with another's references
- Drafting this report from the generated results files

**My own:**

- Choosing the leaderboard, the systems and the evaluation sets
- Catching that the first candidate third dataset was not a board ranking set,
  which would have left a quarter of the subset with nothing to compare against
- Deciding to validate the reproduction against the board's own published
  outputs before trusting any of it, which section 1's diagnosis then depended on
- Pressing on the verbatim-transcription claim until it was checked three ways.
  The first version of that finding was overstated, and the re-validation
  narrowed it to something better supported
- Excluding the datasets I could not pair to the board's outputs inside the time
  I had, rather than including them quietly, and recording what I did not try
- Noticing that the section 2 ranking claim rested on two systems outside the
  reproduction set, and cutting them from the report, so that every comparison in
  it uses the same three systems throughout

**Verified by hand:** the board's decode configuration was read from its submit
scripts, every deviation from it was measured on real clips before adoption, and
the scorer was validated against 64 published results before any model ran.
