# Accent Bias in Speech Recognition

Does OpenAI's Whisper transcribe English less accurately for some accents than others, and does
the gap between accent groups change as the model gets bigger?

This repo audits Whisper (tiny → large/turbo) on Mozilla Common Voice English, grouped by
self-reported accent, using word error rate (WER).

_Findings: coming in Phase 4._

## Repo layout

```
accent-bias-asr/
├── README.md
├── data/        # data card + processed metadata (never the audio)
├── src/         # pipeline code
├── tests/       # tests that run on a fake mini-corpus
├── notebooks/   # Colab notebooks (Phase 3+)
├── results/     # CSVs and figures
└── paper/       # drafts and final PDF
```

## Phase 2: build the dataset

### 1. Set up

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

### 2. Get Common Voice English

Download Common Voice English from [Mozilla Data Collective](https://datacollective.mozillafoundation.org/)
(free account). Extract it, or at least `validated.tsv` and `clip_durations.tsv`, into `data/raw/`, so you
have something like `data/raw/cv-corpus-<version>/en/validated.tsv`. `data/raw/` is git-ignored.

### 3. Explore the accent labels

```bash
.venv/bin/python src/prepare_dataset.py --cv-dir data/raw/cv-corpus-<version>/en --explore-only
```

This prints how many clips have no accent, how many clips land in each group, and the biggest labels
the rules don't handle yet. It also writes `data/accent_label_mapping.csv`, which has every distinct label.

### 4. Review the grouping rules

Open [`src/accent_groups.py`](src/accent_groups.py). It holds the 8 groups and the ordered rules that map
labels to them, each with a reason. Add rules for big unmapped labels, or change the groups, and re-run
step 3 until you're happy. **This is your call as the researcher; write down why in the research log.**

### 5. Build the sample and data card

```bash
.venv/bin/python src/prepare_dataset.py --cv-dir data/raw/cv-corpus-<version>/en
```

Writes:

- `data/clips_selected.csv`: the final clips (ID, group, sentence, pseudonymous speaker, ...)
- `data/DATA_CARD.md`: source, version, every filter with counts, every grouping rule and mapping,
  final counts per group, gender/age per group
- `data/accent_label_mapping.csv`: every raw label and where it went

Options: `--per-group 400`, `--max-per-speaker 10`, `--min-words 4`, `--seed 42`. Run with `--help` to see all of them.

### 6. Pull out just the audio you need (for Colab in Phase 3)

```bash
.venv/bin/python src/extract_clips.py --cv-dir data/raw/cv-corpus-<version>/en --zip
```

This also works straight from the download with `--archive path/to/download.tar.gz`, so you don't have to
unpack everything. Upload `data/audio_subset.zip` (about 3,200 clips) to Google Drive.

### How the sample is drawn

1. Drop clips with a blank accent, and clips whose labels don't map cleanly to one group.
2. Drop speakers whose clips map to more than one group.
3. Drop sentences under 4 words (WER on tiny sentences is too noisy) and audio under 1 second.
4. **Speaker-balanced, sentence-first sampling.** Sample in passes: the first pass allows 1 clip per
   speaker, the next allows 2, and so on up to 10. Within each pass, visit sentences from the
   most-shared across groups to the least-shared, and have each group that has a sentence take one clip
   of it. Stop each group at 400 clips.

Because of step 4, each group is spread over as many different speakers as it has, and groups read
the same sentences wherever possible.

## Phase 3: transcribe and score

Transcription needs a GPU, so it runs in Google Colab.

1. Pack the audio, clip list and scripts into one file:
   ```bash
   .venv/bin/python src/make_colab_bundle.py
   ```
2. Upload `colab_bundle.zip` to a Google Drive folder named `accent-bias-asr`.
3. Open [`notebooks/03_transcribe.ipynb`](notebooks/03_transcribe.ipynb) in Colab (File → Upload notebook),
   switch the runtime to a T4 GPU, and run the cells in order: tiny first (check it by hand), then base,
   small, medium, large-v3, turbo, and optionally wav2vec 2.0.
4. Download `My Drive/accent-bias-asr/results/results.csv` into `results/`.

**Disconnects are fine.** Each transcript is written to Drive as soon as it's done. Re-run the setup cells
and the same model cell, and finished clips are skipped.

**Decoding settings, the same for every Whisper model:** language forced to English, beam size 5 with
temperature fallback (the Whisper paper's settings), and a fixed random seed per clip. Clips are processed
in a fixed shuffled order so a partial run still covers every group.

**Scoring** ([`src/score.py`](src/score.py)): the reference and prediction both go through Whisper's
`EnglishTextNormalizer`, then jiwer computes WER per clip. `results/results.csv` has one row per
clip × model: `clip_id, accent_group, speaker, model, reference, prediction, reference_norm,
prediction_norm, wer, substitutions, deletions, insertions, ref_words`.

### Tests

```bash
.venv/bin/pytest -q
```

## Ethics

Results are reported only at group level. Common Voice audio is not redistributed; speaker IDs are
replaced with project-local pseudonyms. Findings describe where the *system* underperforms, not how
anyone "should" speak.
