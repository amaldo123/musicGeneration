# ML Training Dataset Requirements

This document specifies what a **MIDI corpus** must provide for the JAX count-prior training pipeline in `aimusic/ml/`. It covers corpus layout, MIDI content constraints, how files are converted into training examples, vocabulary alignment, quality guidelines, and validation checks.

The v1 trainer (`train_prior_from_corpus`) learns **factorized transition counts** over eight structural token streams. It does not consume pre-annotated JSON, CSV, or NPZ datasets directly; the only supported input format is **Standard MIDI Files** in a directory tree.

---

## 1. Pipeline overview

Training follows this path:

```text
MIDI corpus directory
  └─► parse_midi_beats()        # beat-quantized harmonic + drum activity
        └─► annotate_beat_grid() # BeatState sequence via vocabulary heuristics
              └─► build_transition_examples()  # consecutive (prev, next) pairs
                    └─► examples_to_tokenized()  # 8-stream token ids
                          └─► train_counts()     # JAX bincount transition tables
                                └─► save_prior_bundle()
```

Each **transition example** mirrors Method A prior context:

- **History length:** 1 prior beat
- **Future hint length:** 1 upcoming beat
- **Factorization:** `PriorFactorization.FACTORIZED` (independent per-stream tables)

The trained artifact bundle stores:

- `manifest.json` — model family, EDO, checkpoint paths
- `vocabularies.json` — token tables + embedded `StyleConfig` snapshot
- `counts/` — Orbax checkpoint of JAX transition count matrices

At inference time, generation loads the bundle via `--prior-bundle` and scores transitions using the same vocabularies and EDO recorded in the bundle.

---

## 2. Corpus layout

### 2.1 Directory structure

| Requirement | Detail |
|---|---|
| **Input type** | A single directory path (`--midi-dir`) |
| **Discovery** | Recursive glob: `**/*.mid` and `**/*.midi` |
| **Ordering** | Files processed in sorted path order |
| **Nesting** | Subdirectories are allowed and recommended for organization |
| **Symlinks** | Followed by `pathlib` glob behavior on the host OS |

Example layout:

```text
corpus/
├── rock/
│   ├── piece_001.mid
│   └── piece_002.mid
├── fusion/
│   └── tune_a.mid
└── exercises/
    └── ii_v_i.mid
```

### 2.2 File naming

Filenames are not interpreted semantically. Any valid `.mid` / `.midi` name is acceptable. Duplicate basenames in different folders are fine.

### 2.3 Minimum corpus size

| Threshold | Requirement |
|---|---|
| **Hard minimum** | At least one `.mid` or `.midi` file in the tree |
| **Useful minimum** | At least one file that yields **≥ 2 annotated beats**, producing **≥ 1 transition** |
| **Recommended** | Many files with diverse harmonic motion, meters, and groove activity |

An empty directory or a corpus where every file fails annotation produces **zero transitions**. Training may still write a bundle with all-zero count tables, but the resulting prior is not useful for generation.

---

## 3. MIDI file requirements

### 3.1 Format

| Field | Requirement |
|---|---|
| **Standard** | Standard MIDI File (SMF), Type 0 or Type 1 |
| **Extensions** | `.mid` or `.midi` only (case-sensitive on some platforms) |
| **Encoding** | Must be readable by `mido` |
| **Validity** | Corrupt or non-MIDI files cause the entire training run to fail (no per-file skip) |

### 3.2 Tempo and meter metadata

| Field | Default if missing | Notes |
|---|---|---|
| **Tempo** | 120 BPM | Updated by `set_tempo` meta events |
| **Time signature** | 4/4 | Updated by `time_signature` meta events |
| **Ticks per beat** | From file header | Used for beat quantization |

**Recommendation:** Include an explicit `time_signature` meta event at the start of each file. If a piece changes meter mid-file, later beats use the active signature at that tick.

### 3.3 Beat quantization

Notes are assigned to beat windows by:

```text
beat_index = midi_tick // ticks_per_beat
```

Implications:

- All harmonic and drum activity is **quantized to the notated beat grid**.
- Sub-beat timing within a beat is collapsed; only the beat index matters.
- The grid spans from beat `0` through the highest beat index with any activity, inclusive. Intermediate silent beats are included as empty windows.

### 3.4 Tracks and channels

All tracks are merged into a single chronological message stream.

| Channel | Role in ingestion |
|---|---|
| **0–8, 10–15** | Harmonic content: `note_on` pitches are recorded for that beat |
| **9** (GM drums) | Drum activity: each `note_on` increments `drum_onsets` for that beat |

Velocity-zero `note_on` events are treated as `note_off`.

### 3.5 Harmonic content per beat (critical)

Harmonic pitch classes for a beat come **only from `note_on` events occurring in that beat window**. Sustained notes from earlier beats are **not** carried forward.

| Scenario | Result |
|---|---|
| Chord attacked on beat 0, sustained through beat 3 without re-attack | Only beat 0 receives harmonic pitches; beats 1–3 may be empty |
| New voicing re-attacked each beat | Each beat can be annotated |
| Rest / silence beat | No harmonic pitches → beat skipped during annotation |

**Requirement for usable training data:** Provide **explicit harmonic attacks** on beats you want annotated, or accept that sustained-passage MIDI will yield sparse examples.

### 3.6 Drum content

Drum onsets (channel 9) do not produce harmonic annotations by themselves. They influence the **groove** token via a simple onset-count heuristic:

| Drum onsets in beat | Groove token label assigned |
|---|---|
| ≥ 4 | `straight_16ths` |
| ≥ 2 | `syncopated_8ths` |
| < 2 | `straight_8ths` |

The assigned label must exist in the training `StyleConfig.groove_families` vocabulary (see §5).

---

## 4. Annotation requirements

Each beat window is converted to a `BeatState`:

```text
St = (meter_id, beat_in_bar, boundary_lvl, key_id, chord_id, role_id, head_id, groove_id)
```

A beat is **included** in the training sequence only if all of the following hold:

1. Its meter signature (e.g. `"4/4"`) is in the training vocabulary (`StyleConfig.allowed_meters`).
2. At least one harmonic pitch class is present on that beat.
3. A **best-match chord** from the vocabulary can be inferred from those pitch classes (template overlap heuristic).

Beats that fail any check are **silently skipped** (no error, no placeholder state).

### 4.1 Meter

| Rule | Detail |
|---|---|
| **Matching** | Exact string match on signature (`"numerator/denominator"`) |
| **Default training meters** | `4/4`, `5/4`, `7/4` (`StyleConfig()` default) |
| **CLI override** | `--meter 4/4 --meter 3/4` sets `allowed_meters` |
| **Failure mode** | Beats in unsupported meters are dropped |

**Important:** `3/4` is **not** in the default `StyleConfig.allowed_meters` even though it appears in `DEFAULT_METER_SIGNATURES`. Files in 3/4 require an explicit `--meter 3/4` (or a custom `StyleConfig`).

### 4.2 Chord inference

Pitch classes are derived as `midi_note % edo` (default `edo=12`).

The annotator selects the vocabulary chord whose pitch-class template has the **largest overlap** with observed pitch classes. Ties favor the last best-scoring chord in vocabulary iteration order.

| Requirement | Detail |
|---|---|
| **Minimum overlap** | At least one shared pitch class |
| **Supported qualities** | `maj`, `min`, `7`, `dim` |
| **Default vocabulary size** | 48 chords = 12 roots × 4 qualities |
| **Label format** | `{root}{quality}` e.g. `Cmaj`, `G7`, `Bdim` |

Beats with only non-template pitch sets (e.g. sus4 clusters with no overlap) are skipped.

### 4.3 Key, role, head, boundary

These fields are **heuristic** and do not require separate ground-truth labels in the MIDI:

| Field | Derivation |
|---|---|
| **key_id** | Root pitch class of the matched chord |
| **role_id** | `hold`, `prep`, `cad`, or `change` from consecutive chord analysis |
| **head_id** | Highest pitch class vs chord tones (`root`, `third`, `fifth`, `seventh`, `extension`, or `rest`) |
| **boundary_lvl** | Bar position: downbeat local/phrase/section levels from bar index |

### 4.4 Minimum annotated sequence length

Per file:

```text
annotated_beats >= 2  →  transitions = annotated_beats - 1
annotated_beats < 2   →  0 transitions from that file
```

---

## 5. Vocabulary and StyleConfig alignment

Training builds vocabularies from `StyleConfig` via `build_default_vocabularies()`. The same configuration is serialized into `vocabularies.json` inside the artifact bundle.

### 5.1 Parameters that must stay consistent

When using a trained prior at generation time, these must match the bundle:

| Parameter | Training source | Inference source |
|---|---|---|
| **allowed_meters** | `StyleConfig` / CLI `--meter` | Embedded in `vocabularies.json` |
| **groove_families** | `StyleConfig` / CLI `--groove-family` | Embedded in `vocabularies.json` |
| **edo** | `--edo` (default 12) | `manifest.json` → `expected_edo` |
| **chord/key vocabulary sizes** | `StyleConfig` fields | Embedded in `vocabularies.json` |

Mismatch between generation config and bundle vocabulary produces incorrect or invalid token scoring.

### 5.2 Default StyleConfig (CLI with no overrides)

```python
StyleConfig(
    allowed_meters=("4/4", "5/4", "7/4"),
    groove_families=("straight", "syncopated"),
    chord_vocabulary_size=48,
    key_vocabulary_size=12,
)
```

### 5.3 Structural token streams

Training examples tokenize into eight independent streams (factorized mode):

| Stream | BeatState / token source |
|---|---|
| `meter` | `meter_id` |
| `beat_position` | `beat_in_bar` |
| `boundary` | `boundary_lvl` |
| `key` | `key_id` |
| `chord` | `chord_id` |
| `role` | `role_id` |
| `head` | `head_id` |
| `groove` | `groove_id` |

Each stream gets its own `vocab_size × vocab_size` transition count matrix smoothed with Dirichlet-style `alpha` (default `1.0`).

### 5.4 Groove family tokens

Groove tokens are named `{family}_{subdivision}ths`:

| Family | Generated labels |
|---|---|
| `straight` | `straight_8ths`, `straight_16ths` |
| `syncopated` | `syncopated_8ths`, `syncopated_16ths` |
| `swing` | `swing_8ths` |

The annotation heuristic emits `straight_8ths`, `syncopated_8ths`, or `straight_16ths`. Include **`straight` and `syncopated`** in `groove_families` at minimum. Add `swing` only if you intend to use swing groove tokens elsewhere; the current MIDI heuristic does not assign `swing_*` labels.

---

## 6. EDO (equal division of the octave)

| Setting | Default | Notes |
|---|---|---|
| **`--edo`** | `12` | Passed to pitch-class reduction and chord templates |
| **19-EDO** | Supported in API | Requires consistent `edo` at train and inference time |
| **Manifest** | `expected_edo` stored | Loaded bundle exposes this for validation |

All pitch classes in the corpus are interpreted modulo `edo`. Chord templates from `aimusic.theory.tonal` must support the chosen EDO.

---

## 7. Recommended corpus characteristics

### 7.1 Style alignment

Train on MIDI that resembles the music you plan to generate:

- Progressive rock / jazz fusion harmonic language (major, minor, dominant 7th, diminished)
- Meters you will use at generation time
- Similar rhythmic density and drum activity if groove tokens matter

### 7.2 Harmonic density

Prefer files where harmonic parts **re-articulate** voicings at or near the beat grid. Pads and long sustains without re-attacks contribute fewer annotated beats.

### 7.3 Transition diversity

The count prior learns **observed transitions**. Underrepresented moves (e.g. rare modulations, odd meters) receive high smoothing and weak statistical support. For robust priors:

- Include multiple pieces per meter and key area
- Include varied chord progressions (not only I–IV–V)
- Aim for broad coverage of `role` transitions (`hold`, `change`, `prep`, `cad`)

### 7.4 Length guidelines

| Corpus size | Expected effect |
|---|---|
| **Tiny (1–2 short files, few beats)** | Overfits local progressions; high variance |
| **Small (10–50 pieces)** | Usable prototype / smoke-test bundles |
| **Medium+ (100+ pieces, 1000+ transitions)** | More stable transition estimates |
| **Large** | Diminishing returns unless style-filtered |

There is no hard upper bound; all matching files in the tree are loaded into memory as transition queries before counting.

### 7.5 Content to exclude or isolate

| Content | Recommendation |
|---|---|
| **Drum-only tracks** | No harmonic annotations → no transitions |
| **Non-musical MIDI** | May produce spurious chord matches; exclude |
| **Heavily microtonal material** | Use matching `edo`; verify chord templates apply |
| **Unsupported meters** | Filter out or expand `--meter` |
| **Corrupt exports** | Validate before adding; one failure aborts training |

---

## 8. Training CLI reference

Install ML dependencies:

```bash
pip install -e ".[ml,dev]"
```

Train:

```bash
python -m aimusic.ml.cli train \
  --midi-dir ./corpus \
  --output ./artifacts/prior_v1 \
  --edo 12 \
  --alpha 1.0 \
  --model-version v1 \
  --meter 4/4 \
  --meter 5/4 \
  --groove-family straight \
  --groove-family syncopated
```

| Flag | Default | Dataset impact |
|---|---|---|
| `--midi-dir` | required | Corpus root |
| `--output` | required | Artifact bundle destination |
| `--edo` | `12` | Pitch-class reduction |
| `--alpha` | `1.0` | Smoothing (not a dataset filter) |
| `--model-version` | `v1` | Manifest label only |
| `--meter` | (StyleConfig default) | Filters which beats are annotated |
| `--groove-family` | (StyleConfig default) | Groove token vocabulary |

Programmatic API:

```python
from aimusic.core.config import StyleConfig
from aimusic.ml.train import train_prior_from_corpus

result = train_prior_from_corpus(
    "./corpus",
    "./artifacts/prior_v1",
    style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated")),
    edo=12,
    alpha=1.0,
)
# result.transition_count, result.midi_files_processed
```

---

## 9. Validation checklist

Before training a production bundle, verify:

- [ ] Corpus directory exists and contains only intended `.mid` / `.midi` files
- [ ] Each target meter appears in `--meter` / `StyleConfig.allowed_meters`
- [ ] Harmonic parts re-attack on beats you care about (not sustain-only voicings)
- [ ] Time signatures are present and correct in MIDI meta events
- [ ] `--edo` matches the tuning system of the corpus
- [ ] `--groove-family` includes families referenced by annotation labels
- [ ] Spot-check: `load_corpus_transitions(midi_dir, style_config)` returns `len(queries) > 0`
- [ ] Training output reports `transition_count > 0` and `midi_files_processed` matches expectation
- [ ] Bundle loads: `manifest.json`, `vocabularies.json`, and `counts/` exist
- [ ] Generation smoke test: `python -m aimusic.app.cli generate --beats 8 --prior-bundle ./artifacts/prior_v1`

### Minimal valid fixture pattern

The test corpus uses a two-beat C major → G major progression with explicit `note_on` per beat, 4/4 time signature, and 120 BPM tempo. This is the smallest pattern that yields annotated transitions:

```text
Beat 0: note_on C4, E4, G4  →  Cmaj
Beat 1: note_on G3, B3, D4  →  G7 or Gmaj (best overlap)
```

---

## 10. Known limitations (v1)

These are current pipeline constraints, not dataset author errors:

| Limitation | Impact |
|---|---|
| **Heuristic annotation** | Labels are inferred, not ground truth from human annotation |
| **No sustain carry-over** | Long held chords without re-attack produce sparse beats |
| **Single chord winner per beat** | Polytonality / slash chords reduced to one template |
| **GM drum channel assumption** | Drum groove features require channel 9 |
| **No per-file error isolation** | One unreadable MIDI file fails the whole run |
| **Count prior only** | v1 does not train a Flax neural model from raw MIDI features |
| **Factorized independence** | Streams modeled separately; no cross-stream interaction in counts |
| **Silent skip** | Skipped beats are not logged in training output (diagnostics API is internal) |

Future dataset formats (precomputed `BeatState` JSONL, multi-instrument role tags, human labels) would require pipeline extensions beyond the current MIDI-only path.

---

## 11. Related code

| Module | Responsibility |
|---|---|
| `aimusic/ml/midi_ingest.py` | MIDI → `MidiBeatGrid` |
| `aimusic/ml/annotate.py` | Beat grid → `BeatState` heuristics |
| `aimusic/ml/dataset.py` | Transition example building and JAX batch packing |
| `aimusic/ml/train.py` | End-to-end training entry point |
| `aimusic/ml/bundle.py` | Artifact save/load |
| `aimusic/core/vocab.py` | Vocabulary construction from `StyleConfig` |
| `aimusic/scoring/priors.py` | `PriorQuery`, tokenization, neural prior seam |

---

## 12. Summary

| Category | Requirement |
|---|---|
| **Input format** | Directory of Standard MIDI Files (`.mid`, `.midi`) |
| **Discovery** | Recursive; all matching files included |
| **Meter** | Must match `StyleConfig.allowed_meters` or beats are dropped |
| **Harmony** | At least one `note_on` per annotated beat on non-drum channels |
| **Chord match** | Pitch classes must overlap a vocabulary chord template |
| **Sequence length** | ≥ 2 annotated beats per file to emit transitions |
| **Corpus yield** | ≥ 1 transition total for a useful prior |
| **Consistency** | Same `StyleConfig`, `edo`, and vocabulary sizes at train and inference |
| **Quality** | Re-attacked harmonic content, explicit time signatures, style-aligned repertoire |
