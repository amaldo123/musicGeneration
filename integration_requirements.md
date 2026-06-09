# Neural Integration Requirements

This document describes the exact integration contract the neural model team must satisfy to plug into the current `aiMusic` codebase.

## 1. NeuralPrior interface contract

Primary integration file: `aimusic/scoring/priors.py`.

The graph/planning code consumes the `Prior` protocol:

```python
class Prior(Protocol):
    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: Optional[PriorContext] = None,
    ) -> float: ...
```

Optional batch protocol:

```python
class BatchedPrior(Prior, Protocol):
    def logp_next_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]: ...
```

The external model object passed into `NeuralPrior(model=...)` must satisfy:

```python
class NeuralPriorModel(Protocol):
    def score_transition(self, query: TokenizedPriorQuery) -> float: ...
```

Optional batched model protocol:

```python
class BatchedNeuralPriorModel(NeuralPriorModel, Protocol):
    def score_transition_batch(
        self,
        queries: Sequence[TokenizedPriorQuery],
    ) -> Tuple[float, ...]: ...
```

Current wrapper class:

```python
@dataclass(frozen=True)
class NeuralPrior:
    config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    manifest: Optional[NeuralPriorManifest] = None
    model: Optional[NeuralPriorModel] = field(default=None, repr=False, compare=False)
```

Required behavior:

- The neural team does not need to subclass `NeuralPrior`.
- They must provide a model object that satisfies `NeuralPriorModel`.
- `NeuralPrior.logp_next(...)` is what the planner ultimately calls.
- If batch scoring is implemented, `score_transition_batch(...)` must return exactly one score per input query, in the same order.
- Scalar and batch outputs must be numerically consistent. The repository tests already enforce scalar/batch parity expectations on the wrapper seam.

## 2. Input specification

### 2.1 Raw structural state

`BeatState` is the canonical symbolic state:

```python
@dataclass(frozen=True)
class BeatState:
    meter_id: int
    beat_in_bar: int
    boundary_lvl: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int
```

All fields are validated as non-negative `int`.

### 2.2 Model-facing query payload

The neural model does not receive raw `BeatState` directly when wrapped by `NeuralPrior`. It receives:

```python
@dataclass(frozen=True)
class TokenizedPriorQuery:
    prev_event: StructuralEventTokens
    next_event: StructuralEventTokens
    time_index: int
    history_tokens: StructuralTokenSequence = field(default_factory=StructuralTokenSequence)
    future_hint_tokens: StructuralTokenSequence = field(default_factory=StructuralTokenSequence)
    section_name: Optional[str] = None
    metadata: MetadataPairs = ()
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED
```

With:

```python
@dataclass(frozen=True)
class StructuralEventTokens:
    meter_id: int
    beat_position: int
    boundary_level: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int
```

And:

```python
@dataclass(frozen=True)
class StructuralTokenSequence:
    meter_ids: Tuple[int, ...] = ()
    beat_positions: Tuple[int, ...] = ()
    boundary_levels: Tuple[int, ...] = ()
    key_ids: Tuple[int, ...] = ()
    chord_ids: Tuple[int, ...] = ()
    role_ids: Tuple[int, ...] = ()
    head_ids: Tuple[int, ...] = ()
    groove_ids: Tuple[int, ...] = ()
```

### 2.3 What the planner actually passes today

Current graph construction builds `PriorContext` as:

- `history=(source_state,)`
- `future_hints=end_layer.states[: min(3, len(end_layer.states))]`
- `metadata=(("graph_time", str(time_index)),)`
- `section_name=None`

In the current `run_method_a(...)` path, `start_layer` and `end_layer` are singleton endpoint distributions, so during actual Method A graph expansion the prior typically sees:

- one previous state
- one next-state candidate
- `history_tokens` length `1`
- `future_hint_tokens` length `1`
- no section label

There is a second use of batched prior scoring in candidate chord proposal. `_top_k_prior_chord_ids(...)` constructs one `PriorQuery` per chord token and ranks all chord options. With default vocabulary this is a batch of `48` tokenized transition queries.

### 2.4 Vocabulary sizes and default token sets

All eight streams are discrete token IDs.

Default vocabulary sizes from `DEFAULT_VOCABULARIES`:

- `meter`: `4`
- `beat_position`: `7`
- `boundary`: `4`
- `key`: `12`
- `chord`: `48`
- `role`: `4`
- `head`: `8`
- `groove`: `5`

Default tokens:

`meter`

| id | label | beats_per_bar | strong_beats |
|---|---|---:|---|
| 0 | `4/4` | 4 | `(0, 2)` |
| 1 | `3/4` | 3 | `(0,)` |
| 2 | `5/4` | 5 | `(0, 3)` |
| 3 | `7/4` | 7 | `(0, 4)` |

`beat_position`

| id | label | index |
|---|---|---:|
| 0 | `beat_1` | 0 |
| 1 | `beat_2` | 1 |
| 2 | `beat_3` | 2 |
| 3 | `beat_4` | 3 |
| 4 | `beat_5` | 4 |
| 5 | `beat_6` | 5 |
| 6 | `beat_7` | 6 |

`boundary`

| id | label | level |
|---|---|---:|
| 0 | `none` | 0 |
| 1 | `local` | 1 |
| 2 | `phrase` | 2 |
| 3 | `section` | 3 |

`key`

| id | label | root_pc |
|---|---|---:|
| 0 | `C` | 0 |
| 1 | `C#` | 1 |
| 2 | `D` | 2 |
| 3 | `Eb` | 3 |
| 4 | `E` | 4 |
| 5 | `F` | 5 |
| 6 | `F#` | 6 |
| 7 | `G` | 7 |
| 8 | `Ab` | 8 |
| 9 | `A` | 9 |
| 10 | `Bb` | 10 |
| 11 | `B` | 11 |

`role`

| id | label | description |
|---|---|---|
| 0 | `hold` | Maintain the current harmonic function. |
| 1 | `prep` | Prepare an upcoming structural change. |
| 2 | `change` | Introduce a local structural departure. |
| 3 | `cad` | Drive toward a cadential arrival. |

`head`

| id | label | description |
|---|---|---|
| 0 | `rest` | No melodic head anchor on this beat. |
| 1 | `root` | Anchor the melodic head on the chord root. |
| 2 | `third` | Anchor the melodic head on the chord third. |
| 3 | `fifth` | Anchor the melodic head on the chord fifth. |
| 4 | `seventh` | Anchor the melodic head on the chord seventh. |
| 5 | `extension` | Anchor the head on a color tone or extension. |
| 6 | `upper_approach` | Approach the target head from above. |
| 7 | `lower_approach` | Approach the target head from below. |

`groove`

| id | label | family | subdivision |
|---|---|---|---:|
| 0 | `straight_8ths` | `straight` | 8 |
| 1 | `straight_16ths` | `straight` | 16 |
| 2 | `syncopated_8ths` | `syncopated` | 8 |
| 3 | `syncopated_16ths` | `syncopated` | 16 |
| 4 | `swing_8ths` | `swing` | 8 |

`chord`

- Quality order is fixed: `maj`, `min`, `7`, `dim`.
- Default token ID rule is `id = (root_pc * 4) + quality_index`.
- Default labels are:
  - `0 Cmaj`, `1 Cmin`, `2 C7`, `3 Cdim`
  - `4 C#maj`, `5 C#min`, `6 C#7`, `7 C#dim`
  - `8 Dmaj`, `9 Dmin`, `10 D7`, `11 Ddim`
  - `12 Ebmaj`, `13 Ebmin`, `14 Eb7`, `15 Ebdim`
  - `16 Emaj`, `17 Emin`, `18 E7`, `19 Edim`
  - `20 Fmaj`, `21 Fmin`, `22 F7`, `23 Fdim`
  - `24 F#maj`, `25 F#min`, `26 F#7`, `27 F#dim`
  - `28 Gmaj`, `29 Gmin`, `30 G7`, `31 Gdim`
  - `32 Abmaj`, `33 Abmin`, `34 Ab7`, `35 Abdim`
  - `36 Amaj`, `37 Amin`, `38 A7`, `39 Adim`
  - `40 Bbmaj`, `41 Bbmin`, `42 Bb7`, `43 Bbdim`
  - `44 Bmaj`, `45 Bmin`, `46 B7`, `47 Bdim`

### 2.5 Configurable vocabulary dimensions

The neural team must not hard-code default counts as immutable system constants.

`StyleConfig` can change:

- `allowed_meters`
- `groove_families`
- `chord_vocabulary_size`
- `key_vocabulary_size`

`NeuralPriorConfig.factorization_mode` can also vary:

- `PriorFactorization.WHOLE_STATE`
- `PriorFactorization.FACTORIZED`
- `PriorFactorization.MIXED`

Current code standardizes on `FACTORIZED`.

## 3. Output specification

The scorer must return a single scalar `float` per transition query.

Semantic contract:

- Return value is treated as a data-prior log-score.
- Higher is better.
- More positive means the transition is preferred by the data prior.
- It is not an energy.
- It does not need to be normalized across all outgoing states.

The graph builder combines the prior with symbolic energy as:

```python
(lambda_data * data_logp) - (lambda_gttm * gttm_energy)
```

Where:

- `data_logp = prior.logp_next(...)`
- `gttm_energy = calculate_gttm_energy(...)`

Required numeric constraints:

- Must be finite.
- Must never be `NaN`.
- Must never be `+inf` or `-inf`.
- Batched return type must be `Tuple[float, ...]`.
- Batch length must exactly equal query count.

## 4. Vocabulary export requirements

The symbolic side must hand the neural team:

- Full token enumerations for all 8 streams.
- A `StyleConfig` snapshot used to generate those vocabularies.
- A `NeuralPriorManifest`.
- The expected EDO cardinality.

Required fields per stream:

- `meter`: `id`, `label`, `beats_per_bar`, `strong_beats`
- `beat_position`: `id`, `label`, `index`
- `boundary`: `id`, `label`, `level`
- `key`: `id`, `label`, `root_pc`
- `chord`: `id`, `label`, `root_pc`, `quality`
- `role`: `id`, `label`, `description`
- `head`: `id`, `label`, `description`
- `groove`: `id`, `label`, `family`, `subdivision`

Required manifest shape:

```python
@dataclass(frozen=True)
class NeuralPriorManifest:
    manifest_version: int = 1
    model_family: str = "external_neural_prior"
    model_version: str = "placeholder-v1"
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED
    token_streams: Tuple[str, ...] = STRUCTURAL_STREAM_NAMES
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    supports_batch_scoring: bool = True
    expected_edo: Optional[int] = None
    metadata: MetadataPairs = ()
```

Required export format:

- Use JSON for the manifest, because the repository already implements `save_neural_prior_manifest(...)` and `load_neural_prior_manifest(...)`.
- Export the vocabularies as JSON objects with stable field names matching the token dataclasses above.
- Do not use pickle as the primary interchange format.

## 5. Schrödinger Bridge coupling

The neural scorer is not called inside the SB solver iterations themselves. It is used earlier, during sparse graph construction, to assign each edge a static log weight.

Coupling path:

1. `build_sparse_graph(...)` proposes legal `BeatState` successors.
2. For each retained transition edge, it calls `calculate_transition_log_weight(...)`.
3. That computes:
   - `data_logp = prior.logp_next(...)`
   - `gttm_energy = calculate_gttm_energy(...)`
4. Edge weight is:

```python
edge.log_weight = (lambda_data * data_logp) - (lambda_gttm * gttm_energy)
```

5. `build_sb_problem(...)` converts those edge weights into the SB problem.
6. `solve_sb(...)` uses `edge.log_weight / sb_config.temperature` as the sparse log-kernel.

Implication for the neural team:

- Your model supplies the data-prior term only.
- You are not replacing GTTM.
- You are not directly returning bridge potentials or SB marginals.
- Your score acts as a static transition log-bias before SB conditioning.

## 6. Batching and performance requirements

Current default sparsity controls from `SBConfig`:

- `horizon_t = 64`
- `k_max = 64`
- `d_max = 8`
- `batch_size` in `NeuralPriorConfig` defaults to `32`

What this means operationally:

- Graph expansion retains at most `d_max` scored outgoing edges per source state.
- A non-final layer retains at most `k_max` states after pruning.
- At default settings, a layer can therefore retain at most about `64 * 8 = 512` scored edges before later pruning effects.
- Across a 64-step horizon, the full planning pass can require many thousands of scalar transition scores.

Current batch-usage reality:

- The optional batch prior API exists.
- `prior_logps(...)` will use `logp_next_batch(...)` when available.
- Current graph edge scoring batches per-source transition queries via `calculate_transition_log_weights(...)`.
- Current chord proposal ranking is the only path that definitely batches today, with up to one query per chord token. Default batch size there is `48`.

Requirements:

- Scalar inference support is mandatory.
- Efficient batch inference is strongly recommended because graph edge scoring now uses the batch prior path when available.
- GPU inference is not required by the interface.
- CPU inference must be acceptable because the codebase has no framework-specific GPU runtime hooks.
- The model must be fast enough to handle repeated small-batch or scalar calls without destabilizing planning latency.

## 7. Realizer interface

The repository’s symbolic score contract is:

```python
@dataclass(frozen=True)
class NoteEvent:
    ton: int
    toff: int
    h: int
    v: float
    e: Tuple[float, ...] = ()
    track: str = "default"
```

```python
@dataclass(frozen=True)
class Score:
    note_events: Tuple[NoteEvent, ...] = ()
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0
```

Constraints:

- `ton` and `toff` are integer tick positions.
- `toff > ton`.
- `h` is an integer pitch height in EDO steps, not a MIDI note number.
- `v` is a float in `[0.0, 1.0]`.
- `e` is a tuple of finite floats for expressive controls.
- `track` must be a non-empty string.

Current decoder emits four track names:

- `bass`
- `comping`
- `lead`
- `drums`

The current MIDI renderer does not consume `Score` directly. It consumes:

```python
@dataclass(frozen=True)
class SymbolicNote:
    pitch_height: int
    start_time: float
    end_time: float
    velocity: int = 64
    timbre: int | None = None
    pressure: int | None = None
```

Therefore a multitrack neural realizer is only integration-compatible today if it can either:

- emit repository-native `Score` / `NoteEvent` objects and a separate adapter converts them to `SymbolicNote`, or
- replace the current decode-plus-render path with a new adapter layer agreed with the symbolic team.

There is now a diagnostics artifact layer in `aimusic/core/diagnostics.py`, but it does not change the rendering contract. It defines:

```python
@dataclass(frozen=True)
class RunManifest:
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    sb_stats: SBDiagnostics = field(default_factory=SBDiagnostics)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"
```

This is a run-reporting/inspection artifact, not a score or renderer input type.

EPIC 9 also added diagnostic payload contracts that matter if the neural team is expected to interoperate with run reports or regression fixtures:

```python
@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    label: str
```

```python
@dataclass
class StructuralDiagnostics:
    key_timeline: List[TimelineEvent] = field(default_factory=list)
    chord_timeline: List[TimelineEvent] = field(default_factory=list)
    role_timeline: List[TimelineEvent] = field(default_factory=list)
    groove_timeline: List[TimelineEvent] = field(default_factory=list)
    boundaries: List[float] = field(default_factory=list)
    tension_curve: List[Tuple[float, float]] = field(default_factory=list)
```

```python
@dataclass
class SBDiagnostics:
    iterations_run: int = 0
    converged: bool = False
    final_max_delta: float = 0.0
    layer_sizes: List[int] = field(default_factory=list)
    pruned_nodes: int = 0
    effective_entropy: float = 0.0
```

These are not neural-prior inputs, but they are now part of the observable end-to-end integration surface around reproducibility and inspection.

Renderer constraints from `midi_render.py`:

- MIDI channel `0` is reserved as MPE master.
- Note channels are allocated from `1..15`.
- More than `15` overlapping notes raises `ValueError`.
- Non-12-EDO rendering currently uses `MicrotonalRendering.MPE`.
- `MicrotonalRendering.MTS` raises `NotImplementedError`.
- Per-note timbre, when present, is emitted as MIDI CC `74`.
- Per-note pressure, when present, is emitted as MIDI `aftertouch`.
- The renderer writes deterministic event ordering and includes a track-name meta event plus a tempo meta event.

EPIC 8 also added a renderer-inspection helper:

```python
@dataclass(frozen=True)
class MidiSummary:
    total_notes: int
    unique_channels: Tuple[int, ...]
    pitch_bend_events: int
    timbre_events: int
    pressure_events: int
```

and:

```python
def summarize_midi(filepath: str) -> MidiSummary: ...
```

This means rendered output is now expected to be inspectable in terms of note count, channel usage, pitch-bend usage, timbre events, and pressure events.

## 8. Hard constraints and invariants

- The model must be deterministic for a fixed input query. No RNG is passed into the prior interface.
- The model must never return `NaN` or infinite values.
- If batch scoring is implemented, it must preserve query order and exact output count.
- The model must accept all token IDs present in the exported vocabularies.
- The model must not assume undocumented tokens outside the manifest’s `token_streams`.
- The model must tolerate empty `history_tokens` and empty `future_hint_tokens`, because `PriorContext` allows both.
- The model must tolerate `section_name=None`.
- The model must tolerate arbitrary metadata tuples of string pairs.
- The model must work with `PriorFactorization.FACTORIZED` at minimum.
- The model must not violate purity assumptions by mutating query objects.

Planner-side hard gating means the neural model will not see arbitrary transitions. Candidate generation already enforces:

- meter continuity and meter-change legality
- contiguous beat progression
- boundary placement legality
- role legality
- key-change legality
- groove-family change legality
- head/chord compatibility legality

The model scores only transitions that survived those symbolic gates.

## 9. Open questions and hard blockers

These items are unresolved in code and should be treated as blockers or clarification items before full neural integration.

1. `NeuralPrior` is still a placeholder seam.
The current `NeuralPrior` wrapper contains `_score_placeholder(...)` logic and defaults to `PlaceholderPriorMode.STRUCTURED` or `NEUTRAL`. A production model implementation is not present in the repository.

2. Whole-state and mixed factorization are declared but not concretely specified.
`PriorFactorization.WHOLE_STATE` and `PriorFactorization.MIXED` exist in config, but the repository only defines a concrete factorized token contract. There is no whole-state token schema to train against. This is a hard blocker if the neural team is asked to target `WHOLE_STATE` or `MIXED`.

3. Checkpoint and tokenizer artifact formats are unspecified.
`NeuralPriorConfig` and `NeuralPriorManifest` contain `checkpoint_path` and `tokenizer_path`, but the file formats, serialization libraries, loader expectations, and versioning policy are not defined.

4. The MIDI renderer is not wired to `Score` directly.
`decode_path_to_score(...)` returns `Score`, while `render_midi(...)` expects `List[SymbolicNote]`. The adapter contract between symbolic score output and renderer input is missing. This is a hard blocker for any neural realizer expected to plug directly into the current renderer.

5. Tempo and time-signature handling are incomplete in the renderer.
`Score` carries `tempo_bpm`, but `render_midi(...)` hardcodes `120` BPM in the MIDI meta message. Time-signature meta events are not emitted. This is a hard blocker if the neural realizer is expected to control rendered tempo or meter metadata.

6. The current renderer still does not accept `Score` directly despite EPIC 8 export progress.
EPIC 8 established playable MIDI export plus inspection helpers, but the concrete boundary is still `List[SymbolicNote] -> render_midi(...)`, not `Score -> render_midi(...)`. Any neural realizer handoff still needs an adapter.

7. `MicrotonalRendering.MTS` is explicitly unimplemented.
Only the MPE path is operational for non-12-EDO MIDI rendering.

8. CLI export is still a placeholder.
`aimusic/app/cli.py` contains an `Export command placeholder invoked.` path rather than a finished export workflow.

9. CLI `generate` currently emits mock diagnostics, not a real generation pipeline run.
`aimusic/app/cli.py` now creates `RunManifest`, `StructuralDiagnostics`, and `SBDiagnostics`, but its `generate` subcommand still fabricates placeholder structural timelines and mock SB statistics instead of invoking `run_method_a(...)`, decoding, or rendering. The neural team should not treat the CLI manifest path as authoritative pipeline integration yet.

10. EPIC 9 diagnostics exist, but they are only partially connected to the real planning/render path.
The diagnostics dataclasses and end-to-end regression test surface now exist, but there is still no authoritative app-layer path that runs planning, decode, MIDI export, and manifest emission as one implemented production flow. This matters if the neural team is expected to validate integration via CLI/app commands rather than direct Python APIs.

11. `SBBackend.JAX` is explicitly unimplemented.
Only the NumPy SB backend exists today. This is not a blocker for neural prior integration by itself, but it blocks any assumption that a JAX-native bridge pipeline already exists.
