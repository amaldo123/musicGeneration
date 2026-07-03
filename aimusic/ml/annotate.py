from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import ChordToken, Vocabularies
from aimusic.ml.midi_ingest import BeatWindow, MidiBeatGrid
from aimusic.theory.tonal import chord_pitch_classes, is_dominant, is_fifth_down


@dataclass(frozen=True)
class AnnotationDiagnostics:
    skipped_beats: int = 0
    reasons: Tuple[str, ...] = ()


def _pitch_classes_from_midi_notes(notes: Sequence[int], edo: int = 12) -> frozenset[int]:
    return frozenset(note % edo for note in notes)


def _best_chord_token(
    pitch_classes: frozenset[int],
    vocabularies: Vocabularies,
    edo: int = 12,
) -> Optional[ChordToken]:
    if not pitch_classes:
        return None

    best: Optional[ChordToken] = None
    best_score = -1
    for chord in vocabularies.chords:
        try:
            template = chord_pitch_classes(chord.root_pc, chord.quality, edo=edo)
        except ValueError:
            continue
        overlap = len(pitch_classes & template)
        if overlap > best_score:
            best_score = overlap
            best = chord
    if best_score <= 0:
        return None
    return best


def _key_id_for_root(root_pc: int, vocabularies: Vocabularies, *, edo: int = 12) -> int:
    target = root_pc % edo
    for key in vocabularies.keys:
        if key.root_pc == target:
            return key.id
    return vocabularies.keys.token_for_id(0).id


def _head_id_for_melody(
    pitch_classes: frozenset[int],
    chord: ChordToken,
    vocabularies: Vocabularies,
    edo: int = 12,
) -> int:
    if not pitch_classes:
        return vocabularies.heads.token_for_label("rest").id

    melody_pc = max(pitch_classes)
    template = chord_pitch_classes(chord.root_pc, chord.quality, edo=edo)
    if melody_pc == chord.root_pc % edo:
        return vocabularies.heads.token_for_label("root").id
    if (chord.root_pc + 4) % edo in pitch_classes and melody_pc == (chord.root_pc + 4) % edo:
        return vocabularies.heads.token_for_label("third").id
    if (chord.root_pc + 7) % edo in pitch_classes and melody_pc == (chord.root_pc + 7) % edo:
        return vocabularies.heads.token_for_label("fifth").id
    if (chord.root_pc + 10) % edo in pitch_classes and melody_pc == (chord.root_pc + 10) % edo:
        return vocabularies.heads.token_for_label("seventh").id
    return vocabularies.heads.token_for_label("extension").id


def _role_id(
    prev_chord: Optional[ChordToken],
    next_chord: ChordToken,
    vocabularies: Vocabularies,
    edo: int = 12,
) -> int:
    if prev_chord is None:
        return vocabularies.roles.token_for_label("hold").id
    if prev_chord.id == next_chord.id:
        return vocabularies.roles.token_for_label("hold").id
    if is_dominant(next_chord.quality):
        return vocabularies.roles.token_for_label("prep").id
    if is_fifth_down(prev_chord.root_pc, next_chord.root_pc, edo=edo):
        return vocabularies.roles.token_for_label("cad").id
    return vocabularies.roles.token_for_label("change").id


def _boundary_level(beat: BeatWindow) -> int:
    if beat.beat_in_bar != 0:
        return 0
    if beat.bar_index > 0 and beat.bar_index % 8 == 0:
        return 3
    if beat.bar_index > 0 and beat.bar_index % 4 == 0:
        return 2
    if beat.bar_index > 0:
        return 1
    return 0


def _groove_id(beat: BeatWindow, vocabularies: Vocabularies) -> int:
    if beat.drum_onsets >= 4:
        return vocabularies.grooves.token_for_label("straight_16ths").id
    if beat.drum_onsets >= 2:
        return vocabularies.grooves.token_for_label("syncopated_8ths").id
    return vocabularies.grooves.token_for_label("straight_8ths").id


def annotate_beat_grid(
    grid: MidiBeatGrid,
    vocabularies: Vocabularies,
    *,
    edo: int = 12,
) -> Tuple[BeatState, ...]:
    """Convert a beat grid into BeatState annotations using vocabulary-aware heuristics."""
    states: list[BeatState] = []
    prev_chord: Optional[ChordToken] = None

    meter_labels = {token.label: token.id for token in vocabularies.meters}

    for beat in grid.beats:
        meter_id = meter_labels.get(beat.meter_signature)
        if meter_id is None:
            continue

        pitch_classes = _pitch_classes_from_midi_notes(beat.harmonic_pitches, edo=edo)
        chord = _best_chord_token(pitch_classes, vocabularies, edo=edo)
        if chord is None:
            continue

        key_id = _key_id_for_root(chord.root_pc, vocabularies, edo=edo)
        boundary_label = ("none", "local", "phrase", "section")[_boundary_level(beat)]
        boundary_lvl = vocabularies.boundaries.token_for_label(boundary_label).id

        state = BeatState(
            meter_id=meter_id,
            beat_in_bar=beat.beat_in_bar,
            boundary_lvl=boundary_lvl,
            key_id=key_id,
            chord_id=chord.id,
            role_id=_role_id(prev_chord, chord, vocabularies, edo=edo),
            head_id=_head_id_for_melody(pitch_classes, chord, vocabularies, edo=edo),
            groove_id=_groove_id(beat, vocabularies),
        )
        states.append(state)
        prev_chord = chord

    return tuple(states)
