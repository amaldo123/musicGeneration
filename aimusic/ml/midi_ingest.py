from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import mido


@dataclass(frozen=True)
class BeatWindow:
    """Notes active during one beat window."""

    beat_index: int
    meter_signature: str
    beat_in_bar: int
    bar_index: int
    harmonic_pitches: Tuple[int, ...]
    drum_onsets: int


@dataclass(frozen=True)
class MidiBeatGrid:
    """Beat-quantized view of a MIDI file."""

    ticks_per_beat: int
    tempo_bpm: float
    beats: Tuple[BeatWindow, ...]


def _meter_signature(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}"


def _merge_tracks_to_messages(midi_file: mido.MidiFile) -> list[tuple[int, mido.Message]]:
    merged: list[tuple[int, mido.Message]] = []
    for track in midi_file.tracks:
        tick = 0
        for message in track:
            tick += message.time
            merged.append((tick, message))
    merged.sort(key=lambda item: item[0])
    return merged


def _initial_meter(messages: Sequence[tuple[int, mido.Message]]) -> tuple[int, int]:
    for _, message in messages:
        if message.type == "time_signature":
            return int(message.numerator), int(message.denominator)
    return 4, 4


def _tick_to_beat(tick: int, ticks_per_beat: int) -> int:
    return tick // ticks_per_beat


def parse_midi_beats(path: str | Path) -> MidiBeatGrid:
    """Parse a MIDI file into beat-aligned harmonic and drum activity windows."""
    midi_path = Path(path)
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    midi_file = mido.MidiFile(str(midi_path))
    messages = _merge_tracks_to_messages(midi_file)
    ticks_per_beat = midi_file.ticks_per_beat

    tempo = mido.bpm2tempo(120)
    numerator, denominator = _initial_meter(messages)
    meter_signature = _meter_signature(numerator, denominator)

    active_notes: dict[tuple[int, int], int] = {}
    beats_per_bar = numerator
    beat_windows: dict[int, dict[str, object]] = {}

    for tick, message in messages:
        beat_index = _tick_to_beat(tick, ticks_per_beat)

        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "time_signature":
            numerator = int(message.numerator)
            denominator = int(message.denominator)
            beats_per_bar = numerator
            meter_signature = _meter_signature(numerator, denominator)
        elif message.type == "note_on" and message.velocity > 0:
            channel = getattr(message, "channel", 0)
            key = (channel, message.note)
            active_notes[key] = beat_index
            window = beat_windows.setdefault(
                beat_index,
                {
                    "harmonic": set(),
                    "drum_onsets": 0,
                    "meter_signature": meter_signature,
                    "beats_per_bar": beats_per_bar,
                },
            )
            if channel == 9:
                window["drum_onsets"] = int(window["drum_onsets"]) + 1
            else:
                cast_harmonic = window["harmonic"]
                assert isinstance(cast_harmonic, set)
                cast_harmonic.add(int(message.note))
        elif message.type in {"note_off", "note_on"}:
            channel = getattr(message, "channel", 0)
            key = (channel, message.note)
            active_notes.pop(key, None)

    if not beat_windows:
        return MidiBeatGrid(
            ticks_per_beat=ticks_per_beat,
            tempo_bpm=float(mido.tempo2bpm(tempo)),
            beats=(),
        )

    max_beat = max(beat_windows)
    beats: list[BeatWindow] = []
    for beat_index in range(max_beat + 1):
        window = beat_windows.get(
            beat_index,
            {
                "harmonic": set(),
                "drum_onsets": 0,
                "meter_signature": meter_signature,
                "beats_per_bar": beats_per_bar,
            },
        )
        bpb = int(window["beats_per_bar"])
        beats.append(
            BeatWindow(
                beat_index=beat_index,
                meter_signature=str(window["meter_signature"]),
                beat_in_bar=beat_index % bpb,
                bar_index=beat_index // bpb,
                harmonic_pitches=tuple(sorted(window["harmonic"])),
                drum_onsets=int(window["drum_onsets"]),
            )
        )

    return MidiBeatGrid(
        ticks_per_beat=ticks_per_beat,
        tempo_bpm=float(mido.tempo2bpm(tempo)),
        beats=tuple(beats),
    )
