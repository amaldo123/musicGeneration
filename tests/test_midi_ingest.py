from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from aimusic.ml.midi_ingest import parse_midi_beats


def write_simple_c_g_progression(path: Path) -> None:
    """Write a two-beat MIDI with C major then G major harmony."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # Beat 0: C major (C4 E4 G4)
    track.append(mido.Message("note_on", note=60, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_on", note=64, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_on", note=67, velocity=80, time=0, channel=0))

    # Beat 1: G major (G3 B3 D4) after one quarter note
    track.append(mido.Message("note_off", note=60, velocity=80, time=480, channel=0))
    track.append(mido.Message("note_off", note=64, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_off", note=67, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_on", note=55, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_on", note=59, velocity=80, time=0, channel=0))
    track.append(mido.Message("note_on", note=62, velocity=80, time=0, channel=0))
    track.append(mido.MetaMessage("end_of_track", time=480))
    mid.save(str(path))


class TestMidiIngest(unittest.TestCase):
    def test_parse_midi_beats_extracts_two_beats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = Path(tmp_dir) / "simple.mid"
            write_simple_c_g_progression(midi_path)
            grid = parse_midi_beats(midi_path)

        self.assertEqual(grid.ticks_per_beat, 480)
        self.assertGreaterEqual(len(grid.beats), 2)
        self.assertEqual(grid.beats[0].meter_signature, "4/4")
        self.assertIn(60, grid.beats[0].harmonic_pitches)


if __name__ == "__main__":
    unittest.main()
