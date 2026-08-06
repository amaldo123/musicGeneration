import unittest

from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.midi_ingest import BeatWindow, MidiBeatGrid
from tests.test_midi_ingest import write_simple_c_g_progression
import tempfile
from pathlib import Path
from aimusic.ml.midi_ingest import parse_midi_beats


class TestAnnotate(unittest.TestCase):
    def test_annotate_real_midi_produces_valid_beat_states(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = Path(tmp_dir) / "simple.mid"
            write_simple_c_g_progression(midi_path)
            grid = parse_midi_beats(midi_path)
            states = annotate_beat_grid(grid, DEFAULT_VOCABULARIES)

        self.assertGreaterEqual(len(states), 2)
        for state in states:
            self.assertGreaterEqual(state.meter_id, 0)
            self.assertGreaterEqual(state.chord_id, 0)

    def test_annotate_synthetic_grid(self):
        grid = MidiBeatGrid(
            ticks_per_beat=480,
            tempo_bpm=120.0,
            beats=(
                BeatWindow(
                    beat_index=0,
                    meter_signature="4/4",
                    beat_in_bar=0,
                    bar_index=0,
                    harmonic_pitches=(60, 64, 67),
                    drum_onsets=0,
                ),
                BeatWindow(
                    beat_index=1,
                    meter_signature="4/4",
                    beat_in_bar=1,
                    bar_index=0,
                    harmonic_pitches=(55, 59, 62),
                    drum_onsets=2,
                ),
            ),
        )
        states = annotate_beat_grid(grid, DEFAULT_VOCABULARIES)
        self.assertEqual(len(states), 2)
        cmaj = DEFAULT_VOCABULARIES.chords.token_for_label("Cmaj").id
        self.assertEqual(states[0].chord_id, cmaj)


if __name__ == "__main__":
    unittest.main()
