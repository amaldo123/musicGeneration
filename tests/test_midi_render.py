import os
import tempfile
import unittest
import mido

from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.theory.edo import EDO
from aimusic.render.midi_render import SymbolicNote, render_midi

class TestMidiRender12EDO(unittest.TestCase):
    def setUp(self):
        # Create a standard 12-EDO config where pitch height 0 = Middle C (60)
        self.config = EDOConfig(
            n=12,
            base_tuning=60,
            pitch_bend_range=2,
            microtonal_rendering_method=MicrotonalRendering.MPE
        )
        self.edo = EDO(self.config)
        
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.test_dir.name, "test_out.mid")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_direct_12_edo_mapping(self):
        """Tests that pitch heights map exactly to integer MIDI notes in 12-EDO."""
        
        notes = [
            SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0), # Middle C
            SymbolicNote(pitch_height=2, start_time=1.0, end_time=2.0), # D
            SymbolicNote(pitch_height=4, start_time=2.0, end_time=3.0), # E
        ]
        
        render_midi(notes, self.edo, self.output_path)
        
        self.assertTrue(os.path.exists(self.output_path))
        mid = mido.MidiFile(self.output_path)
        
        note_on_events = [msg for msg in mid.tracks[0] if msg.type == 'note_on']
        
        self.assertEqual(len(note_on_events), 3)
        self.assertEqual(note_on_events[0].note, 60) # C4
        self.assertEqual(note_on_events[1].note, 62) # D4
        self.assertEqual(note_on_events[2].note, 64) # E4

    def test_rendering_raises_on_unsupported_edo(self):
        """Ensures it block 19-EDO."""
        config_19 = EDOConfig(n=19, base_tuning=60, pitch_bend_range=48, microtonal_rendering_method=MicrotonalRendering.MPE)
        edo_19 = EDO(config_19)
        notes = [SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0)]
        
        with self.assertRaises(NotImplementedError):
            render_midi(notes, edo_19, self.output_path)

if __name__ == "__main__":
    unittest.main()