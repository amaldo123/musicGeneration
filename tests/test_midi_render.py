import os
import tempfile
import unittest
import mido

from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.theory.edo import EDO
from aimusic.render.midi_render import SymbolicNote, render_midi

class TestMidiRender(unittest.TestCase):
    def setUp(self):
        # 12-EDO setup
        self.config_12 = EDOConfig(
            n=12, base_tuning=60, pitch_bend_range=2, microtonal_rendering_method=MicrotonalRendering.MPE
        )
        self.edo_12 = EDO(self.config_12)

        # 19-EDO setup
        self.config_19 = EDOConfig(
            n=19, base_tuning=60, pitch_bend_range=48, microtonal_rendering_method=MicrotonalRendering.MPE
        )
        self.edo_19 = EDO(self.config_19)
        
        self.test_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.test_dir.name, "test_out.mid")

    def tearDown(self):
        self.test_dir.cleanup()

    def test_direct_12_edo_mapping(self):
        """Tests that pitch heights map exactly to integer MIDI notes with no pitch bends."""
        notes = [
            SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0),
            SymbolicNote(pitch_height=2, start_time=1.0, end_time=2.0),
        ]
        
        render_midi(notes, self.edo_12, self.output_path)
        
        mid = mido.MidiFile(self.output_path)
        note_on_events = [msg for msg in mid.tracks[0] if msg.type == 'note_on']
        pitchwheel_events = [msg for msg in mid.tracks[0] if msg.type == 'pitchwheel']
        
        self.assertEqual(len(note_on_events), 2)
        self.assertEqual(note_on_events[0].note, 60)
        self.assertEqual(note_on_events[1].note, 62)
        self.assertEqual(len(pitchwheel_events), 0)

    def test_mpe_19_edo_rendering_and_channel_allocation(self):
        """Tests microtonal pitch bends and MPE channel allocation for overlapping notes."""
        
        notes = [
            SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0),
            SymbolicNote(pitch_height=1, start_time=0.0, end_time=1.0),
        ]
        
        render_midi(notes, self.edo_19, self.output_path)
        
        mid = mido.MidiFile(self.output_path)
        note_on_events = [msg for msg in mid.tracks[0] if msg.type == 'note_on']
        pitchwheel_events = [msg for msg in mid.tracks[0] if msg.type == 'pitchwheel']
        
        self.assertEqual(len(note_on_events), 2)
        
        self.assertNotEqual(note_on_events[0].channel, note_on_events[1].channel)
        
        self.assertGreater(len(pitchwheel_events), 0)
        
        detuned_note_channel = note_on_events[1].channel
        bend_channel = pitchwheel_events[0].channel
        self.assertEqual(detuned_note_channel, bend_channel)
    
    def test_mts_rendering_is_deferred(self):
        """Ensures MTS rendering is clearly documented as deferred with an exception."""
        config_mts = EDOConfig(
            n=19, base_tuning=60, pitch_bend_range=48, microtonal_rendering_method=MicrotonalRendering.MTS
        )
        edo_mts = EDO(config_mts)
        notes = [SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0)]
        
        with self.assertRaises(NotImplementedError) as context:
            render_midi(notes, edo_mts, self.output_path)
            
        self.assertIn("deferred", str(context.exception))

if __name__ == "__main__":
    unittest.main()