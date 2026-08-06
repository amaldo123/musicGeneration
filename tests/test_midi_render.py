import os
import tempfile
import unittest
import mido

from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.core.core_types import NoteEvent, Score
from aimusic.theory.edo import EDO
from aimusic.render.midi_render import (
    DEFAULT_DRUM_CHANNEL,
    SymbolicNote,
    render_midi,
    summarize_midi,
)

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
        self.assertEqual(len(pitchwheel_events), 2)
        self.assertEqual(pitchwheel_events[0].pitch, 0)
        self.assertEqual(pitchwheel_events[1].pitch, 0)

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
    
    def test_mts_rendering_produces_sysex_and_notes(self):
        """MTS uses one MIDI key per EDO step and emits accurate tuning data."""
        config_mts = EDOConfig(
            n=19, base_tuning=60, pitch_bend_range=48, microtonal_rendering_method=MicrotonalRendering.MTS
        )
        edo_mts = EDO(config_mts)
        notes = [
            SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0),
            SymbolicNote(pitch_height=1, start_time=1.0, end_time=2.0),
        ]

        render_midi(notes, edo_mts, self.output_path)

        mid = mido.MidiFile(self.output_path)
        all_msgs = [msg for track in mid.tracks for msg in track]
        sysex_events = [msg for msg in all_msgs if msg.type == "sysex"]
        note_on_events = [msg for msg in all_msgs if msg.type == "note_on"]

        self.assertEqual(len(sysex_events), 1)
        self.assertEqual([event.note for event in note_on_events], [60, 61])

        data = list(sysex_events[0].data)
        self.assertEqual(data[:5], [0x7E, 0x7F, 0x08, 0x01, 0x00])
        self.assertEqual(len(data), 406)
        checksum = 0
        for value in data[:-1]:
            checksum ^= value
        self.assertEqual(data[-1], checksum)

        entry_offset = 5 + 16 + 60 * 3
        root_entry = data[entry_offset:entry_offset + 3]
        step_entry = data[entry_offset + 3:entry_offset + 6]

        def decoded_pitch(entry):
            return entry[0] + (((entry[1] << 7) | entry[2]) / 16384.0)

        self.assertAlmostEqual(decoded_pitch(root_entry), 60.0, places=6)
        self.assertAlmostEqual(decoded_pitch(step_entry), 60.0 + 12.0 / 19.0, places=4)

    def test_19_edo_drum_notes_remain_general_midi_notes(self):
        score = Score(
            note_events=(
                NoteEvent(0, 120, 36, 1.0, track="drums"),
            )
        )

        render_midi(score, self.edo_19, self.output_path)

        note_ons = [
            message
            for track in mido.MidiFile(self.output_path).tracks
            for message in track
            if message.type == "note_on" and message.velocity > 0
        ]
        self.assertEqual(note_ons[0].note, 36)
        self.assertEqual(note_ons[0].channel, DEFAULT_DRUM_CHANNEL)

    def test_expressive_controls_rendered(self):
        """Tests that MPE Timbre (CC74) and Pressure (Aftertouch) are correctly written."""
        notes = [SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0, velocity=100, 
                timbre=85, pressure=60)]
        
        render_midi(notes, self.edo_12, self.output_path)
        
        mid = mido.MidiFile(self.output_path)
        
        # Filter specifically for CC 74 rather than grabbing all control changes
        timbre_events = [msg for msg in mid.tracks[0] if msg.type == 'control_change' and msg.control == 74]
        at_events = [msg for msg in mid.tracks[0] if msg.type == 'aftertouch']
        
        # Verify Timbre (CC74)
        self.assertEqual(len(timbre_events), 1)
        self.assertEqual(timbre_events[0].value, 85)
        
        # Verify Channel Pressure
        self.assertEqual(len(at_events), 1)
        self.assertEqual(at_events[0].value, 60)
    
    def test_summarize_midi_helper(self):
        """Tests that the inspection helper accurately tallies the MPE events in a file."""
        notes = [
            SymbolicNote(pitch_height=0, start_time=0.0, end_time=1.0),
            SymbolicNote(pitch_height=1, start_time=0.0, end_time=1.0, timbre=100, pressure=50),
        ]
        
        render_midi(notes, self.edo_19, self.output_path)
        
        summary = summarize_midi(self.output_path)
        
        # Verify the stats match with 2-note input
        self.assertEqual(summary.total_notes, 2)
        self.assertEqual(len(summary.unique_channels), 2)  
        
        # EDO 19 pitch height '1' requires a pitch bend
        self.assertGreater(summary.pitch_bend_events, 0)
        
        # Only the second note had expressive controls
        self.assertEqual(summary.timbre_events, 1)
        self.assertEqual(summary.pressure_events, 1)

    def test_mpe_polyphony_limit_enforcement(self):
        """Ensures the renderer safely fails instead of corrupting MPE channels when > 15 notes overlap."""
        
        # Create 16 notes that all play at the exact same time (0.0 to 1.0)
        # Since MPE only has 15 free channels (1-15), the 16th note must trigger the safe failure.
        notes = [SymbolicNote(pitch_height=i, start_time=0.0, end_time=1.0) for i in range(16)]
        
        with self.assertRaises(ValueError) as context:
            render_midi(notes, self.edo_12, self.output_path)
            
        self.assertIn("MPE polyphony limit exceeded", str(context.exception))
if __name__ == "__main__":
    unittest.main()
