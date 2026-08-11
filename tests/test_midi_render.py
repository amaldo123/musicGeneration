import math
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

    def test_mpe_realized_pitch_is_cent_accurate(self):
        """Reconstruct sounding pitch from MIDI bytes and compare it with EDO pitch."""
        pitch_heights = tuple(range(-19, 39))
        notes = [
            SymbolicNote(
                pitch_height=pitch_height,
                start_time=float(index),
                end_time=float(index + 1),
            )
            for index, pitch_height in enumerate(pitch_heights)
        ]

        render_midi(notes, self.edo_19, self.output_path)

        bends_by_channel = {}
        realized_pitches = []
        for message in mido.MidiFile(self.output_path).tracks[0]:
            if message.type == "pitchwheel":
                bends_by_channel[message.channel] = message.pitch
            elif message.type == "note_on" and message.velocity > 0:
                bend = bends_by_channel[message.channel]
                realized_pitches.append(
                    message.note
                    + (bend / (8191 if bend >= 0 else 8192))
                    * self.config_19.pitch_bend_range
                )

        self.assertEqual(len(realized_pitches), len(pitch_heights))
        for pitch_height, realized_pitch in zip(pitch_heights, realized_pitches):
            expected_pitch = self.config_19.base_tuning + (pitch_height * 12 / 19)
            cents_error = abs(realized_pitch - expected_pitch) * 100
            with self.subTest(pitch_height=pitch_height):
                self.assertLessEqual(cents_error, 0.3)
    
    def test_mts_rendering_encodes_cent_accurate_frequency_words(self):
        """Decode MTS bytes and verify that representative notes realize 19-EDO."""
        config_mts = EDOConfig(
            n=19, base_tuning=60, pitch_bend_range=48, microtonal_rendering_method=MicrotonalRendering.MTS
        )
        edo_mts = EDO(config_mts)
        pitch_heights = (-19, -7, -1, 0, 1, 7, 18, 19, 31)
        notes = [
            SymbolicNote(
                pitch_height=pitch_height,
                start_time=float(index),
                end_time=float(index + 1),
            )
            for index, pitch_height in enumerate(pitch_heights)
        ]

        render_midi(notes, edo_mts, self.output_path)

        mid = mido.MidiFile(self.output_path)
        all_msgs = [msg for track in mid.tracks for msg in track]
        sysex_events = [msg for msg in all_msgs if msg.type == "sysex"]
        note_on_events = [
            msg for msg in all_msgs if msg.type == "note_on" and msg.velocity > 0
        ]

        self.assertEqual(len(sysex_events), 1)
        data = tuple(sysex_events[0].data)
        self.assertEqual(data[:5], (0x7E, 0x7F, 0x08, 0x01, 0x00))
        self.assertEqual(len(data), 5 + 16 + (128 * 3) + 1)
        self.assertTrue(all(0 <= value <= 0x7F for value in data))

        checksum = 0
        for value in data:
            checksum ^= value
        self.assertEqual(checksum, 0)

        self.assertEqual(
            [message.note for message in note_on_events],
            [60 + pitch_height for pitch_height in pitch_heights],
        )
        entries_offset = 5 + 16
        for pitch_height, message in zip(pitch_heights, note_on_events):
            entry_offset = entries_offset + (message.note * 3)
            semitone, fraction_msb, fraction_lsb = data[entry_offset:entry_offset + 3]
            fraction = (fraction_msb << 7) | fraction_lsb
            realized_midi_pitch = semitone + fraction / 16384.0
            expected_midi_pitch = 60 + pitch_height * (12.0 / 19)
            realized_frequency = 440.0 * 2.0 ** ((realized_midi_pitch - 69) / 12.0)
            expected_frequency = 440.0 * 2.0 ** ((expected_midi_pitch - 69) / 12.0)
            cents_error = abs(
                1200.0 * math.log2(realized_frequency / expected_frequency)
            )
            with self.subTest(pitch_height=pitch_height):
                self.assertLessEqual(cents_error, 0.01)

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

    def test_score_render_validates_tracks_channels_tempo_programs_and_note_count(self):
        score = Score(
            note_events=(
                NoteEvent(0, 480, 36, 0.8, track="bass"),
                NoteEvent(0, 480, 48, 0.7, track="comping"),
                NoteEvent(0, 480, 52, 0.7, track="comping"),
                NoteEvent(0, 240, 60, 0.9, track="lead"),
                NoteEvent(0, 120, 36, 1.0, track="drums"),
            ),
            ticks_per_beat=480,
            tempo_bpm=96.0,
        )

        render_midi(score, self.edo_12, self.output_path)
        midi_file = mido.MidiFile(self.output_path)

        self.assertEqual(len(midi_file.tracks), 5)
        names = [
            next(message.name for message in track if message.type == "track_name")
            for track in midi_file.tracks
        ]
        self.assertEqual(names, ["Conductor", "bass", "comping", "lead", "drums"])

        tempo = next(
            message.tempo
            for message in midi_file.tracks[0]
            if message.type == "set_tempo"
        )
        self.assertEqual(tempo, mido.bpm2tempo(score.tempo_bpm))

        expected_programs = {"bass": 33, "comping": 4, "lead": 81}
        melodic_channels = set()
        midi_note_count = 0
        for track_name, track in zip(names[1:], midi_file.tracks[1:]):
            note_ons = [
                message
                for message in track
                if message.type == "note_on" and message.velocity > 0
            ]
            midi_note_count += len(note_ons)
            channels = {message.channel for message in note_ons}
            self.assertEqual(len(channels), 1)
            channel = next(iter(channels))
            if track_name == "drums":
                self.assertEqual(channel, DEFAULT_DRUM_CHANNEL)
                self.assertFalse(any(message.type == "program_change" for message in track))
            else:
                self.assertNotEqual(channel, DEFAULT_DRUM_CHANNEL)
                self.assertTrue(melodic_channels.isdisjoint(channels))
                melodic_channels.update(channels)
                programs = [
                    message.program
                    for message in track
                    if message.type == "program_change"
                ]
                self.assertEqual(programs, [expected_programs[track_name]])

        self.assertEqual(midi_note_count, len(score.note_events))
        self.assertEqual(summarize_midi(self.output_path).total_notes, len(score.note_events))
if __name__ == "__main__":
    unittest.main()
