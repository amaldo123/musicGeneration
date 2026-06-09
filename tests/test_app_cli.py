from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import mido

from aimusic.app import cli
from aimusic.core.core_types import NoteEvent, Score


class TestAppCli(unittest.TestCase):
    def test_export_command_renders_multitrack_score_json(self) -> None:
        score = Score(
            note_events=(
                NoteEvent(ton=0, toff=480, h=36, v=0.7, track="bass"),
                NoteEvent(ton=0, toff=480, h=60, v=0.9, track="lead"),
            ),
            ticks_per_beat=480,
            tempo_bpm=104.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            score_path = os.path.join(tmp_dir, "score.json")
            midi_path = os.path.join(tmp_dir, "score.mid")
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(score.to_dict(), f)

            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "aimusic",
                    "export",
                    score_path,
                    "--out",
                    midi_path,
                ],
            ):
                with redirect_stdout(stdout):
                    cli.main()

            self.assertTrue(os.path.exists(midi_path))
            self.assertIn("Exported multitrack MIDI", stdout.getvalue())

            mid = mido.MidiFile(midi_path)
            track_names = [
                next((msg.name for msg in track if msg.type == "track_name"), None)
                for track in mid.tracks
            ]
            self.assertEqual(track_names, ["Conductor", "bass", "lead"])

    def test_export_command_accepts_track_program_and_drum_overrides(self) -> None:
        score = Score(
            note_events=(
                NoteEvent(ton=0, toff=480, h=60, v=0.9, track="lead"),
                NoteEvent(ton=0, toff=120, h=42, v=0.7, track="kit"),
            ),
            ticks_per_beat=480,
            tempo_bpm=104.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            score_path = os.path.join(tmp_dir, "score.json")
            midi_path = os.path.join(tmp_dir, "score.mid")
            with open(score_path, "w", encoding="utf-8") as f:
                json.dump(score.to_dict(), f)

            with patch(
                "sys.argv",
                [
                    "aimusic",
                    "export",
                    score_path,
                    "--out",
                    midi_path,
                    "--track-program",
                    "lead=88",
                    "--drum-track",
                    "kit",
                ],
            ):
                cli.main()

            mid = mido.MidiFile(midi_path)
            lead_track = next(
                track
                for track in mid.tracks
                if any(msg.type == "track_name" and msg.name == "lead" for msg in track)
            )
            kit_track = next(
                track
                for track in mid.tracks
                if any(msg.type == "track_name" and msg.name == "kit" for msg in track)
            )

            lead_program = next(msg for msg in lead_track if msg.type == "program_change")
            kit_note_on = next(msg for msg in kit_track if msg.type == "note_on" and msg.velocity > 0)
            self.assertEqual(lead_program.program, 88)
            self.assertEqual(kit_note_on.channel, 9)

    def test_generate_command_produces_score_midi_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stdout = io.StringIO()
            with patch(
                "sys.argv",
                [
                    "aimusic",
                    "generate",
                    "--seed",
                    "11",
                    "--beats",
                    "4",
                    "--meter",
                    "4/4",
                    "--groove-family",
                    "straight",
                    "--out",
                    tmp_dir,
                ],
            ):
                with redirect_stdout(stdout):
                    cli.main()

            files = sorted(os.listdir(tmp_dir))
            self.assertEqual(len(files), 3)

            manifest_name = next(name for name in files if name.endswith("_manifest.json"))
            midi_name = next(name for name in files if name.endswith(".mid"))
            score_name = next(name for name in files if name.endswith("_score.json"))

            with open(os.path.join(tmp_dir, manifest_name), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            with open(os.path.join(tmp_dir, score_name), "r", encoding="utf-8") as f:
                score = json.load(f)

            self.assertEqual(manifest["seed"], 11)
            self.assertTrue(manifest["sb_stats"]["converged"])
            self.assertEqual(manifest["config"]["meter"], "4/4")
            self.assertEqual(manifest["config"]["groove_family"], "straight")
            self.assertGreater(score["event_count"], 0)

            mid = mido.MidiFile(os.path.join(tmp_dir, midi_name))
            track_names = [
                next((msg.name for msg in track if msg.type == "track_name"), None)
                for track in mid.tracks
            ]
            self.assertIn("Conductor", track_names)
            self.assertTrue(any(name in ("bass", "comping", "lead", "drums") for name in track_names))
            self.assertIn("Generated multitrack MIDI", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
