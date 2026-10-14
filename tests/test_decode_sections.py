"""tests/test_decode_sections.py — Section-aware decoding tests for Step 9."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Dict, List
import unittest

from aimusic.core.config import DecodeConfig, PriorWeights
from aimusic.core.core_types import BeatState, Score
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_default_vocabularies
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import PlanningSection
from tests.req04_support.fixtures import (
    EDO19_SMOKE,
    RISE_16,
    SEEDS,
    TWO_SECTION_16,
    run_fixture,
)
from tests.req04_support.report import ReportLogger
from tests.test_regression_fixtures import (
    FIXTURE_DIR,
    TestLongHorizonDeterministicFixture,
    _canonical_json_hash,
    _load_fixture,
)


class TestDecodeSections(unittest.TestCase):
    """Step 9: Section density and tension-driven expression in the decoder."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)

    def test_1_no_section_regression(self) -> None:
        """sections=None produces identical Score to omitted sections, matching stored hashes."""
        self.logger.case("no_section_regression")

        # 1. Golden short passage fixture
        short_fix = _load_fixture("short_passage_golden.json")
        golden_score = Score.from_dict(short_fix["score"])
        golden_hash = _canonical_json_hash(golden_score.to_dict())
        self.assertEqual(golden_hash, "c2a15af8ca6e61f0b7d94555c658632825711ad1dd8dea82f9fc79827b399a53")
        self.logger.metric("short_passage_golden_hash", golden_hash)

        # 2. Long horizon deterministic fixture
        lh_fix = _load_fixture("long_horizon_smoke.json")
        lh_runner = TestLongHorizonDeterministicFixture()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_midi = Path(tmp_dir) / "lh.mid"
            render_result = lh_runner._render_fixture(lh_fix, tmp_midi)
            expected_score_hash = "cc1d41eb69c124023d3de4de4be700579019842824f848b38e987a258c2670cc"
            self.assertEqual(render_result["score_sha256"], expected_score_hash)
            self.logger.metric("long_horizon_score_sha256", render_result["score_sha256"])

        # 3. Upstream baseline path seed 11, 4 beats, 12 EDO
        baseline_path_file = FIXTURE_DIR / "req04_baseline.json"
        with baseline_path_file.open(encoding="utf-8") as f:
            baseline_data = json.load(f)
        raw_states = baseline_data["paths"]["seed_11_beats_4_edo_12"]
        states = tuple(
            BeatState(
                meter_id=s[0],
                beat_in_bar=s[1],
                boundary_lvl=s[2],
                key_id=s[3],
                chord_id=s[4],
                role_id=s[5],
                head_id=s[6],
                groove_id=s[7],
            )
            for s in raw_states
        )
        score_omitted, _ = decode_path_to_score(
            states,
            decode_config=DecodeConfig(),
            vocabularies=DEFAULT_VOCABULARIES,
            edo=12,
            key=RNGKey(seed=11),
        )
        score_none, _ = decode_path_to_score(
            states,
            decode_config=DecodeConfig(),
            vocabularies=DEFAULT_VOCABULARIES,
            edo=12,
            sections=None,
            key=RNGKey(seed=11),
        )
        self.assertEqual(score_omitted, score_none)
        score_hash = _canonical_json_hash(score_none.to_dict())
        self.assertEqual(score_hash, "83b7f96ce3e5ee8d1243a0005582ac5e45623cfe347a7b957197ad742bfddaec")
        self.logger.metric("baseline_seed11_score_sha256", score_hash)

    def test_2_density_monotonicity(self) -> None:
        """Mean note count per track strictly increases with target density on RISE_16."""
        self.logger.case("density_monotonicity")
        target_densities = (0.2, 0.5, 0.8)
        tracks = ("bass", "comping", "lead", "drums")

        # Pre-plan RISE_16 paths for the 5 standard seeds
        planned_runs = [
            run_fixture(RISE_16, seed=seed, weights=PriorWeights())
            for seed in SEEDS
        ]
        total_beats = RISE_16.total_beats * len(SEEDS)  # 16 * 5 = 80 beats

        table_rows: List[List[str]] = []

        for deterministic in (True, False):
            mode_label = "deterministic" if deterministic else "stochastic"
            density_means: Dict[float, Dict[str, float]] = {}

            for d in target_densities:
                sec = PlanningSection(
                    name="Rise",
                    start_time=0,
                    end_time=16,
                    boundary_level=3,
                    target_tension_arc=(0.15, 0.55),
                    target_density=d,
                )
                track_totals = {tr: 0 for tr in tracks}

                for seed, run in zip(SEEDS, planned_runs):
                    cfg = DecodeConfig(
                        deterministic_activation=deterministic,
                        subbeats_per_beat=4,
                    )
                    score, _ = decode_path_to_score(
                        run.path,
                        decode_config=cfg,
                        vocabularies=DEFAULT_VOCABULARIES,
                        edo=12,
                        sections=(sec,),
                        key=RNGKey(seed=seed),
                    )
                    counts = score.track_event_counts()
                    for tr in tracks:
                        track_totals[tr] += counts.get(tr, 0)

                means = {tr: track_totals[tr] / total_beats for tr in tracks}
                total_mean = sum(means.values())
                density_means[d] = means

                table_rows.append([
                    f"{d:.1f}",
                    mode_label,
                    f"{means['bass']:.4f}",
                    f"{means['comping']:.4f}",
                    f"{means['lead']:.4f}",
                    f"{means['drums']:.4f}",
                    f"{total_mean:.4f}",
                ])

            # Assert strict monotonicity across target densities (0.2 < 0.5 < 0.8) for all tracks
            for tr in tracks:
                m_low = density_means[0.2][tr]
                m_mid = density_means[0.5][tr]
                m_high = density_means[0.8][tr]
                self.assertLess(
                    m_low,
                    m_mid,
                    f"[{mode_label}] {tr} density 0.2 ({m_low:.4f}) not < 0.5 ({m_mid:.4f})",
                )
                self.assertLess(
                    m_mid,
                    m_high,
                    f"[{mode_label}] {tr} density 0.5 ({m_mid:.4f}) not < 0.8 ({m_high:.4f})",
                )

        headers = ["Target Density", "Activation Mode", "Bass", "Comping", "Lead", "Drums", "Total Mean NPB"]
        self.logger.table("RISE_16 Density Monotonicity Across 5 Seeds", headers, table_rows)

    def test_3_section_contrast(self) -> None:
        """Two-section plan with target densities (0.3, 0.8) yields higher notes/beat in second half."""
        self.logger.case("section_contrast")
        contrast_sections = (
            PlanningSection(
                name="A",
                start_time=0,
                end_time=8,
                boundary_level=2,
                target_tension_arc=(0.15, 0.55),
                target_density=0.3,
            ),
            PlanningSection(
                name="B",
                start_time=8,
                end_time=16,
                boundary_level=3,
                target_tension_arc=(0.55, 0.15),
                target_density=0.8,
            ),
        )

        planned_runs = [
            run_fixture(TWO_SECTION_16, seed=seed, weights=PriorWeights())
            for seed in SEEDS
        ]
        ticks_split = 8 * 480  # beat 8 boundary
        notes_first_half = 0
        notes_second_half = 0

        for seed, run in zip(SEEDS, planned_runs):
            score, _ = decode_path_to_score(
                run.path,
                decode_config=DecodeConfig(),
                vocabularies=DEFAULT_VOCABULARIES,
                edo=12,
                sections=contrast_sections,
                key=RNGKey(seed=seed),
            )
            for ev in score.note_events:
                if ev.ton < ticks_split:
                    notes_first_half += 1
                else:
                    notes_second_half += 1

        total_beats_half = 8 * len(SEEDS)  # 40 beats per half
        mean_first_half = notes_first_half / total_beats_half
        mean_second_half = notes_second_half / total_beats_half

        self.logger.metric("mean_notes_per_beat_section_A", mean_first_half)
        self.logger.metric("mean_notes_per_beat_section_B", mean_second_half)

        contrast_headers = ["Section", "Target Density", "Total Notes", "Mean Notes/Beat"]
        contrast_rows = [
            ["Section A (beats 0-8)", "0.3", str(notes_first_half), f"{mean_first_half:.4f}"],
            ["Section B (beats 8-16)", "0.8", str(notes_second_half), f"{mean_second_half:.4f}"],
        ]
        self.logger.table("TWO_SECTION_16 Section Contrast Across 5 Seeds", contrast_headers, contrast_rows)

        self.assertGreater(
            mean_second_half,
            mean_first_half,
            f"Second half notes/beat ({mean_second_half:.4f}) must exceed first half ({mean_first_half:.4f})",
        )

    def test_4_expression_isolation(self) -> None:
        """target_expression_blend 0 vs 1 alters expression without changing pitches or onsets."""
        self.logger.case("expression_isolation")
        run = run_fixture(RISE_16, seed=11, weights=PriorWeights())
        sections = (
            PlanningSection(
                name="Rise",
                start_time=0,
                end_time=16,
                boundary_level=3,
                target_tension_arc=(0.15, 0.55),
                target_density=0.5,
            ),
        )

        cfg_blend0 = DecodeConfig(target_expression_blend=0.0)
        cfg_blend1 = DecodeConfig(target_expression_blend=1.0)

        score_b0, _ = decode_path_to_score(
            run.path,
            decode_config=cfg_blend0,
            vocabularies=DEFAULT_VOCABULARIES,
            edo=12,
            sections=sections,
            key=RNGKey(seed=42),
        )
        score_b1, _ = decode_path_to_score(
            run.path,
            decode_config=cfg_blend1,
            vocabularies=DEFAULT_VOCABULARIES,
            edo=12,
            sections=sections,
            key=RNGKey(seed=42),
        )

        self.assertEqual(len(score_b0.note_events), len(score_b1.note_events))
        self.assertGreater(len(score_b0.note_events), 0)

        differing_expression_count = 0
        for ev0, ev1 in zip(score_b0.note_events, score_b1.note_events):
            self.assertEqual(ev0.h, ev1.h, "Pitch (h) must not change with expression blend")
            self.assertEqual(ev0.ton, ev1.ton, "Onset tick must not change with expression blend")
            self.assertEqual(ev0.toff - ev0.ton, ev1.toff - ev1.ton, "Duration must not change")
            self.assertEqual(ev0.track, ev1.track, "Track must not change")
            if ev0.e != ev1.e:
                differing_expression_count += 1

        self.logger.metric("total_notes", len(score_b0.note_events))
        self.logger.metric("differing_expression_notes", differing_expression_count)
        self.assertGreater(
            differing_expression_count,
            0,
            "At least some notes must have modified expression between blend=0.0 and blend=1.0",
        )

    def test_5_19_edo_smoke(self) -> None:
        """19-EDO decoded notes have pitches strictly within configured track registers."""
        self.logger.case("19_edo_smoke")
        run = run_fixture(EDO19_SMOKE, seed=11, weights=PriorWeights())
        vocabs_19 = build_default_vocabularies(edo=19)
        cfg = DecodeConfig(target_expression_blend=0.5)

        score, _ = decode_path_to_score(
            run.path,
            decode_config=cfg,
            vocabularies=vocabs_19,
            edo=19,
            sections=EDO19_SMOKE.sections,
            key=RNGKey(seed=11),
        )
        self.assertGreater(len(score.note_events), 0)

        for ev in score.note_events:
            if ev.track == "bass":
                self.assertGreaterEqual(ev.h, cfg.bass_register[0])
                self.assertLessEqual(ev.h, cfg.bass_register[1])
            elif ev.track == "comping":
                self.assertGreaterEqual(ev.h, cfg.comping_register[0])
                self.assertLessEqual(ev.h, cfg.comping_register[1])
            elif ev.track == "lead":
                self.assertGreaterEqual(ev.h, cfg.lead_register[0])
                self.assertLessEqual(ev.h, cfg.lead_register[1])
            elif ev.track == "drums":
                self.assertIn(ev.h, (36, 38, 42))

        self.logger.metric("19edo_total_events", len(score.note_events))
        self.logger.metric("19edo_track_counts", score.track_event_counts())


if __name__ == "__main__":
    unittest.main()
