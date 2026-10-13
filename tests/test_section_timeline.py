"""test_section_timeline.py — Tests for SectionTimeline, plan validation, and time indexing.

Covers REQ-04 Step 2:
  - arrival-state lookup at beats 7, 8, 15, 16 for TWO_SECTION_16; edge_view(7).section is B; edge_view(6).section is A
  - gaps, overlaps, wrong start, wrong end, duplicate names, arc with one point, arc value 1.2, density 1.5 each raise SectionPlanError
  - interpolation: first and last beat of a section equal first and last arc point; monotone arcs stay monotone; a 1-beat section does not divide by zero
  - validate_executable: aligned plan passes; start at beat 6 in 4/4 raises; with snap_to_bars=True it moves to beat 4 and reports change; multi-meter style returns warnings only
  - 19-EDO smoke: timeline behaviour does not depend on EDO
  - log one table per fixture: k, section, offset, target, is_start
"""
from __future__ import annotations

import unittest

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.planning.sections import (
    PlanningSection,
    SectionPlanError,
    SectionTimeline,
    interpolate_arc,
)
from tests.req04_support.fixtures import (
    EDO19_SMOKE,
    FALL_16,
    PLATEAU_16,
    RISE_16,
    THREE_SECTION_24,
    TWO_SECTION_16,
)
from tests.req04_support.report import ReportLogger


class TestSectionTimeline(unittest.TestCase):
    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs12 = build_default_vocabularies(edo=12)
        self.vocabs19 = build_default_vocabularies(edo=19)

    def tearDown(self) -> None:
        self.logger.finish()

    def test_arrival_state_and_edge_views_two_section_16(self) -> None:
        """Arrival-state lookup at beats 7, 8, 15, 16 for TWO_SECTION_16."""
        self.logger.case("test_arrival_state_and_edge_views_two_section_16")
        timeline = SectionTimeline(TWO_SECTION_16.sections, TWO_SECTION_16.total_beats)

        # State lookups
        s7 = timeline.at_state(7)
        s8 = timeline.at_state(8)
        s15 = timeline.at_state(15)
        s16 = timeline.at_state(16)

        self.assertEqual(s7.name, "A")
        self.assertEqual(s8.name, "B")
        self.assertEqual(s15.name, "B")
        self.assertEqual(s16.name, "B")

        # Edge views: edge t scores arrival state k = t + 1
        ev6 = timeline.edge_view(6)  # arrival state k=7 (in section A)
        ev7 = timeline.edge_view(7)  # arrival state k=8 (in section B)
        ev15 = timeline.edge_view(15)  # arrival state k=16 (in section B)

        self.assertEqual(ev6.section.name, "A")
        self.assertEqual(ev6.arrival_time, 7)
        self.assertFalse(ev6.is_section_start)

        self.assertEqual(ev7.section.name, "B")
        self.assertEqual(ev7.arrival_time, 8)
        self.assertTrue(ev7.is_section_start)

        self.assertEqual(ev15.section.name, "B")
        self.assertEqual(ev15.arrival_time, 16)
        self.assertFalse(ev15.is_section_start)

        self.logger.metric("edge_view_6_section", ev6.section.name)
        self.logger.metric("edge_view_7_section", ev7.section.name)
        self.logger.gate("D1_time_index", True, "edge_view(6) arrives at A; edge_view(7) arrives at B (section start)")

    def test_plan_validation_errors(self) -> None:
        """Gaps, overlaps, wrong start, wrong end, duplicate names, invalid arcs/densities raise SectionPlanError."""
        self.logger.case("test_plan_validation_errors")

        # 1. Gap
        with self.assertRaises(SectionPlanError) as ctx:
            SectionTimeline(
                (
                    PlanningSection("A", 0, 4, 2),
                    PlanningSection("B", 5, 8, 3),
                ),
                total_beats=8,
            )
        self.assertIn("gap", str(ctx.exception).lower())
        self.assertIn("B", str(ctx.exception))

        # 2. Overlap
        with self.assertRaises(SectionPlanError) as ctx:
            SectionTimeline(
                (
                    PlanningSection("A", 0, 5, 2),
                    PlanningSection("B", 4, 8, 3),
                ),
                total_beats=8,
            )
        self.assertIn("overlap", str(ctx.exception).lower())
        self.assertIn("B", str(ctx.exception))

        # 3. Wrong start
        with self.assertRaises(SectionPlanError) as ctx:
            SectionTimeline(
                (PlanningSection("A", 1, 8, 2),),
                total_beats=8,
            )
        self.assertIn("start_time", str(ctx.exception).lower())
        self.assertIn("A", str(ctx.exception))

        # 4. Wrong end
        with self.assertRaises(SectionPlanError) as ctx:
            SectionTimeline(
                (PlanningSection("A", 0, 8, 2),),
                total_beats=10,
            )
        self.assertIn("end_time", str(ctx.exception).lower())
        self.assertIn("A", str(ctx.exception))

        # 5. Duplicate names
        with self.assertRaises(SectionPlanError) as ctx:
            SectionTimeline(
                (
                    PlanningSection("A", 0, 4, 2),
                    PlanningSection("A", 4, 8, 3),
                ),
                total_beats=8,
            )
        self.assertIn("duplicate", str(ctx.exception).lower())
        self.assertIn("A", str(ctx.exception))

        # 6. Arc with one point
        with self.assertRaises(SectionPlanError) as ctx:
            PlanningSection("A", 0, 4, 2, target_tension_arc=(0.5,))
        self.assertIn("target_tension_arc", str(ctx.exception).lower())
        self.assertIn("two", str(ctx.exception).lower())

        # 7. Arc value 1.2
        with self.assertRaises(SectionPlanError) as ctx:
            PlanningSection("A", 0, 4, 2, target_tension_arc=(0.2, 1.2))
        self.assertIn("target_tension_arc", str(ctx.exception).lower())
        self.assertIn("[0, 1]", str(ctx.exception))

        # 8. Density 1.5
        with self.assertRaises(SectionPlanError) as ctx:
            PlanningSection("A", 0, 4, 2, target_density=1.5)
        self.assertIn("target_density", str(ctx.exception).lower())
        self.assertIn("[0, 1]", str(ctx.exception))

        self.logger.gate("plan_validation", True, "All structural plan violations rejected with descriptive SectionPlanError")

    def test_interpolation_properties(self) -> None:
        """First/last beat equal first/last arc point; monotone stays monotone; 1-beat section doesn't divide by zero."""
        self.logger.case("test_interpolation_properties")

        # First and last beat of a section equal first and last arc point
        arc = (0.2, 0.8)
        self.assertAlmostEqual(interpolate_arc(arc, 0, 4), 0.2, places=6)
        self.assertAlmostEqual(interpolate_arc(arc, 3, 4), 0.8, places=6)

        # Multi-segment arc endpoint equality
        arc3 = (0.1, 0.6, 0.9)
        self.assertAlmostEqual(interpolate_arc(arc3, 0, 10), 0.1, places=6)
        self.assertAlmostEqual(interpolate_arc(arc3, 9, 10), 0.9, places=6)

        # Monotone arcs stay monotone
        vals = [interpolate_arc(arc3, offset, 10) for offset in range(10)]
        for i in range(len(vals) - 1):
            self.assertLessEqual(vals[i], vals[i + 1])

        # 1-beat section does not divide by zero
        val_1beat = interpolate_arc((0.35, 0.75), 0, 1)
        self.assertAlmostEqual(val_1beat, 0.35, places=6)

        self.logger.metric("interp_first", interpolate_arc(arc, 0, 4))
        self.logger.metric("interp_last", interpolate_arc(arc, 3, 4))
        self.logger.metric("interp_1beat", val_1beat)
        self.logger.gate("interpolation", True, "Endpoints matched, monotonicity preserved, length=1 division by zero avoided")

    def test_validate_executable_rules(self) -> None:
        """Aligned passes; misaligned in 4/4 raises; snap_to_bars moves to beat 4; multi-meter warns only."""
        self.logger.case("test_validate_executable_rules")

        style_44 = StyleConfig(allowed_meters=("4/4",))

        # Aligned plan passes
        tl_aligned = SectionTimeline(
            (
                PlanningSection("A", 0, 8, 2),
                PlanningSection("B", 8, 16, 3),
            ),
            total_beats=16,
        )
        warnings_aligned = tl_aligned.validate_executable(style_44, self.vocabs12, snap_to_bars=False)
        self.assertEqual(warnings_aligned, [])

        # Start at beat 6 in 4/4 raises
        tl_misaligned = SectionTimeline(
            (
                PlanningSection("A", 0, 6, 2),
                PlanningSection("B", 6, 16, 3),
            ),
            total_beats=16,
        )
        with self.assertRaises(SectionPlanError) as ctx:
            tl_misaligned.validate_executable(style_44, self.vocabs12, snap_to_bars=False)
        self.assertIn("downbeat", str(ctx.exception).lower())
        self.assertIn("B", str(ctx.exception))

        # With snap_to_bars=True it moves to beat 4 and reports the change
        new_tl, changes = tl_misaligned.validate_executable(style_44, self.vocabs12, snap_to_bars=True)
        self.assertEqual(new_tl.sections[0].end_time, 4)
        self.assertEqual(new_tl.sections[1].start_time, 4)
        self.assertTrue(len(changes) > 0)
        self.assertTrue(any("4" in c and "snapped" in c for c in changes))

        # Multi-meter style returns warnings only (does NOT raise)
        style_multi = StyleConfig(allowed_meters=("4/4", "3/4"))
        warnings_multi = tl_misaligned.validate_executable(style_multi, self.vocabs12, snap_to_bars=False)
        self.assertIsInstance(warnings_multi, list)
        self.assertGreater(len(warnings_multi), 0)

        self.logger.metric("snap_changes_count", len(changes))
        self.logger.metric("multi_meter_warnings_count", len(warnings_multi))
        self.logger.gate("validate_executable", True, "Aligned passes, misaligned raises or snaps to bar, multi-meter returns warnings")

    def test_edo19_smoke_timeline_independence(self) -> None:
        """19-EDO smoke: timeline behavior does not depend on EDO."""
        self.logger.case("test_edo19_smoke_timeline_independence")

        timeline = SectionTimeline(EDO19_SMOKE.sections, EDO19_SMOKE.total_beats)
        self.assertEqual(timeline.total_beats, 4)
        self.assertEqual(timeline.at_state(0).name, "EDO19")
        self.assertEqual(timeline.at_state(4).name, "EDO19")

        style_19 = StyleConfig(allowed_meters=("4/4",))
        warnings = timeline.validate_executable(style_19, self.vocabs19)
        self.assertEqual(warnings, [])

        # Target tension curve works identically
        curve = timeline.as_curve()
        self.assertEqual(len(curve), 5)
        self.logger.gate("19edo_smoke", True, "SectionTimeline operates consistently with 19-EDO vocabularies")

    def test_log_tables_per_fixture(self) -> None:
        """Log one table per fixture: k, section, offset, target, is_start."""
        self.logger.case("test_log_tables_per_fixture")

        fixtures = [RISE_16, FALL_16, PLATEAU_16, TWO_SECTION_16, THREE_SECTION_24, EDO19_SMOKE]
        for fix in fixtures:
            timeline = SectionTimeline(fix.sections, fix.total_beats)
            headers = ["k", "section", "offset", "target", "is_start"]
            rows = []
            for k in range(fix.total_beats + 1):
                sec = timeline.at_state(k)
                offset = timeline.offset_in_section(k)
                target = timeline.target_tension(k)
                is_start = timeline.is_section_start(k)
                rows.append([
                    str(k),
                    sec.name,
                    str(offset),
                    f"{target:.3f}",
                    str(is_start),
                ])
            self.logger.table(f"Fixture: {fix.name}", headers, rows)

        self.logger.gate("fixture_tables", True, "Emitted per-beat section timeline tables for all 6 fixtures")


if __name__ == "__main__":
    unittest.main()
