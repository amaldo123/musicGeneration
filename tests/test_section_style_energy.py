"""test_section_style_energy.py — Tests for component-level section style energy and key region.

Covers REQ-04 Step 5:
  - Each component in isolation: zero when satisfied, positive when violated,
    scales linearly with its weight; setting one weight to 0 removes only that component.
  - Boundary logic at section start versus interior.
  - Start alignment at beat_in_bar 0 versus 3, and soft multi-meter alignment factor.
  - Key region distance in 12-EDO and 19-EDO (equal root gives 0; fifth away gives 0.5 before weighting).
  - A1-style tests with lambda_section_style=1, lambda_target_tension=0, fixed seeds:
    changing ONLY one preference (key_region, groove_family, allowed_meters, preferred_roles, boundary_level)
    changes the path; setting the corresponding component weight to 0 reproduces the baseline path.
  - Log a table of component violation rates along the path with the term off versus on across fixtures and seeds.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List, Tuple

from aimusic.core.config import PriorWeights, SectionStyleWeights, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_tonal_context
from aimusic.planning.plans import PlanningSection
from aimusic.planning.sections import SectionTimeline
from aimusic.scoring.tension import section_style_breakdown, section_style_energy
from tests.req04_support.fixtures import (
    Fixture,
    run_fixture,
)
from tests.req04_support.report import ReportLogger


class TestSectionStyleEnergy(unittest.TestCase):
    """Component-level tests for section style energies, key regions, and path steering."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs = DEFAULT_VOCABULARIES
        # Identify sample tokens for testing
        # Meters: "4/4" (id 0)
        self.meter_4_4_id = self.vocabs.meters.token_for_label("4/4").id
        # Roles: "hold", "prep", "change", "cad"
        self.role_hold_id = self.vocabs.roles.token_for_label("hold").id
        self.role_prep_id = self.vocabs.roles.token_for_label("prep").id
        self.role_cad_id = self.vocabs.roles.token_for_label("cad").id
        # Grooves: family "straight" vs "syncopated"
        straight_tokens = [t for t in self.vocabs.grooves.tokens if t.family == "straight"]
        sync_tokens = [t for t in self.vocabs.grooves.tokens if t.family == "syncopated"]
        self.groove_straight_id = straight_tokens[0].id if straight_tokens else 0
        self.groove_sync_id = sync_tokens[0].id if sync_tokens else 1
        # Keys: root 0 (C) and root 7 (G) in 12-EDO
        keys_c = [t for t in self.vocabs.keys.tokens if t.root_pc == 0]
        keys_g = [t for t in self.vocabs.keys.tokens if t.root_pc == 7]
        self.key_c_id = keys_c[0].id if keys_c else 0
        self.key_g_id = keys_g[0].id if keys_g else 7

    def _make_state(
        self,
        *,
        meter_id: int = 0,
        beat_in_bar: int = 0,
        boundary_lvl: int = 0,
        key_id: int = 0,
        role_id: int = 0,
        groove_id: int = 0,
    ) -> BeatState:
        return BeatState(
            meter_id=meter_id,
            beat_in_bar=beat_in_bar,
            boundary_lvl=boundary_lvl,
            key_id=key_id,
            chord_id=0,
            role_id=role_id,
            head_id=1,
            groove_id=groove_id,
        )

    # =========================================================================
    # 1. Components in isolation: zero when satisfied, positive when violated,
    #    scales linearly with weight, 0 when weight=0
    # =========================================================================

    def test_meter_component_in_isolation(self) -> None:
        """Meter component: 0 when in allowed_meters, 1 when not, scales linearly with meter weight."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=1, allowed_meters=("4/4",))
        satisfied_state = self._make_state(meter_id=self.meter_4_4_id)
        other_meter_token = [t for t in self.vocabs.meters.tokens if t.label != "4/4"][0]
        violated_state = self._make_state(meter_id=other_meter_token.id)

        # Raw breakdown
        bd_sat = section_style_breakdown(None, satisfied_state, sec, self.vocabs, edo=12)
        bd_vio = section_style_breakdown(None, violated_state, sec, self.vocabs, edo=12)
        self.assertEqual(bd_sat["meter"], 0.0)
        self.assertEqual(bd_vio["meter"], 1.0)

        # Isolation energy with only meter weight active
        w_default = SectionStyleWeights(meter=1.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)
        w_scaled = SectionStyleWeights(meter=2.5, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)
        w_zero = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)

        e_sat = section_style_energy(None, satisfied_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_vio = section_style_energy(None, violated_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_scaled = section_style_energy(None, violated_state, sec, self.vocabs, edo=12, style_weights=w_scaled)
        e_zero = section_style_energy(None, violated_state, sec, self.vocabs, edo=12, style_weights=w_zero)

        self.assertEqual(e_sat, 0.0)
        self.assertEqual(e_vio, 1.0)
        self.assertAlmostEqual(e_scaled, 2.5)
        self.assertEqual(e_zero, 0.0)

        self.logger.case("meter_component_isolation", satisfied_energy=e_sat, violated_energy=e_vio, scaled_energy=e_scaled)
        self.logger.gate(
            "meter_isolation",
            (e_sat == 0.0 and e_vio == 1.0 and e_scaled == 2.5 and e_zero == 0.0),
            "Meter component is 0 when satisfied, 1 when violated, scales linearly with weight",
        )

    def test_key_region_component_in_isolation(self) -> None:
        """Key region component: 0 when root in region, min d/(1+d) when outside, scales linearly."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=1, key_region=(0,))
        sat_state = self._make_state(key_id=self.key_c_id)  # root 0
        vio_state = self._make_state(key_id=self.key_g_id)  # root 7 (fifth away: d=1 -> 0.5)

        bd_sat = section_style_breakdown(None, sat_state, sec, self.vocabs, edo=12)
        bd_vio = section_style_breakdown(None, vio_state, sec, self.vocabs, edo=12)
        self.assertEqual(bd_sat["key_region"], 0.0)
        self.assertAlmostEqual(bd_vio["key_region"], 0.5)

        w_default = SectionStyleWeights(meter=0.0, key_region=0.5, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)
        w_scaled = SectionStyleWeights(meter=0.0, key_region=1.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)
        w_zero = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)

        e_sat = section_style_energy(None, sat_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_vio = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_scaled = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_scaled)
        e_zero = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_zero)

        self.assertEqual(e_sat, 0.0)
        self.assertAlmostEqual(e_vio, 0.25)  # 0.5 * 0.5
        self.assertAlmostEqual(e_scaled, 0.5)  # 1.0 * 0.5
        self.assertEqual(e_zero, 0.0)

        self.logger.case("key_region_isolation", e_sat=e_sat, e_vio=e_vio, e_scaled=e_scaled)
        self.logger.gate(
            "key_region_isolation",
            (e_sat == 0.0 and abs(e_vio - 0.25) < 1e-6 and abs(e_scaled - 0.5) < 1e-6 and e_zero == 0.0),
            "Key region component is 0 when satisfied, d/(1+d) outside, scales linearly with weight",
        )

    def test_groove_family_component_in_isolation(self) -> None:
        """Groove family component: 0 when family matches, 1 when differs, scales linearly."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=1, groove_family="straight")
        sat_state = self._make_state(groove_id=self.groove_straight_id)
        vio_state = self._make_state(groove_id=self.groove_sync_id)

        bd_sat = section_style_breakdown(None, sat_state, sec, self.vocabs, edo=12)
        bd_vio = section_style_breakdown(None, vio_state, sec, self.vocabs, edo=12)
        self.assertEqual(bd_sat["groove_family"], 0.0)
        self.assertEqual(bd_vio["groove_family"], 1.0)

        w_default = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.3, role=0.0, boundary=0.0, start_alignment=0.0)
        w_scaled = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.9, role=0.0, boundary=0.0, start_alignment=0.0)
        w_zero = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)

        e_sat = section_style_energy(None, sat_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_vio = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_scaled = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_scaled)
        e_zero = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_zero)

        self.assertEqual(e_sat, 0.0)
        self.assertAlmostEqual(e_vio, 0.3)
        self.assertAlmostEqual(e_scaled, 0.9)
        self.assertEqual(e_zero, 0.0)

        self.logger.case("groove_family_isolation", e_sat=e_sat, e_vio=e_vio, e_scaled=e_scaled)
        self.logger.gate(
            "groove_family_isolation",
            (e_sat == 0.0 and abs(e_vio - 0.3) < 1e-6 and abs(e_scaled - 0.9) < 1e-6 and e_zero == 0.0),
            "Groove family component is 0 when matched, 1 when differs, scales linearly with weight",
        )

    def test_role_component_in_isolation(self) -> None:
        """Role component: 0 when label in preferred_roles, 1 when not, scales linearly."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=1, preferred_roles=("hold", "prep"))
        sat_state = self._make_state(role_id=self.role_hold_id)
        vio_state = self._make_state(role_id=self.role_cad_id)

        bd_sat = section_style_breakdown(None, sat_state, sec, self.vocabs, edo=12)
        bd_vio = section_style_breakdown(None, vio_state, sec, self.vocabs, edo=12)
        self.assertEqual(bd_sat["role"], 0.0)
        self.assertEqual(bd_vio["role"], 1.0)

        w_default = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.2, boundary=0.0, start_alignment=0.0)
        w_scaled = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.6, boundary=0.0, start_alignment=0.0)
        w_zero = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0)

        e_sat = section_style_energy(None, sat_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_vio = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_default)
        e_scaled = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_scaled)
        e_zero = section_style_energy(None, vio_state, sec, self.vocabs, edo=12, style_weights=w_zero)

        self.assertEqual(e_sat, 0.0)
        self.assertAlmostEqual(e_vio, 0.2)
        self.assertAlmostEqual(e_scaled, 0.6)
        self.assertEqual(e_zero, 0.0)

        self.logger.case("role_isolation", e_sat=e_sat, e_vio=e_vio, e_scaled=e_scaled)
        self.logger.gate(
            "role_isolation",
            (e_sat == 0.0 and abs(e_vio - 0.2) < 1e-6 and abs(e_scaled - 0.6) < 1e-6 and e_zero == 0.0),
            "Role component is 0 when preferred, 1 when not, scales linearly with weight",
        )

    # =========================================================================
    # 2. Boundary logic (section start vs interior) & Start alignment
    # =========================================================================

    def test_boundary_component_start_vs_interior(self) -> None:
        """Boundary logic: at section-start arrival state abs(lvl - sec_lvl); in interior 1 if lvl >= sec_lvl >= 2."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=3)

        # At section start (is_section_start=True):
        bd_start_3 = section_style_breakdown(None, self._make_state(boundary_lvl=3), sec, self.vocabs, edo=12, is_section_start=True)
        bd_start_2 = section_style_breakdown(None, self._make_state(boundary_lvl=2), sec, self.vocabs, edo=12, is_section_start=True)
        bd_start_0 = section_style_breakdown(None, self._make_state(boundary_lvl=0), sec, self.vocabs, edo=12, is_section_start=True)

        self.assertEqual(bd_start_3["boundary"], 0.0)  # |3 - 3| = 0
        self.assertEqual(bd_start_2["boundary"], 1.0)  # |2 - 3| = 1
        self.assertEqual(bd_start_0["boundary"], 3.0)  # |0 - 3| = 3

        # In interior (is_section_start=False):
        # A section-grade boundary (>= 3) must not appear inside a section
        bd_int_0 = section_style_breakdown(None, self._make_state(boundary_lvl=0), sec, self.vocabs, edo=12, is_section_start=False)
        bd_int_1 = section_style_breakdown(None, self._make_state(boundary_lvl=1), sec, self.vocabs, edo=12, is_section_start=False)
        bd_int_2 = section_style_breakdown(None, self._make_state(boundary_lvl=2), sec, self.vocabs, edo=12, is_section_start=False)
        bd_int_3 = section_style_breakdown(None, self._make_state(boundary_lvl=3), sec, self.vocabs, edo=12, is_section_start=False)

        self.assertEqual(bd_int_0["boundary"], 0.0)
        self.assertEqual(bd_int_1["boundary"], 0.0)
        self.assertEqual(bd_int_2["boundary"], 0.0)  # lvl 2 < sec_lvl 3 -> 0
        self.assertEqual(bd_int_3["boundary"], 1.0)  # lvl 3 >= sec_lvl 3 >= 2 -> 1.0

        # Section with boundary_level=1 (sub-section grade):
        sec_sub = PlanningSection(name="Sub", start_time=0, end_time=4, boundary_level=1)
        bd_sub_int_2 = section_style_breakdown(None, self._make_state(boundary_lvl=2), sec_sub, self.vocabs, edo=12, is_section_start=False)
        self.assertEqual(bd_sub_int_2["boundary"], 0.0)  # section.boundary_level < 2 -> 0.0

        self.logger.case("boundary_start_vs_interior", start_lvl3=bd_start_3["boundary"], start_lvl0=bd_start_0["boundary"], int_lvl3=bd_int_3["boundary"])
        self.logger.gate(
            "boundary_start_vs_interior",
            (bd_start_3["boundary"] == 0.0 and bd_start_0["boundary"] == 3.0 and bd_int_3["boundary"] == 1.0),
            "Boundary component penalizes deviation at section start and forbids section-grade boundaries inside",
        )

    def test_start_alignment_at_beat_0_vs_3(self) -> None:
        """start_alignment: at section-start arrival state, 1 if beat_in_bar != 0 else 0; 0 in interior."""
        sec = PlanningSection(name="Sec", start_time=0, end_time=4, boundary_level=1)

        # Section start
        bd_start_beat0 = section_style_breakdown(None, self._make_state(beat_in_bar=0), sec, self.vocabs, edo=12, is_section_start=True)
        bd_start_beat3 = section_style_breakdown(None, self._make_state(beat_in_bar=3), sec, self.vocabs, edo=12, is_section_start=True)
        self.assertEqual(bd_start_beat0["start_alignment"], 0.0)
        self.assertEqual(bd_start_beat3["start_alignment"], 1.0)

        # Section interior
        bd_int_beat3 = section_style_breakdown(None, self._make_state(beat_in_bar=3), sec, self.vocabs, edo=12, is_section_start=False)
        self.assertEqual(bd_int_beat3["start_alignment"], 0.0)

        # Multi-meter alignment scaling
        w = SectionStyleWeights(meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=1.0)
        e_single = section_style_energy(
            None, self._make_state(beat_in_bar=3), sec, self.vocabs, edo=12,
            style_weights=w, is_section_start=True, multi_meter=False,
        )
        e_multi = section_style_energy(
            None, self._make_state(beat_in_bar=3), sec, self.vocabs, edo=12,
            style_weights=w, is_section_start=True, multi_meter=True, multi_meter_alignment_factor=3.0,
        )
        self.assertAlmostEqual(e_single, 1.0)
        self.assertAlmostEqual(e_multi, 3.0)

        self.logger.case("start_alignment", e_single=e_single, e_multi=e_multi)
        self.logger.gate(
            "start_alignment",
            (bd_start_beat0["start_alignment"] == 0.0 and bd_start_beat3["start_alignment"] == 1.0 and bd_int_beat3["start_alignment"] == 0.0 and e_multi == 3.0),
            "Start alignment is 0 at downbeat, 1 at offbeat on section start, scaled by multi_meter factor",
        )

    # =========================================================================
    # 3. Key region distance in 12-EDO and 19-EDO
    # =========================================================================

    def test_key_region_distance_12_and_19_edo(self) -> None:
        """Equal root gives 0; fifth away gives 0.5 before weighting in both 12-EDO and 19-EDO."""
        # 12-EDO
        sec_12 = PlanningSection(name="Sec12", start_time=0, end_time=4, boundary_level=1, key_region=(0,))
        bd_12_equal = section_style_breakdown(None, self._make_state(key_id=self.key_c_id), sec_12, self.vocabs, edo=12)
        bd_12_fifth = section_style_breakdown(None, self._make_state(key_id=self.key_g_id), sec_12, self.vocabs, edo=12)
        self.assertEqual(bd_12_equal["key_region"], 0.0)
        self.assertAlmostEqual(bd_12_fifth["key_region"], 0.5)

        # Multi-root region (0, 7)
        sec_12_multi = PlanningSection(name="Sec12Multi", start_time=0, end_time=4, boundary_level=1, key_region=(0, 7))
        bd_12_multi_c = section_style_breakdown(None, self._make_state(key_id=self.key_c_id), sec_12_multi, self.vocabs, edo=12)
        bd_12_multi_g = section_style_breakdown(None, self._make_state(key_id=self.key_g_id), sec_12_multi, self.vocabs, edo=12)
        self.assertEqual(bd_12_multi_c["key_region"], 0.0)
        self.assertEqual(bd_12_multi_g["key_region"], 0.0)

        # 19-EDO
        tonal_19 = build_tonal_context(19)
        vocabs_19 = tonal_19.vocabularies
        keys_19_0 = [t for t in vocabs_19.keys.tokens if t.root_pc == 0][0]
        # In 19-EDO, a fifth is 11 steps: root_pc == 11
        keys_19_11 = [t for t in vocabs_19.keys.tokens if t.root_pc == 11][0]

        sec_19 = PlanningSection(name="Sec19", start_time=0, end_time=4, boundary_level=1, key_region=(0,))
        st_19_0 = BeatState(meter_id=0, beat_in_bar=0, boundary_lvl=0, key_id=keys_19_0.id, chord_id=0, role_id=0, head_id=1, groove_id=0)
        st_19_11 = BeatState(meter_id=0, beat_in_bar=0, boundary_lvl=0, key_id=keys_19_11.id, chord_id=0, role_id=0, head_id=1, groove_id=0)

        bd_19_equal = section_style_breakdown(None, st_19_0, sec_19, vocabs_19, edo=19)
        bd_19_fifth = section_style_breakdown(None, st_19_11, sec_19, vocabs_19, edo=19)
        self.assertEqual(bd_19_equal["key_region"], 0.0)
        self.assertAlmostEqual(bd_19_fifth["key_region"], 0.5)

        self.logger.case(
            "key_region_12_and_19",
            bd_12_equal=bd_12_equal["key_region"],
            bd_12_fifth=bd_12_fifth["key_region"],
            bd_19_equal=bd_19_equal["key_region"],
            bd_19_fifth=bd_19_fifth["key_region"],
        )
        self.logger.gate(
            "key_region_12_and_19_edo",
            (
                bd_12_equal["key_region"] == 0.0
                and abs(bd_12_fifth["key_region"] - 0.5) < 1e-6
                and bd_19_equal["key_region"] == 0.0
                and abs(bd_19_fifth["key_region"] - 0.5) < 1e-6
            ),
            "Key region distance is 0 for matching root and 0.5 for fifth in both 12-EDO and 19-EDO",
        )

    # =========================================================================
    # 4. A1-style tests: changing ONLY one preference changes the path;
    #    with weight=0 reproduces baseline
    # =========================================================================

    def test_a1_preference_steering_and_ablation_groove(self) -> None:
        """Changing groove_family steers the path; setting groove_family weight to 0 reproduces baseline."""
        seed = 11
        w_baseline = PriorWeights(lambda_section_style=0.0)
        w_style_on = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(groove_family=1.0, meter=0.0, key_region=0.0, role=0.0, boundary=0.0, start_alignment=0.0),
        )
        w_ablated = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(groove_family=0.0, meter=0.0, key_region=0.0, role=0.0, boundary=0.0, start_alignment=0.0),
        )

        style_cfg = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))

        fix = Fixture(
            name="GrooveSteered",
            total_beats=16,
            sections=(PlanningSection(name="A", start_time=0, end_time=16, boundary_level=3, groove_family="syncopated"),),
            style_config=style_cfg,
        )
        base_run = run_fixture(fix, seed=seed, weights=w_baseline)
        steered_run = run_fixture(fix, seed=seed, weights=w_style_on)
        ablated_run = run_fixture(fix, seed=seed, weights=w_ablated)

        path_changed = (steered_run.path != base_run.path)
        path_reproduced = (ablated_run.path == base_run.path)

        self.logger.case("groove_steering", path_changed=path_changed, path_reproduced=path_reproduced)
        self.logger.gate(
            "groove_steering_and_ablation",
            (path_changed and path_reproduced),
            "Groove preference steers the path and ablating groove weight reproduces baseline",
        )
        self.assertTrue(path_changed, "Path must change when groove_family is requested with lambda_section_style=1.0")
        self.assertTrue(path_reproduced, "Path must match baseline when groove_family weight is ablated to 0.0")

    def test_a1_preference_steering_and_ablation_role(self) -> None:
        """Changing preferred_roles steers the path; setting role weight to 0 reproduces baseline."""
        seed = 11
        w_baseline = PriorWeights(lambda_section_style=0.0)
        w_style_on = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(role=1.0, meter=0.0, key_region=0.0, groove_family=0.0, boundary=0.0, start_alignment=0.0),
        )
        w_ablated = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(role=0.0, meter=0.0, key_region=0.0, groove_family=0.0, boundary=0.0, start_alignment=0.0),
        )

        fix = Fixture(
            name="RoleSteered",
            total_beats=16,
            sections=(PlanningSection(name="A", start_time=0, end_time=16, boundary_level=3, preferred_roles=("cad",)),),
        )
        base_run = run_fixture(fix, seed=seed, weights=w_baseline)
        steered_run = run_fixture(fix, seed=seed, weights=w_style_on)
        ablated_run = run_fixture(fix, seed=seed, weights=w_ablated)

        path_changed = (steered_run.path != base_run.path)
        path_reproduced = (ablated_run.path == base_run.path)

        self.logger.case("role_steering", path_changed=path_changed, path_reproduced=path_reproduced)
        self.logger.gate(
            "role_steering_and_ablation",
            (path_changed and path_reproduced),
            "Role preference steers the path and ablating role weight reproduces baseline",
        )
        self.assertTrue(path_changed, "Path must change when preferred_roles is requested with lambda_section_style=1.0")
        self.assertTrue(path_reproduced, "Path must match baseline when role weight is ablated to 0.0")

    def test_a1_preference_steering_and_ablation_key_region(self) -> None:
        """Changing key_region steers the path; setting key_region weight to 0 reproduces baseline."""
        seed = 11
        w_baseline = PriorWeights(lambda_section_style=0.0)
        w_style_on = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(key_region=1.0, meter=0.0, role=0.0, groove_family=0.0, boundary=0.0, start_alignment=0.0),
        )
        w_ablated = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(key_region=0.0, meter=0.0, role=0.0, groove_family=0.0, boundary=0.0, start_alignment=0.0),
        )

        fix = Fixture(
            name="KeySteered",
            total_beats=16,
            sections=(PlanningSection(name="A", start_time=0, end_time=16, boundary_level=3, key_region=(6,)),),
        )
        base_run = run_fixture(fix, seed=seed, weights=w_baseline)
        steered_run = run_fixture(fix, seed=seed, weights=w_style_on)
        ablated_run = run_fixture(fix, seed=seed, weights=w_ablated)

        path_changed = (steered_run.path != base_run.path)
        path_reproduced = (ablated_run.path == base_run.path)

        self.logger.case("key_region_steering", path_changed=path_changed, path_reproduced=path_reproduced)
        self.logger.gate(
            "key_region_steering_and_ablation",
            (path_changed and path_reproduced),
            "Key region preference steers the path and ablating key region weight reproduces baseline",
        )
        self.assertTrue(path_changed, "Path must change when key_region is requested with lambda_section_style=1.0")
        self.assertTrue(path_reproduced, "Path must match baseline when key_region weight is ablated to 0.0")

    def test_a1_preference_steering_and_ablation_boundary(self) -> None:
        """Changing boundary_level steers the path; setting boundary weight to 0 reproduces baseline."""
        seed = 11
        w_baseline = PriorWeights(lambda_section_style=0.0)
        w_style_on = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(boundary=1.0, meter=0.0, key_region=0.0, role=0.0, groove_family=0.0, start_alignment=0.0),
        )
        w_ablated = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(boundary=0.0, meter=0.0, key_region=0.0, role=0.0, groove_family=0.0, start_alignment=0.0),
        )

        fix = Fixture(
            name="BoundSteered",
            total_beats=16,
            sections=(
                PlanningSection(name="A", start_time=0, end_time=8, boundary_level=2),
                PlanningSection(name="B", start_time=8, end_time=16, boundary_level=1),
            ),
        )
        base_run = run_fixture(fix, seed=seed, weights=w_baseline)
        steered_run = run_fixture(fix, seed=seed, weights=w_style_on)
        ablated_run = run_fixture(fix, seed=seed, weights=w_ablated)

        path_changed = (steered_run.path != base_run.path)
        path_reproduced = (ablated_run.path == base_run.path)

        self.logger.case("boundary_steering", path_changed=path_changed, path_reproduced=path_reproduced)
        self.logger.gate(
            "boundary_steering_and_ablation",
            (path_changed and path_reproduced),
            "Boundary preference steers the path and ablating boundary weight reproduces baseline",
        )
        self.assertTrue(path_changed, "Path must change when boundary_level is requested with lambda_section_style=1.0")
        self.assertTrue(path_reproduced, "Path must match baseline when boundary weight is ablated to 0.0")

    def test_a1_preference_steering_and_ablation_meter(self) -> None:
        """Changing allowed_meters steers the path; setting meter weight to 0 reproduces baseline."""
        seed = 11
        w_baseline = PriorWeights(lambda_section_style=0.0)
        w_style_on = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(meter=1.0, boundary=0.0, key_region=0.0, role=0.0, groove_family=0.0, start_alignment=0.0),
        )
        w_ablated = PriorWeights(
            lambda_section_style=1.0,
            style_weights=SectionStyleWeights(meter=0.0, boundary=0.0, key_region=0.0, role=0.0, groove_family=0.0, start_alignment=0.0),
        )

        style_cfg = StyleConfig(allowed_meters=("4/4", "3/4"))
        fix = Fixture(
            name="MeterSteered",
            total_beats=12,
            sections=(PlanningSection(name="A", start_time=0, end_time=12, boundary_level=3, allowed_meters=("3/4",)),),
            style_config=style_cfg,
        )
        base_run = run_fixture(fix, seed=seed, weights=w_baseline)
        steered_run = run_fixture(fix, seed=seed, weights=w_style_on)
        ablated_run = run_fixture(fix, seed=seed, weights=w_ablated)

        path_changed = (steered_run.path != base_run.path)
        path_reproduced = (ablated_run.path == base_run.path)

        self.logger.case("meter_steering", path_changed=path_changed, path_reproduced=path_reproduced)
        self.logger.gate(
            "meter_steering_and_ablation",
            (path_changed and path_reproduced),
            "Meter preference steers the path and ablating meter weight reproduces baseline",
        )
        self.assertTrue(path_changed, "Path must change when allowed_meters is requested with lambda_section_style=1.0")
        self.assertTrue(path_reproduced, "Path must match baseline when meter weight is ablated to 0.0")

    # =========================================================================
    # 5. Component violation-rate table along the path
    # =========================================================================

    def test_violation_rate_table_logged(self) -> None:
        """Measure and log component violation rate along the path with term off versus on."""
        style_cfg = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))

        fixture = Fixture(
            name="MultiPrefFixture",
            total_beats=16,
            sections=(
                PlanningSection(
                    name="A",
                    start_time=0,
                    end_time=8,
                    boundary_level=2,
                    groove_family="syncopated",
                    preferred_roles=("prep", "change"),
                    key_region=(7,),
                ),
                PlanningSection(
                    name="B",
                    start_time=8,
                    end_time=16,
                    boundary_level=3,
                    groove_family="straight",
                    preferred_roles=("cad", "hold"),
                    key_region=(0,),
                ),
            ),
            style_config=style_cfg,
        )

        timeline = SectionTimeline(fixture.sections, fixture.total_beats)

        def compute_violation_rates(path: Tuple[BeatState, ...]) -> Dict[str, float]:
            totals: Dict[str, float] = {
                "meter": 0.0,
                "key_region": 0.0,
                "groove_family": 0.0,
                "role": 0.0,
                "boundary": 0.0,
                "start_alignment": 0.0,
            }
            steps = len(path) - 1
            for t in range(steps):
                edge_view = timeline.edge_view(t)
                bd = section_style_breakdown(
                    path[t], path[t + 1], edge_view, self.vocabs, edo=12
                )
                for comp, val in bd.items():
                    totals[comp] += float(val)
            return {comp: totals[comp] / float(steps) for comp in totals}

        seeds = (7, 11, 23)
        rows: List[Dict[str, Any]] = []

        for seed in seeds:
            run_off = run_fixture(fixture, seed=seed, weights=PriorWeights(lambda_section_style=0.0))
            run_on = run_fixture(fixture, seed=seed, weights=PriorWeights(lambda_section_style=2.0))

            rates_off = compute_violation_rates(run_off.path)
            rates_on = compute_violation_rates(run_on.path)

            for comp in ("groove_family", "role", "key_region", "boundary", "start_alignment"):
                rows.append({
                    "seed": seed,
                    "component": comp,
                    "off_rate": rates_off[comp],
                    "on_rate": rates_on[comp],
                    "delta": rates_on[comp] - rates_off[comp],
                })

        # Log table and metrics
        headers = ["seed", "component", "off_rate", "on_rate", "delta"]
        table_rows = [
            [r["seed"], r["component"], f"{r['off_rate']:.4f}", f"{r['on_rate']:.4f}", f"{r['delta']:+.4f}"]
            for r in rows
        ]
        self.logger.table("component_violation_rates", headers, table_rows)

        # Average violation across seeds
        avg_groove_off = sum(r["off_rate"] for r in rows if r["component"] == "groove_family") / len(seeds)
        avg_groove_on = sum(r["on_rate"] for r in rows if r["component"] == "groove_family") / len(seeds)
        avg_role_off = sum(r["off_rate"] for r in rows if r["component"] == "role") / len(seeds)
        avg_role_on = sum(r["on_rate"] for r in rows if r["component"] == "role") / len(seeds)
        avg_key_off = sum(r["off_rate"] for r in rows if r["component"] == "key_region") / len(seeds)
        avg_key_on = sum(r["on_rate"] for r in rows if r["component"] == "key_region") / len(seeds)

        self.logger.metric("avg_groove_violation_off", avg_groove_off)
        self.logger.metric("avg_groove_violation_on", avg_groove_on)
        self.logger.metric("avg_role_violation_off", avg_role_off)
        self.logger.metric("avg_role_violation_on", avg_role_on)
        self.logger.metric("avg_key_violation_off", avg_key_off)
        self.logger.metric("avg_key_violation_on", avg_key_on)

        self.logger.gate(
            "groove_violation_reduced_or_maintained",
            avg_groove_on <= avg_groove_off,
            f"groove off={avg_groove_off:.4f} -> on={avg_groove_on:.4f}",
        )
        self.logger.gate(
            "role_violation_reduced_or_maintained",
            avg_role_on <= avg_role_off,
            f"role off={avg_role_off:.4f} -> on={avg_role_on:.4f}",
        )
        self.logger.gate(
            "key_region_violation_reduced_or_maintained",
            avg_key_on <= avg_key_off,
            f"key_region off={avg_key_off:.4f} -> on={avg_key_on:.4f}",
        )


if __name__ == "__main__":
    unittest.main()
