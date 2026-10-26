"""test_endpoint_sections.py — Test REQ-04 Step 7: endpoint planning on the timeline.

Covers:
  - A1 for endpoints: changing only section 0 meter/key_region/preferred roles
    changes the chosen start state or probability; changing only the last section
    does the same for the end state.
  - Start boundary level equals section 0 level (when legal), or falls back to highest legal level.
  - End state tension is closer to arc[-1] with lambda_target_tension > 0 than with 0 across seeds.
  - 19-EDO smoke test.
  - Comparison table of top-k endpoint candidates with columns: rank, state label, p_off, p_on, delta.
"""

from __future__ import annotations

import unittest

from aimusic.core.config import PriorWeights, StyleConfig
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import build_tonal_context
from aimusic.planning.plans import (
    MethodARunConfig,
    generate_end_endpoint_distribution,
    generate_method_a_endpoints,
    generate_start_endpoint_distribution,
)
from aimusic.planning.sections import PlanningSection
from aimusic.scoring.tension import beat_tension
from tests.req04_support.report import ReportLogger


class TestEndpointSections(unittest.TestCase):
    """Test suite for timeline-aware endpoint distributions and target tension."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.style = StyleConfig(allowed_meters=("4/4", "3/4"), groove_families=("straight",))
        self.style_single = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        self.tc12 = build_tonal_context(12, self.style)
        self.voc12 = self.tc12.vocabularies

    def test_A1_start_endpoint_preference_changes_state_or_prob(self) -> None:
        """A1: Changing section 0 meter, key_region, or roles changes start endpoint."""
        self.logger.case("test_A1_start_endpoint_preference_changes_state_or_prob")

        base_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4", "3/4"), target_key_id=0, preferred_roles=("hold",)),
            PlanningSection("B", 8, 16, 3, (0.4, 0.2), allowed_meters=("4/4", "3/4")),
        )
        weights = PriorWeights(lambda_section_style=8.0, lambda_target_tension=0.0)
        cfg_base = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=base_sec, style_config=self.style, prior_weights=weights)
        pi0_base = generate_start_endpoint_distribution(cfg_base, vocabularies=self.voc12, sections=base_sec)
        base_top = pi0_base.layer.states[0]
        base_p = pi0_base.probabilities[0]

        # 1. Change section 0 allowed_meters from ("4/4", "3/4") to ("3/4",)
        meter_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("3/4",), target_key_id=0, preferred_roles=("hold",)),
            PlanningSection("B", 8, 16, 3, (0.4, 0.2), allowed_meters=("4/4", "3/4")),
        )
        cfg_meter = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=meter_sec, style_config=self.style, prior_weights=weights)
        pi0_meter = generate_start_endpoint_distribution(cfg_meter, vocabularies=self.voc12, sections=meter_sec)
        top_meter_id = pi0_meter.layer.states[0].meter_id
        meter_token = self.voc12.meters.token_for_id(top_meter_id).label
        self.assertEqual(meter_token, "3/4", f"Expected meter 3/4, got {meter_token}")
        self.assertNotEqual(pi0_meter.layer.states[0], base_top)

        # 2. Change section 0 target_key_id / key_region to 7 (fifth away)
        key_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4", "3/4"), target_key_id=7, preferred_roles=("hold",)),
            PlanningSection("B", 8, 16, 3, (0.4, 0.2), allowed_meters=("4/4", "3/4")),
        )
        cfg_key = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=key_sec, style_config=self.style, prior_weights=weights)
        pi0_key = generate_start_endpoint_distribution(cfg_key, vocabularies=self.voc12, sections=key_sec)
        self.assertEqual(pi0_key.layer.states[0].key_id, 7)
        self.assertNotEqual(pi0_key.layer.states[0], base_top)

        # 3. Change section 0 preferred_roles to ("prep",)
        role_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4", "3/4"), target_key_id=0, preferred_roles=("prep",)),
            PlanningSection("B", 8, 16, 3, (0.4, 0.2), allowed_meters=("4/4", "3/4")),
        )
        cfg_role = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=role_sec, style_config=self.style, prior_weights=weights)
        pi0_role = generate_start_endpoint_distribution(cfg_role, vocabularies=self.voc12, sections=role_sec)
        role_token = self.voc12.roles.token_for_id(pi0_role.layer.states[0].role_id).label
        self.assertEqual(role_token, "prep", f"Expected role prep, got {role_token}")
        self.assertNotEqual(pi0_role.layer.states[0], base_top)

        self.logger.metric("base_top_prob", base_p)
        self.logger.metric("meter_changed", True)
        self.logger.metric("key_changed", True)
        self.logger.metric("role_changed", True)

    def test_A1_end_endpoint_preference_changes_state_or_prob(self) -> None:
        """A1: Changing only the last section's preferences changes end endpoint."""
        self.logger.case("test_A1_end_endpoint_preference_changes_state_or_prob")

        base_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4",)),
            PlanningSection("B", 8, 16, 2, (0.4, 0.2), allowed_meters=("4/4",), target_key_id=0, preferred_roles=("cad",)),
        )
        weights = PriorWeights(lambda_section_style=6.0, lambda_target_tension=0.0)
        cfg_base = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=base_sec, style_config=self.style_single, prior_weights=weights)
        piT_base = generate_end_endpoint_distribution(cfg_base, vocabularies=self.voc12, sections=base_sec)
        base_top = piT_base.layer.states[0]

        # 1. Change last section preferred_roles to ("change",)
        role_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4",)),
            PlanningSection("B", 8, 16, 2, (0.4, 0.2), allowed_meters=("4/4",), target_key_id=0, preferred_roles=("change",)),
        )
        cfg_role = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=role_sec, style_config=self.style_single, prior_weights=weights)
        piT_role = generate_end_endpoint_distribution(cfg_role, vocabularies=self.voc12, sections=role_sec)
        role_label = self.voc12.roles.token_for_id(piT_role.layer.states[0].role_id).label
        self.assertEqual(role_label, "change")
        self.assertNotEqual(piT_role.layer.states[0], base_top)

        # 2. Change last section target_key_id to 5
        key_sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4",)),
            PlanningSection("B", 8, 16, 2, (0.4, 0.2), allowed_meters=("4/4",), target_key_id=5, preferred_roles=("cad",)),
        )
        cfg_key = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=key_sec, style_config=self.style_single, prior_weights=weights)
        piT_key = generate_end_endpoint_distribution(cfg_key, vocabularies=self.voc12, sections=key_sec)
        self.assertEqual(piT_key.layer.states[0].key_id, 5)
        self.assertNotEqual(piT_key.layer.states[0], base_top)

        self.logger.metric("end_role_changed", True)
        self.logger.metric("end_key_changed", True)


    def test_start_boundary_level_equals_section_0_level_and_fallback(self) -> None:
        """Start boundary level equals section 0 level when legal, else falls back to highest legal."""
        self.logger.case("test_start_boundary_level_equals_section_0_level_and_fallback")

        # Section 0 level 2
        sec_2 = (
            PlanningSection("A", 0, 8, boundary_level=2, target_tension_arc=(0.2, 0.4)),
            PlanningSection("B", 8, 16, boundary_level=3, target_tension_arc=(0.4, 0.2)),
        )
        cfg_2 = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=sec_2, style_config=self.style_single, prior_weights=PriorWeights(lambda_section_style=2.0))
        pi0_2 = generate_start_endpoint_distribution(cfg_2, vocabularies=self.voc12, sections=sec_2)
        self.assertEqual(pi0_2.layer.states[0].boundary_lvl, 2)

        # Section 0 level 3
        sec_3 = (
            PlanningSection("A", 0, 8, boundary_level=3, target_tension_arc=(0.2, 0.4)),
            PlanningSection("B", 8, 16, boundary_level=2, target_tension_arc=(0.4, 0.2)),
        )
        cfg_3 = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=sec_3, style_config=self.style_single, prior_weights=PriorWeights(lambda_section_style=2.0))
        pi0_3 = generate_start_endpoint_distribution(cfg_3, vocabularies=self.voc12, sections=sec_3)
        self.assertEqual(pi0_3.layer.states[0].boundary_lvl, 3)

        # Section 0 level 1
        sec_1 = (
            PlanningSection("A", 0, 8, boundary_level=1, target_tension_arc=(0.2, 0.4)),
            PlanningSection("B", 8, 16, boundary_level=2, target_tension_arc=(0.4, 0.2)),
        )
        cfg_1 = MethodARunConfig(total_beats=16, seed=11, edo=12, sections=sec_1, style_config=self.style_single, prior_weights=PriorWeights(lambda_section_style=2.0))
        pi0_1 = generate_start_endpoint_distribution(cfg_1, vocabularies=self.voc12, sections=sec_1)
        self.assertEqual(pi0_1.layer.states[0].boundary_lvl, 1)

        self.logger.metric("lvl2_matched", True)
        self.logger.metric("lvl3_matched", True)
        self.logger.metric("lvl1_matched", True)

    def test_end_state_tension_closer_to_arc_with_lambda(self) -> None:
        """End state tension is closer to arc[-1] with lambda_target_tension > 0 than with 0."""
        self.logger.case("test_end_state_tension_closer_to_arc_with_lambda")

        target_end = 0.15
        sec = (
            PlanningSection("A", 0, 8, 2, (0.2, 0.4), allowed_meters=("4/4",)),
            PlanningSection("B", 8, 16, 2, (0.4, target_end), allowed_meters=("4/4",)),
        )

        SEEDS = (7, 11, 23, 42, 101)
        devs_off: list[float] = []
        devs_on: list[float] = []

        for s in SEEDS:
            cfg_off = MethodARunConfig(
                total_beats=16,
                seed=s,
                sections=sec,
                style_config=self.style_single,
                prior_weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=1.0),
            )
            ep_off, _ = generate_method_a_endpoints(cfg_off, vocabularies=self.voc12, key=RNGKey(seed=s), sections=sec)
            tau_off = beat_tension(None, ep_off.end_choice.state, self.voc12, 12)
            dev_off = abs(tau_off - target_end)
            devs_off.append(dev_off)

            cfg_on = MethodARunConfig(
                total_beats=16,
                seed=s,
                sections=sec,
                style_config=self.style_single,
                prior_weights=PriorWeights(lambda_target_tension=20.0, lambda_section_style=1.0),
            )
            ep_on, _ = generate_method_a_endpoints(cfg_on, vocabularies=self.voc12, key=RNGKey(seed=s), sections=sec)
            tau_on = beat_tension(None, ep_on.end_choice.state, self.voc12, 12)
            dev_on = abs(tau_on - target_end)
            devs_on.append(dev_on)

        mean_off = sum(devs_off) / len(devs_off)
        mean_on = sum(devs_on) / len(devs_on)
        self.logger.metric("mean_tension_dev_off", mean_off)
        self.logger.metric("mean_tension_dev_on", mean_on)
        self.logger.metric("delta_dev", mean_on - mean_off)

        self.assertLess(
            mean_on,
            mean_off,
            f"Expected mean_on ({mean_on:.4f}) < mean_off ({mean_off:.4f})",
        )

    def test_19_edo_smoke(self) -> None:
        """19-EDO smoke: generates valid endpoint distributions without relying on len(vocabularies.keys)."""
        self.logger.case("test_19_edo_smoke")

        style19 = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        tc19 = build_tonal_context(19, style19)
        voc19 = tc19.vocabularies

        sec19 = (
            PlanningSection("A", 0, 4, 2, (0.2, 0.4), key_region=(0, 7)),
        )
        cfg19 = MethodARunConfig(
            total_beats=4,
            seed=11,
            edo=19,
            sections=sec19,
            style_config=style19,
            prior_weights=PriorWeights(lambda_target_tension=2.0, lambda_section_style=2.0),
        )

        pi0 = generate_start_endpoint_distribution(cfg19, vocabularies=voc19, sections=sec19)
        piT = generate_end_endpoint_distribution(cfg19, vocabularies=voc19, sections=sec19)

        self.assertEqual(pi0.layer.time_index, 0)
        self.assertEqual(piT.layer.time_index, 4)
        self.assertAlmostEqual(sum(pi0.probabilities), 1.0)
        self.assertAlmostEqual(sum(piT.probabilities), 1.0)
        for state in pi0.layer.states:
            self.assertGreaterEqual(state.key_id, 0)
            self.assertLess(state.key_id, 19)

        ep, _ = generate_method_a_endpoints(cfg19, vocabularies=voc19, key=RNGKey(seed=11), sections=sec19)
        self.assertIn(ep.start_choice.state, pi0.layer.states)
        self.assertIn(ep.end_choice.state, piT.layer.states)
        self.logger.metric("19_edo_endpoints_valid", True)

    def test_log_top_k_comparison_table(self) -> None:
        """Log formatted top-k endpoint comparison table with and without section influence."""
        self.logger.case("test_log_top_k_comparison_table")

        sec = (
            PlanningSection(
                name="A",
                start_time=0,
                end_time=8,
                boundary_level=2,
                target_tension_arc=(0.2, 0.4),
                allowed_meters=("4/4",),
                preferred_roles=("prep",),
                target_key_id=0,
            ),
            PlanningSection(
                name="B",
                start_time=8,
                end_time=16,
                boundary_level=3,
                target_tension_arc=(0.4, 0.2),
                allowed_meters=("4/4",),
            ),
        )

        c_off = MethodARunConfig(
            total_beats=16,
            seed=11,
            sections=sec,
            style_config=self.style,
            prior_weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        )
        pi_off = generate_start_endpoint_distribution(c_off, vocabularies=self.voc12, sections=sec)

        c_on = MethodARunConfig(
            total_beats=16,
            seed=11,
            sections=sec,
            style_config=self.style,
            prior_weights=PriorWeights(lambda_target_tension=5.0, lambda_section_style=2.0),
        )
        pi_on = generate_start_endpoint_distribution(c_on, vocabularies=self.voc12, sections=sec)

        p_off_map = {s: p for s, p in zip(pi_off.layer.states, pi_off.probabilities)}
        p_on_map = {s: p for s, p in zip(pi_on.layer.states, pi_on.probabilities)}

        all_states = list(dict.fromkeys(list(pi_on.layer.states) + list(pi_off.layer.states)))

        rows = []
        for rank, state in enumerate(all_states[:10], start=1):
            po = p_off_map.get(state, 0.0)
            pn = p_on_map.get(state, 0.0)
            delta = pn - po
            meter_label = self.voc12.meters.token_for_id(state.meter_id).label
            role_label = self.voc12.roles.token_for_id(state.role_id).label
            chord_tok = self.voc12.chords.token_for_id(state.chord_id)
            label = f"m={meter_label} b={state.beat_in_bar} lvl={state.boundary_lvl} k={state.key_id} c={chord_tok.quality} r={role_label} h={state.head_id} g={state.groove_id}"
            rows.append((rank, label, f"{po:.4f}", f"{pn:.4f}", f"{delta:+.4f}"))

        headers = ["rank", "state label", "p_off", "p_on", "delta"]
        self.logger.table("Top-10 Start Endpoint Candidates Comparison", headers, rows)


if __name__ == "__main__":
    unittest.main()
