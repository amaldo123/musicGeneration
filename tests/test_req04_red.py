"""test_req04_red.py — Red defect proofs for REQ-04.

Proves defects F1, F2 (boundary level & target curve index), F4, F5, F7.
Every test is decorated with @unittest.expectedFailure, asserting the desired
correct behavior so that each test currently fails on the observed defect.
"""

from __future__ import annotations

import unittest
from math import isclose

from aimusic.core.config import DecodeConfig, PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.decode import decode_path_to_score
from aimusic.planning.graph import _rescore_retained_edges_with_windows, build_sparse_graph
from aimusic.planning.plans import (
    MethodARunConfig,
    PlanningSection,
    SectionTimeline,
    run_method_a,
)
from aimusic.scoring.priors import NullPrior
from tests.req04_support.fixtures import RISE_16, TWO_SECTION_16, run_fixture
from tests.req04_support.report import ReportLogger



class TestReq04Defects(unittest.TestCase):
    """Proves known architectural defects in REQ-04 candidate and planning code."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs = DEFAULT_VOCABULARIES
        self.addCleanup(self.logger.finish)

    def test_F1_rescored_weights_depend_on_tension_weight(self) -> None:
        """F1: Rescoring retained graph edges must incorporate tension weights into edge log_weights."""
        # Run a baseline fixture to get a realistic planning graph with retained edges
        run_res = run_fixture(
            RISE_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        )
        graph = run_res.graph
        end_layer = graph.layers[-1]
        timeline = SectionTimeline(RISE_16.sections, RISE_16.total_beats)

        # Rescore the exact same retained edge support with tension weight 20.0 vs 0.0
        w_20 = PriorWeights(lambda_target_tension=20.0, lambda_section_style=0.0)
        w_0 = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)

        rescored_20, _ = _rescore_retained_edges_with_windows(
            graph.edges_by_time,
            end_layer=end_layer,
            prior=NullPrior(),
            weights=w_20,
            vocabularies=self.vocabs,
            edo=12,
            timeline=timeline,
        )
        rescored_0, _ = _rescore_retained_edges_with_windows(
            graph.edges_by_time,
            end_layer=end_layer,
            prior=NullPrior(),
            weights=w_0,
            vocabularies=self.vocabs,
            edo=12,
            timeline=timeline,
        )

        weights_20 = [edge.log_weight for layer in rescored_20 for edge in layer]
        weights_0 = [edge.log_weight for layer in rescored_0 for edge in layer]
        max_diff = max(abs(a - b) for a, b in zip(weights_20, weights_0)) if weights_20 else 0.0

        self.logger.case("test_F1_rescored_weights_depend_on_tension_weight", lambda_high=20.0, lambda_low=0.0)
        self.logger.metric("max_edge_weight_diff", max_diff, expected="> 0.0")
        self.logger.gate(
            "F1",
            passed=max_diff > 0.0,
            detail=f"rescored edge weights max_diff={max_diff:.6f} expected > 0 (defect F1: tension term missing from rescored edges)",
        )
        self.logger.finish()

        # Desired: Edge weights after rescoring must reflect lambda_target_tension
        self.assertNotEqual(weights_20, weights_0)

    def test_F2_section_start_state_carries_section_boundary_level(self) -> None:
        """F2: The arrival state at a section start beat must carry that section's boundary level."""
        # Case A: Section A [0,8) level 2, Section B [8,16) level 3
        sec_23 = (
            PlanningSection(name="A", start_time=0, end_time=8, boundary_level=2, target_tension_arc=(0.2, 0.4)),
            PlanningSection(name="B", start_time=8, end_time=16, boundary_level=3, target_tension_arc=(0.4, 0.2)),
        )
        # Case B: Section A [0,8) level 3, Section B [8,16) level 2
        sec_32 = (
            PlanningSection(name="A", start_time=0, end_time=8, boundary_level=3, target_tension_arc=(0.2, 0.4)),
            PlanningSection(name="B", start_time=8, end_time=16, boundary_level=2, target_tension_arc=(0.4, 0.2)),
        )

        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        sb = SBConfig(horizon_t=16, k_max=16, d_max=4)
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=2.0)

        res_23, _ = run_method_a(
            MethodARunConfig(total_beats=16, seed=11, sections=sec_23, style_config=style, sb_config=sb, prior_weights=weights),
            key=RNGKey(seed=11),
        )
        res_32, _ = run_method_a(
            MethodARunConfig(total_beats=16, seed=11, sections=sec_32, style_config=style, sb_config=sb, prior_weights=weights),
            key=RNGKey(seed=11),
        )

        lvl_23_at_8 = res_23.path[8].boundary_lvl
        lvl_32_at_8 = res_32.path[8].boundary_lvl

        self.logger.case("test_F2_section_start_state_carries_section_boundary_level")
        self.logger.metric("path_23[8].boundary_lvl", lvl_23_at_8, expected=3)
        self.logger.metric("path_32[8].boundary_lvl", lvl_32_at_8, expected=2)
        passed = (lvl_23_at_8 == 3 and lvl_32_at_8 == 2)
        self.logger.gate(
            "F2_boundary",
            passed=passed,
            detail=f"path[8].boundary_lvl={lvl_23_at_8} expected=3 (for levels 2->3); path[8].boundary_lvl={lvl_32_at_8} expected=2 (for levels 3->2)",
        )
        self.logger.beat_table(res_23.path, sec_23, self.vocabs, 12, limit=10)
        self.logger.finish()

        # Desired: arrival state at beat 8 belongs to Section B and has Section B's boundary level
        self.assertEqual(lvl_23_at_8, 3)
        self.assertEqual(lvl_32_at_8, 2)

    def test_F2_target_index_matches_evaluation_curve(self) -> None:
        """F2: The target used for arrival beat k in transition diagnostics must match target curve index k."""
        run_res = run_fixture(
            TWO_SECTION_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.0),
        )
        diags = run_res.diagnostics.transition_diagnostics
        target_curve_map = dict(run_res.target_curve)

        # Transition 0 is transition from beat 0 to arrival beat 1
        d0_target = diags[0].target_tension
        target_at_beat_1 = target_curve_map.get(1.0, 0.0)

        self.logger.case("test_F2_target_index_matches_evaluation_curve")
        self.logger.metric("diag_0_target_tension", d0_target, expected_at_beat_1=target_at_beat_1)
        passed = isclose(d0_target, target_at_beat_1, abs_tol=1e-5)
        self.logger.gate(
            "F2_target_index",
            passed=passed,
            detail=f"transition[0] target_tension={d0_target:.4f} expected={target_at_beat_1:.4f} (arrival beat k=1 target tension)",
        )
        self.logger.finish()

        # Desired: target tension for edge arriving at beat 1 should equal target tension at beat 1
        self.assertAlmostEqual(d0_target, target_at_beat_1, places=4)

    def test_F4_target_density_changes_decoded_note_count(self) -> None:
        """F4: Distinct section target_density values must change the decoded note event count."""
        run_res = run_fixture(
            RISE_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        )
        path = run_res.path

        sec_low = (PlanningSection(name="Low", start_time=0, end_time=16, boundary_level=3, target_density=0.2),)
        sec_high = (PlanningSection(name="High", start_time=0, end_time=16, boundary_level=3, target_density=0.9),)

        score_low, _ = decode_path_to_score(
            path,
            key=RNGKey(seed=42),
            decode_config=DecodeConfig(),
            vocabularies=self.vocabs,
            edo=12,
            sections=sec_low,
        )
        score_high, _ = decode_path_to_score(
            path,
            key=RNGKey(seed=42),
            decode_config=DecodeConfig(),
            vocabularies=self.vocabs,
            edo=12,
            sections=sec_high,
        )

        n_low = len(score_low.note_events)
        n_high = len(score_high.note_events)

        self.logger.case("test_F4_target_density_changes_decoded_note_count", density_low=0.2, density_high=0.9)
        self.logger.metric("notes_low_density", n_low)
        self.logger.metric("notes_high_density", n_high)
        self.logger.gate(
            "F4",
            passed=(n_low != n_high),
            detail=f"notes_low={n_low} notes_high={n_high} (defect F4: section target_density ignored during decoding)",
        )
        self.logger.finish()

        # Desired: Higher density target must decode to a different note count than lower density
        self.assertNotEqual(n_low, n_high)


    def test_F5_transition_total_equals_sum_of_parts(self) -> None:
        """F5: TransitionDiagnostic.total_log_weight must equal data - energy - penalties within tolerance 1e-9."""
        run_res = run_fixture(
            TWO_SECTION_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5),
        )
        diags = run_res.diagnostics.transition_diagnostics
        self.assertTrue(len(diags) > 0)

        d0 = diags[0]
        expected_total = float(
            d0.data_logp - d0.gttm_energy - d0.target_tension_penalty - d0.section_style_penalty
        )
        observed_total = float(d0.total_log_weight)
        diff = abs(observed_total - expected_total)

        self.logger.case("test_F5_transition_total_equals_sum_of_parts")
        self.logger.metric("observed_total_log_weight", observed_total)
        self.logger.metric("sum_of_parts", expected_total)
        self.logger.metric("discrepancy", diff)
        self.logger.gate(
            "F5",
            passed=(diff < 1e-9),
            detail=f"diag[0].total_log_weight={observed_total:.6f} sum_of_parts={expected_total:.6f} diff={diff:.6f} (defect F5: mismatch with edge log_weight)",
        )
        self.logger.finish()

        # Desired: total_log_weight must exactly equal sum of components
        self.assertAlmostEqual(observed_total, expected_total, delta=1e-9)

    def test_F7_style_weight_zero_does_not_change_candidate_pool(self) -> None:
        """F7: Changing lambda_section_style must not alter candidate proposal pool when steering is not requested."""
        start_state = BeatState(0, 0, 3, 0, 0, 0, 1, 0)
        end_state = BeatState(0, 0, 3, 0, 0, 3, 1, 0)

        from aimusic.core.core_types import Layer
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=4, states=(end_state,))
        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))
        sb = SBConfig(horizon_t=4, k_max=16, d_max=4)
        sections = (
            PlanningSection(
                name="A",
                start_time=0,
                end_time=4,
                boundary_level=3,
                groove_family="syncopated",
                preferred_roles=("cad", "prep"),
            ),
        )

        # Graph built with style weight 0.0 vs style weight 2.0 (proposal steering not requested)
        w_0 = PriorWeights(lambda_section_style=0.0)
        w_2 = PriorWeights(lambda_section_style=2.0)

        graph_0, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=NullPrior(),
            weights=w_0,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=sections,
        )
        graph_2, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=NullPrior(),
            weights=w_2,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=sections,
        )


        layer_states_0 = [set(layer.states) for layer in graph_0.layers]
        layer_states_2 = [set(layer.states) for layer in graph_2.layers]

        diff_counts = [
            len(s0 ^ s2)
            for s0, s2 in zip(layer_states_0, layer_states_2)
        ]
        total_diff = sum(diff_counts)

        self.logger.case("test_F7_style_weight_zero_does_not_change_candidate_pool")
        self.logger.metric("total_layer_state_diffs", total_diff)
        self.logger.gate(
            "F7",
            passed=(total_diff == 0),
            detail=f"candidate pool differs by {total_diff} states across layers when lambda_section_style changes (defect F7)",
        )
        self.logger.finish()

        # Desired: Candidate pool / layer states must be identical when only scoring weight changes
        self.assertEqual(layer_states_0, layer_states_2)


if __name__ == "__main__":
    unittest.main()
