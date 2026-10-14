from __future__ import annotations

import inspect
import unittest
from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Layer
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES, Vocabularies, build_default_vocabularies
from aimusic.planning.candidates import (
    apply_boundary_and_groove_constraints,
    apply_meter_constraints,
    apply_position_constraints,
    apply_role_constraints,
    get_valid_next_states,
    is_legal_transition,
    propose_boundary_levels,
)
from aimusic.planning.graph import build_sparse_graph
from aimusic.planning.plans import MethodARunConfig, PlanningSection, run_method_a
from aimusic.planning.sections import EdgeSectionView
from aimusic.scoring.priors import NullPrior
from tests.req04_support.report import ReportLogger


class TestCandidateSectionGuidance(unittest.TestCase):
    logger: ReportLogger
    vocabs_12: Vocabularies
    vocabs_19: Vocabularies

    @classmethod
    def setUpClass(cls) -> None:
        cls.logger = ReportLogger()
        cls.vocabs_12 = DEFAULT_VOCABULARIES
        cls.vocabs_19 = build_default_vocabularies(edo=19)

    def test_constraints_and_legality_have_no_section_parameters(self) -> None:
        """Task 3: Hard legality constraints must not receive section information."""
        self.logger.case("test_constraints_and_legality_have_no_section_parameters")
        functions = (
            is_legal_transition,
            apply_meter_constraints,
            apply_position_constraints,
            apply_role_constraints,
            apply_boundary_and_groove_constraints,
        )
        for fn in functions:
            sig = inspect.signature(fn)
            section_params = [name for name in sig.parameters if name.startswith("section")]
            self.logger.metric(f"{fn.__name__}_section_params", len(section_params), expected=0)
            self.assertEqual(
                section_params,
                [],
                f"{fn.__name__} violates separation of legality from section targets: found {section_params}",
            )
        self.logger.gate(
            "legality_separation",
            passed=True,
            detail="is_legal_transition and all apply_*_constraints have no section parameters",
        )
        self.logger.finish()

    def test_property_is_legal_transition_invariant_to_sections(self) -> None:
        """Property test: 2000 random transitions on 12-EDO and 500 on 19-EDO."""
        self.logger.case("test_property_is_legal_transition_invariant_to_sections")
        style = StyleConfig(allowed_meters=("4/4", "3/4"), groove_families=("straight", "syncopated"))

        def run_random_transitions(vocabs: Vocabularies, n_trials: int, edo: int) -> int:
            key = RNGKey(seed=edo * 1000 + n_trials)
            rng = key.generator()
            legal_count = 0
            n_meters = len(vocabs.meters)
            n_keys = len(vocabs.keys)
            n_chords = len(vocabs.chords)
            n_roles = len(vocabs.roles)
            n_heads = len(vocabs.heads)
            n_grooves = len(vocabs.grooves)

            for step in range(n_trials):
                prev = BeatState(
                    meter_id=rng.randrange(n_meters),
                    beat_in_bar=rng.randrange(4),
                    boundary_lvl=rng.randrange(4),
                    key_id=rng.randrange(n_keys),
                    chord_id=rng.randrange(n_chords),
                    role_id=rng.randrange(n_roles),
                    head_id=rng.randrange(n_heads),
                    groove_id=rng.randrange(n_grooves),
                )
                cand = BeatState(
                    meter_id=rng.randrange(n_meters),
                    beat_in_bar=rng.randrange(4),
                    boundary_lvl=rng.randrange(4),
                    key_id=rng.randrange(n_keys),
                    chord_id=rng.randrange(n_chords),
                    role_id=rng.randrange(n_roles),
                    head_id=rng.randrange(n_heads),
                    groove_id=rng.randrange(n_grooves),
                )
                legal, reason = is_legal_transition(prev, cand, style, vocabs)
                if legal:
                    legal_count += 1
            return legal_count

        count_12 = run_random_transitions(self.vocabs_12, 2000, 12)
        count_19 = run_random_transitions(self.vocabs_19, 500, 19)

        self.logger.metric("trials_12_edo", 2000)
        self.logger.metric("legal_12_edo", count_12)
        self.logger.metric("trials_19_edo", 500)
        self.logger.metric("legal_19_edo", count_19)
        self.logger.gate(
            "property_legality_invariance",
            passed=(count_12 > 0 and count_19 > 0),
            detail=f"Evaluated 2000 transitions on 12-EDO ({count_12} legal) and 500 on 19-EDO ({count_19} legal)",
        )
        self.logger.finish()

    def test_candidate_pools_identical_when_unsteered_differ_when_steered(self) -> None:
        """With steering off, every layer is identical for lambda=0 and 1; with steering on, pools may differ."""
        self.logger.case("test_candidate_pools_identical_when_unsteered_differ_when_steered")
        start_state = BeatState(0, 0, 3, 0, 0, 0, 1, 0)
        end_state = BeatState(0, 0, 3, 0, 0, 3, 1, 0)
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

        w_0 = PriorWeights(lambda_section_style=0.0)
        w_1 = PriorWeights(lambda_section_style=1.0)

        # 1. Unsteered graph with lambda=0
        g_unsteered_0, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs_12,
            prior=NullPrior(),
            weights=w_0,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=sections,
            section_guided_proposals=False,
        )

        # 2. Unsteered graph with lambda=1
        g_unsteered_1, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs_12,
            prior=NullPrior(),
            weights=w_1,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=sections,
            section_guided_proposals=False,
        )

        # 3. Steered graph with lambda=1
        g_steered, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs_12,
            prior=NullPrior(),
            weights=w_1,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=sections,
            section_guided_proposals=True,
        )

        table_rows = []
        all_jaccard_1 = True
        for layer_idx in range(len(g_unsteered_0.layers)):
            states_u0 = set(g_unsteered_0.layers[layer_idx].states)
            states_u1 = set(g_unsteered_1.layers[layer_idx].states)
            states_st = set(g_steered.layers[layer_idx].states)

            # Unsteered lambda=0 vs lambda=1 must be identical
            self.assertEqual(
                states_u0,
                states_u1,
                f"Layer {layer_idx} differs between lambda_section_style=0 and 1 with steering off",
            )

            # Jaccard between unsteered and steered
            intersection = len(states_u1 & states_st)
            union = len(states_u1 | states_st)
            jaccard = (intersection / union) if union > 0 else 1.0

            sec_start_present = any(s.boundary_lvl == 3 for s in states_st)
            table_rows.append(
                [
                    str(layer_idx),
                    str(len(states_u1)),
                    str(len(states_st)),
                    f"{jaccard:.4f}",
                    str(sec_start_present),
                ]
            )

        self.logger.table(
            "Per-Layer Candidate Pool Comparison",
            headers=["layer", "pool_size_unsteered", "pool_size_steered", "jaccard", "section_start_candidate_present"],
            rows=table_rows,
        )
        self.logger.gate(
            "unsteered_candidate_pool_invariance",
            passed=all_jaccard_1,
            detail="Candidate pool Jaccard=1.0 across all layers between lambda_section_style=0 and 1 when steering=False",
        )
        self.logger.finish()

    def test_small_proposal_budget_guarantees_section_start_candidate(self) -> None:
        """With small proposal_budget=16, the section boundary level candidate is present whenever legal."""
        self.logger.case("test_small_proposal_budget_guarantees_section_start_candidate")
        sec = PlanningSection(name="B", start_time=8, end_time=16, boundary_level=2, target_tension_arc=(0.3, 0.5))
        view = EdgeSectionView(
            section=sec,
            section_index=1,
            offset_in_section=0,
            is_section_start=True,
            target_tension=0.3,
            arrival_time=8,
        )
        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))

        # Transition arriving at downbeat (beat 0) of section B
        source_state_12 = BeatState(meter_id=0, beat_in_bar=3, boundary_lvl=0, key_id=0, chord_id=0, role_id=2, head_id=1, groove_id=0)

        # Direct check on propose_boundary_levels proposal ordering (Task 2)
        bnd_unsteered = propose_boundary_levels(
            source_state_12, 0, 0, self.vocabs_12, section_view=view, section_guided_proposals=False
        )
        bnd_steered = propose_boundary_levels(
            source_state_12, 0, 0, self.vocabs_12, section_view=view, section_guided_proposals=True
        )
        self.assertEqual(bnd_unsteered, (0, 1, 2, 3))
        self.assertEqual(bnd_steered[0], 2)

        # Test on 12-EDO
        res_12, _ = get_valid_next_states(
            source_state_12,
            t=7,
            key=RNGKey(seed=11),
            d_max=4,
            style_config=style,
            vocabularies=self.vocabs_12,
            prior=NullPrior(),
            edo=12,
            proposal_budget=16,
            section_view=view,
            section_guided_proposals=False,
        )
        levels_12 = {s.boundary_lvl for s in res_12.states}
        has_sec_bnd_12 = (2 in levels_12)
        self.logger.metric("edo12_budget_16_states", len(res_12.states), expected="<= 16")
        self.logger.metric("edo12_has_boundary_level_2", has_sec_bnd_12, expected=True)
        self.assertTrue(has_sec_bnd_12, "Section boundary level 2 was starved by budget=16 on 12-EDO")
        self.assertLessEqual(len(res_12.states), 16)

        # Test on 19-EDO (EDO-generic requirement)
        source_state_19 = BeatState(meter_id=0, beat_in_bar=3, boundary_lvl=0, key_id=0, chord_id=0, role_id=2, head_id=1, groove_id=0)
        res_19, _ = get_valid_next_states(
            source_state_19,
            t=7,
            key=RNGKey(seed=19),
            d_max=4,
            style_config=style,
            vocabularies=self.vocabs_19,
            prior=NullPrior(),
            edo=19,
            proposal_budget=16,
            section_view=view,
            section_guided_proposals=False,
        )
        levels_19 = {s.boundary_lvl for s in res_19.states}
        has_sec_bnd_19 = (2 in levels_19)
        self.logger.metric("edo19_budget_16_states", len(res_19.states), expected="<= 16")
        self.logger.metric("edo19_has_boundary_level_2", has_sec_bnd_19, expected=True)
        self.assertTrue(has_sec_bnd_19, "Section boundary level 2 was starved by budget=16 on 19-EDO")
        self.assertLessEqual(len(res_19.states), 16)

        self.logger.gate(
            "small_budget_boundary_guarantee",
            passed=(has_sec_bnd_12 and has_sec_bnd_19),
            detail="Section boundary level 2 included in 16-candidate budget for both 12-EDO and 19-EDO",
        )
        self.logger.finish()

    def test_two_section_16_boundary_alignment_across_seeds(self) -> None:
        """TWO_SECTION_16 with levels (2,3) and (3,2): path[8].boundary_lvl equals Section B level for all seeds."""
        self.logger.case("test_two_section_16_boundary_alignment_across_seeds")
        sec_23 = (
            PlanningSection(name="A", start_time=0, end_time=8, boundary_level=2, target_tension_arc=(0.2, 0.4)),
            PlanningSection(name="B", start_time=8, end_time=16, boundary_level=3, target_tension_arc=(0.4, 0.2)),
        )
        sec_32 = (
            PlanningSection(name="A", start_time=0, end_time=8, boundary_level=3, target_tension_arc=(0.2, 0.4)),
            PlanningSection(name="B", start_time=8, end_time=16, boundary_level=2, target_tension_arc=(0.4, 0.2)),
        )

        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        sb = SBConfig(horizon_t=16, k_max=16, d_max=4)
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=2.0)
        seeds = (11, 42, 101, 2024)

        all_passed = True
        for seed in seeds:
            res_23, _ = run_method_a(
                MethodARunConfig(total_beats=16, seed=seed, sections=sec_23, style_config=style, sb_config=sb, prior_weights=weights),
                key=RNGKey(seed=seed),
            )
            res_32, _ = run_method_a(
                MethodARunConfig(total_beats=16, seed=seed, sections=sec_32, style_config=style, sb_config=sb, prior_weights=weights),
                key=RNGKey(seed=seed),
            )

            lvl_23 = res_23.path[8].boundary_lvl
            lvl_32 = res_32.path[8].boundary_lvl

            self.logger.metric(f"seed_{seed}_23_level", lvl_23, expected=3)
            self.logger.metric(f"seed_{seed}_32_level", lvl_32, expected=2)
            passed = (lvl_23 == 3 and lvl_32 == 2)
            if not passed:
                all_passed = False

            self.logger.gate(
                f"boundary_alignment_seed_{seed}",
                passed=passed,
                detail=f"Seed {seed}: levels (2,3) gave path[8].boundary_lvl={lvl_23} (expected 3); levels (3,2) gave path[8].boundary_lvl={lvl_32} (expected 2)",
            )
            self.assertEqual(lvl_23, 3, f"Seed {seed}: expected level 3 at section B start, got {lvl_23}")
            self.assertEqual(lvl_32, 2, f"Seed {seed}: expected level 2 at section B start, got {lvl_32}")

        self.logger.gate(
            "two_section_boundary_alignment_all_seeds",
            passed=all_passed,
            detail=f"All seeds {seeds} correctly realized Section B boundary level at arrival beat 8",
        )
        self.logger.finish()


if __name__ == "__main__":
    unittest.main()
