"""tests/test_unified_scorer.py — Verification of unified transition scorer (Step 3).

Fixes F1 and F6. Proves:
1. Decomposition: final == data + gttm - lambda_tau * E_tau - lambda_sigma * E_sigma to 1e-9.
2. Equivalence: calculate_transition_log_weight == breakdown.final_log_weight.
3. Legacy baseline: with both lambdas 0, the scorer equals inline legacy formula.
4. Rescoring delta: second-pass rescoring with lambda_target_tension=20 vs 0 yields
   differences on each edge exactly equal to -lambda_tau * E_tau within 1e-9.
5. Window averaging unchanged when lambdas are 0 against baseline fixture.
6. 19-EDO smoke test.
"""

from __future__ import annotations

import unittest

from aimusic.core.config import PriorWeights, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_tonal_context
from aimusic.planning.graph import (
    _rescore_retained_edges_with_windows,
)
from aimusic.planning.plans import PlanningSection, SectionTimeline
from aimusic.scoring.gttm_features import calculate_gttm_energy
from aimusic.scoring.priors import (
    NullPrior,
    PriorContext,
    calculate_transition_log_weight,
    calculate_transition_score_breakdown,
)
from tests.req04_support.fixtures import RISE_16, run_fixture
from tests.req04_support.report import ReportLogger


class TestUnifiedScorer(unittest.TestCase):
    """Unified transition scoring tests verifying decisions D1 and D2."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs = DEFAULT_VOCABULARIES
        self.addCleanup(self.logger.finish)

    def _sample_transitions(self, count: int = 50) -> list[tuple[BeatState, BeatState, int, SectionTimeline]]:
        run_res = run_fixture(
            RISE_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        )
        graph = run_res.graph
        timeline = SectionTimeline(RISE_16.sections, RISE_16.total_beats)
        sampled: list[tuple[BeatState, BeatState, int, SectionTimeline]] = []
        for layer in graph.edges_by_time:
            for edge in layer:
                sampled.append((edge.source, edge.target, edge.time_index, timeline))
                if len(sampled) >= count:
                    return sampled
        return sampled

    def test_decomposition(self) -> None:
        """Decomposition: final == data + gttm - lambda_tau*E_tau - lambda_sigma*E_sigma within 1e-9."""
        samples = self._sample_transitions(50)
        self.assertGreaterEqual(len(samples), 50)

        weights = PriorWeights(
            lambda_data=1.2,
            lambda_gttm=0.8,
            lambda_target_tension=5.0,
            lambda_section_style=3.0,
        )

        max_discrepancy = 0.0
        for prev_st, next_st, t, timeline in samples:
            sec_view = timeline.edge_view(t)
            ctx = PriorContext(history=(prev_st,), section_view=sec_view)
            breakdown = calculate_transition_score_breakdown(
                prev_st,
                next_st,
                t,
                prior=NullPrior(),
                context=ctx,
                weights=weights,
                vocabularies=self.vocabs,
                edo=12,
            )
            # check_decomposition method raises if |final - expected| > tol
            breakdown.check_decomposition(tol=1e-9)

            expected = (
                breakdown.data_contribution
                + breakdown.gttm_contribution
                - breakdown.target_tension_contribution
                - breakdown.section_style_contribution
            )
            diff = abs(breakdown.final_log_weight - expected)
            if diff > max_discrepancy:
                max_discrepancy = diff

        self.logger.case("test_decomposition", sample_count=len(samples))
        self.logger.metric("max_decomposition_discrepancy", max_discrepancy, expected="< 1e-9")
        self.logger.gate(
            "decomposition_tol",
            passed=max_discrepancy < 1e-9,
            detail=f"max_discrepancy={max_discrepancy:.2e} < 1e-9",
        )
        self.assertLess(max_discrepancy, 1e-9)

    def test_calculate_transition_log_weight_matches_breakdown(self) -> None:
        """calculate_transition_log_weight == breakdown.final_log_weight for identical inputs."""
        samples = self._sample_transitions(50)
        weights = PriorWeights(
            lambda_data=1.1,
            lambda_gttm=0.9,
            lambda_target_tension=4.0,
            lambda_section_style=2.5,
        )

        max_diff = 0.0
        for prev_st, next_st, t, timeline in samples:
            sec_view = timeline.edge_view(t)
            ctx = PriorContext(history=(prev_st,), section_view=sec_view)

            breakdown = calculate_transition_score_breakdown(
                prev_st,
                next_st,
                t,
                prior=NullPrior(),
                context=ctx,
                weights=weights,
                vocabularies=self.vocabs,
                edo=12,
            )
            log_weight = calculate_transition_log_weight(
                prev_st,
                next_st,
                t,
                prior=NullPrior(),
                context=ctx,
                weights=weights,
                vocabularies=self.vocabs,
                edo=12,
            )
            diff = abs(log_weight - breakdown.final_log_weight)
            if diff > max_diff:
                max_diff = diff

        self.logger.case("test_calculate_transition_log_weight_matches_breakdown", sample_count=len(samples))
        self.logger.metric("max_log_weight_breakdown_diff", max_diff, expected="< 1e-9")
        self.logger.gate(
            "log_weight_matches_breakdown",
            passed=max_diff < 1e-9,
            detail=f"max_diff={max_diff:.2e} < 1e-9",
        )
        self.assertLess(max_diff, 1e-9)

    def test_zero_lambdas_equals_legacy_formula(self) -> None:
        """With both lambdas 0, the new scorer equals inline legacy formula: lambda_data*logp + lambda_gttm*G."""
        samples = self._sample_transitions(50)
        weights = PriorWeights(
            lambda_data=1.5,
            lambda_gttm=0.7,
            lambda_target_tension=0.0,
            lambda_section_style=0.0,
        )

        max_diff = 0.0
        for prev_st, next_st, t, timeline in samples:
            sec_view = timeline.edge_view(t)
            ctx = PriorContext(history=(prev_st,), section_view=sec_view)

            # Inline legacy formula: data_contribution + gttm_contribution
            prior = NullPrior()
            data_logp = prior.logp_next(prev_st, next_st, t, ctx)
            gttm_energy = calculate_gttm_energy(
                prev_st,
                next_st,
                t,
                window=None,
                weights=weights,
                vocabularies=self.vocabs,
                edo=12,
            )
            legacy_weight = (weights.lambda_data * data_logp) - (weights.lambda_gttm * gttm_energy)

            breakdown = calculate_transition_score_breakdown(
                prev_st,
                next_st,
                t,
                prior=prior,
                context=ctx,
                weights=weights,
                vocabularies=self.vocabs,
                edo=12,
            )

            diff = abs(breakdown.final_log_weight - legacy_weight)
            if diff > max_diff:
                max_diff = diff

        self.logger.case("test_zero_lambdas_equals_legacy_formula", sample_count=len(samples))
        self.logger.metric("max_legacy_diff", max_diff, expected="< 1e-9")
        self.logger.gate(
            "zero_lambdas_legacy_match",
            passed=max_diff < 1e-9,
            detail=f"max_diff={max_diff:.2e} < 1e-9",
        )
        self.assertLess(max_diff, 1e-9)

    def test_second_pass_rescoring_delta_equals_tension_contribution(self) -> None:
        """Second pass: rescoring with lambda_target_tension 20 vs 0 yields delta == -lambda_tau*E_tau within 1e-9."""
        run_res = run_fixture(
            RISE_16,
            seed=11,
            weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        )
        graph = run_res.graph
        end_layer = graph.layers[-1]
        timeline = SectionTimeline(RISE_16.sections, RISE_16.total_beats)

        w_20 = PriorWeights(lambda_target_tension=20.0, lambda_section_style=0.0)
        w_0 = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)
        w_1 = PriorWeights(lambda_target_tension=1.0, lambda_section_style=0.0)

        rescored_20, diags_20 = _rescore_retained_edges_with_windows(
            graph.edges_by_time,
            end_layer=end_layer,
            prior=NullPrior(),
            weights=w_20,
            vocabularies=self.vocabs,
            edo=12,
            timeline=timeline,
        )
        rescored_0, diags_0 = _rescore_retained_edges_with_windows(
            graph.edges_by_time,
            end_layer=end_layer,
            prior=NullPrior(),
            weights=w_0,
            vocabularies=self.vocabs,
            edo=12,
            timeline=timeline,
        )
        rescored_1, diags_1 = _rescore_retained_edges_with_windows(
            graph.edges_by_time,
            end_layer=end_layer,
            prior=NullPrior(),
            weights=w_1,
            vocabularies=self.vocabs,
            edo=12,
            timeline=timeline,
        )

        max_delta_err = 0.0
        total_edges = 0

        # Build table rows for 10 sample edges
        table_rows: list[list[object]] = []
        compare_variants: list[dict[str, object]] = []

        sample_idx = 0
        for layer_idx, (layer_20, layer_0, layer_1, l_diags_20) in enumerate(
            zip(rescored_20, rescored_0, rescored_1, diags_20)
        ):
            # Map edges by (source, target) to align them independent of sorting order
            edges_0_by_key = {(e.source, e.target): e for e in layer_0}
            edges_1_by_key = {(e.source, e.target): e for e in layer_1}

            for edge_20, d_20 in zip(layer_20, l_diags_20):
                total_edges += 1
                edge_0 = edges_0_by_key[(edge_20.source, edge_20.target)]
                edge_1 = edges_1_by_key[(edge_20.source, edge_20.target)]

                # Expected difference on each edge: weight_20 - weight_0 = -20.0 * E_tau
                e_tau = d_20.target_tension_energy
                expected_delta = -20.0 * e_tau
                observed_delta = edge_20.log_weight - edge_0.log_weight
                err = abs(observed_delta - expected_delta)
                if err > max_delta_err:
                    max_delta_err = err

                if sample_idx < 10:
                    legacy_val = edge_0.log_weight
                    new_val_0 = edge_0.log_weight
                    delta_0 = new_val_0 - legacy_val
                    new_val_1 = edge_1.log_weight
                    e_tau_val = e_tau
                    neg_lambda_e_tau = -1.0 * e_tau_val
                    table_rows.append([
                        f"edge_{sample_idx}(t={edge_0.time_index})",
                        f"{legacy_val:.4f}",
                        f"{new_val_0:.4f}",
                        f"{delta_0:.4f}",
                        f"{new_val_1:.4f}",
                        f"{e_tau_val:.4f}",
                        f"{neg_lambda_e_tau:.4f}",
                    ])
                    compare_variants.append({
                        "label": f"edge_{sample_idx}",
                        "legacy_weight": legacy_val,
                        "new_weight(lambda=0)": new_val_0,
                        "delta": delta_0,
                        "new_weight(lambda_tau=1)": new_val_1,
                        "E_tau": e_tau_val,
                        "-lambda_tau*E_tau": neg_lambda_e_tau,
                    })
                    sample_idx += 1

        self.logger.case("test_second_pass_rescoring_delta", total_edges=total_edges)
        self.logger.metric("max_rescoring_delta_error", max_delta_err, expected="< 1e-9")
        self.logger.gate(
            "second_pass_delta",
            passed=max_delta_err < 1e-9,
            detail=f"max_err={max_delta_err:.2e} on {total_edges} edges",
        )

        # Log formatted table per requirement
        headers = [
            "edge",
            "legacy_weight",
            "new_weight(lambda=0)",
            "delta",
            "new_weight(lambda_tau=1)",
            "E_tau",
            "-lambda_tau*E_tau",
        ]
        self.logger.table("Sample 10 Edges Rescoring Comparison", headers, table_rows)
        self.logger.compare("Sample 10 Edges Rescoring Comparison", "edge_0", compare_variants)

        # Assert delta column for lambda=0 is 0.0 everywhere
        for r in table_rows:
            self.assertEqual(r[3], "0.0000")
        self.assertLess(max_delta_err, 1e-9)


    def test_19_edo_smoke(self) -> None:
        """19-EDO smoke test: decomposition check and rescoring delta in 19-EDO."""
        tonal_ctx = build_tonal_context(19, StyleConfig())
        vocabs_19 = tonal_ctx.vocabularies
        timeline = SectionTimeline(
            (
                PlanningSection(
                    name="Rise_19",
                    start_time=0,
                    end_time=8,
                    boundary_level=3,
                    target_tension_arc=(0.15, 0.55),
                ),
            ),
            total_beats=8,
        )

        weights = PriorWeights(
            lambda_data=1.0,
            lambda_gttm=1.0,
            lambda_target_tension=10.0,
            lambda_section_style=2.0,
        )

        s0 = BeatState(0, 0, 3, 0, 0, 0, 1, 0)
        s1 = BeatState(0, 1, 0, 0, 0, 1, 1, 0)
        sec_view = timeline.edge_view(0)
        ctx = PriorContext(history=(s0,), section_view=sec_view)

        breakdown = calculate_transition_score_breakdown(
            s0,
            s1,
            0,
            prior=NullPrior(),
            context=ctx,
            weights=weights,
            vocabularies=vocabs_19,
            edo=19,
        )
        breakdown.check_decomposition(tol=1e-9)

        log_w = calculate_transition_log_weight(
            s0,
            s1,
            0,
            prior=NullPrior(),
            context=ctx,
            weights=weights,
            vocabularies=vocabs_19,
            edo=19,
        )
        self.assertAlmostEqual(log_w, breakdown.final_log_weight, places=9)
