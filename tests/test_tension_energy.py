"""test_tension_energy.py — Tests for target tension energy, arrival-beat indexing, and sweep.

Covers REQ-04 Step 4:
  - Energy is 0 at equality, symmetric, grows with |deviation|, halves when sigma doubles squared
    (E(sigma=0.2) == E(sigma=0.1)/4), unit deviation costs 0.5 nat, sigma > 0 validation.
  - target_tension_curve(timeline)[k] equals timeline.target_tension(k) for all k on 3 fixtures.
  - Realized tension curve index convention equals the optimizer's convention:
    the target used for edge t equals curve[t+1].
  - compare_tension_curves returns zero error when realized equals target,
    and non-zero error when shifted by one beat (guards against re-introducing F2).
  - 19-EDO smoke test.
  - Informational sweep on RISE_16 across seeds (7, 11, 23),
    lambda_target_tension in {0, 0.5, 1, 2, 5}, sigma_tension in {0.05, 0.1, 0.2}.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, List

from aimusic.core.config import PriorWeights
from aimusic.planning.sections import SectionTimeline
from aimusic.scoring.tension import (
    compare_tension_curves,
    target_tension_at_time,
    target_tension_curve,
    transition_target_tension_energy,
)
from tests.req04_support.fixtures import (
    EDO19_SMOKE,
    FALL_16,
    RISE_16,
    TWO_SECTION_16,
    run_fixture,
)
from tests.req04_support.report import ReportLogger


class TestTensionEnergy(unittest.TestCase):
    """Test tension deviation energy, interpolation delegation, and curve alignment."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)

    def tearDown(self) -> None:
        self.logger.finish()

    def test_energy_formula_properties(self) -> None:
        """Energy is 0 at equality, symmetric, strictly increasing, scales with 1/(2*sigma^2)."""
        self.logger.case("test_energy_formula_properties")

        # 1. Zero at equality
        self.assertAlmostEqual(transition_target_tension_energy(0.4, 0.4, 0.1), 0.0)
        self.assertAlmostEqual(transition_target_tension_energy(0.0, 0.0, 0.1), 0.0)
        self.assertAlmostEqual(transition_target_tension_energy(0.6, 0.6, 0.2), 0.0)

        # 2. Symmetric
        e_forward = transition_target_tension_energy(0.2, 0.5, 0.1)
        e_reverse = transition_target_tension_energy(0.5, 0.2, 0.1)
        self.assertAlmostEqual(e_forward, e_reverse)

        # 3. Grows with |deviation|
        e_dev1 = transition_target_tension_energy(0.3, 0.4, 0.1)  # dev = 0.1
        e_dev2 = transition_target_tension_energy(0.3, 0.5, 0.1)  # dev = 0.2
        e_dev3 = transition_target_tension_energy(0.3, 0.6, 0.1)  # dev = 0.3
        self.assertLess(e_dev1, e_dev2)
        self.assertLess(e_dev2, e_dev3)

        # 4. Halves when sigma doubles squared: E(sigma=0.2) == E(sigma=0.1) / 4
        e_sig01 = transition_target_tension_energy(0.3, 0.5, 0.1)
        e_sig02 = transition_target_tension_energy(0.3, 0.5, 0.2)
        self.assertAlmostEqual(e_sig02, e_sig01 / 4.0, places=7)

        # 5. Tolerance unit scale: a deviation equal to sigma costs exactly 0.5 nat
        self.assertAlmostEqual(transition_target_tension_energy(0.3, 0.3 + 0.1, 0.1), 0.5)
        self.assertAlmostEqual(transition_target_tension_energy(0.3, 0.3 + 0.05, 0.05), 0.5)
        self.assertAlmostEqual(transition_target_tension_energy(0.5, 0.5 + 0.2, 0.2), 0.5)

        # 6. Validation: sigma must be > 0 and a real number
        with self.assertRaises(ValueError):
            transition_target_tension_energy(0.3, 0.4, 0.0)
        with self.assertRaises(ValueError):
            transition_target_tension_energy(0.3, 0.4, -0.1)
        with self.assertRaises(TypeError):
            transition_target_tension_energy(0.3, 0.4, True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            transition_target_tension_energy(0.3, 0.4, "0.1")  # type: ignore[arg-type]

        self.logger.gate(
            "TENSION_ENERGY_PROPERTIES",
            True,
            "Tension energy verifies zero at equality, symmetry, growth with dev, E(2*sig)=E/4, and 0.5 nat scale",
        )

    def test_target_curve_delegates_to_timeline_and_arrival_convention(self) -> None:
        """target_tension_curve delegates to SectionTimeline; optimizer target matches curve[t+1]."""
        self.logger.case("test_target_curve_delegates_to_timeline_and_arrival_convention")

        fixtures = (RISE_16, FALL_16, TWO_SECTION_16)
        for fix in fixtures:
            timeline = SectionTimeline(fix.sections, fix.total_beats)
            curve_from_timeline = target_tension_curve(timeline)
            curve_from_sections = target_tension_curve(fix.sections)

            # Both signatures produce identical curves
            self.assertEqual(curve_from_timeline, curve_from_sections)
            # Length covers 0..total_beats (total_beats + 1 points)
            self.assertEqual(len(curve_from_timeline), fix.total_beats + 1)

            # For all arrival beats k, curve[k] equals timeline.target_tension(k)
            for k in range(fix.total_beats + 1):
                time_k, val_k = curve_from_timeline[k]
                self.assertEqual(time_k, float(k))
                self.assertAlmostEqual(val_k, timeline.target_tension(k), places=7)

        # Verify optimizer arrival convention: edge t scores arrival state t + 1
        weights = PriorWeights(lambda_target_tension=0.0)
        run = run_fixture(RISE_16, seed=11, weights=weights)
        timeline_rise = SectionTimeline(RISE_16.sections, RISE_16.total_beats)

        self.assertEqual(len(run.target_curve), RISE_16.total_beats + 1)
        self.assertEqual(len(run.diagnostics.transition_diagnostics), RISE_16.total_beats)

        for t, trans_diag in enumerate(run.diagnostics.transition_diagnostics):
            expected_target = run.target_curve[t + 1][1]
            self.assertAlmostEqual(
                trans_diag.target_tension,
                expected_target,
                places=7,
                msg=f"Edge {t} diagnostic target {trans_diag.target_tension} != curve[{t+1}] {expected_target}",
            )
            self.assertAlmostEqual(
                trans_diag.target_tension,
                timeline_rise.target_tension(t + 1),
                places=7,
            )

        self.logger.gate(
            "TENSION_CURVE_DELEGATION_AND_CONVENTION",
            True,
            "target_tension_curve delegates to SectionTimeline, and optimizer target matches curve[t+1]",
        )

    def test_compare_tension_curves_alignment_and_shift_sensitivity(self) -> None:
        """compare_tension_curves returns 0 error when matching, non-zero when shifted by 1 beat."""
        self.logger.case("test_compare_tension_curves_alignment_and_shift_sensitivity")

        target = target_tension_curve(RISE_16.sections)
        # 1. Perfectly aligned realized curve
        identical_realized = list(target)
        rep_exact = compare_tension_curves(target, identical_realized, RISE_16.sections)
        self.assertAlmostEqual(rep_exact.mean_absolute_error, 0.0, places=7)
        self.assertAlmostEqual(rep_exact.max_absolute_error, 0.0, places=7)
        self.assertAlmostEqual(rep_exact.shape_correlation, 1.0, places=7)

        # 2. Shifted by 1 beat (simulates F2 off-by-one bug: evaluating at t instead of t + 1)
        # In RISE_16, tension rises from 0.15 to 0.55 over 16 beats (slope = 0.025/beat)
        shifted_realized = [
            (float(t), target_tension_at_time(RISE_16.sections, max(0, t - 1)))
            for t in range(len(target))
        ]
        rep_shifted = compare_tension_curves(target, shifted_realized, RISE_16.sections)
        self.assertGreater(rep_shifted.mean_absolute_error, 0.01)
        self.assertGreater(rep_shifted.max_absolute_error, 0.02)

        self.logger.gate(
            "COMPARE_TENSION_SENSITIVITY",
            True,
            f"Zero error on exact alignment, MAE={rep_shifted.mean_absolute_error:.4f} when shifted by 1 beat",
        )

    def test_edo19_smoke_tension_energy_and_curves(self) -> None:
        """19-EDO smoke test: tension energy, curves, and planning execute cleanly."""
        self.logger.case("test_edo19_smoke_tension_energy_and_curves")

        weights = PriorWeights(lambda_target_tension=1.0, sigma_tension=0.1, lambda_section_style=0.0)
        run = run_fixture(EDO19_SMOKE, seed=7, weights=weights, edo=19)

        self.assertEqual(len(run.path), EDO19_SMOKE.total_beats + 1)
        self.assertEqual(len(run.diagnostics.transition_diagnostics), EDO19_SMOKE.total_beats)
        self.assertEqual(len(run.realized_curve), EDO19_SMOKE.total_beats + 1)
        self.assertEqual(len(run.target_curve), EDO19_SMOKE.total_beats + 1)

        for t, diag in enumerate(run.diagnostics.transition_diagnostics):
            self.assertAlmostEqual(diag.target_tension, run.target_curve[t + 1][1], places=7)

        rep = compare_tension_curves(run.target_curve, run.realized_curve, EDO19_SMOKE.sections)
        self.assertGreaterEqual(rep.mean_absolute_error, 0.0)

        self.logger.gate(
            "EDO19_TENSION_SMOKE",
            True,
            f"19-EDO completed with MAE={rep.mean_absolute_error:.4f}, max_err={rep.max_absolute_error:.4f}",
        )

    def test_informational_sweep_rise_16(self) -> None:
        """Informational sweep (no pass/fail assertion, logged only) feeding Step 12."""
        self.logger.case("test_informational_sweep_rise_16")

        seeds = (7, 11, 23)
        sigmas = (0.05, 0.1, 0.2)

        # Baseline: lambda = 0.0
        base_runs = [
            run_fixture(
                RISE_16,
                seed=s,
                weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
            )
            for s in seeds
        ]
        base_reps = [
            compare_tension_curves(r.target_curve, r.realized_curve, RISE_16.sections)
            for r in base_runs
        ]
        base_mae = sum(rep.mean_absolute_error for rep in base_reps) / len(base_reps)
        base_max_err = sum(rep.max_absolute_error for rep in base_reps) / len(base_reps)
        base_corr = sum(rep.shape_correlation for rep in base_reps) / len(base_reps)

        variants: List[Dict[str, Any]] = [
            {
                "label": "baseline_lam0",
                "MAE": round(base_mae, 4),
                "max_err": round(base_max_err, 4),
                "corr": round(base_corr, 4),
            }
        ]

        sweep_rows: List[List[Any]] = [
            ["baseline (lam=0)", "-", round(base_mae, 4), round(base_max_err, 4), round(base_corr, 4), "0.0000"]
        ]

        for lam in (0.5, 1.0, 2.0, 5.0):
            for sig in sigmas:
                w = PriorWeights(
                    lambda_target_tension=lam,
                    sigma_tension=sig,
                    lambda_section_style=0.0,
                )
                reps = []
                for s in seeds:
                    run = run_fixture(RISE_16, seed=s, weights=w)
                    rep = compare_tension_curves(run.target_curve, run.realized_curve, RISE_16.sections)
                    reps.append(rep)
                mean_mae = sum(rep.mean_absolute_error for rep in reps) / len(reps)
                mean_max_err = sum(rep.max_absolute_error for rep in reps) / len(reps)
                mean_corr = sum(rep.shape_correlation for rep in reps) / len(reps)

                lbl = f"lam={lam}_sig={sig}"
                variants.append({
                    "label": lbl,
                    "MAE": round(mean_mae, 4),
                    "max_err": round(mean_max_err, 4),
                    "corr": round(mean_corr, 4),
                })
                delta_mae = mean_mae - base_mae
                sweep_rows.append([
                    f"lam={lam}",
                    f"sig={sig}",
                    round(mean_mae, 4),
                    round(mean_max_err, 4),
                    round(mean_corr, 4),
                    f"{delta_mae:+.4f}",
                ])

        # Record structured comparison
        self.logger.compare(
            title="RISE_16 Tension Sweep (Seeds 7, 11, 23 Mean)",
            baseline_label="baseline_lam0",
            variants=variants,
            lower_is_better=["MAE", "max_err"],
        )

        self.logger.table(
            title="RISE_16 Tension Sweep Table",
            headers=["lambda_tau", "sigma_tau", "MAE", "max_err", "corr", "delta_MAE"],
            rows=sweep_rows,
        )


if __name__ == "__main__":
    unittest.main()
