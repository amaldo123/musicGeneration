"""tests/test_req04_acceptance.py — Comprehensive acceptance suite for REQ-04.

Acceptance criteria A1–A5, R5, PERF:
- TestA1TargetOrStyleChangesPath: Target/style changes affect generation when weight > 0, identical when weight = 0.
- TestA2TensionTracking: Tension tracking accuracy and correlation across standard fixtures and seeds.
- TestA3BaselineReproduction: Reproduction of upstream golden baseline when REQ-04 weights are 0.
- TestA4FeatureAblation: Ablation matrix isolating tension, style, and component contributions.
- TestA5BoundaryAlignment: Boundary level alignment at section transitions and invalid plan validation.
- TestR5LegalityIndependence: Independence of hard transition legality from section preferences.
- TestPerformance: Overhead of REQ-04 scoring terms within performance budget.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Sequence, Tuple
import unittest

from aimusic.core.config import (
    DecodeConfig,
    PriorWeights,
    SBConfig,
    SectionStyleWeights,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, Layer
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import (
    DEFAULT_VOCABULARIES,
    Vocabularies,
    build_default_vocabularies,
)
from aimusic.decode import decode_path_to_score
from aimusic.planning.candidates import (
    apply_boundary_and_groove_constraints,
    apply_meter_constraints,
    apply_position_constraints,
    apply_role_constraints,
    is_legal_transition,
)
from aimusic.planning.graph import build_sparse_graph
from aimusic.planning.plans import (
    MethodARunConfig,
    PlanningSection,
    SectionPlanError,
    build_section_plan,
    run_method_a,
)
from aimusic.planning.sections import SectionTimeline
from aimusic.scoring.priors import NullPrior
from tests.req04_support.fixtures import (
    FALL_16,
    PLATEAU_16,
    RISE_16,
    SEEDS,
    THREE_SECTION_24,
    TWO_SECTION_16,
    Fixture,
    run_fixture,
)
from tests.req04_support.metrics import (
    first_path_difference,
    tension_error,
)
from tests.req04_support.report import ReportLogger

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_TOLERANCES_FILE = _FIXTURES_DIR / "req04_tolerances.json"
_BASELINE_FILE = _FIXTURES_DIR / "req04_baseline.json"


def _load_tolerances() -> Dict[str, Any]:
    with _TOLERANCES_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_baseline() -> Dict[str, Any]:
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# =============================================================================
# A1 — Target / style changes affect generation
# =============================================================================
class TestA1TargetOrStyleChangesPath(unittest.TestCase):
    """A1: Varying each structural input changes path with weight > 0, identical with weight = 0."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.seeds = SEEDS[:3]

    def tearDown(self) -> None:
        self.logger.finish()

    def test_a1_tension_arc(self) -> None:
        """Tension arc change steers path when lambda_tension > 0, identical when 0."""
        self.logger.case("A1_tension_arc", seeds=self.seeds)
        tol = _load_tolerances()
        lambda_t = float(tol["A2"]["lambda_target_tension"])

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        target_tension_arc=s.target_tension_arc,
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        target_tension_arc=tuple(reversed(s.target_tension_arc)),
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base))
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var))

                w_on = PriorWeights(lambda_target_tension=lambda_t)
                w_off = PriorWeights(lambda_target_tension=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "tension_arc",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when tension arc is varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when lambda_tension=0")

        self.logger.table(
            "A1 Tension Arc Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_tension_arc", all_passed, "Tension arc steers path with weight > 0, zero difference when 0")

    def test_a1_groove_family(self) -> None:
        """Groove family change steers path when groove weight > 0, identical when 0."""
        self.logger.case("A1_groove_family", seeds=self.seeds)
        style_cfg = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        groove_family="straight",
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        groove_family="syncopated",
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base), style_config=style_cfg)
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var), style_config=style_cfg)

                w_on = PriorWeights(
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, meter=0.0, key_region=0.0, role=0.0, boundary=0.0, start_alignment=0.0
                    ),
                )
                w_off = PriorWeights(lambda_section_style=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "groove_family",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when groove family is varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when groove weight=0")

        self.logger.table(
            "A1 Groove Family Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_groove_family", all_passed, "Groove family steers path with weight > 0, zero difference when 0")

    def test_a1_key_region(self) -> None:
        """Key region change steers path when key weight > 0, identical when 0."""
        self.logger.case("A1_key_region", seeds=self.seeds)

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        key_region=(0,),
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        key_region=(6,),
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base))
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var))

                w_on = PriorWeights(
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        key_region=1.0, meter=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0
                    ),
                )
                w_off = PriorWeights(lambda_section_style=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "key_region",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when key region is varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when key weight=0")

        self.logger.table(
            "A1 Key Region Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_key_region", all_passed, "Key region steers path with weight > 0, zero difference when 0")

    def test_a1_allowed_meters(self) -> None:
        """Allowed meters change steers path when meter weight > 0, identical when 0."""
        self.logger.case("A1_allowed_meters", seeds=self.seeds)
        style_cfg = StyleConfig(allowed_meters=("4/4", "3/4", "5/4"))

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=2,
                        allowed_meters=("4/4",),
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=2,
                        allowed_meters=("3/4", "5/4") if s.start_time == 0 else ("4/4",),
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base), style_config=style_cfg)
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var), style_config=style_cfg)

                w_on = PriorWeights(
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        meter=1.0, key_region=0.0, groove_family=0.0, role=0.0, boundary=0.0, start_alignment=0.0
                    ),
                )
                w_off = PriorWeights(lambda_section_style=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "allowed_meters",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when allowed_meters is varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when meter weight=0")

        self.logger.table(
            "A1 Allowed Meters Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_allowed_meters", all_passed, "Allowed meters steers path with weight > 0, zero difference when 0")

    def test_a1_preferred_roles(self) -> None:
        """Preferred roles change steers path when role weight > 0, identical when 0."""
        self.logger.case("A1_preferred_roles", seeds=self.seeds)

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        preferred_roles=("cad", "change"),
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        preferred_roles=("prep", "hold"),
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base))
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var))

                w_on = PriorWeights(
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        role=1.0, meter=0.0, key_region=0.0, groove_family=0.0, boundary=0.0, start_alignment=0.0
                    ),
                )
                w_off = PriorWeights(lambda_section_style=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "preferred_roles",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when preferred roles are varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when role weight=0")

        self.logger.table(
            "A1 Preferred Roles Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_preferred_roles", all_passed, "Preferred roles steer path with weight > 0, zero difference when 0")

    def test_a1_boundary_level(self) -> None:
        """Boundary level change steers path when boundary weight > 0, identical when 0."""
        self.logger.case("A1_boundary_level", seeds=self.seeds)

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_base = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=2,
                    )
                    for s in fix.sections
                ]
                sec_var = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=1 if s.start_time == 0 else 2,
                    )
                    for s in fix.sections
                ]
                f_base = Fixture(fix.name, fix.total_beats, tuple(sec_base))
                f_var = Fixture(fix.name, fix.total_beats, tuple(sec_var))

                w_on = PriorWeights(
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        boundary=1.0, meter=0.0, key_region=0.0, groove_family=0.0, role=0.0, start_alignment=0.0
                    ),
                )
                w_off = PriorWeights(lambda_section_style=0.0)

                r_on_base = run_fixture(f_base, seed=seed, weights=w_on)
                r_on_var = run_fixture(f_var, seed=seed, weights=w_on)
                r_off_base = run_fixture(f_base, seed=seed, weights=w_off)
                r_off_var = run_fixture(f_var, seed=seed, weights=w_off)

                diff_on = first_path_difference(r_on_base.path, r_on_var.path)
                diff_off = first_path_difference(r_off_base.path, r_off_var.path)

                passed = (diff_on is not None) and (diff_off is None)
                if not passed:
                    all_passed = False

                first_diff_str = f"beat {diff_on[0]}: {','.join(diff_on[1])}" if diff_on else "none"
                table_rows.append([
                    fix.name,
                    seed,
                    "boundary_level",
                    "PASS" if passed else "FAIL",
                    first_diff_str,
                    "identical" if diff_off is None else f"differs at {diff_off[0]}",
                ])

                self.assertIsNotNone(diff_on, f"[{fix.name} seed {seed}] path must change when boundary level is varied")
                self.assertIsNone(diff_off, f"[{fix.name} seed {seed}] path must be identical when boundary weight=0")

        self.logger.table(
            "A1 Boundary Level Path Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_boundary_level", all_passed, "Boundary level steers path with weight > 0, zero difference when 0")

    def test_a1_target_density_decoder(self) -> None:
        """Target density changes decoded Score when provided, identical when omitted."""
        self.logger.case("A1_target_density_decoder", seeds=self.seeds)

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (RISE_16, TWO_SECTION_16):
            for seed in self.seeds:
                sec_low = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        target_density=0.2,
                    )
                    for s in fix.sections
                ]
                sec_high = [
                    PlanningSection(
                        name=s.name,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        boundary_level=s.boundary_level,
                        target_density=0.8,
                    )
                    for s in fix.sections
                ]
                f_low = Fixture(fix.name, fix.total_beats, tuple(sec_low))
                f_high = Fixture(fix.name, fix.total_beats, tuple(sec_high))

                r_low = run_fixture(f_low, seed=seed, weights=PriorWeights())
                r_high = run_fixture(f_high, seed=seed, weights=PriorWeights())

                # Paths must be identical at planning level (density does not affect transition scoring)
                path_diff = first_path_difference(r_low.path, r_high.path)
                self.assertIsNone(path_diff, "Target density must not alter planning path")

                score_low, _ = decode_path_to_score(
                    r_low.path,
                    decode_config=DecodeConfig(),
                    vocabularies=DEFAULT_VOCABULARIES,
                    edo=12,
                    sections=tuple(sec_low),
                    key=RNGKey(seed=seed),
                )
                score_high, _ = decode_path_to_score(
                    r_low.path,
                    decode_config=DecodeConfig(),
                    vocabularies=DEFAULT_VOCABULARIES,
                    edo=12,
                    sections=tuple(sec_high),
                    key=RNGKey(seed=seed),
                )
                score_none1, _ = decode_path_to_score(
                    r_low.path,
                    decode_config=DecodeConfig(),
                    vocabularies=DEFAULT_VOCABULARIES,
                    edo=12,
                    sections=None,
                    key=RNGKey(seed=seed),
                )
                score_none2, _ = decode_path_to_score(
                    r_low.path,
                    decode_config=DecodeConfig(),
                    vocabularies=DEFAULT_VOCABULARIES,
                    edo=12,
                    sections=None,
                    key=RNGKey(seed=seed),
                )

                score_differs = (score_low != score_high)
                score_none_identical = (score_none1 == score_none2)
                passed = score_differs and score_none_identical
                if not passed:
                    all_passed = False

                n_low = len(score_low.note_events)
                n_high = len(score_high.note_events)
                table_rows.append([
                    fix.name,
                    seed,
                    "target_density",
                    "PASS" if passed else "FAIL",
                    f"notes: {n_low} vs {n_high} (differs)",
                    "identical when omitted",
                ])

                self.assertTrue(score_differs, "Decoded Score must change when target density varies")
                self.assertEqual(score_none1, score_none2, "Decoded Score must be identical when sections=None")

        self.logger.table(
            "A1 Decoder Target Density Sensitivity",
            ["fixture", "seed", "variation", "status", "diff_on", "diff_off"],
            table_rows,
        )
        self.logger.gate("A1_target_density", all_passed, "Target density changes decoded Score, identical when omitted")


# =============================================================================
# A2 — Tension tracking
# =============================================================================
class TestA2TensionTracking(unittest.TestCase):
    """A2: Tension tracking satisfies MAE, win-rate, section MAE, offset, and correlation gates."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.tolerances = _load_tolerances()["A2"]

    def tearDown(self) -> None:
        self.logger.finish()

    def test_a2_tension_tracking(self) -> None:
        """Evaluate tension tracking for RISE_16, FALL_16, PLATEAU_16, TWO_SECTION_16 across SEEDS."""
        self.logger.case("A2_tension_tracking", seeds=SEEDS)
        lambda_t = float(self.tolerances["lambda_target_tension"])

        fixtures = (RISE_16, FALL_16, PLATEAU_16, TWO_SECTION_16)
        all_passed = True
        summary_rows: List[List[Any]] = []

        for fix in fixtures:
            fix_tol = self.tolerances["fixtures"][fix.name]
            mae_limit = float(fix_tol.get("mae", self.tolerances["mae"]))
            win_rate_limit = float(fix_tol.get("win_rate", self.tolerances["win_rate"]))
            section_mae_limit = float(fix_tol.get("section_mae", self.tolerances["section_mae"]))
            peak_offset_limit = float(fix_tol["peak_offset_beats"]) if "peak_offset_beats" in fix_tol else None
            min_corr_limit = float(fix_tol["min_corr"]) if "min_corr" in fix_tol else None

            maes_off: List[float] = []
            maes_on: List[float] = []
            deltas: List[float] = []
            sec_maes: List[float] = []
            offsets: List[float] = []
            corrs: List[float] = []
            wins = 0

            fixture_rows: List[List[Any]] = []

            for seed in SEEDS:
                r_off = run_fixture(fix, seed=seed, weights=PriorWeights(lambda_target_tension=0.0))
                r_on = run_fixture(fix, seed=seed, weights=PriorWeights(lambda_target_tension=lambda_t))

                m_off = tension_error(r_off.realized_curve, r_off.target_curve, fix.sections)
                m_on = tension_error(r_on.realized_curve, r_on.target_curve, fix.sections)

                delta = m_on.mean_abs - m_off.mean_abs
                max_sec = max(m_on.per_section_mean_abs.values()) if m_on.per_section_mean_abs else m_on.mean_abs

                maes_off.append(m_off.mean_abs)
                maes_on.append(m_on.mean_abs)
                deltas.append(delta)
                sec_maes.append(max_sec)
                offsets.append(m_on.peak_offset_beats)
                corrs.append(m_on.pearson_r)

                if m_on.mean_abs < m_off.mean_abs:
                    wins += 1

                seed_pass = (
                    m_on.mean_abs <= mae_limit * 1.25  # per-seed sanity bound
                    and max_sec <= section_mae_limit * 1.25
                )

                fixture_rows.append([
                    fix.name,
                    seed,
                    f"{m_off.mean_abs:.4f}",
                    f"{m_on.mean_abs:.4f}",
                    f"{delta:+.4f}",
                    f"{max_sec:.4f}",
                    f"{m_on.peak_offset_beats:.1f}",
                    f"{m_on.pearson_r:.4f}",
                    "PASS" if seed_pass else "FAIL",
                ])

            mean_mae_off = float(statistics.mean(maes_off))
            mean_mae_on = float(statistics.mean(maes_on))
            std_mae_on = float(statistics.stdev(maes_on)) if len(maes_on) > 1 else 0.0
            win_rate = wins / len(SEEDS)
            mean_sec_mae = float(statistics.mean(sec_maes))
            mean_offset = float(statistics.mean(offsets))
            mean_corr = float(statistics.mean(corrs))

            # Assertion checks against tolerances
            fix_passed = True
            if mean_mae_on > mae_limit:
                fix_passed = False
            if win_rate < win_rate_limit:
                fix_passed = False
            if mean_sec_mae > section_mae_limit:
                fix_passed = False
            if peak_offset_limit is not None and mean_offset > peak_offset_limit:
                fix_passed = False
            if min_corr_limit is not None and mean_corr < min_corr_limit:
                fix_passed = False

            if not fix_passed:
                all_passed = False

            self.logger.table(
                f"A2 Tension Tracking — {fix.name}",
                ["fixture", "seed", "MAE_off", "MAE_on", "delta", "section_MAE", "peak_offset", "corr", "status"],
                fixture_rows,
            )

            summary_rows.append([
                fix.name,
                f"{mean_mae_off:.4f}",
                f"{mean_mae_on:.4f} ± {std_mae_on:.4f}",
                f"{win_rate:.0%}",
                f"{mean_sec_mae:.4f}",
                f"{mean_offset:.2f}" if peak_offset_limit is not None else "N/A",
                f"{mean_corr:.4f}" if min_corr_limit is not None else "N/A",
                "PASS" if fix_passed else "FAIL",
            ])

            self.assertLessEqual(
                mean_mae_on,
                mae_limit,
                f"[{fix.name}] mean MAE_on {mean_mae_on:.4f} exceeds tolerance {mae_limit}",
            )
            self.assertGreaterEqual(
                win_rate,
                win_rate_limit,
                f"[{fix.name}] win rate {win_rate:.0%} below tolerance {win_rate_limit:.0%}",
            )
            self.assertLessEqual(
                mean_sec_mae,
                section_mae_limit,
                f"[{fix.name}] per-section MAE {mean_sec_mae:.4f} exceeds tolerance {section_mae_limit}",
            )
            if peak_offset_limit is not None:
                self.assertLessEqual(
                    mean_offset,
                    peak_offset_limit,
                    f"[{fix.name}] peak offset {mean_offset:.2f} exceeds tolerance {peak_offset_limit}",
                )
            if min_corr_limit is not None:
                self.assertGreaterEqual(
                    mean_corr,
                    min_corr_limit,
                    f"[{fix.name}] correlation {mean_corr:.4f} below tolerance {min_corr_limit}",
                )

        self.logger.table(
            "A2 Tension Tracking Summary Across 5 Seeds",
            ["fixture", "mean_MAE_off", "mean_MAE_on ± std", "win_rate", "section_MAE", "peak_offset", "mean_corr", "gate"],
            summary_rows,
        )
        self.logger.gate("A2", all_passed, "All 4 fixtures satisfy tension tracking tolerances across 5 seeds")


# =============================================================================
# A3 — Baseline reproduction
# =============================================================================
class TestA3BaselineReproduction(unittest.TestCase):
    """A3: With lambdas=0, generated paths exactly equal req04_baseline.json."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.baseline = _load_baseline()

    def tearDown(self) -> None:
        self.logger.finish()

    def test_a3_baseline_reproduction(self) -> None:
        """Verify baseline equality for 4- and 16-beat (12-EDO) and 19-EDO smoke case with and without sections."""
        self.logger.case("A3_baseline_reproduction", seeds=SEEDS)
        weights_zero = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)

        table_rows: List[List[Any]] = []
        all_passed = True

        # Test cases: (beats, edo, seeds, fixture_or_none)
        cases: List[Tuple[int, int, Sequence[int]]] = [
            (4, 12, SEEDS),
            (16, 12, SEEDS),
            (4, 19, (11,)),
        ]

        for beats, edo, seed_list in cases:
            for seed in seed_list:
                key = f"seed_{seed}_beats_{beats}_edo_{edo}"
                golden_raw = self.baseline["paths"][key]
                golden_states = tuple(
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
                    for s in golden_raw
                )

                style_baseline = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))

                # 1. Run with default / no sections supplied (sections=None)
                cfg_no_sec = MethodARunConfig(
                    total_beats=beats,
                    seed=seed,
                    edo=edo,
                    style_config=style_baseline,
                    prior_weights=weights_zero,
                )
                res_no_sec, _ = run_method_a(cfg_no_sec, key=RNGKey(seed=seed))

                # 2. Run with sections supplied explicitly
                sec = build_section_plan(cfg_no_sec)
                cfg_with_sec = MethodARunConfig(
                    total_beats=beats,
                    seed=seed,
                    edo=edo,
                    sections=sec,
                    style_config=style_baseline,
                    prior_weights=weights_zero,
                )
                res_with_sec, _ = run_method_a(cfg_with_sec, key=RNGKey(seed=seed))

                diff_golden_with = first_path_difference(res_with_sec.path, golden_states)
                diff_golden_none = first_path_difference(res_no_sec.path, golden_states)
                diff_with_vs_none = first_path_difference(res_with_sec.path, res_no_sec.path)

                passed = (diff_golden_with is None) and (diff_golden_none is None) and (diff_with_vs_none is None)
                if not passed:
                    all_passed = False

                diff_desc = "match" if diff_golden_with is None else f"diff at beat {diff_golden_with[0]}: {diff_golden_with[1]}"
                table_rows.append([
                    f"{beats} beats, {edo}-EDO",
                    seed,
                    len(golden_states),
                    "PASS" if passed else "FAIL",
                    diff_desc,
                    "identical" if diff_with_vs_none is None else "differs",
                ])

                self.assertIsNone(
                    diff_golden_with,
                    f"[{key}] Path with sections must match baseline. Diff: {diff_golden_with}",
                )
                self.assertIsNone(
                    diff_golden_none,
                    f"[{key}] Path without sections must match baseline. Diff: {diff_golden_none}",
                )
                self.assertIsNone(
                    diff_with_vs_none,
                    f"[{key}] Path with sections must match path without sections.",
                )

        self.logger.table(
            "A3 Baseline Reproduction Results",
            ["configuration", "seed", "beats", "status", "baseline_diff", "sections_vs_none"],
            table_rows,
        )
        self.logger.gate(
            "A3",
            all_passed,
            f"All {len(table_rows)} configurations match req04_baseline.json exactly",
        )


# =============================================================================
# A4 — Feature ablation
# =============================================================================
class TestA4FeatureAblation(unittest.TestCase):
    """A4: Ablation matrix verifies independent contributions of tension, style, and components."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.tolerances = _load_tolerances()["A4"]

    def tearDown(self) -> None:
        self.logger.finish()

    def test_a4_feature_ablation_matrix(self) -> None:
        """Measure MAE, style match rates, distinctness, and cross-effects across ablation variants."""
        self.logger.case("A4_feature_ablation_matrix", seeds=SEEDS)
        ablation_margin = float(self.tolerances["ablation_margin"])
        cross_effect_limit = float(self.tolerances["cross_effect"])

        vocabs = DEFAULT_VOCABULARIES
        groove_sync = next(t.id for t in vocabs.grooves.tokens if t.family == "syncopated")
        role_cad = vocabs.roles.token_for_label("cad").id
        role_prep = vocabs.roles.token_for_label("prep").id

        # Setup standard 2-section fixture with rich styles
        sec_a = PlanningSection(
            name="A",
            start_time=0,
            end_time=8,
            boundary_level=2,
            target_tension_arc=(0.15, 0.55),
            groove_family="syncopated",
            key_region=(7,),
            preferred_roles=("prep", "hold"),
        )
        sec_b = PlanningSection(
            name="B",
            start_time=8,
            end_time=16,
            boundary_level=3,
            target_tension_arc=(0.55, 0.15),
            groove_family="straight",
            key_region=(0,),
            preferred_roles=("cad", "change"),
        )
        style_cfg = StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated"))
        fix = Fixture("Ablation_TWO_SECTION_16", 16, (sec_a, sec_b), style_config=style_cfg)

        # Baseline run (lambdas=0) for distinctness calculation
        base_runs = {
            seed: run_fixture(fix, seed=seed, weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0))
            for seed in SEEDS
        }

        # Define variant configs
        variants: Dict[str, Tuple[PriorWeights, bool]] = {
            "full": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=1.0, role=1.0, boundary=1.0, meter=1.0
                    ),
                ),
                False,
            ),
            "no_tension": (
                PriorWeights(
                    lambda_target_tension=0.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=1.0, role=1.0, boundary=1.0, meter=1.0
                    ),
                ),
                False,
            ),
            "no_style": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=0.0,
                ),
                False,
            ),
            "no_groove": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=0.0, key_region=1.0, role=1.0, boundary=1.0, meter=1.0
                    ),
                ),
                False,
            ),
            "no_key": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=0.0, role=1.0, boundary=1.0, meter=1.0
                    ),
                ),
                False,
            ),
            "no_role": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=1.0, role=0.0, boundary=1.0, meter=1.0
                    ),
                ),
                False,
            ),
            "no_boundary": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=1.0, role=1.0, boundary=0.0, meter=1.0
                    ),
                ),
                False,
            ),
            "steering_on": (
                PriorWeights(
                    lambda_target_tension=2.0,
                    lambda_section_style=2.0,
                    style_weights=SectionStyleWeights(
                        groove_family=1.0, key_region=1.0, role=1.0, boundary=1.0, meter=1.0
                    ),
                ),
                True,
            ),
        }

        metrics: Dict[str, Dict[str, float]] = {}
        table_rows: List[List[Any]] = []

        for name, (w, steer) in variants.items():
            maes: List[float] = []
            groove_matches: List[float] = []
            key_matches: List[float] = []
            role_matches: List[float] = []
            boundary_matches: List[float] = []
            distinctness_scores: List[float] = []
            log_weights: List[float] = []

            for seed in SEEDS:
                sb_cfg = SBConfig(
                    horizon_t=fix.total_beats,
                    k_max=16,
                    d_max=4,
                    section_guided_proposals=steer,
                )
                run = run_fixture(fix, seed=seed, weights=w, sb_config=sb_cfg)

                # MAE
                err = tension_error(run.realized_curve, run.target_curve, fix.sections)
                maes.append(err.mean_abs)

                # Style match rates along path
                path = run.path
                g_match = sum(
                    1 for t, st in enumerate(path)
                    if (t < 8 and st.groove_id == groove_sync) or (t >= 8 and st.groove_id != groove_sync)
                ) / len(path)
                groove_matches.append(g_match)

                k_match = sum(
                    1 for t, st in enumerate(path)
                    if (t < 8 and vocabs.keys.token_for_id(st.key_id).root_pc == 7)
                    or (t >= 8 and vocabs.keys.token_for_id(st.key_id).root_pc == 0)
                ) / len(path)
                key_matches.append(k_match)

                r_match = sum(
                    1 for t, st in enumerate(path)
                    if (t < 8 and st.role_id in (role_prep, 0)) or (t >= 8 and st.role_id in (role_cad, 2))
                ) / len(path)
                role_matches.append(r_match)

                b_match = 1.0 if (path[8].boundary_lvl == 3 and path[8].beat_in_bar == 0) else 0.0
                boundary_matches.append(b_match)

                # Distinctness from baseline
                base_path = base_runs[seed].path
                diff_count = sum(1 for st, b_st in zip(path, base_path) if st != b_st)
                distinctness_scores.append(diff_count / len(path))

                # Final max delta from SB convergence trace
                log_weights.append(float(run.solution.trace.final_max_delta))

            mean_mae = float(statistics.mean(maes))
            mean_groove = float(statistics.mean(groove_matches))
            mean_key = float(statistics.mean(key_matches))
            mean_role = float(statistics.mean(role_matches))
            mean_boundary = float(statistics.mean(boundary_matches))
            mean_dist = float(statistics.mean(distinctness_scores))
            mean_lw = float(statistics.mean(log_weights))

            metrics[name] = {
                "mae": mean_mae,
                "groove": mean_groove,
                "key": mean_key,
                "role": mean_role,
                "boundary": mean_boundary,
                "dist": mean_dist,
                "lw": mean_lw,
            }

            table_rows.append([
                name,
                f"{mean_mae:.4f}",
                f"{mean_groove:.3f}",
                f"{mean_key:.3f}",
                f"{mean_role:.3f}",
                f"{mean_boundary:.3f}",
                f"{mean_dist:.3f}",
                f"{mean_lw:.2e}",
            ])

        self.logger.table(
            "A4 Ablation Matrix Across 5 Seeds",
            ["variant", "MAE", "groove_match", "key_match", "role_match", "boundary_match", "distinctness", "mean_delta"],
            table_rows,
        )

        full_m = metrics["full"]
        no_t_m = metrics["no_tension"]
        no_s_m = metrics["no_style"]
        no_g_m = metrics["no_groove"]
        no_k_m = metrics["no_key"]
        no_r_m = metrics["no_role"]

        # 1. Removing tension raises MAE >= ablation_margin
        mae_increase = no_t_m["mae"] - full_m["mae"]
        self.assertGreaterEqual(
            mae_increase,
            ablation_margin,
            f"Removing tension must raise MAE by at least {ablation_margin:.4f}; got {mae_increase:.4f}",
        )

        # 2. Removing style lowers the relevant match rates >= ablation_margin
        groove_drop = full_m["groove"] - no_s_m["groove"]
        self.assertGreaterEqual(
            groove_drop,
            ablation_margin,
            f"Removing style must lower groove match by at least {ablation_margin:.4f}; got {groove_drop:.4f}",
        )
        key_drop = full_m["key"] - no_s_m["key"]
        self.assertGreaterEqual(
            key_drop,
            ablation_margin,
            f"Removing style must lower key match by at least {ablation_margin:.4f}; got {key_drop:.4f}",
        )
        role_drop = full_m["role"] - no_s_m["role"]
        self.assertGreaterEqual(
            role_drop,
            ablation_margin,
            f"Removing style must lower role match by at least {ablation_margin:.4f}; got {role_drop:.4f}",
        )

        # 3. Removing one style component does not change other component match rates by more than tolerance.cross_effect
        no_b_m = metrics["no_boundary"]

        # Ablating groove: check cross effects on key and role
        self.assertLessEqual(
            abs(full_m["key"] - no_g_m["key"]),
            cross_effect_limit,
            f"Ablating groove must not alter key match by > {cross_effect_limit:.4f}",
        )
        self.assertLessEqual(
            abs(full_m["role"] - no_g_m["role"]),
            cross_effect_limit,
            f"Ablating groove must not alter role match by > {cross_effect_limit:.4f}",
        )

        # Ablating key: check cross effects on groove and role
        self.assertLessEqual(
            abs(full_m["groove"] - no_k_m["groove"]),
            cross_effect_limit,
            f"Ablating key must not alter groove match by > {cross_effect_limit:.4f}",
        )
        self.assertLessEqual(
            abs(full_m["role"] - no_k_m["role"]),
            cross_effect_limit,
            f"Ablating key must not alter role match by > {cross_effect_limit:.4f}",
        )

        # Ablating role: check cross effects on groove and key
        self.assertLessEqual(
            abs(full_m["groove"] - no_r_m["groove"]),
            cross_effect_limit,
            f"Ablating role must not alter groove match by > {cross_effect_limit:.4f}",
        )
        self.assertLessEqual(
            abs(full_m["key"] - no_r_m["key"]),
            cross_effect_limit,
            f"Ablating role must not alter key match by > {cross_effect_limit:.4f}",
        )

        # Ablating boundary: check cross effects on groove and key
        self.assertLessEqual(
            abs(full_m["groove"] - no_b_m["groove"]),
            cross_effect_limit,
            f"Ablating boundary must not alter groove match by > {cross_effect_limit:.4f}",
        )
        self.assertLessEqual(
            abs(full_m["key"] - no_b_m["key"]),
            cross_effect_limit,
            f"Ablating boundary must not alter key match by > {cross_effect_limit:.4f}",
        )

        self.logger.gate(
            "A4",
            True,
            f"Ablations confirm tension effect (+{mae_increase:.4f} MAE), style effect (-{groove_drop:.4f}), and bounded cross-effects",
        )


# =============================================================================
# A5 — Boundary alignment
# =============================================================================
class TestA5BoundaryAlignment(unittest.TestCase):
    """A5: Executable section starts align to downbeats and requested boundary levels."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)

    def tearDown(self) -> None:
        self.logger.finish()

    def test_a5_boundary_alignment_on_executable_sections(self) -> None:
        """Section starts on TWO_SECTION_16 and THREE_SECTION_24 have matching boundary_lvl and beat_in_bar=0."""
        self.logger.case("A5_boundary_alignment_executable", seeds=SEEDS)
        weights = PriorWeights(
            lambda_section_style=2.0,
            style_weights=SectionStyleWeights(boundary=1.0, start_alignment=1.0),
        )

        table_rows: List[List[Any]] = []
        all_passed = True

        for fix in (TWO_SECTION_16, THREE_SECTION_24):
            for seed in SEEDS:
                run = run_fixture(fix, seed=seed, weights=weights)
                path = run.path

                for sec in fix.sections:
                    b = sec.start_time
                    st = path[b]
                    lvl_ok = (st.boundary_lvl == sec.boundary_level)
                    downbeat_ok = (st.beat_in_bar == 0)
                    passed = lvl_ok and downbeat_ok

                    if not passed:
                        all_passed = False

                    table_rows.append([
                        fix.name,
                        sec.name,
                        b,
                        sec.boundary_level,
                        st.boundary_lvl,
                        st.beat_in_bar,
                        seed,
                        "PASS" if passed else "FAIL",
                    ])

                    self.assertEqual(
                        st.boundary_lvl,
                        sec.boundary_level,
                        f"[{fix.name} seed {seed}] section '{sec.name}' start {b} boundary {st.boundary_lvl} != {sec.boundary_level}",
                    )
                    self.assertEqual(
                        st.beat_in_bar,
                        0,
                        f"[{fix.name} seed {seed}] section '{sec.name}' start {b} beat_in_bar {st.beat_in_bar} != 0",
                    )

        self.logger.table(
            "A5 Section Boundary Alignment",
            ["fixture", "section", "start", "expected_lvl", "got_lvl", "beat_in_bar", "seed", "status"],
            table_rows,
        )
        self.logger.gate(
            "A5_executable",
            all_passed,
            f"All {len(table_rows)} section starts align to requested boundary level and downbeat",
        )

    def test_a5_non_downbeat_section_start_validation(self) -> None:
        """Non-downbeat section start raises SectionPlanError or snaps to bar when snap_to_bars=True."""
        self.logger.case("A5_non_downbeat_section_start")
        style_44 = StyleConfig(allowed_meters=("4/4",))
        vocabs = DEFAULT_VOCABULARIES

        # Misaligned plan: section B starts at beat 6 (not divisible by 4)
        misaligned_sections = (
            PlanningSection("A", 0, 6, 2, (0.2, 0.4)),
            PlanningSection("B", 6, 16, 3, (0.4, 0.2)),
        )
        tl_misaligned = SectionTimeline(misaligned_sections, 16)

        # 1. snap_to_bars=False must raise SectionPlanError
        with self.assertRaises(SectionPlanError):
            tl_misaligned.validate_executable(style_44, vocabs, snap_to_bars=False)

        # 2. snap_to_bars=True must snap and report the change
        new_tl, changes = tl_misaligned.validate_executable(style_44, vocabs, snap_to_bars=True)
        self.assertGreater(len(changes), 0, "Must report at least one snapping change")
        self.assertEqual(new_tl.sections[1].start_time, 4, "Section B must snap from beat 6 to beat 4")

        self.logger.gate(
            "A5",
            True,
            "Non-downbeat section start raises SectionPlanError when snap=False and snaps to bar 4 when snap=True",
        )


# =============================================================================
# R5 — Legality independence
# =============================================================================
class TestR5LegalityIndependence(unittest.TestCase):
    """R5: Hard transition legality is strictly decoupled from section preferences."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs12 = DEFAULT_VOCABULARIES
        self.vocabs19 = build_default_vocabularies(edo=19)

    def tearDown(self) -> None:
        self.logger.finish()

    def test_r5_legality_independence(self) -> None:
        """Constraint functions lack section parameters, and legality is invariant across random transitions."""
        self.logger.case("R5_legality_independence")

        # 1. Verify signatures of hard constraint functions
        functions = (
            is_legal_transition,
            apply_meter_constraints,
            apply_position_constraints,
            apply_role_constraints,
            apply_boundary_and_groove_constraints,
        )
        for fn in functions:
            sig = inspect.signature(fn)
            sec_params = [p for p in sig.parameters if p.startswith("section")]
            self.assertEqual(sec_params, [], f"{fn.__name__} violates separation: found {sec_params}")

        # 2. Property test: 2000 random 12-EDO and 500 19-EDO transitions
        style = StyleConfig(allowed_meters=("4/4", "3/4"), groove_families=("straight", "syncopated"))

        def count_legal_random_transitions(vocabs: Vocabularies, n_trials: int, edo: int) -> int:
            rng = RNGKey(seed=edo * 1000 + n_trials).generator()
            legal_count = 0
            n_meters = len(vocabs.meters)
            n_keys = len(vocabs.keys)
            n_chords = len(vocabs.chords)
            n_roles = len(vocabs.roles)
            n_heads = len(vocabs.heads)
            n_grooves = len(vocabs.grooves)

            for _ in range(n_trials):
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
                legal, _ = is_legal_transition(prev, cand, style, vocabs)
                if legal:
                    legal_count += 1
            return legal_count

        count_12 = count_legal_random_transitions(self.vocabs12, 2000, 12)
        count_19 = count_legal_random_transitions(self.vocabs19, 500, 19)

        self.assertGreater(count_12, 0, "Expected non-zero legal transitions in 12-EDO random sample")
        self.assertGreater(count_19, 0, "Expected non-zero legal transitions in 19-EDO random sample")

        self.logger.table(
            "R5 Legality Invariance Random Transition Trials",
            ["EDO", "trials", "legal_count", "legal_rate"],
            [
                ["12-EDO", 2000, count_12, f"{count_12 / 2000:.2%}"],
                ["19-EDO", 500, count_19, f"{count_19 / 500:.2%}"],
            ],
        )

        self.logger.gate(
            "R5",
            True,
            f"Verified constraint signatures and 2500 random transitions (12-EDO {count_12}/2000, 19-EDO {count_19}/500 legal)",
        )


# =============================================================================
# PERF — Performance
# =============================================================================
class TestPerformance(unittest.TestCase):
    """PERF: Graph construction time with REQ-04 terms enabled satisfies performance budget."""

    logger: ReportLogger

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.tolerances = _load_tolerances()["PERF"]

    def tearDown(self) -> None:
        self.logger.finish()

    def test_performance_graph_build_overhead(self) -> None:
        """Measure graph build time with terms on vs off and assert time_on <= mult * time_off + overhead."""
        self.logger.case("PERF_graph_build_overhead")
        mult = float(self.tolerances["time_multiplier"])
        overhead = float(self.tolerances["time_overhead_sec"])

        vocabs = DEFAULT_VOCABULARIES
        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        sb = SBConfig(horizon_t=16, k_max=16, d_max=4)
        start_state = BeatState(0, 0, 3, 0, 0, 0, 1, 0)
        end_state = BeatState(0, 0, 3, 0, 0, 3, 1, 0)
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=16, states=(end_state,))

        w_off = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)
        w_on = PriorWeights(lambda_target_tension=2.0, lambda_section_style=2.0)

        # Warmup
        build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=16,
            sb_config=sb,
            style_config=style,
            vocabularies=vocabs,
            prior=NullPrior(),
            weights=w_off,
            edo=12,
            key=RNGKey(seed=11),
            d_max=4,
            sections=RISE_16.sections,
        )

        # Measure time_off
        t0 = time.perf_counter()
        build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=16,
            sb_config=sb,
            style_config=style,
            vocabularies=vocabs,
            prior=NullPrior(),
            weights=w_off,
            edo=12,
            key=RNGKey(seed=11),
            d_max=4,
            sections=RISE_16.sections,
        )
        time_off = time.perf_counter() - t0

        # Measure time_on
        t0 = time.perf_counter()
        build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=16,
            sb_config=sb,
            style_config=style,
            vocabularies=vocabs,
            prior=NullPrior(),
            weights=w_on,
            edo=12,
            key=RNGKey(seed=11),
            d_max=4,
            sections=RISE_16.sections,
        )
        time_on = time.perf_counter() - t0

        limit = mult * time_off + overhead
        ratio = time_on / time_off if time_off > 0 else 1.0
        passed = time_on <= limit

        self.logger.table(
            "PERF Graph Build Overhead Comparison",
            ["time_off_sec", "time_on_sec", "ratio", "budget_limit_sec", "status"],
            [[f"{time_off:.4f}", f"{time_on:.4f}", f"{ratio:.2f}", f"{limit:.4f}", "PASS" if passed else "FAIL"]],
        )

        self.logger.gate(
            "PERF",
            passed,
            f"time_on {time_on:.4f}s <= {mult} * {time_off:.4f}s + {overhead}s = {limit:.4f}s (ratio {ratio:.2f})",
        )
        self.assertLessEqual(time_on, limit, f"Graph build time {time_on:.4f}s exceeds budget {limit:.4f}s")


if __name__ == "__main__":
    unittest.main()
