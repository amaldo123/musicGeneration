"""fixtures.py — Named fixtures and runners for REQ-04 generative section tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.planning.graph import SparseGraph
from aimusic.planning.plans import (
    MethodAPlanDiagnostics,
    MethodARunConfig,
    PlanningSection,
    run_method_a,
)
from aimusic.planning.sb import SBSolution
from aimusic.scoring.tension import realized_tension_curve, target_tension_curve

SEEDS: Tuple[int, ...] = (7, 11, 23, 42, 101)


@dataclass(frozen=True)
class Fixture:
    name: str
    total_beats: int
    sections: Tuple[PlanningSection, ...]
    edo: int = 12
    style_config: Optional[StyleConfig] = None
    sb_config: Optional[SBConfig] = None


@dataclass(frozen=True)
class FixtureRun:
    path: Tuple[BeatState, ...]
    graph: SparseGraph
    solution: SBSolution
    diagnostics: MethodAPlanDiagnostics
    realized_curve: Tuple[Tuple[float, float], ...]
    target_curve: Tuple[Tuple[float, float], ...]


# Fixture arcs grounded in empirical reachable range [p5, p95] = [0.0000, 0.6000]
# (sampled across 2121 legal transitions in tests/fixtures/req04_reachable_range.json)

RISE_16 = Fixture(
    name="RISE_16",
    total_beats=16,
    sections=(
        PlanningSection(
            name="Rise",
            start_time=0,
            end_time=16,
            boundary_level=3,
            # Control points (0.15, 0.55) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.15, 0.55),
        ),
    ),
)

FALL_16 = Fixture(
    name="FALL_16",
    total_beats=16,
    sections=(
        PlanningSection(
            name="Fall",
            start_time=0,
            end_time=16,
            boundary_level=3,
            # Control points (0.55, 0.15) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.55, 0.15),
        ),
    ),
)

PLATEAU_16 = Fixture(
    name="PLATEAU_16",
    total_beats=16,
    sections=(
        PlanningSection(
            name="Plateau",
            start_time=0,
            end_time=16,
            boundary_level=3,
            # Control points (0.35, 0.35) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.35, 0.35),
        ),
    ),
)

TWO_SECTION_16 = Fixture(
    name="TWO_SECTION_16",
    total_beats=16,
    sections=(
        PlanningSection(
            name="A",
            start_time=0,
            end_time=8,
            boundary_level=2,
            # Control points (0.15, 0.55) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.15, 0.55),
        ),
        PlanningSection(
            name="B",
            start_time=8,
            end_time=16,
            boundary_level=3,
            # Control points (0.55, 0.15) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.55, 0.15),
        ),
    ),
)

THREE_SECTION_24 = Fixture(
    name="THREE_SECTION_24",
    total_beats=24,
    sections=(
        PlanningSection(
            name="A",
            start_time=0,
            end_time=8,
            boundary_level=2,
            # Control points (0.20, 0.40) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.20, 0.40),
        ),
        PlanningSection(
            name="B",
            start_time=8,
            end_time=16,
            boundary_level=3,
            # Control points (0.40, 0.60) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.40, 0.60),
        ),
        PlanningSection(
            name="C",
            start_time=16,
            end_time=24,
            boundary_level=2,
            # Control points (0.60, 0.20) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.60, 0.20),
        ),
    ),
)

EDO19_SMOKE = Fixture(
    name="EDO19_SMOKE",
    total_beats=4,
    sections=(
        PlanningSection(
            name="EDO19",
            start_time=0,
            end_time=4,
            boundary_level=3,
            # Control points (0.20, 0.50) lie within [p5=0.0000, p95=0.6000]
            target_tension_arc=(0.20, 0.50),
        ),
    ),
    edo=19,
)


def run_fixture(
    fixture: Fixture,
    *,
    seed: int,
    weights: PriorWeights,
    edo: int = 12,
    **overrides: Any,
) -> FixtureRun:
    """Run Method A planning on the given fixture and return a FixtureRun bundle."""
    actual_edo = overrides.pop("edo", fixture.edo if fixture.edo != 12 else edo)
    style_config = overrides.pop(
        "style_config",
        fixture.style_config or StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
    )
    sb_config = overrides.pop(
        "sb_config",
        fixture.sb_config or SBConfig(horizon_t=fixture.total_beats, k_max=16, d_max=4),
    )

    cfg = MethodARunConfig(
        total_beats=fixture.total_beats,
        seed=seed,
        edo=actual_edo,
        sections=fixture.sections,
        prior_weights=weights,
        style_config=style_config,
        sb_config=sb_config,
        **overrides,
    )
    plan_result, _ = run_method_a(cfg, key=RNGKey(seed=seed))
    r_curve = tuple(realized_tension_curve(plan_result.path, plan_result.vocabularies, actual_edo))
    t_curve = tuple(target_tension_curve(fixture.sections))
    return FixtureRun(
        path=plan_result.path,
        graph=plan_result.graph,
        solution=plan_result.sb_solution,
        diagnostics=plan_result.diagnostics,
        realized_curve=r_curve,
        target_curve=t_curve,
    )
