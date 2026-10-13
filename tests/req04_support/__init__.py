"""tests/req04_support — shared logger, fixtures, and metrics for REQ-04 tests."""

from tests.req04_support.fixtures import (
    EDO19_SMOKE,
    FALL_16,
    PLATEAU_16,
    RISE_16,
    SEEDS,
    THREE_SECTION_24,
    TWO_SECTION_16,
    Fixture,
    FixtureRun,
    run_fixture,
)
from tests.req04_support.metrics import (
    TensionMetrics,
    first_path_difference,
    path_signature,
    tension_error,
)
from tests.req04_support.report import ReportLogger

__all__ = [
    "EDO19_SMOKE",
    "FALL_16",
    "PLATEAU_16",
    "RISE_16",
    "SEEDS",
    "THREE_SECTION_24",
    "TWO_SECTION_16",
    "Fixture",
    "FixtureRun",
    "ReportLogger",
    "TensionMetrics",
    "first_path_difference",
    "path_signature",
    "run_fixture",
    "tension_error",
]
