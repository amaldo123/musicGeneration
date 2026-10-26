"""tests/req04_support — shared logger and fixtures for section generation tests."""

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
    "run_fixture",
]
