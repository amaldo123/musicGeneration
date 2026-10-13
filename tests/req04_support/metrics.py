"""metrics.py — Metrics, comparisons, and path diagnostics for REQ-04."""

from __future__ import annotations

import hashlib
import statistics
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from aimusic.core.core_types import BeatState
from aimusic.planning.plans import PlanningSection
from aimusic.scoring.tension import compare_tension_curves

_STATE_FIELDS = (
    "meter_id",
    "beat_in_bar",
    "boundary_lvl",
    "key_id",
    "chord_id",
    "role_id",
    "head_id",
    "groove_id",
)


@dataclass(frozen=True)
class TensionMetrics:
    mean_abs: float
    max_abs: float
    per_section_mean_abs: Dict[str, float]
    peak_offset_beats: float
    pearson_r: float


def tension_error(
    realized: Sequence[Tuple[float, float]],
    target: Sequence[Tuple[float, float]],
    sections: Optional[Sequence[PlanningSection]] = None,
) -> TensionMetrics:
    """Compute tension tracking error metrics between realized and target curves."""
    if sections:
        rep = compare_tension_curves(target, realized, sections)
        return TensionMetrics(
            mean_abs=rep.mean_absolute_error,
            max_abs=rep.max_absolute_error,
            per_section_mean_abs=rep.section_errors,
            peak_offset_beats=rep.peak_timing_offset,
            pearson_r=rep.shape_correlation,
        )

    # Fallback when no section objects are provided
    target_map = {t: v for t, v in target}
    common_times = [t for t, _ in realized if t in target_map]
    if not common_times:
        return TensionMetrics(
            mean_abs=0.0,
            max_abs=0.0,
            per_section_mean_abs={},
            peak_offset_beats=0.0,
            pearson_r=0.0,
        )

    errors = [abs(dict(realized)[t] - target_map[t]) for t in common_times]
    mean_abs = float(statistics.mean(errors))
    max_abs = float(max(errors))

    r_map = dict(realized)
    target_peak = max(common_times, key=lambda t: target_map[t])
    realized_peak = max(common_times, key=lambda t: r_map[t])
    peak_offset = float(abs(realized_peak - target_peak))

    xs = [target_map[t] for t in common_times]
    ys = [r_map[t] for t in common_times]
    pearson_r = 0.0
    if len(xs) >= 2 and len(set(xs)) > 1 and len(set(ys)) > 1:
        pearson_r = float(statistics.correlation(xs, ys))

    return TensionMetrics(
        mean_abs=mean_abs,
        max_abs=max_abs,
        per_section_mean_abs={},
        peak_offset_beats=peak_offset,
        pearson_r=pearson_r,
    )


def first_path_difference(
    path_a: Sequence[BeatState],
    path_b: Sequence[BeatState],
) -> Optional[Tuple[int, Tuple[str, ...]]]:
    """Find the first differing beat index and the BeatState field names that differ."""
    min_len = min(len(path_a), len(path_b))
    for k in range(min_len):
        diff_fields = tuple(
            field
            for field in _STATE_FIELDS
            if getattr(path_a[k], field) != getattr(path_b[k], field)
        )
        if diff_fields:
            return (k, diff_fields)

    if len(path_a) != len(path_b):
        return (min_len, ("length",))

    return None


def path_signature(path: Sequence[BeatState]) -> str:
    """Return a deterministic string hash of a BeatState path."""
    raw = repr([
        (
            s.meter_id,
            s.beat_in_bar,
            s.boundary_lvl,
            s.key_id,
            s.chord_id,
            s.role_id,
            s.head_id,
            s.groove_id,
        )
        for s in path
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
