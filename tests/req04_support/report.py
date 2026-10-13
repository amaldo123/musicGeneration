"""report.py — Shared structured logger for REQ-04 red/green tests.

Output directory: logs/req04/<run_id>/
  run_id  = env REQ04_RUN_ID  or  UTC timestamp (YYYYMMDDTHHmmss)
  log dir = env REQ04_LOG_DIR or  <repo_root>/logs

Files written by finish():
  events.jsonl  — one JSON object per call
  report.md     — human-readable table / gate summary
  summary.json  — machine-readable metadata + gate results
"""

from __future__ import annotations

import json
import logging
import os
import time
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence

log = logging.getLogger("req04.report")


# ---------------------------------------------------------------------------
# Helper: find repo root (two levels above this file: tests/req04_support/)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # tests/ -> repo root

_DEFAULT_RUN_ID: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def _default_log_dir() -> Path:
    env = os.environ.get("REQ04_LOG_DIR", "")
    if env:
        return Path(env)
    return _REPO_ROOT / "logs"


def _run_id() -> str:
    return os.environ.get("REQ04_RUN_ID", _DEFAULT_RUN_ID)


# ---------------------------------------------------------------------------
# Event types stored in events.jsonl
# ---------------------------------------------------------------------------
@dataclass
class _Event:
    kind: str
    data: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "ts": self.ts, **self.data}


# ---------------------------------------------------------------------------
# ReportLogger
# ---------------------------------------------------------------------------
class ReportLogger:
    """Structured test logger for one REQ-04 test run."""

    _instances: ClassVar[Dict[str, ReportLogger]] = {}

    def __new__(
        cls,
        run_id: Optional[str] = None,
        log_dir: Optional[Path] = None,
    ) -> ReportLogger:
        actual_run_id = run_id or _run_id()
        base = log_dir or _default_log_dir()
        key = f"{base}:{actual_run_id}"
        if key in cls._instances:
            return cls._instances[key]
        inst = super().__new__(cls)
        cls._instances[key] = inst
        return inst

    def __init__(self, run_id: Optional[str] = None, log_dir: Optional[Path] = None) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._run_id = run_id or _run_id()
        base = log_dir or _default_log_dir()
        self._out_dir: Path = base / "req04" / self._run_id
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._events: List[_Event] = []
        self._gates: List[Dict[str, Any]] = []
        self._current_case: Optional[str] = None

        logging.basicConfig(
            format="%(levelname)s %(name)s %(message)s",
            level=logging.INFO,
        )

    # ------------------------------------------------------------------
    # Convenience constructor for unittest.TestCase subclasses
    # ------------------------------------------------------------------
    @classmethod
    def for_test(cls, testcase: unittest.TestCase) -> ReportLogger:
        """Create or get a ReportLogger for the testcase."""
        test_name = testcase.__class__.__name__ + "." + testcase._testMethodName
        run_id = os.environ.get("REQ04_RUN_ID", _run_id())
        rl = cls(run_id=run_id)
        rl._current_case = test_name
        rl._record("case_start", {"name": test_name})
        return rl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def case(self, name: str, **config: Any) -> None:
        """Start a named test case; record its full configuration."""
        self._current_case = name
        self._record("case_start", {"name": name, "config": config})
        log.info("[REQ04] case: %s  config=%s", name, _short(config))

    def metric(self, name: str, value: Any, **context: Any) -> None:
        """Record a single named metric with optional context."""
        self._record("metric", {"name": name, "value": value, "context": context})
        log.info("[REQ04] metric %s=%r  %s", name, value, _short(context))

    def table(
        self,
        title: str,
        headers: Sequence[str],
        rows: Sequence[Sequence[Any]],
    ) -> None:
        """Write a fixed-width table to the log and accumulate for report.md."""
        self._record("table", {"title": title, "headers": list(headers), "rows": [list(r) for r in rows]})
        lines = _format_table(title, headers, rows)
        for line in lines:
            log.info(line[:160])

    def compare(
        self,
        title: str,
        baseline_label: str,
        variants: Sequence[Dict[str, Any]],
        lower_is_better: Optional[Sequence[str]] = None,
    ) -> None:
        """Tabular comparison: rows = variants, columns = metrics, delta vs baseline."""
        self._record("compare", {
            "title": title,
            "baseline_label": baseline_label,
            "variants": list(variants),
            "lower_is_better": lower_is_better,
        })
        if not variants:
            return
        metric_keys = [k for k in variants[0] if k != "label"]
        headers = ["variant"] + metric_keys + [f"Δ{k}" for k in metric_keys]
        baseline_row = next((v for v in variants if v.get("label") == baseline_label), variants[0])
        rows: List[List[Any]] = []
        for v in variants:
            row: List[Any] = [v.get("label", "?")]
            deltas: List[Any] = []
            for k in metric_keys:
                row.append(v.get(k, ""))
                delta = ""
                if k in v and k in baseline_row:
                    try:
                        delta = f"{v[k] - baseline_row[k]:+.4f}"
                    except TypeError:
                        pass
                deltas.append(delta)
            rows.append(row + deltas)
        self.table(title, headers, rows)

    def beat_table(
        self,
        path: Sequence[Any],
        timeline_or_sections: Any,
        vocabularies: Any,
        edo: int,
        limit: Optional[int] = None,
    ) -> None:
        """Per-beat table: beat, section, tau*, tau_hat, deviation, boundary_lvl, role, groove, chord."""
        from aimusic.scoring.tension import (
            beat_tension,
            target_tension_curve,
        )

        path_list = list(path)
        if limit is not None:
            path_list = path_list[:limit]

        # Build target curve
        sections = timeline_or_sections if hasattr(timeline_or_sections, "__iter__") else []
        try:
            target_map = {int(t): v for t, v in target_tension_curve(sections)}
        except Exception:
            target_map = {}

        headers = ["k", "section", "tau*", "tau_hat", "dev", "bnd_lvl", "role", "groove", "chord"]
        rows = []
        for k, state in enumerate(path_list):
            prev = path_list[k - 1] if k > 0 else None
            try:
                tau_hat = beat_tension(prev, state, vocabularies, edo)
            except Exception:
                tau_hat = float("nan")
            tau_star = target_map.get(k, float("nan"))
            dev = tau_hat - tau_star if not (tau_hat != tau_hat or tau_star != tau_star) else float("nan")  # noqa: PLR0124

            section_name = "?"
            for s in sections:
                if hasattr(s, "start_time") and hasattr(s, "end_time"):
                    if s.start_time <= k < s.end_time:
                        section_name = s.name
                        break

            try:
                role = vocabularies.roles.token_for_id(state.role_id).label
            except Exception:
                role = str(state.role_id)
            try:
                groove = vocabularies.grooves.token_for_id(state.groove_id).family
            except Exception:
                groove = str(state.groove_id)
            try:
                chord = vocabularies.chords.token_for_id(state.chord_id).label
            except Exception:
                chord = str(state.chord_id)

            rows.append([
                k,
                section_name,
                f"{tau_star:.3f}" if tau_star == tau_star else "nan",  # noqa: PLR0124
                f"{tau_hat:.3f}" if tau_hat == tau_hat else "nan",  # noqa: PLR0124
                f"{dev:+.3f}" if dev == dev else "nan",  # noqa: PLR0124
                state.boundary_lvl,
                role,
                groove,
                chord,
            ])

        self.table("Beat Table", headers, rows)

    def gate(self, criterion: str, passed: bool, detail: str) -> None:
        """Record a named gate check; prints [REQ04][criterion][PASS|FAIL] detail."""
        status = "PASS" if passed else "FAIL"
        msg = f"[REQ04][{criterion}][{status}] {detail}"
        log.info(msg[:160])
        self._gates.append({"criterion": criterion, "passed": passed, "detail": detail})
        self._record("gate", {"criterion": criterion, "passed": passed, "detail": detail})

    def finish(self) -> Path:
        """Write events.jsonl, report.md, summary.json to the output directory."""
        # events.jsonl
        events_path = self._out_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as fh:
            for evt in self._events:
                fh.write(json.dumps(evt.to_dict()) + "\n")

        # report.md
        report_path = self._out_dir / "report.md"
        self._write_report_md(report_path)

        # summary.json
        summary_path = self._out_dir / "summary.json"
        n_pass = sum(1 for g in self._gates if g["passed"])
        n_fail = sum(1 for g in self._gates if not g["passed"])
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump({
                "run_id": self._run_id,
                "gates": self._gates,
                "n_pass": n_pass,
                "n_fail": n_fail,
            }, fh, indent=2)

        log.info("[REQ04] Report written to %s", self._out_dir)
        return report_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record(self, kind: str, data: Dict[str, Any]) -> None:
        self._events.append(_Event(kind=kind, data=data))

    def _write_report_md(self, path: Path) -> None:
        lines: List[str] = [
            f"# REQ-04 Test Report — run_id `{self._run_id}`\n",
            "",
        ]
        for evt in self._events:
            if evt.kind == "case_start":
                lines.append(f"\n## Case: {evt.data.get('name', '?')}")
                cfg = evt.data.get("config", {})
                if cfg:
                    lines.append(f"\n**Config:** `{cfg}`")
            elif evt.kind == "metric":
                lines.append(
                    f"\n- **{evt.data['name']}** = `{evt.data['value']}`"
                    + (f"  ({evt.data.get('context', '')})" if evt.data.get("context") else "")
                )
            elif evt.kind == "table":
                lines.append(f"\n### {evt.data['title']}")
                lines.append("")
                lines.extend(_format_table("", evt.data["headers"], evt.data["rows"]))

            elif evt.kind == "gate":
                status = "✅ PASS" if evt.data["passed"] else "❌ FAIL"
                lines.append(
                    f"\n**Gate [{evt.data['criterion']}]** {status}: {evt.data['detail']}"
                )

        # Gate summary table
        if self._gates:
            lines += [
                "\n## Gate Summary",
                "",
                "| Criterion | Status | Detail |",
                "|-----------|--------|--------|",
            ]
            for g in self._gates:
                status = "✅" if g["passed"] else "❌"
                detail_escaped = str(g["detail"]).replace("|", "\\|")[:120]
                lines.append(f"| {g['criterion']} | {status} | {detail_escaped} |")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _short(d: Any, max_len: int = 120) -> str:
    s = repr(d)
    return s if len(s) <= max_len else s[:max_len] + "…"


def _format_table(
    title: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> List[str]:
    """Return lines for a fixed-width ASCII table."""
    all_rows = [list(headers)] + [list(r) for r in rows]
    col_widths = [
        max(len(str(all_rows[row_i][col_i])) for row_i in range(len(all_rows)))
        for col_i in range(len(headers))
    ]

    def _fmt_row(row: Sequence[Any]) -> str:
        cells = [str(row[i]).ljust(col_widths[i]) for i in range(len(headers))]
        return "| " + " | ".join(cells) + " |"

    sep = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    out = []
    if title:
        out.append(f"### {title}")
    out.append(_fmt_row(headers))
    out.append(sep)
    for row in rows:
        out.append(_fmt_row(row))
    return out
