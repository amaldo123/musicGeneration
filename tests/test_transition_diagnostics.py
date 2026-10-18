"""test_transition_diagnostics.py — Per-transition diagnostics tests for REQ-04.

Tests decomposition invariant, graph score provenance, JSON schema compatibility,
CLI generation integration with section plans and reporting, and auditable score tables.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aimusic.core.config import PriorWeights
from aimusic.core.diagnostics import DIAGNOSTICS_SCHEMA_VERSION, RunManifest
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning import graph as graph_module
from aimusic.planning.plans import TransitionDiagnostic
import aimusic.scoring.priors as priors_module
from tests.req04_support.fixtures import (
    EDO19_SMOKE,
    RISE_16,
    SEEDS,
    THREE_SECTION_24,
    TWO_SECTION_16,
    run_fixture,
)
from tests.req04_support.report import ReportLogger


class TestTransitionDiagnostics(unittest.TestCase):
    """Focused test suite for per-transition diagnostics and CLI integration."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs = DEFAULT_VOCABULARIES
        self.addCleanup(self.logger.finish)

    def test_1_decomposition_invariant(self) -> None:
        """Verify decomposition invariant holds within 1e-9 for all selected transitions."""
        self.logger.case("test_1_decomposition_invariant")
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5)

        test_cases = [
            (RISE_16, 12),
            (TWO_SECTION_16, 12),
            (THREE_SECTION_24, 12),
        ]

        total_transitions_checked = 0
        max_discrepancy = 0.0

        for fixture, edo in test_cases:
            for seed in SEEDS:
                run_res = run_fixture(fixture, seed=seed, weights=weights, edo=edo)
                diags = run_res.diagnostics.transition_diagnostics
                self.assertEqual(len(diags), fixture.total_beats)

                for diag in diags:
                    total_transitions_checked += 1
                    # Verify using method
                    self.assertTrue(diag.verify_decomposition(tol=1e-9))

                    expected = (
                        diag.data_contribution
                        + diag.gttm_contribution
                        - diag.target_tension_contribution
                        - diag.section_style_contribution
                    )
                    diff = abs(diag.final_log_weight - expected)
                    if diff > max_discrepancy:
                        max_discrepancy = diff

        # 19-EDO case
        run_19 = run_fixture(EDO19_SMOKE, seed=7, weights=weights, edo=19)
        diags_19 = run_19.diagnostics.transition_diagnostics
        self.assertEqual(len(diags_19), EDO19_SMOKE.total_beats)
        for diag in diags_19:
            total_transitions_checked += 1
            self.assertTrue(diag.verify_decomposition(tol=1e-9))
            expected = (
                diag.data_contribution
                + diag.gttm_contribution
                - diag.target_tension_contribution
                - diag.section_style_contribution
            )
            diff = abs(diag.final_log_weight - expected)
            if diff > max_discrepancy:
                max_discrepancy = diff

        self.logger.metric("total_transitions_checked", total_transitions_checked)
        self.logger.metric("max_decomposition_discrepancy", max_discrepancy)
        self.logger.gate(
            "DECOMPOSITION_INVARIANT",
            passed=(max_discrepancy <= 1e-9),
            detail=f"Checked {total_transitions_checked} transitions across fixtures & seeds, max diff={max_discrepancy:.2e} <= 1e-9",
        )

    def test_2_diagnostics_use_stored_graph_scores(self) -> None:
        """Patch graph scorer to increase weights by 0.5; verify diagnostic reflects exact change."""
        self.logger.case("test_2_diagnostics_use_stored_graph_scores")
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5)

        # Baseline run
        base_res = run_fixture(TWO_SECTION_16, seed=11, weights=weights)
        base_diags = base_res.diagnostics.transition_diagnostics

        # Patch with_section_terms in graph module so final_log_weight increases by exactly 0.5
        orig_with_section_terms = priors_module.with_section_terms

        def patched_with_section_terms(*args, **kwargs):
            res = orig_with_section_terms(*args, **kwargs)
            return dataclasses.replace(
                res,
                data_contribution=res.data_contribution + 0.5,
                final_log_weight=res.final_log_weight + 0.5,
            )

        with patch.object(priors_module, "with_section_terms", side_effect=patched_with_section_terms):
            with patch.object(graph_module, "with_section_terms", side_effect=patched_with_section_terms):
                patched_res = run_fixture(TWO_SECTION_16, seed=11, weights=weights)
                patched_diags = patched_res.diagnostics.transition_diagnostics

        self.assertEqual(len(base_diags), len(patched_diags))
        weight_diffs = [
            p.final_log_weight - b.final_log_weight
            for b, p in zip(base_diags, patched_diags)
        ]

        for idx, diff in enumerate(weight_diffs):
            self.assertAlmostEqual(
                diff,
                0.5,
                places=7,
                msg=f"Transition {idx} final_log_weight diff {diff} != 0.5",
            )

        self.logger.metric("min_weight_diff", min(weight_diffs))
        self.logger.metric("max_weight_diff", max(weight_diffs))
        self.logger.gate(
            "STORED_SCORES_PROVENANCE",
            passed=all(abs(d - 0.5) < 1e-7 for d in weight_diffs),
            detail="Diagnostics reflect patched graph edge weight (+0.5) exactly; no independent re-evaluation.",
        )

    def test_3_json_compatibility(self) -> None:
        """New-schema JSON roundtrip reproduces record; old-schema manifest loads cleanly."""
        self.logger.case("test_3_json_compatibility")
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5)
        run_res = run_fixture(TWO_SECTION_16, seed=11, weights=weights)
        d0 = run_res.diagnostics.transition_diagnostics[0]

        # 1. New-schema round trip
        d0_dict = d0.to_dict()
        json_str = json.dumps(d0_dict)
        loaded_dict = json.loads(json_str)
        d0_reconstructed = TransitionDiagnostic.from_dict(loaded_dict)

        self.assertEqual(d0.time_index, d0_reconstructed.time_index)
        self.assertEqual(d0.arrival_time, d0_reconstructed.arrival_time)
        self.assertEqual(d0.section_name, d0_reconstructed.section_name)
        self.assertEqual(d0.section_index, d0_reconstructed.section_index)
        self.assertEqual(d0.is_section_start, d0_reconstructed.is_section_start)
        self.assertEqual(d0.source_state, d0_reconstructed.source_state)
        self.assertEqual(d0.target_state, d0_reconstructed.target_state)
        self.assertAlmostEqual(d0.final_log_weight, d0_reconstructed.final_log_weight, places=9)
        self.assertAlmostEqual(d0.data_contribution, d0_reconstructed.data_contribution, places=9)
        self.assertAlmostEqual(d0.gttm_contribution, d0_reconstructed.gttm_contribution, places=9)
        self.assertAlmostEqual(d0.target_tension, d0_reconstructed.target_tension, places=9)
        self.assertAlmostEqual(d0.realized_tension, d0_reconstructed.realized_tension, places=9)

        # 2. Old-schema record loading
        old_record = {
            "time_index": 0,
            "source_state": d0.source_state.to_dict(),
            "target_state": d0.target_state.to_dict(),
            "data_logp": -1.25,
            "gttm_energy": 2.50,
            "gttm_family_breakdown": {"meter": 1.0},
            "target_tension": 0.35,
            "realized_tension": 0.40,
            "tension_deviation": 0.05,
            "target_tension_penalty": 0.125,
            "section_style_penalty": 0.0,
            "section_style_breakdown": {},
            "total_log_weight": -3.875,
        }
        loaded_old = TransitionDiagnostic.from_dict(old_record)
        self.assertEqual(loaded_old.time_index, 0)
        self.assertEqual(loaded_old.arrival_time, 1)
        self.assertAlmostEqual(loaded_old.final_log_weight, -3.875, places=6)
        self.assertAlmostEqual(loaded_old.total_log_weight, -3.875, places=6)
        self.assertAlmostEqual(loaded_old.target_tension_penalty, 0.125, places=6)

        # 3. Old-schema RunManifest loading
        old_manifest_data = {
            "run_id": "old-run-123",
            "timestamp": 1600000000.0,
            "version": "0.1.0",
            "tension_model_version": "1.0.0",
            "seed": 42,
            "config": {"edo": 12},
            "structure": {
                "boundaries": [0.0, 8.0],
                "tension_curve": [(0.0, 0.2), (1.0, 0.3)],
                "target_tension_curve": [(0.0, 0.2), (1.0, 0.3)],
                "tension_deviation": {"mean_absolute_error": 0.0},
                "transition_diagnostics": [old_record],
            },
            "sb_stats": {
                "iterations_run": 5,
                "converged": True,
                "final_max_delta": 1e-12,
                "layer_sizes": [1, 2, 1],
                "pruned_nodes": 0,
                "effective_entropy": 0.0,
            },
        }
        manifest_loaded = RunManifest.from_dict(old_manifest_data)
        self.assertEqual(manifest_loaded.run_id, "old-run-123")
        self.assertEqual(manifest_loaded.version, "0.1.0")
        self.assertEqual(len(manifest_loaded.structural_stats.transition_diagnostics), 1)

        self.logger.gate(
            "JSON_SCHEMA_COMPATIBILITY",
            passed=True,
            detail=f"New schema ({DIAGNOSTICS_SCHEMA_VERSION}) roundtrip passed; legacy schema v0.1.0 loaded cleanly.",
        )

    def test_4_cli_integration(self) -> None:
        """CLI generates valid report-dir manifest & summary table; gaps exit with SectionPlanError."""
        self.logger.case("test_4_cli_integration")
        fixtures_path = Path("tests/fixtures/req04_two_sections.json").resolve()
        self.assertTrue(fixtures_path.exists())

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "out"
            report_path = Path(tmp_dir) / "report"

            cmd = [
                sys.executable,
                "-m",
                "aimusic.app.cli",
                "generate",
                "--seed",
                "11",
                "--beats",
                "16",
                "--sections-file",
                str(fixtures_path),
                "--report-dir",
                str(report_path),
                "--out",
                str(out_path),
                "--lambda-target-tension",
                "2.0",
                "--lambda-section-style",
                "1.5",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=f"CLI failed with error:\n{res.stderr}\nStdout:\n{res.stdout}")

            # Verify summary table printed to stdout
            self.assertIn("=== REQ-04 Section Generation Summary ===", res.stdout)
            self.assertIn("Section", res.stdout)
            self.assertIn("MAE Tension", res.stdout)
            self.assertIn("A", res.stdout)
            self.assertIn("B", res.stdout)

            # Verify report manifest
            manifest_file = report_path / "manifest.json"
            self.assertTrue(manifest_file.exists())
            with manifest_file.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            trans_diags = manifest_data["structure"]["transition_diagnostics"]
            self.assertEqual(len(trans_diags), 16)
            self.assertIn("arrival_time", trans_diags[0])
            self.assertIn("section_name", trans_diags[0])
            self.assertIn("final_log_weight", trans_diags[0])

            # Verify summary.json and report.md
            self.assertTrue((report_path / "summary.json").exists())
            self.assertTrue((report_path / "report.md").exists())

            # Test section plan containing a gap
            gap_plan = [
                {
                    "name": "Sec1",
                    "start_time": 0,
                    "end_time": 4,
                    "boundary_level": 2,
                    "target_tension_arc": [0.2, 0.4],
                },
                {
                    "name": "Sec2",
                    "start_time": 6,  # Gap between 4 and 6
                    "end_time": 12,
                    "boundary_level": 3,
                    "target_tension_arc": [0.4, 0.2],
                },
            ]
            gap_file = Path(tmp_dir) / "gap_plan.json"
            gap_file.write_text(json.dumps(gap_plan), encoding="utf-8")

            cmd_gap = [
                sys.executable,
                "-m",
                "aimusic.app.cli",
                "generate",
                "--sections-file",
                str(gap_file),
                "--beats",
                "12",
            ]
            res_gap = subprocess.run(cmd_gap, capture_output=True, text=True)
            self.assertNotEqual(res_gap.returncode, 0)
            combined_output = res_gap.stdout + res_gap.stderr
            self.assertIn("SectionPlanError", combined_output)
            self.assertIn("Gap detected", combined_output)

        self.logger.gate(
            "CLI_INTEGRATION",
            passed=True,
            detail="CLI produced summary table, manifest with per-transition diags, and exited with SectionPlanError on gap.",
        )

    def test_5_auditable_score_table(self) -> None:
        """Log first ten transitions with columns k, section, tau*, tau_hat, deviation, data, gttm, -lt*E_tau, -ls*E_style, final."""
        self.logger.case("test_5_auditable_score_table")
        weights = PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5)
        run_res = run_fixture(TWO_SECTION_16, seed=11, weights=weights)
        diags = run_res.diagnostics.transition_diagnostics
        self.assertGreaterEqual(len(diags), 10)

        headers = [
            "k",
            "section",
            "tau*",
            "tau_hat",
            "deviation",
            "data",
            "gttm",
            "-lt*E_tau",
            "-ls*E_style",
            "final",
        ]

        rows = []
        for d in diags[:10]:
            k = d.arrival_time
            sec = d.section_name
            tau_star = f"{d.target_tension:.4f}"
            tau_hat = f"{d.realized_tension:.4f}"
            dev = f"{d.tension_deviation:.4f}"
            data_score = f"{d.data_contribution:.4f}"
            gttm_score = f"{d.gttm_contribution:.4f}"
            tension_term = f"{-d.target_tension_contribution:.4f}"
            style_term = f"{-d.section_style_contribution:.4f}"
            final_score = f"{d.final_log_weight:.4f}"
            rows.append([
                k,
                sec,
                tau_star,
                tau_hat,
                dev,
                data_score,
                gttm_score,
                tension_term,
                style_term,
                final_score,
            ])

        self.logger.table("Auditable Transition Score Decomposition (First 10 Transitions)", headers, rows)

        # Print for immediate terminal inspection
        print("\n=== Auditable Score Table (First 10 Transitions) ===")
        hdr_str = " | ".join(f"{h:<11}" for h in headers)
        print(hdr_str)
        print("-" * len(hdr_str))
        for row in rows:
            print(" | ".join(f"{str(cell):<11}" for cell in row))
        print("=" * len(hdr_str) + "\n")

        self.logger.gate(
            "AUDITABLE_SCORE_TABLE",
            passed=True,
            detail=f"Logged {len(rows)} transitions with exact decomposition breakdown.",
        )


if __name__ == "__main__":
    unittest.main()
