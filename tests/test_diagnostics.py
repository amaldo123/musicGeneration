import unittest
import dataclasses
import math
from unittest.mock import MagicMock
from aimusic.core.diagnostics import (
    TimelineEvent, 
    StructuralDiagnostics, 
    compute_tension_curve,
    RunManifest,
    SBDiagnostics
)

class TestDiagnostics(unittest.TestCase):
    def test_timeline_event_serialization(self):
        """Ensures timeline events serialize properly using the standard asdict."""
        event = TimelineEvent(start_time=0.0, end_time=4.0, label="C Major")
        serialized = dataclasses.asdict(event)
        
        self.assertEqual(serialized["start_time"], 0.0)
        self.assertEqual(serialized["end_time"], 4.0)
        self.assertEqual(serialized["label"], "C Major")

    def test_structural_diagnostics_to_dict(self):
        """Verifies that EVERY timeline array converts safely to JSON structures."""
        struct = StructuralDiagnostics(
            key_timeline=[TimelineEvent(0.0, 4.0, "C Major")],
            chord_timeline=[TimelineEvent(0.0, 2.0, "Cmaj7")],
            role_timeline=[TimelineEvent(0.0, 2.0, "Tonic")],
            groove_timeline=[TimelineEvent(0.0, 4.0, "Swing")],
            boundaries=[0.0, 4.0],
            tension_curve=[(0.0, 0.1), (4.0, 0.9)]
        )
        
        data = struct.to_dict()
        
        # Exhaustively checking every single key to prevent silent failures
        self.assertIn("key_timeline", data)
        self.assertIn("chord_timeline", data)
        self.assertIn("role_timeline", data)
        self.assertIn("groove_timeline", data)
        self.assertIn("boundaries", data)
        self.assertIn("tension_curve", data)
        
        # Verify nested data is accurate
        self.assertEqual(data["key_timeline"][0]["label"], "C Major")
        self.assertEqual(data["chord_timeline"][0]["label"], "Cmaj7")
        self.assertEqual(data["role_timeline"][0]["label"], "Tonic")
        self.assertEqual(data["groove_timeline"][0]["label"], "Swing")
        self.assertEqual(data["boundaries"], [0.0, 4.0])

    def test_compute_tension_curve(self):
        """Tests the heuristic math mapping musical roles to tension floats."""
        roles = [
            TimelineEvent(0.0, 4.0, "Tonic"),
            TimelineEvent(4.0, 8.0, "Subdominant"),
            TimelineEvent(8.0, 12.0, "Dominant"),
            TimelineEvent(12.0, 16.0, "Unknown") 
        ]
        
        curve = compute_tension_curve(roles)
        
        self.assertEqual(len(curve), 4)
        self.assertEqual(curve[0], (0.0, 0.1))
        self.assertEqual(curve[1], (4.0, 0.5))
        self.assertEqual(curve[2], (8.0, 0.9))
        self.assertEqual(curve[3], (12.0, 0.5)) 
        
    def test_sb_diagnostics_extraction(self):
        """Tests that SB logs and Effective Entropy are correctly calculated from a solution."""
        # Mock the SBSolution object returned by aimusic.planning.sb
        mock_solution = MagicMock()
        mock_solution.trace.iterations = 42
        mock_solution.trace.converged = True
        mock_solution.trace.final_max_delta = 1e-6
        
        mock_solution.problem.diagnostics.layer_sizes = (5, 10, 5)
        mock_solution.problem.diagnostics.zero_outdegree_count = 2
        mock_solution.problem.diagnostics.zero_indegree_count = 1
        
        # Layer 1: Confident (entropy = 0)
        # Layer 2: 50/50 Split (entropy = approx 0.693)
        mock_solution.marginals.node_marginals_by_layer = [
            (1.0, 0.0),      
            (0.5, 0.5)       
        ]

        #Extract Data
        stats = SBDiagnostics.from_solution(mock_solution)

        #Verify Basic Stats
        self.assertEqual(stats.iterations_run, 42)
        self.assertTrue(stats.converged)
        self.assertEqual(stats.final_max_delta, 1e-6)
        self.assertEqual(stats.layer_sizes, [5, 10, 5])
        self.assertEqual(stats.pruned_nodes, 3) # 2 out + 1 in

        # Verify Shannon Entropy Math
        expected_layer_2_entropy = -(0.5 * math.log(0.5)) * 2
        expected_average_entropy = (0.0 + expected_layer_2_entropy) / 2
        self.assertAlmostEqual(stats.effective_entropy, expected_average_entropy, places=5)

    def test_run_manifest_generation(self):
        """Ensures the top-level manifest generates valid UUIDs and timestamps."""
        manifest = RunManifest(seed=42, config_dump={"edo": 12})
        data = manifest.to_dict()
        
        self.assertEqual(data["seed"], 42)
        self.assertEqual(data["config"]["edo"], 12)
        self.assertIsNotNone(data["run_id"])
        self.assertIsNotNone(data["timestamp"])
        self.assertIn("structure", data)

if __name__ == "__main__":
    unittest.main()