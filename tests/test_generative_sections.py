import unittest
from aimusic.core.config import (
    DecodeConfig,
    PriorWeights,
    SBConfig,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, Layer
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.candidates import get_valid_next_states
from aimusic.planning.graph import build_sparse_graph
from aimusic.planning.plans import (
    MethodARunConfig,
    PlanningSection,
    TransitionDiagnostic,
    get_section_at_time,
    run_method_a,
)
from aimusic.scoring.priors import (
    NullPrior,
    PriorContext,
    PriorQuery,
    calculate_transition_log_weight,
    calculate_transition_log_weights,
)
from aimusic.scoring.tension import (
    beat_tension,
    compare_tension_curves,
    realized_tension_curve,
    section_style_energy,
    target_tension_at_time,
    target_tension_curve,
    transition_target_tension_energy,
)


class TestGenerativeSections(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabs = DEFAULT_VOCABULARIES
        self.key = RNGKey(seed=42)

    def test_planning_section_extension_and_helpers(self) -> None:
        section1 = PlanningSection(
            name="Verse",
            start_time=0,
            end_time=4,
            boundary_level=2,
            target_tension_arc=(0.2, 0.8),
            allowed_meters=("4/4",),
            groove_family="syncopated",
            target_key_id=0,
            target_density=0.6,
            preferred_roles=("prep", "change"),
        )
        section2 = PlanningSection(
            name="Chorus",
            start_time=4,
            end_time=8,
            boundary_level=3,
            target_tension_arc=(0.8, 0.3),
        )
        sections = (section1, section2)

        self.assertEqual(get_section_at_time(sections, 2), section1)
        self.assertEqual(get_section_at_time(sections, 5), section2)
        self.assertIsNone(get_section_at_time(sections, 10))

        # Target tension interpolation
        t0_val = target_tension_at_time(sections, 0)
        self.assertAlmostEqual(t0_val, 0.2, places=4)

        t2_val = target_tension_at_time(sections, 2)
        self.assertAlmostEqual(t2_val, 0.5, places=4)

    def test_custom_section_plan_supplied_to_method_a(self) -> None:
        """Custom section plan supplied to MethodARunConfig reaches Method A and endpoints."""
        custom_sections = (
            PlanningSection(
                name="Intro",
                start_time=0,
                end_time=4,
                boundary_level=2,
                target_tension_arc=(0.1, 0.4),
                allowed_meters=("4/4",),
                groove_family="straight",
                target_key_id=0,
                target_density=0.3,
            ),
            PlanningSection(
                name="Climax",
                start_time=4,
                end_time=8,
                boundary_level=3,
                target_tension_arc=(0.7, 0.95),
                allowed_meters=("4/4",),
                groove_family="syncopated",
                target_key_id=0,
                target_density=0.9,
            ),
        )
        cfg = MethodARunConfig(
            total_beats=8,
            seed=42,
            sections=custom_sections,
            prior_weights=PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5),
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )
        res, _ = run_method_a(cfg, key=self.key)
        self.assertEqual(res.endpoints.sections, custom_sections)
        self.assertEqual(res.diagnostics.section_tags, ("Intro", "Climax"))
        self.assertEqual(res.diagnostics.target_tension_arcs, ((0.1, 0.4), (0.7, 0.95)))

    def test_baseline_reproduction_when_lambdas_zero(self) -> None:
        """Disabling conditioning (lambda_target_tension=0, lambda_section_style=0) reproduces baseline graph/path exactly."""
        w_zero = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)

        cfg_baseline = MethodARunConfig(
            total_beats=8,
            seed=42,
            prior_weights=w_zero,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )
        res_baseline, _ = run_method_a(cfg_baseline, key=self.key)

        custom_sections = (
            PlanningSection(
                name="Intro",
                start_time=0,
                end_time=4,
                boundary_level=3,
                target_tension_arc=(0.1, 0.9),
                groove_family="syncopated",
            ),
            PlanningSection(
                name="Outro",
                start_time=4,
                end_time=8,
                boundary_level=3,
                target_tension_arc=(0.9, 0.1),
                groove_family="straight",
            ),
        )
        cfg_custom_zero = MethodARunConfig(
            total_beats=8,
            seed=42,
            sections=custom_sections,
            prior_weights=w_zero,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )
        res_custom_zero, _ = run_method_a(cfg_custom_zero, key=self.key)

        self.assertEqual(res_baseline.graph.diagnostics.layer_sizes, res_custom_zero.graph.diagnostics.layer_sizes)
        self.assertEqual(res_baseline.path, res_custom_zero.path)

    def test_target_arc_only_change_alters_path(self) -> None:
        """Changing target_tension_arc while holding seed identical alters path and tension profile."""
        sec_low = (
            PlanningSection(name="Low", start_time=0, end_time=8, boundary_level=3, target_tension_arc=(0.1, 0.2)),
        )
        sec_high = (
            PlanningSection(name="High", start_time=0, end_time=8, boundary_level=3, target_tension_arc=(0.85, 0.95)),
        )
        w_tension = PriorWeights(lambda_target_tension=3.0, lambda_section_style=0.0)

        cfg_low = MethodARunConfig(
            total_beats=8, seed=42, sections=sec_low, prior_weights=w_tension,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4)
        )
        cfg_high = MethodARunConfig(
            total_beats=8, seed=42, sections=sec_high, prior_weights=w_tension,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4)
        )

        res_low, _ = run_method_a(cfg_low, key=self.key)
        res_high, _ = run_method_a(cfg_high, key=self.key)

        self.assertNotEqual(res_low.path, res_high.path)
        curve_low = realized_tension_curve(res_low.path, self.vocabs, 12)
        curve_high = realized_tension_curve(res_high.path, self.vocabs, 12)
        mean_low = sum(val for _, val in curve_low) / len(curve_low)
        mean_high = sum(val for _, val in curve_high) / len(curve_high)
        self.assertGreater(mean_high, mean_low)

    def test_style_preference_only_change_alters_path(self) -> None:
        """Changing style preferences while holding seed identical alters selected path."""
        sec_straight = (
            PlanningSection(name="Straight", start_time=0, end_time=8, boundary_level=3, groove_family="straight"),
        )
        sec_synco = (
            PlanningSection(name="Synco", start_time=0, end_time=8, boundary_level=3, groove_family="syncopated"),
        )
        w_style = PriorWeights(lambda_target_tension=0.0, lambda_section_style=3.0)

        cfg_straight = MethodARunConfig(
            total_beats=8, seed=42, sections=sec_straight, prior_weights=w_style,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4)
        )
        cfg_synco = MethodARunConfig(
            total_beats=8, seed=42, sections=sec_synco, prior_weights=w_style,
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4)
        )

        res_straight, _ = run_method_a(cfg_straight, key=self.key)
        res_synco, _ = run_method_a(cfg_synco, key=self.key)

        self.assertNotEqual(res_straight.path, res_synco.path)

    def test_section_boundary_alignment(self) -> None:
        """Section boundary level is specific to section boundaries, not repeated on every downbeat."""
        sec = PlanningSection(name="A", start_time=0, end_time=8, boundary_level=3)
        b_levels = []
        for beat_in_bar in range(4):
            proposals = get_valid_next_states(
                BeatState(0, (beat_in_bar - 1) % 4, 0, 0, 0, 0, 0, 0),
                t=beat_in_bar,
                key=self.key,
                d_max=8,
                vocabularies=self.vocabs,
                section=sec,
                section_guided_proposals=True,
            )[0]
            b_levels.append(max(s.boundary_lvl for s in proposals.states))
        self.assertEqual(b_levels[0], 3)  # Section boundary at start
        self.assertLess(b_levels[1], 3)   # Non-boundary beats inside section

    def test_tension_tracking_error_reduction(self) -> None:
        """Enabling target tension reduces MAE compared to unconditioned generation."""
        sections = (
            PlanningSection(name="A", start_time=0, end_time=4, boundary_level=2, target_tension_arc=(0.1, 0.3)),
            PlanningSection(name="B", start_time=4, end_time=8, boundary_level=3, target_tension_arc=(0.8, 0.9)),
        )
        target = target_tension_curve(sections)

        cfg_off = MethodARunConfig(
            total_beats=8, seed=42, sections=sections,
            prior_weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )
        cfg_on = MethodARunConfig(
            total_beats=8, seed=42, sections=sections,
            prior_weights=PriorWeights(lambda_target_tension=6.0, lambda_section_style=0.0),
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )

        res_off, _ = run_method_a(cfg_off, key=self.key)
        res_on, _ = run_method_a(cfg_on, key=self.key)

        rep_off = compare_tension_curves(target, realized_tension_curve(res_off.path, self.vocabs, 12), sections)
        rep_on = compare_tension_curves(target, realized_tension_curve(res_on.path, self.vocabs, 12), sections)

        self.assertLess(rep_on.mean_absolute_error, rep_off.mean_absolute_error)

    def test_feature_ablation_variants(self) -> None:
        """Verify feature ablation variants produce distinct paths and feature contributions."""
        weights_variants = [
            PriorWeights(lambda_target_tension=2.0, lambda_section_style=2.0),
            PriorWeights(lambda_target_tension=2.0, lambda_section_style=0.0),
            PriorWeights(lambda_target_tension=0.0, lambda_section_style=2.0),
            PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
        ]
        paths = []
        for w in weights_variants:
            cfg = MethodARunConfig(
                total_beats=8, seed=42, prior_weights=w,
                sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
            )
            res, _ = run_method_a(cfg, key=self.key)
            paths.append(res.path)

        self.assertEqual(len(paths), 4)
        # Paths with different conditioning weights should not all be identical
        unique_paths = set(paths)
        self.assertGreater(len(unique_paths), 1)

    def test_transition_diagnostics_completeness(self) -> None:
        """Transition diagnostics record target tension, realized tension, deviation, and penalties."""
        cfg = MethodARunConfig(
            total_beats=8, seed=42,
            prior_weights=PriorWeights(lambda_target_tension=2.0, lambda_section_style=1.5),
            sb_config=SBConfig(horizon_t=8, k_max=16, d_max=4),
        )
        res, _ = run_method_a(cfg, key=self.key)
        diags = res.diagnostics.transition_diagnostics

        self.assertEqual(len(diags), 8)
        for d in diags:
            self.assertIsInstance(d, TransitionDiagnostic)
            self.assertIsInstance(d.time_index, int)
            self.assertIsInstance(d.target_tension, float)
            self.assertIsInstance(d.realized_tension, float)
            self.assertAlmostEqual(d.tension_deviation, d.realized_tension - d.target_tension, places=5)
            self.assertGreaterEqual(d.target_tension_penalty, 0.0)
            self.assertGreaterEqual(d.section_style_penalty, 0.0)


if __name__ == "__main__":
    unittest.main()


