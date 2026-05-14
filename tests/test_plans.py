import unittest

from aimusic.core.config import PlanConfig, SectioningStrategy, StyleConfig
from aimusic.planning.plans import (
    MethodARunConfig,
    build_section_plan,
    generate_end_endpoint_distribution,
    generate_method_a_endpoints,
    generate_start_endpoint_distribution,
    run_method_a,
)


class TestMethodAEndpointPlanning(unittest.TestCase):
    def test_endpoint_generation_is_reproducible_and_aligned(self):
        run_config = MethodARunConfig(total_beats=4, seed=7)

        pi0 = generate_start_endpoint_distribution(run_config)
        piT = generate_end_endpoint_distribution(run_config)

        self.assertEqual(pi0.layer.time_index, 0)
        self.assertEqual(piT.layer.time_index, 4)
        self.assertAlmostEqual(sum(pi0.probabilities), 1.0)
        self.assertAlmostEqual(sum(piT.probabilities), 1.0)
        self.assertEqual(
            generate_start_endpoint_distribution(run_config),
            pi0,
        )
        self.assertEqual(
            generate_end_endpoint_distribution(run_config),
            piT,
        )

    def test_section_plan_supports_single_and_section_wise_modes(self):
        single_run = MethodARunConfig(total_beats=8)
        section_run = MethodARunConfig(
            total_beats=8,
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("intro", "outro"),
            ),
        )

        single_sections = build_section_plan(single_run)
        section_wise_sections = build_section_plan(section_run)

        self.assertEqual(len(single_sections), 1)
        self.assertEqual(single_sections[0].start_time, 0)
        self.assertEqual(single_sections[0].end_time, 8)
        self.assertEqual([section.name for section in section_wise_sections], ["intro", "outro"])
        self.assertEqual(section_wise_sections[-1].end_time, 8)

    def test_section_wise_rejects_more_sections_than_beats(self):
        with self.assertRaises(ValueError):
            MethodARunConfig(
                total_beats=1,
                plan_config=PlanConfig(
                    sectioning_strategy=SectioningStrategy.SECTION_WISE,
                    section_names=("intro", "outro"),
                ),
            )


class TestMethodAOrchestration(unittest.TestCase):
    def test_generate_method_a_endpoints_returns_sections(self):
        run_config = MethodARunConfig(total_beats=4)

        endpoints = generate_method_a_endpoints(run_config)

        self.assertEqual(endpoints.pi0.layer.time_index, 0)
        self.assertEqual(endpoints.piT.layer.time_index, 4)
        self.assertEqual(len(endpoints.sections), 1)
        self.assertIn(endpoints.start_choice.state, endpoints.pi0.layer.states)
        self.assertIn(endpoints.end_choice.state, endpoints.piT.layer.states)

    def test_run_method_a_map_smoke(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=11,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )

        result = run_method_a(run_config)

        self.assertEqual(len(result.path), run_config.total_beats + 1)
        self.assertEqual(result.path[0], result.diagnostics.chosen_start_state)
        self.assertEqual(result.path[-1], result.diagnostics.chosen_end_state)
        self.assertEqual(len(result.graph.layers[0].states), 1)
        self.assertEqual(len(result.graph.layers[-1].states), 1)
        self.assertEqual(result.endpoints.pi0.layer.time_index, 0)
        self.assertEqual(result.endpoints.start_choice.selection_mode, "argmax")
        self.assertEqual(result.diagnostics.endpoint_selection_mode, "argmax")
        self.assertGreater(result.diagnostics.chosen_start_probability, 0.0)
        self.assertGreater(result.diagnostics.chosen_end_probability, 0.0)
        self.assertEqual(result.diagnostics.path_mode, "map")
        self.assertIsNotNone(result.path_score)
        self.assertTrue(result.sb_solution.trace.converged)

    def test_run_method_a_sampling_is_seed_reproducible(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=23,
            use_sampling=True,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )

        first = run_method_a(run_config)
        second = run_method_a(run_config)

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.sampled_path, second.sampled_path)
        self.assertEqual(first.diagnostics.path_mode, "sample")
        self.assertEqual(first.diagnostics.endpoint_selection_mode, "sample")
        self.assertEqual(first.endpoints.start_choice, second.endpoints.start_choice)
        self.assertEqual(first.endpoints.end_choice, second.endpoints.end_choice)

    def test_run_method_a_section_wise_smoke(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=5,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("intro", "outro"),
            ),
        )

        result = run_method_a(run_config)

        self.assertEqual(result.diagnostics.section_tags, ("intro", "outro"))
        self.assertEqual(len(result.endpoints.sections), 2)
        self.assertEqual(result.endpoints.sections[-1].end_time, 4)


if __name__ == "__main__":
    unittest.main()
