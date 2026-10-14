"""test_prior_section_tokens.py — Tests for REQ-04 Step 8: Section Context for Priors.

Verifies:
1. Arrival convention: Edge t scores arrival state t+1. Edge 7 in TWO_SECTION_16 maps to Section B with is_section_start=True.
2. Graph propagation: A stub neural prior receives expected SectionTokens during build_sparse_graph; logs example payload.
3. Manifest compatibility: Manifests with and without 'section' stream load successfully; unknown streams are rejected.
4. Behavior preservation: Graph weights and transition scores are identical with and without section tokens for NullPrior and placeholder NeuralPrior.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Sequence, Tuple
import unittest

from aimusic.core.config import (
    NeuralPriorConfig,
    PlaceholderPriorMode,
    PriorWeights,
    SBConfig,
    StyleConfig,
)
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.graph import Layer, build_sparse_graph
from aimusic.planning.sections import PlanningSection, SectionTimeline
from aimusic.scoring.priors import (
    STRUCTURAL_STREAM_NAMES,
    SUPPORTED_STREAM_NAMES,
    NeuralPrior,
    NeuralPriorManifest,
    NullPrior,
    Prior,
    PriorContext,
    PriorQuery,
    SectionTokens,
    TokenizedPriorQuery,
    load_neural_prior_manifest,
    prior_logps,
    save_neural_prior_manifest,
)
from tests.req04_support.fixtures import TWO_SECTION_16
from tests.req04_support.report import ReportLogger


class RecordingPriorModel:
    """Stub NeuralPriorModel recording all tokenized queries received during scoring."""

    def __init__(self) -> None:
        self.recorded_queries: list[TokenizedPriorQuery] = []

    def score_transition(self, query: TokenizedPriorQuery) -> float:
        self.recorded_queries.append(query)
        return 0.0

    def score_transition_batch(
        self,
        queries: Sequence[TokenizedPriorQuery],
    ) -> Tuple[float, ...]:
        self.recorded_queries.extend(queries)
        return tuple(0.0 for _ in queries)


class StripSectionTokensPrior:
    """Prior wrapper that strips section_view and SectionTokens from contexts to test prior invariance."""

    def __init__(self, wrapped: Prior) -> None:
        self.wrapped = wrapped

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: PriorContext | None = None,
    ) -> float:
        stripped_ctx = None
        if context is not None:
            stripped_ctx = PriorContext(
                history=context.history,
                future_hints=context.future_hints,
                section_name=context.section_name,
                section_view=None,
                metadata=context.metadata,
                history_tokens=context.history_tokens,
                future_hint_tokens=context.future_hint_tokens,
            )
        return float(self.wrapped.logp_next(prev_state, next_state, t, stripped_ctx))

    def logp_next_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
        stripped = tuple(
            PriorQuery(
                prev_state=q.prev_state,
                next_state=q.next_state,
                time_index=q.time_index,
                context=(
                    PriorContext(
                        history=q.context.history,
                        future_hints=q.context.future_hints,
                        section_name=q.context.section_name,
                        section_view=None,
                        metadata=q.context.metadata,
                        history_tokens=q.context.history_tokens,
                        future_hint_tokens=q.context.future_hint_tokens,
                    )
                    if q.context is not None
                    else None
                ),
            )
            for q in queries
        )
        return prior_logps(self.wrapped, stripped)


class TestPriorSectionTokens(unittest.TestCase):
    """Test suite for SectionTokens, query tokenization, manifest versioning, and prior invariance."""

    def setUp(self) -> None:
        self.logger = ReportLogger.for_test(self)
        self.vocabs = DEFAULT_VOCABULARIES

    def test_arrival_state_convention(self) -> None:
        """Assert arrival-state convention: edge t scores arrival state t+1 on TWO_SECTION_16."""
        self.logger.case("test_arrival_state_convention")
        timeline = SectionTimeline(
            TWO_SECTION_16.sections,
            total_beats=TWO_SECTION_16.total_beats,
        )

        s_prev = BeatState(0, 2, 0, 0, 0, 0, 1, 0)
        s_next = BeatState(0, 3, 0, 0, 0, 1, 1, 0)

        # Edge 6: arrival time 7 -> Section A (0 to 8), offset 7, is_section_start=False
        view_6 = timeline.edge_view(6)
        query_6 = PriorQuery(s_prev, s_next, time_index=6, context=PriorContext(section_view=view_6))
        tok_6 = query_6.tokenize()
        self.assertIsNotNone(tok_6.section_tokens)
        st_6 = tok_6.section_tokens
        assert st_6 is not None
        self.assertEqual(st_6.section_name, "A")
        self.assertEqual(st_6.section_index, 0)
        self.assertEqual(st_6.offset_in_section, 7)
        self.assertEqual(st_6.section_length, 8)
        self.assertFalse(st_6.is_section_start)

        # Edge 7: arrival time 8 -> Section B (8 to 16), offset 0, is_section_start=True
        view_7 = timeline.edge_view(7)
        query_7 = PriorQuery(s_prev, s_next, time_index=7, context=PriorContext(section_view=view_7))
        tok_7 = query_7.tokenize()
        self.assertIsNotNone(tok_7.section_tokens)
        st_7 = tok_7.section_tokens
        assert st_7 is not None
        self.assertEqual(st_7.section_name, "B")
        self.assertEqual(st_7.section_index, 1)
        self.assertEqual(st_7.offset_in_section, 0)
        self.assertEqual(st_7.section_length, 8)
        self.assertTrue(st_7.is_section_start)

        # Edge 8: arrival time 9 -> Section B (8 to 16), offset 1, is_section_start=False
        view_8 = timeline.edge_view(8)
        query_8 = PriorQuery(s_prev, s_next, time_index=8, context=PriorContext(section_view=view_8))
        tok_8 = query_8.tokenize()
        self.assertIsNotNone(tok_8.section_tokens)
        st_8 = tok_8.section_tokens
        assert st_8 is not None
        self.assertEqual(st_8.section_name, "B")
        self.assertEqual(st_8.section_index, 1)
        self.assertEqual(st_8.offset_in_section, 1)
        self.assertEqual(st_8.section_length, 8)
        self.assertFalse(st_8.is_section_start)

        # Query without section view: section_tokens is None
        query_unsteered = PriorQuery(s_prev, s_next, time_index=7, context=PriorContext())
        tok_unsteered = query_unsteered.tokenize()
        self.assertIsNone(tok_unsteered.section_tokens)

        self.logger.metric("edge_6_section", st_6.section_name, expected="A")
        self.logger.metric("edge_6_offset", st_6.offset_in_section, expected=7)
        self.logger.metric("edge_6_is_start", st_6.is_section_start, expected=False)
        self.logger.metric("edge_7_section", st_7.section_name, expected="B")
        self.logger.metric("edge_7_offset", st_7.offset_in_section, expected=0)
        self.logger.metric("edge_7_is_start", st_7.is_section_start, expected=True)
        self.logger.metric("edge_8_section", st_8.section_name, expected="B")
        self.logger.metric("edge_8_offset", st_8.offset_in_section, expected=1)
        self.logger.metric("edge_8_is_start", st_8.is_section_start, expected=False)

        self.logger.gate(
            "arrival_state_convention_boundary",
            passed=(st_7.section_name == "B" and st_7.is_section_start is True and st_7.offset_in_section == 0),
            detail="Edge 7 correctly maps to arrival beat 8 at Section B start with offset 0",
        )
        self.logger.finish()

    def test_graph_propagation(self) -> None:
        """Assert that a stub prior receives SectionTokens during build_sparse_graph and log payload."""
        self.logger.case("test_graph_propagation")
        recording_model = RecordingPriorModel()
        prior = NeuralPrior(
            manifest=NeuralPriorManifest(
                manifest_version=2,
                token_streams=SUPPORTED_STREAM_NAMES,
                supports_batch_scoring=True,
            ),
            model=recording_model,
        )

        start_state = BeatState(0, 0, 2, 0, 0, 0, 1, 0)
        end_state = BeatState(0, 0, 3, 0, 0, 3, 1, 0)
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=16, states=(end_state,))
        sb = SBConfig(horizon_t=16, k_max=8, d_max=4)
        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))

        graph, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=16,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=prior,
            edo=12,
            key=RNGKey(seed=42),
            d_max=4,
            sections=TWO_SECTION_16.sections,
        )

        self.assertGreater(len(recording_model.recorded_queries), 0)
        all_have_tokens = all(
            q.section_tokens is not None for q in recording_model.recorded_queries
        )
        self.assertTrue(all_have_tokens, "Some queries during build_sparse_graph lacked section_tokens.")

        # Check edge 7 boundary queries arriving at section B
        edge_7_queries = [
            q for q in recording_model.recorded_queries if q.time_index == 7
        ]
        self.assertGreater(len(edge_7_queries), 0)
        all_edge_7_is_start = all(
            q.section_tokens is not None and q.section_tokens.is_section_start is True and q.section_tokens.section_name == "B"
            for q in edge_7_queries
        )
        self.assertTrue(all_edge_7_is_start)

        # Log one example token payload
        example_tokens: SectionTokens = recording_model.recorded_queries[0].section_tokens  # type: ignore[assignment]
        example_payload = example_tokens.to_dict()
        import logging
        logging.getLogger("req04.report").info(
            "Logged Example SectionTokens payload:\n%s",
            json.dumps(example_payload, indent=2),
        )
        self.logger.table(
            "Example SectionTokens Payload",
            headers=["field", "value"],
            rows=[[k, str(v)] for k, v in example_payload.items()],
        )
        self.logger.metric("total_recorded_queries", len(recording_model.recorded_queries))
        self.logger.metric("edge_7_queries_count", len(edge_7_queries))
        self.logger.metric("example_section_name", example_tokens.section_name)
        self.logger.metric("example_target_tension", example_tokens.target_tension)

        self.logger.gate(
            "graph_propagation_receives_section_tokens",
            passed=(all_have_tokens and all_edge_7_is_start),
            detail=f"All {len(recording_model.recorded_queries)} queries in build_sparse_graph carried valid SectionTokens",
        )
        self.logger.finish()

    def test_manifest_compatibility(self) -> None:
        """Verify manifests with and without 'section' stream load successfully and unknown streams are rejected."""
        self.logger.case("test_manifest_compatibility")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Manifest without "section" stream (v1)
            manifest_v1 = NeuralPriorManifest(
                manifest_version=1,
                token_streams=STRUCTURAL_STREAM_NAMES,
                expected_edo=12,
            )
            v1_path = str(tmp_path / "manifest_v1.json")
            save_neural_prior_manifest(manifest_v1, v1_path)
            loaded_v1 = load_neural_prior_manifest(v1_path)

            self.assertEqual(loaded_v1.manifest_version, 1)
            self.assertEqual(loaded_v1.token_streams, STRUCTURAL_STREAM_NAMES)
            self.assertNotIn("section", loaded_v1.token_streams)

            # 2. Manifest with "section" stream (v2)
            manifest_v2 = NeuralPriorManifest(
                manifest_version=2,
                token_streams=SUPPORTED_STREAM_NAMES,
                expected_edo=12,
            )
            v2_path = str(tmp_path / "manifest_v2.json")
            save_neural_prior_manifest(manifest_v2, v2_path)
            loaded_v2 = load_neural_prior_manifest(v2_path)

            self.assertEqual(loaded_v2.manifest_version, 2)
            self.assertEqual(loaded_v2.token_streams, SUPPORTED_STREAM_NAMES)
            self.assertIn("section", loaded_v2.token_streams)

        # 3. Reject unknown stream names
        with self.assertRaises(ValueError):
            NeuralPriorManifest(token_streams=("meter", "unknown_stream"))

        with self.assertRaises(ValueError):
            NeuralPriorManifest.from_dict({"token_streams": ["meter", "invalid_stream"]})

        self.logger.metric("manifest_v1_loaded_version", loaded_v1.manifest_version, expected=1)
        self.logger.metric("manifest_v2_loaded_version", loaded_v2.manifest_version, expected=2)
        self.logger.metric("manifest_v2_has_section", "section" in loaded_v2.token_streams, expected=True)

        self.logger.gate(
            "manifest_compatibility_v1_and_v2",
            passed=("section" not in loaded_v1.token_streams and "section" in loaded_v2.token_streams),
            detail="Manifests with and without 'section' stream load cleanly; unknown streams rejected",
        )
        self.logger.finish()

    def test_behavior_preservation(self) -> None:
        """Verify graph weights are identical with and without section tokens for NullPrior and placeholder NeuralPrior."""
        self.logger.case("test_behavior_preservation")

        sec_4 = (
            PlanningSection(name="Intro", start_time=0, end_time=4, boundary_level=2, target_tension_arc=(0.2, 0.4)),
        )
        timeline_4 = SectionTimeline(sec_4, total_beats=4)
        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        sb = SBConfig(horizon_t=4, k_max=8, d_max=4)
        weights_zero = PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0)

        start_state = BeatState(0, 0, 2, 0, 0, 0, 1, 0)
        end_state = BeatState(0, 0, 2, 0, 0, 3, 1, 0)
        start_layer = Layer(time_index=0, states=(start_state,))
        end_layer = Layer(time_index=4, states=(end_state,))

        view = timeline_4.edge_view(1)
        q_with_tokens = PriorQuery(start_state, end_state, time_index=1, context=PriorContext(section_view=view))
        q_without_tokens = PriorQuery(
            start_state, end_state, time_index=1, context=PriorContext(section_name=view.section.name, section_view=None)
        )

        # Assert tokenization exposes SectionTokens only when section_view is present
        self.assertIsNotNone(q_with_tokens.tokenize().section_tokens)
        self.assertIsNone(q_without_tokens.tokenize().section_tokens)

        # --- A. NullPrior ---
        null_prior = NullPrior(neutral_logp=-0.25)

        # Transition query comparison
        self.assertEqual(
            null_prior.logp_next(q_with_tokens.prev_state, q_with_tokens.next_state, 1, q_with_tokens.context),
            null_prior.logp_next(q_without_tokens.prev_state, q_without_tokens.next_state, 1, q_without_tokens.context),
        )

        # Graph comparison
        g_null_with, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=null_prior,
            weights=weights_zero,
            edo=12,
            key=RNGKey(seed=101),
            d_max=4,
            sections=sec_4,
            section_guided_proposals=False,
        )
        g_null_without, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=StripSectionTokensPrior(null_prior),
            weights=weights_zero,
            edo=12,
            key=RNGKey(seed=101),
            d_max=4,
            sections=sec_4,
            section_guided_proposals=False,
        )
        null_edges_with = [e.log_weight for layer_edges in g_null_with.edges_by_time for e in layer_edges]
        null_edges_without = [e.log_weight for layer_edges in g_null_without.edges_by_time for e in layer_edges]
        self.assertEqual(null_edges_with, null_edges_without)

        # --- B. Placeholder NeuralPrior ---
        neural_placeholder = NeuralPrior(
            config=NeuralPriorConfig(
                placeholder_mode=PlaceholderPriorMode.STRUCTURED,
                default_logp=-0.15,
            )
        )
        # Tokenized queries: placeholder ignores section_tokens
        score_with = neural_placeholder.logp_next(
            q_with_tokens.prev_state, q_with_tokens.next_state, 1, q_with_tokens.context
        )
        score_without = neural_placeholder.logp_next(
            q_without_tokens.prev_state, q_without_tokens.next_state, 1, q_without_tokens.context
        )
        self.assertEqual(score_with, score_without)

        # Graph comparison
        g_neural_with, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=neural_placeholder,
            weights=weights_zero,
            edo=12,
            key=RNGKey(seed=101),
            d_max=4,
            sections=sec_4,
            section_guided_proposals=False,
        )
        g_neural_without, _ = build_sparse_graph(
            start_layer=start_layer,
            end_layer=end_layer,
            total_beats=4,
            sb_config=sb,
            style_config=style,
            vocabularies=self.vocabs,
            prior=StripSectionTokensPrior(neural_placeholder),
            weights=weights_zero,
            edo=12,
            key=RNGKey(seed=101),
            d_max=4,
            sections=sec_4,
            section_guided_proposals=False,
        )
        neural_edges_with = [e.log_weight for layer_edges in g_neural_with.edges_by_time for e in layer_edges]
        neural_edges_without = [e.log_weight for layer_edges in g_neural_without.edges_by_time for e in layer_edges]
        self.assertEqual(neural_edges_with, neural_edges_without)

        self.logger.metric("null_prior_edges_count", len(null_edges_with))
        self.logger.metric("neural_prior_edges_count", len(neural_edges_with))
        self.logger.gate(
            "behavior_preservation_null_and_placeholder",
            passed=(null_edges_with == null_edges_without and neural_edges_with == neural_edges_without),
            detail="Graph edge weights are bit-for-bit identical with and without section tokens for NullPrior and placeholder NeuralPrior",
        )
        self.logger.finish()


if __name__ == "__main__":
    unittest.main()
