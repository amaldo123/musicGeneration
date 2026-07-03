import math
import tempfile
import unittest
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.dataset import (
    build_transition_examples,
    examples_to_tokenized,
    load_corpus_transitions,
    pack_tokenized_batch,
    vocab_sizes_from_vocabularies,
)
from aimusic.scoring.priors import PriorContext, PriorQuery
from tests.conftest import requires_jax
from tests.test_midi_ingest import write_simple_c_g_progression


def _state(chord_label: str, beat: int = 0) -> BeatState:
    return BeatState(
        meter_id=DEFAULT_VOCABULARIES.meters.token_for_label("4/4").id,
        beat_in_bar=beat,
        boundary_lvl=0,
        key_id=DEFAULT_VOCABULARIES.keys.token_for_label("C").id,
        chord_id=DEFAULT_VOCABULARIES.chords.token_for_label(chord_label).id,
        role_id=DEFAULT_VOCABULARIES.roles.token_for_label("hold").id,
        head_id=DEFAULT_VOCABULARIES.heads.token_for_label("root").id,
        groove_id=DEFAULT_VOCABULARIES.grooves.token_for_label("straight_8ths").id,
    )


class TestDataset(unittest.TestCase):
    def test_build_transition_examples_mirrors_method_a_context(self):
        states = (_state("Cmaj", 0), _state("G7", 1), _state("Cmaj", 0))
        queries = build_transition_examples(states)

        self.assertEqual(len(queries), 2)
        self.assertEqual(len(queries[0].context.history), 1)
        self.assertEqual(len(queries[0].context.future_hints), 1)
        self.assertEqual(queries[0].context.metadata, (("graph_time", "0"),))

    def test_load_corpus_transitions_from_fixture_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_simple_c_g_progression(Path(tmp_dir) / "piece.mid")
            queries = load_corpus_transitions(tmp_dir, StyleConfig(allowed_meters=("4/4",)))

        self.assertGreater(len(queries), 0)


@requires_jax
class TestDatasetJAX(unittest.TestCase):
    def test_pack_tokenized_batch_shapes(self):
        states = (_state("Cmaj", 0), _state("G7", 1), _state("Cmaj", 0))
        queries = build_transition_examples(states)
        tokenized = examples_to_tokenized(queries)
        sizes = vocab_sizes_from_vocabularies(DEFAULT_VOCABULARIES)
        packed = pack_tokenized_batch(tokenized, sizes)

        self.assertEqual(int(packed["batch_size"]), len(tokenized))
        self.assertEqual(tuple(packed["prev_chord"].shape), (len(tokenized),))


if __name__ == "__main__":
    unittest.main()
