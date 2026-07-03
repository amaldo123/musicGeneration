import math
import unittest

from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.count_prior import CountPriorConfig, CountPriorModel, train_counts
from aimusic.ml.dataset import build_transition_examples, examples_to_tokenized
from aimusic.core.core_types import BeatState
from aimusic.scoring.priors import StructuralEventTokens, TokenizedPriorQuery
from tests.conftest import requires_jax


def _state(chord_label: str) -> BeatState:
    return BeatState(
        meter_id=DEFAULT_VOCABULARIES.meters.token_for_label("4/4").id,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=DEFAULT_VOCABULARIES.keys.token_for_label("C").id,
        chord_id=DEFAULT_VOCABULARIES.chords.token_for_label(chord_label).id,
        role_id=DEFAULT_VOCABULARIES.roles.token_for_label("hold").id,
        head_id=DEFAULT_VOCABULARIES.heads.token_for_label("root").id,
        groove_id=DEFAULT_VOCABULARIES.grooves.token_for_label("straight_8ths").id,
    )


@requires_jax
class TestCountPrior(unittest.TestCase):
    def test_repeated_transition_scores_highest(self):
        states = (
            _state("Cmaj"),
            _state("G7"),
            _state("Cmaj"),
            _state("G7"),
            _state("Cmaj"),
            _state("G7"),
        )
        tokenized = examples_to_tokenized(build_transition_examples(states))
        state = train_counts(tokenized, DEFAULT_VOCABULARIES, alpha=1.0)
        model = CountPriorModel(
            state=state,
            vocabularies=DEFAULT_VOCABULARIES,
            config=CountPriorConfig(),
        )

        cmaj = _state("Cmaj")
        g7 = _state("G7")
        frequent = model.score_transition(
            TokenizedPriorQuery(
                prev_event=StructuralEventTokens.from_state(cmaj),
                next_event=StructuralEventTokens.from_state(g7),
                time_index=0,
            )
        )
        rare = model.score_transition(
            TokenizedPriorQuery(
                prev_event=StructuralEventTokens.from_state(g7),
                next_event=StructuralEventTokens.from_state(_state("Dmaj")),
                time_index=0,
            )
        )
        self.assertGreater(frequent, rare)

    def test_scalar_and_batch_scores_match(self):
        states = (_state("Cmaj"), _state("G7"), _state("Cmaj"))
        tokenized = examples_to_tokenized(build_transition_examples(states))
        state = train_counts(tokenized, DEFAULT_VOCABULARIES)
        model = CountPriorModel(
            state=state,
            vocabularies=DEFAULT_VOCABULARIES,
            config=CountPriorConfig(),
        )

        scalar = tuple(model.score_transition(query) for query in tokenized)
        batched = model.score_transition_batch(tokenized)
        self.assertEqual(len(scalar), len(batched))
        for left, right in zip(scalar, batched):
            self.assertAlmostEqual(left, right, places=5)
            self.assertTrue(math.isfinite(left))


if __name__ == "__main__":
    unittest.main()
