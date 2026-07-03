import tempfile
import unittest

from aimusic.core.config import PriorFactorization, StyleConfig
from aimusic.scoring.priors import NeuralPriorManifest
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.bundle import load_prior_bundle, save_prior_bundle
from aimusic.ml.count_prior import CountPriorState, train_counts
from aimusic.ml.dataset import build_transition_examples, examples_to_tokenized
from tests.conftest import requires_jax
from tests.test_count_prior import _state


@requires_jax
class TestPriorBundleIO(unittest.TestCase):
    def test_bundle_round_trips_count_state(self):
        import jax.numpy as jnp

        style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        vocabs = build_default_vocabularies(style)
        tokenized = examples_to_tokenized(
            build_transition_examples((_state("Cmaj"), _state("G7"), _state("Cmaj"))),
        )
        count_state = train_counts(tokenized, vocabs, alpha=1.0).to_dict()
        manifest = NeuralPriorManifest(
            model_family="jax_count_prior",
            model_version="test-v1",
            factorization_mode=PriorFactorization.FACTORIZED,
            expected_edo=12,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_prior_bundle(
                tmp_dir,
                manifest=manifest,
                vocabularies=vocabs,
                style_config=style,
                count_state=count_state,
            )
            loaded = load_prior_bundle(tmp_dir)

        self.assertEqual(loaded.manifest.model_family, "jax_count_prior")
        self.assertEqual(loaded.manifest.checkpoint_path, "counts")
        self.assertEqual(loaded.manifest.tokenizer_path, "vocabularies.json")
        self.assertEqual(len(loaded.vocabularies.chords), len(vocabs.chords))

        restored = CountPriorState.from_dict(loaded.count_state)
        original = CountPriorState.from_dict(count_state)
        for stream in restored.tables:
            self.assertTrue(jnp.allclose(restored.tables[stream], original.tables[stream]))
        self.assertTrue(jnp.allclose(restored.alpha, original.alpha))


if __name__ == "__main__":
    unittest.main()
