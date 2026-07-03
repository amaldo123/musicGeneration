import tempfile
import unittest
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.ml.train import train_prior_from_corpus
from tests.conftest import requires_jax
from tests.test_midi_ingest import write_simple_c_g_progression


@requires_jax
class TestMLTrain(unittest.TestCase):
    def test_train_writes_loadable_bundle(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_dir = Path(tmp_dir) / "corpus"
            midi_dir.mkdir()
            write_simple_c_g_progression(midi_dir / "piece.mid")
            output_dir = Path(tmp_dir) / "bundle"
            result = train_prior_from_corpus(
                midi_dir,
                output_dir,
                style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight", "syncopated", "swing")),
            )

            self.assertTrue((result.bundle_dir / "manifest.json").is_file())
            self.assertTrue((result.bundle_dir / "vocabularies.json").is_file())
            self.assertTrue((result.bundle_dir / "counts").is_dir())
            self.assertGreater(result.transition_count, 0)


if __name__ == "__main__":
    unittest.main()
