from __future__ import annotations

from aimusic.core.config import NeuralPriorConfig
from aimusic.ml.bundle import load_prior_bundle
from aimusic.ml.count_prior import CountPriorConfig, CountPriorModel
from aimusic.scoring.priors import NeuralPrior


def load_trained_neural_prior(bundle_dir: str) -> NeuralPrior:
    """Load a trained prior bundle and wrap it for use in graph planning."""
    bundle = load_prior_bundle(bundle_dir)
    model = CountPriorModel.from_count_state(
        bundle.count_state,
        bundle.vocabularies,
        config=CountPriorConfig(),
    )
    config = NeuralPriorConfig(
        model_family=bundle.manifest.model_family,
        model_version=bundle.manifest.model_version,
        factorization_mode=bundle.manifest.factorization_mode,
        checkpoint_path=bundle.manifest.checkpoint_path,
        tokenizer_path=bundle.manifest.tokenizer_path,
        manifest_path=str(bundle.bundle_dir / "manifest.json"),
        supports_batch_scoring=bundle.manifest.supports_batch_scoring,
    )
    return NeuralPrior(config=config, manifest=bundle.manifest, model=model)
