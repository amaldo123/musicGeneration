from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aimusic.core.config import PriorFactorization, StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.bundle import save_prior_bundle
from aimusic.ml.count_prior import CountPriorConfig, CountPriorModel, train_counts
from aimusic.ml.dataset import examples_to_tokenized, load_corpus_transitions
from aimusic.scoring.priors import NeuralPriorManifest


@dataclass(frozen=True)
class TrainResult:
    bundle_dir: Path
    transition_count: int
    midi_files_processed: int


def train_prior_from_corpus(
    midi_dir: str | Path,
    output_dir: str | Path,
    *,
    style_config: Optional[StyleConfig] = None,
    edo: int = 12,
    alpha: float = 1.0,
    model_version: str = "v1",
) -> TrainResult:
    """Train a JAX count prior from a MIDI corpus and write an artifact bundle."""
    resolved_style = StyleConfig() if style_config is None else style_config
    vocabularies = build_default_vocabularies(resolved_style)

    queries = load_corpus_transitions(midi_dir, resolved_style, edo=edo)
    tokenized = examples_to_tokenized(queries, PriorFactorization.FACTORIZED)
    count_state = train_counts(tokenized, vocabularies, alpha=alpha).to_dict()

    root = Path(midi_dir)
    midi_files = list(root.glob("**/*.mid")) + list(root.glob("**/*.midi"))

    manifest = NeuralPriorManifest(
        model_family="jax_count_prior",
        model_version=model_version,
        factorization_mode=PriorFactorization.FACTORIZED,
        supports_batch_scoring=True,
        expected_edo=edo,
    )
    bundle_path = save_prior_bundle(
        output_dir,
        manifest=manifest,
        vocabularies=vocabularies,
        style_config=resolved_style,
        count_state=count_state,
    )
    return TrainResult(
        bundle_dir=bundle_path,
        transition_count=len(queries),
        midi_files_processed=len(midi_files),
    )
