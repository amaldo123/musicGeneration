from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from aimusic.core.config import PriorFactorization, StyleConfig
from aimusic.core.vocab import build_default_vocabularies
from aimusic.ml.bundle import save_prior_bundle
from aimusic.ml.count_prior import train_counts
from aimusic.ml.dataset import examples_to_tokenized, load_corpus_transitions
from aimusic.ml.monitoring import (
    ClearMLConfig,
    TrainingMonitor,
    build_training_run_stats,
    collect_corpus_stats,
    create_training_monitor,
)
from aimusic.scoring.priors import NeuralPriorManifest

logger = logging.getLogger(__name__)


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
    monitor: TrainingMonitor | None = None,
    clearml: ClearMLConfig | None = None,
    quiet: bool = False,
) -> TrainResult:
    """Train a JAX count prior from a MIDI corpus and write an artifact bundle."""
    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_monitor = monitor if monitor is not None else create_training_monitor(clearml)

    resolved_monitor.on_run_start(
        midi_dir=Path(midi_dir),
        output_dir=Path(output_dir),
        style_config=resolved_style,
        edo=edo,
        alpha=alpha,
        model_version=model_version,
    )

    t_total = time.monotonic()
    try:
        t0 = time.monotonic()
        vocabularies = build_default_vocabularies(resolved_style)
        logger.info(
            "[stage 1/5] vocab built in %.2fs | meters=%d keys=%d roles=%d grooves=%d",
            time.monotonic() - t0,
            len(vocabularies.meters),
            len(vocabularies.keys),
            len(vocabularies.roles),
            len(vocabularies.grooves),
        )

        t1 = time.monotonic()
        queries = load_corpus_transitions(
            midi_dir,
            resolved_style,
            edo=edo,
            progress=not quiet,
        )
        logger.info(
            "[stage 2/5] corpus parsed: %d transitions in %.2fs",
            len(queries),
            time.monotonic() - t1,
        )

        t2 = time.monotonic()
        tokenized = examples_to_tokenized(queries, PriorFactorization.FACTORIZED)
        logger.info(
            "[stage 3/5] tokenized %d queries in %.2fs",
            len(tokenized),
            time.monotonic() - t2,
        )

        t3 = time.monotonic()
        count_state = train_counts(tokenized, vocabularies, alpha=alpha).to_dict()
        logger.info(
            "[stage 4/5] trained count tables (8 streams) on %d devices in %.2fs",
            1,
            time.monotonic() - t3,
        )

        root = Path(midi_dir)
        midi_files = list(root.glob("**/*.mid")) + list(root.glob("**/*.midi"))
        t4 = time.monotonic()
        corpus_files = collect_corpus_stats(
            midi_dir,
            resolved_style,
            vocabularies,
            edo=edo,
        )
        logger.info(
            "[stage 5a/5] corpus stats for %d files in %.2fs",
            len(corpus_files),
            time.monotonic() - t4,
        )

        manifest = NeuralPriorManifest(
            model_family="jax_count_prior",
            model_version=model_version,
            factorization_mode=PriorFactorization.FACTORIZED,
            supports_batch_scoring=True,
            expected_edo=edo,
        )
        t5 = time.monotonic()
        bundle_path = save_prior_bundle(
            output_dir,
            manifest=manifest,
            vocabularies=vocabularies,
            style_config=resolved_style,
            count_state=count_state,
        )
        logger.info(
            "[stage 5b/5] saved bundle to %s in %.2fs",
            bundle_path,
            time.monotonic() - t5,
        )

        result = TrainResult(
            bundle_dir=bundle_path,
            transition_count=len(queries),
            midi_files_processed=len(midi_files),
        )
        resolved_monitor.on_run_complete(
            bundle_dir=bundle_path,
            stats=build_training_run_stats(
                transition_count=result.transition_count,
                midi_files_processed=result.midi_files_processed,
                corpus_files=corpus_files,
                vocabularies=vocabularies,
                count_state=count_state,
            ),
        )
        logger.info(
            "TOTAL training time: %.2fs | %d transitions from %d MIDI files",
            time.monotonic() - t_total,
            result.transition_count,
            result.midi_files_processed,
        )
        return result
    finally:
        resolved_monitor.close()
