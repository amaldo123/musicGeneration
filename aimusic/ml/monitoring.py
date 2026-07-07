from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Protocol, Sequence

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import Vocabularies
from aimusic.ml.dataset import build_transition_examples, vocab_sizes_from_vocabularies
from aimusic.ml.midi_ingest import parse_midi_beats
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.scoring.priors import STRUCTURAL_STREAM_NAMES


@dataclass(frozen=True)
class ClearMLConfig:
    """Configuration for optional ClearML experiment tracking."""

    enabled: bool = False
    project_name: str = "aimusic/ml"
    task_name: Optional[str] = None
    tags: tuple[str, ...] = ()
    upload_artifacts: bool = True


@dataclass(frozen=True)
class CorpusFileStats:
    relative_path: str
    annotated_beats: int
    transition_count: int


@dataclass(frozen=True)
class TrainingRunStats:
    transition_count: int
    midi_files_processed: int
    corpus_files: tuple[CorpusFileStats, ...]
    vocab_sizes: dict[str, int]
    stream_stats: dict[str, dict[str, float | int]]


class TrainingMonitor(Protocol):
    """Hook surface for ML training observability backends."""

    def on_run_start(
        self,
        *,
        midi_dir: Path,
        output_dir: Path,
        style_config: StyleConfig,
        edo: int,
        alpha: float,
        model_version: str,
    ) -> None: ...

    def on_run_complete(
        self,
        *,
        bundle_dir: Path,
        stats: TrainingRunStats,
    ) -> None: ...

    def close(self) -> None: ...


class NullTrainingMonitor:
    """No-op monitor used when ClearML is disabled."""

    def on_run_start(self, **kwargs: object) -> None:
        return None

    def on_run_complete(self, **kwargs: object) -> None:
        return None

    def close(self) -> None:
        return None


def collect_corpus_stats(
    midi_dir: str | Path,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
    *,
    edo: int = 12,
) -> tuple[CorpusFileStats, ...]:
    """Summarize per-file annotation yield for monitoring dashboards."""
    root = Path(midi_dir)
    files = sorted(root.glob("**/*.mid")) + sorted(root.glob("**/*.midi"))
    stats: list[CorpusFileStats] = []
    for midi_path in files:
        grid = parse_midi_beats(midi_path)
        states = annotate_beat_grid(grid, vocabularies, edo=edo)
        transitions = build_transition_examples(states)
        try:
            relative_path = str(midi_path.relative_to(root))
        except ValueError:
            relative_path = str(midi_path)
        stats.append(
            CorpusFileStats(
                relative_path=relative_path,
                annotated_beats=len(states),
                transition_count=len(transitions),
            )
        )
    return tuple(stats)


def summarize_count_state(count_state: Mapping[str, object]) -> dict[str, dict[str, float | int]]:
    """Derive per-stream transition-table statistics from trained count tables."""
    import numpy as np

    summary: dict[str, dict[str, float | int]] = {}
    for stream in STRUCTURAL_STREAM_NAMES:
        table = np.asarray(count_state[stream], dtype=np.float64)
        vocab_size = int(table.shape[0])
        total = float(table.sum())
        unique_pairs = int((table > 0).sum())
        capacity = max(vocab_size * vocab_size, 1)
        summary[stream] = {
            "vocab_size": vocab_size,
            "total_transitions": total,
            "unique_prev_next_pairs": unique_pairs,
            "sparsity": 1.0 - (unique_pairs / capacity),
        }
    return summary


def build_training_run_stats(
    *,
    transition_count: int,
    midi_files_processed: int,
    corpus_files: Sequence[CorpusFileStats],
    vocabularies: Vocabularies,
    count_state: Mapping[str, object],
) -> TrainingRunStats:
    return TrainingRunStats(
        transition_count=transition_count,
        midi_files_processed=midi_files_processed,
        corpus_files=tuple(corpus_files),
        vocab_sizes=vocab_sizes_from_vocabularies(vocabularies),
        stream_stats=summarize_count_state(count_state),
    )


def _style_config_payload(style_config: StyleConfig) -> dict[str, object]:
    return {
        "allowed_meters": list(style_config.allowed_meters),
        "subdivision_patterns": list(style_config.subdivision_patterns),
        "groove_families": list(style_config.groove_families),
        "chord_vocabulary_size": style_config.chord_vocabulary_size,
        "key_vocabulary_size": style_config.key_vocabulary_size,
        "bass_register": list(style_config.bass_register),
        "comping_register": list(style_config.comping_register),
        "lead_register": list(style_config.lead_register),
        "typical_density_range": list(style_config.typical_density_range),
    }


class ClearMLTrainingMonitor:
    """ClearML-backed monitor for prior training runs."""

    def __init__(self, config: ClearMLConfig) -> None:
        try:
            from clearml import Task
        except ImportError as exc:
            raise ImportError(
                "ClearML monitoring requires the optional clearml extra.\n"
                "Install with: pip install -e '.[ml,clearml]'"
            ) from exc

        task_name = config.task_name or "jax_count_prior"
        self._task = Task.init(
            project_name=config.project_name,
            task_name=task_name,
            tags=list(config.tags) or None,
            reuse_last_task_id=False,
        )
        self._logger = self._task.get_logger()
        self._upload_artifacts = config.upload_artifacts
        self._closed = False

    @property
    def task_id(self) -> str:
        return self._task.id

    def on_run_start(
        self,
        *,
        midi_dir: Path,
        output_dir: Path,
        style_config: StyleConfig,
        edo: int,
        alpha: float,
        model_version: str,
    ) -> None:
        self._task.connect(
            {
                "midi_dir": str(midi_dir),
                "output_dir": str(output_dir),
                "edo": edo,
                "alpha": alpha,
                "model_version": model_version,
                "model_family": "jax_count_prior",
                "factorization_mode": "factorized",
                "style_config": _style_config_payload(style_config),
            },
            name="training",
        )

    def on_run_complete(
        self,
        *,
        bundle_dir: Path,
        stats: TrainingRunStats,
    ) -> None:
        self._logger.report_single_value("transition_count", float(stats.transition_count))
        self._logger.report_single_value("midi_files_processed", float(stats.midi_files_processed))
        if stats.midi_files_processed > 0:
            self._logger.report_single_value(
                "transitions_per_midi_file",
                stats.transition_count / stats.midi_files_processed,
            )

        for stream, vocab_size in stats.vocab_sizes.items():
            self._logger.report_single_value(f"vocab_size/{stream}", float(vocab_size))

        for stream, stream_stats in stats.stream_stats.items():
            for metric_name, value in stream_stats.items():
                self._logger.report_single_value(
                    f"stream/{stream}/{metric_name}",
                    float(value),
                )

        corpus_rows = [
            [
                item.relative_path,
                item.annotated_beats,
                item.transition_count,
            ]
            for item in stats.corpus_files
        ]
        if corpus_rows:
            self._logger.report_table(
                title="corpus",
                series="midi_files",
                table_plot={"columns": ["path", "annotated_beats", "transitions"], "data": corpus_rows},
            )

        zero_transition_files = sum(
            1 for item in stats.corpus_files if item.transition_count == 0
        )
        self._logger.report_single_value("corpus_files_with_zero_transitions", float(zero_transition_files))

        if self._upload_artifacts and bundle_dir.is_dir():
            self._task.upload_artifact("prior_bundle", artifact_object=str(bundle_dir))
            manifest_path = bundle_dir / "manifest.json"
            vocab_path = bundle_dir / "vocabularies.json"
            if manifest_path.is_file():
                self._task.upload_artifact("manifest", artifact_object=str(manifest_path))
            if vocab_path.is_file():
                self._task.upload_artifact("vocabularies", artifact_object=str(vocab_path))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._task.close()


def create_training_monitor(clearml: ClearMLConfig | None = None) -> TrainingMonitor:
    """Return a ClearML monitor when enabled, otherwise a no-op monitor."""
    if clearml is None or not clearml.enabled:
        return NullTrainingMonitor()
    return ClearMLTrainingMonitor(clearml)
