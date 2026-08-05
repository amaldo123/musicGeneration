import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.ml.monitoring import (
    ClearMLConfig,
    ClearMLTrainingMonitor,
    NullTrainingMonitor,
    build_training_run_stats,
    collect_corpus_stats,
    create_training_monitor,
    summarize_count_state,
)
from aimusic.scoring.priors import STRUCTURAL_STREAM_NAMES
from tests.conftest import requires_jax, skip_unless_jax
from tests.test_midi_ingest import write_simple_c_g_progression


class TestMonitoringHelpers(unittest.TestCase):
    def test_create_training_monitor_returns_null_by_default(self):
        monitor = create_training_monitor(None)
        self.assertIsInstance(monitor, NullTrainingMonitor)

    def test_create_training_monitor_disabled_clearml(self):
        monitor = create_training_monitor(ClearMLConfig(enabled=False))
        self.assertIsInstance(monitor, NullTrainingMonitor)

    def test_create_training_monitor_requires_clearml_package(self):
        with self.assertRaises(ImportError):
            with mock.patch.dict("sys.modules", {"clearml": None}):
                create_training_monitor(ClearMLConfig(enabled=True))

    def test_collect_corpus_stats_reports_per_file_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            write_simple_c_g_progression(Path(tmp_dir) / "piece.mid")
            stats = collect_corpus_stats(
                tmp_dir,
                StyleConfig(allowed_meters=("4/4",)),
                DEFAULT_VOCABULARIES,
            )

        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0].relative_path, "piece.mid")
        self.assertGreater(stats[0].annotated_beats, 0)
        self.assertGreater(stats[0].transition_count, 0)

    @requires_jax
    @skip_unless_jax
    def test_summarize_count_state_reports_stream_metrics(self):
        import jax.numpy as jnp

        count_state = {
            stream: jnp.array([[0.0, 2.0], [1.0, 0.0]], dtype=jnp.float32)
            for stream in STRUCTURAL_STREAM_NAMES
        }
        summary = summarize_count_state(count_state)

        self.assertEqual(len(summary), len(STRUCTURAL_STREAM_NAMES))
        self.assertEqual(summary["chord"]["total_transitions"], 3.0)
        self.assertEqual(summary["chord"]["unique_prev_next_pairs"], 2)


class TestClearMLTrainingMonitor(unittest.TestCase):
    def test_clearml_monitor_logs_run(self):
        fake_task = mock.Mock()
        fake_task.id = "task-123"
        fake_logger = mock.Mock()
        fake_task.get_logger.return_value = fake_logger
        fake_clearml = mock.Mock()
        fake_clearml.Task.init.return_value = fake_task

        with mock.patch.dict("sys.modules", {"clearml": fake_clearml}):
            monitor = ClearMLTrainingMonitor(
                ClearMLConfig(
                    enabled=True,
                    project_name="test/project",
                    task_name="count-prior",
                    tags=("jax",),
                )
            )

        fake_clearml.Task.init.assert_called_once()
        monitor.on_run_start(
            midi_dir=Path("/tmp/corpus"),
            output_dir=Path("/tmp/out"),
            style_config=StyleConfig(allowed_meters=("4/4",)),
            edo=12,
            alpha=1.0,
            model_version="v1",
        )
        fake_task.connect.assert_called_once()

        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_dir = Path(tmp_dir)
            (bundle_dir / "manifest.json").write_text("{}", encoding="utf-8")
            monitor.on_run_complete(
                bundle_dir=bundle_dir,
                stats=build_training_run_stats(
                    transition_count=10,
                    midi_files_processed=2,
                    corpus_files=(),
                    vocabularies=DEFAULT_VOCABULARIES,
                    count_state={
                        stream: [[0.0, 1.0], [1.0, 0.0]]
                        for stream in STRUCTURAL_STREAM_NAMES
                    },
                ),
            )

        fake_logger.report_single_value.assert_any_call("transition_count", 10.0)
        fake_task.upload_artifact.assert_any_call("prior_bundle", artifact_object=str(bundle_dir))
        monitor.close()
        fake_task.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
