from __future__ import annotations

import argparse
import logging
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda")

from aimusic.core.config import StyleConfig
from aimusic.ml.monitoring import ClearMLConfig

_LOG_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
_LOG_DATEFMT = "%H:%M:%S"


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATEFMT,
        stream=sys.stderr,
        force=True,
    )
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src").setLevel(logging.WARNING)
    try:
        import absl.logging as absl_logging

        absl_logging.set_verbosity(absl_logging.WARNING)
    except ImportError:
        pass


def _build_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a JAX count-based neural prior.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train from a MIDI corpus directory")
    train_parser.add_argument("--midi-dir", type=str, required=True)
    train_parser.add_argument("--output", type=str, required=True)
    train_parser.add_argument("--edo", type=int, default=12)
    train_parser.add_argument("--alpha", type=float, default=1.0)
    train_parser.add_argument("--model-version", type=str, default="v1")
    train_parser.add_argument("--meter", action="append", default=None)
    train_parser.add_argument("--groove-family", action="append", default=None)
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logs (only print the final summary line).",
    )
    train_parser.add_argument(
        "--clearml",
        action="store_true",
        help="Enable ClearML experiment tracking (requires pip install -e '.[clearml]')",
    )
    train_parser.add_argument(
        "--clearml-project",
        type=str,
        default="aimusic/ml",
        help="ClearML project name",
    )
    train_parser.add_argument(
        "--clearml-task-name",
        type=str,
        default=None,
        help="ClearML task name (defaults to jax_count_prior)",
    )
    train_parser.add_argument(
        "--clearml-tag",
        action="append",
        default=None,
        help="ClearML tag (repeatable)",
    )
    train_parser.add_argument(
        "--no-clearml-artifacts",
        action="store_true",
        help="Log metrics to ClearML but skip artifact uploads",
    )
    train_parser.set_defaults(func=_handle_train)
    return parser


def _handle_train(args: argparse.Namespace) -> None:
    try:
        from aimusic.ml.train import train_prior_from_corpus
    except ImportError as exc:
        print(
            "Error: ML training requires optional dependencies.\n"
            "Install with: uv pip install -e '.[ml]'",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    if args.meter and args.groove_family:
        style_config = StyleConfig(
            allowed_meters=tuple(args.meter),
            groove_families=tuple(args.groove_family),
        )
    elif args.meter:
        style_config = StyleConfig(allowed_meters=tuple(args.meter))
    elif args.groove_family:
        style_config = StyleConfig(groove_families=tuple(args.groove_family))
    else:
        style_config = StyleConfig()
    clearml_config = None
    if args.clearml:
        clearml_config = ClearMLConfig(
            enabled=True,
            project_name=args.clearml_project,
            task_name=args.clearml_task_name,
            tags=tuple(args.clearml_tag or ()),
            upload_artifacts=not args.no_clearml_artifacts,
        )

    try:
        result = train_prior_from_corpus(
            args.midi_dir,
            args.output,
            style_config=style_config,
            edo=args.edo,
            alpha=args.alpha,
            model_version=args.model_version,
            clearml=clearml_config,
            quiet=args.quiet,
        )
    except ImportError as exc:
        if clearml_config is not None:
            print(
                "Error: ClearML monitoring requires the optional clearml extra.\n"
                "Install with: pip install -e '.[ml,clearml]'",
                file=sys.stderr,
            )
            raise SystemExit(1) from exc
        raise
    print(
        f"Trained jax_count_prior bundle at {result.bundle_dir} "
        f"({result.transition_count} transitions from {result.midi_files_processed} MIDI files)"
    )


def main() -> None:
    parser = _build_train_parser()
    args = parser.parse_args()
    _configure_logging(verbose=not getattr(args, "quiet", False))
    args.func(args)


if __name__ == "__main__":
    main()
