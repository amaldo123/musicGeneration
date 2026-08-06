from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from aimusic.core.config import PriorFactorization, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies, build_default_vocabularies
from aimusic.ml.annotate import annotate_beat_grid
from aimusic.ml.midi_ingest import parse_midi_beats
from aimusic.scoring.priors import PriorContext, PriorQuery, TokenizedPriorQuery

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

logger = logging.getLogger(__name__)


STREAM_FIELD_MAP: Mapping[str, tuple[str, str]] = {
    "meter": ("meter_id", "meter_id"),
    "beat_position": ("beat_position", "beat_position"),
    "boundary": ("boundary_level", "boundary_level"),
    "key": ("key_id", "key_id"),
    "chord": ("chord_id", "chord_id"),
    "role": ("role_id", "role_id"),
    "head": ("head_id", "head_id"),
    "groove": ("groove_id", "groove_id"),
}


def vocab_sizes_from_vocabularies(vocabularies: Vocabularies) -> dict[str, int]:
    return {
        "meter": len(vocabularies.meters),
        "beat_position": len(vocabularies.beat_positions),
        "boundary": len(vocabularies.boundaries),
        "key": len(vocabularies.keys),
        "chord": len(vocabularies.chords),
        "role": len(vocabularies.roles),
        "head": len(vocabularies.heads),
        "groove": len(vocabularies.grooves),
    }


def build_transition_examples(
    states: Sequence[BeatState],
    *,
    history_len: int = 1,
    future_hint_len: int = 1,
) -> Tuple[PriorQuery, ...]:
    """Build transition queries mirroring Method A prior context (history=1, future hint=1)."""
    items = tuple(states)
    if len(items) < 2:
        return ()

    queries: list[PriorQuery] = []
    for index in range(len(items) - 1):
        prev_state = items[index]
        next_state = items[index + 1]
        history_start = max(0, index + 1 - history_len)
        history = items[history_start : index + 1]
        future_hints = items[index + 1 : min(len(items), index + 1 + future_hint_len)]
        queries.append(
            PriorQuery(
                prev_state=prev_state,
                next_state=next_state,
                time_index=index,
                context=PriorContext(
                    history=history,
                    future_hints=future_hints,
                    metadata=(("graph_time", str(index)),),
                ),
            )
        )
    return tuple(queries)


def examples_to_tokenized(
    queries: Sequence[PriorQuery],
    factorization: PriorFactorization = PriorFactorization.FACTORIZED,
) -> Tuple[TokenizedPriorQuery, ...]:
    return tuple(query.tokenize(factorization) for query in queries)


def load_corpus_transitions(
    midi_dir: str | Path,
    style_config: StyleConfig,
    *,
    edo: int = 12,
    progress: bool = True,
) -> Tuple[PriorQuery, ...]:
    """Load all MIDI files in a directory and extract transition queries."""
    root = Path(midi_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"MIDI directory not found: {root}")

    vocabularies = build_default_vocabularies(style_config)

    files = sorted(root.glob("**/*.mid")) + sorted(root.glob("**/*.midi"))
    total = len(files)
    logger.info("Scanning %d MIDI files under %s", total, root)

    if progress and tqdm is not None and total > 0:
        iterator = tqdm(
            files,
            desc="parse+annotate",
            unit="file",
            dynamic_ncols=True,
            mininterval=0.5,
        )
    else:
        iterator = iter(files)

    all_queries: list[PriorQuery] = []
    failed: list[tuple[Path, str]] = []
    t_start = time.monotonic()
    for midi_path in iterator:
        try:
            grid = parse_midi_beats(midi_path)
            states = annotate_beat_grid(grid, vocabularies, edo=edo)
            all_queries.extend(build_transition_examples(states))
        except Exception as exc:
            failed.append((midi_path, str(exc)))
            logger.warning("Failed to parse %s: %s", midi_path, exc)

    elapsed = time.monotonic() - t_start
    rate = total / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Loaded %d transitions from %d files in %.2fs (%.1f files/s)%s",
        len(all_queries),
        total,
        elapsed,
        rate,
        f", {len(failed)} failed" if failed else "",
    )
    if failed and not progress:
        for path, err in failed[:10]:
            logger.warning("  failed: %s (%s)", path, err)
        if len(failed) > 10:
            logger.warning("  ... and %d more", len(failed) - 10)
    return tuple(all_queries)


def pack_tokenized_batch(
    queries: Sequence[TokenizedPriorQuery],
    vocab_sizes: Mapping[str, int],
):
    """Stack tokenized queries into JAX batch arrays for training and scoring."""
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("pack_tokenized_batch requires the optional [ml] extra.") from exc

    items = tuple(queries)
    if not items:
        raise ValueError("queries must not be empty.")

    packed: dict[str, object] = {"batch_size": len(items)}
    for stream, (prev_field, next_field) in STREAM_FIELD_MAP.items():
        packed[f"prev_{stream}"] = jnp.array(
            [getattr(item.prev_event, prev_field) for item in items],
            dtype=jnp.int32,
        )
        packed[f"next_{stream}"] = jnp.array(
            [getattr(item.next_event, next_field) for item in items],
            dtype=jnp.int32,
        )

    max_history = max(len(item.history_tokens) for item in items)
    max_future = max(len(item.future_hint_tokens) for item in items)
    for stream, (field_name, _) in STREAM_FIELD_MAP.items():
        hist_rows = []
        fut_rows = []
        attr = f"{field_name}s" if stream != "beat_position" else "beat_positions"
        if stream == "meter":
            attr = "meter_ids"
        elif stream == "boundary":
            attr = "boundary_levels"
        for item in items:
            history_values = getattr(item.history_tokens, attr)
            future_values = getattr(item.future_hint_tokens, attr)
            hist_rows.append(
                list(history_values)
                + [0] * (max_history - len(history_values))
            )
            fut_rows.append(
                list(future_values)
                + [0] * (max_future - len(future_values))
            )
        packed[f"history_{stream}"] = jnp.array(hist_rows, dtype=jnp.int32)
        packed[f"future_{stream}"] = jnp.array(fut_rows, dtype=jnp.int32)

    packed["time_index"] = jnp.array([item.time_index for item in items], dtype=jnp.int32)
    packed["vocab_sizes"] = dict(vocab_sizes)
    return packed
