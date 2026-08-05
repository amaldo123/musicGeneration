from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp

from aimusic.core.config import PriorFactorization
from aimusic.core.vocab import Vocabularies
from aimusic.ml.dataset import STREAM_FIELD_MAP, vocab_sizes_from_vocabularies
from aimusic.scoring.priors import STRUCTURAL_STREAM_NAMES, TokenizedPriorQuery


@dataclass(frozen=True)
class CountPriorState:
    """Factorized transition count tables stored as JAX arrays."""

    tables: Mapping[str, jnp.ndarray]
    alpha: jnp.ndarray

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            stream: self.tables[stream] for stream in STRUCTURAL_STREAM_NAMES
        }
        result["alpha"] = self.alpha
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "CountPriorState":
        tables = {stream: jnp.asarray(data[stream]) for stream in STRUCTURAL_STREAM_NAMES}
        return cls(tables=tables, alpha=jnp.asarray(data["alpha"]))


@dataclass(frozen=True)
class CountPriorConfig:
    alpha: float = 1.0
    factorization: PriorFactorization = PriorFactorization.FACTORIZED


def _stream_transition_ids(
    queries: Sequence[TokenizedPriorQuery],
    stream: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    prev_field, next_field = STREAM_FIELD_MAP[stream]
    prev_ids = jnp.array(
        [int(getattr(query.prev_event, prev_field)) for query in queries],
        dtype=jnp.int32,
    )
    next_ids = jnp.array(
        [int(getattr(query.next_event, next_field)) for query in queries],
        dtype=jnp.int32,
    )
    return prev_ids, next_ids


def _bincount_transition_matrix(
    prev_ids: jnp.ndarray,
    next_ids: jnp.ndarray,
    vocab_size: int,
) -> jnp.ndarray:
    if prev_ids.size == 0:
        return jnp.zeros((vocab_size, vocab_size), dtype=jnp.float32)
    flat = prev_ids * vocab_size + next_ids
    return jnp.bincount(flat, length=vocab_size * vocab_size).reshape(
        vocab_size, vocab_size
    ).astype(jnp.float32)


def train_counts(
    queries: Sequence[TokenizedPriorQuery],
    vocabularies: Vocabularies,
    *,
    alpha: float = 1.0,
) -> CountPriorState:
    """Aggregate factorized transition counts using JAX bincount."""
    sizes = vocab_sizes_from_vocabularies(vocabularies)
    query_items = tuple(queries)
    tables = {
        stream: _bincount_transition_matrix(
            *_stream_transition_ids(query_items, stream),
            sizes[stream],
        )
        for stream in STRUCTURAL_STREAM_NAMES
    }
    return CountPriorState(
        tables=tables,
        alpha=jnp.array(alpha, dtype=jnp.float32),
    )


def _make_score_fn(state: CountPriorState):
    tables = {stream: state.tables[stream] for stream in STRUCTURAL_STREAM_NAMES}
    alpha = state.alpha

    def score_stream(table: jnp.ndarray, prev_ids: jnp.ndarray, next_ids: jnp.ndarray) -> jnp.ndarray:
        vocab_size = table.shape[0]

        def score_one(prev_id: jnp.ndarray, next_id: jnp.ndarray) -> jnp.ndarray:
            selected = table[prev_id, next_id]
            row_total = table[prev_id].sum()
            return jnp.log((selected + alpha) / (row_total + alpha * vocab_size))

        return jax.vmap(score_one)(prev_ids, next_ids)

    def score_packed_batch(packed: Mapping[str, jnp.ndarray]) -> jnp.ndarray:
        batch_size = packed["prev_chord"].shape[0]
        total = jnp.zeros(batch_size, dtype=jnp.float32)
        for stream in STRUCTURAL_STREAM_NAMES:
            total = total + score_stream(
                tables[stream],
                packed[f"prev_{stream}"],
                packed[f"next_{stream}"],
            )
        return total

    return score_packed_batch


@dataclass
class CountPriorModel:
    """JAX count-based prior implementing the external NeuralPriorModel contract."""

    state: CountPriorState
    vocabularies: Vocabularies
    config: CountPriorConfig

    def __post_init__(self) -> None:
        score_fn = _make_score_fn(self.state)
        self._score_batch_jit = jax.jit(score_fn)

    @classmethod
    def from_count_state(
        cls,
        count_state: Mapping[str, object],
        vocabularies: Vocabularies,
        config: CountPriorConfig | None = None,
    ) -> "CountPriorModel":
        return cls(
            state=CountPriorState.from_dict(count_state),
            vocabularies=vocabularies,
            config=CountPriorConfig() if config is None else config,
        )

    def _pack_single(self, query: TokenizedPriorQuery):
        from aimusic.ml.dataset import pack_tokenized_batch, vocab_sizes_from_vocabularies

        sizes = vocab_sizes_from_vocabularies(self.vocabularies)
        return pack_tokenized_batch((query,), sizes)

    def score_transition(self, query: TokenizedPriorQuery) -> float:
        packed = self._pack_single(query)
        scores = self._score_batch_jit(packed)
        return float(scores[0])

    def score_transition_batch(
        self,
        queries: Sequence[TokenizedPriorQuery],
    ) -> Tuple[float, ...]:
        from aimusic.ml.dataset import pack_tokenized_batch, vocab_sizes_from_vocabularies

        items = tuple(queries)
        if not items:
            return ()
        sizes = vocab_sizes_from_vocabularies(self.vocabularies)
        packed = pack_tokenized_batch(items, sizes)
        scores = self._score_batch_jit(packed)
        return tuple(float(value) for value in scores.tolist())
