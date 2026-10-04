# ==============================================================================
# MUSIC GENERATION PIPELINE - CONSOLIDATED CODEBASE
# High-Level Architecture: Symbolic Music Generation using GTTM-inspired Energies
# and Schrödinger Bridge (SB) Optimal Transport
# ==============================================================================

# ==============================================================================
# 1. FILE: aimusic/core/core_types.py
# ==============================================================================
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, Iterator, Tuple

if TYPE_CHECKING:
    from aimusic.core.vocab import TokenVocabulary, Vocabularies


ExpressiveControls = Tuple[float, ...]


def _require_int(name: str, value: int, *, minimum: int | None = None) -> None:
    """Validate that a field is an integer and optionally above a floor."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float | None = None) -> None:
    """Validate that a field is a finite real number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _safe_token_label(vocabulary: "TokenVocabulary[Any]", token_id: int) -> str | None:
    """Look up a token label while tolerating unknown ids."""
    if vocabulary.has_id(token_id):
        return vocabulary.token_for_id(token_id).label
    return None


def _format_token(field_name: str, token_id: int, label: str | None) -> str:
    """Render a token id in raw or label-aware form for logs."""
    if label is None:
        return f"{field_name}_id={token_id}"
    return f"{field_name}={label}[{token_id}]"


class ScoreValidationError(ValueError):
    """Raised when a serialized score/note payload is malformed or inconsistent."""

def _require_mapping(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ScoreValidationError(f"{name} must be a JSON object, got {type(value).__name__}.")
    return value


def _require_key(data: dict[str, object], key: str, context: str) -> object:
    if key not in data:
        raise ScoreValidationError(f"{context} is missing required field '{key}'.")
    return data[key]


def _require_list(name: str, value: object) -> list:
    if not isinstance(value, list):
        raise ScoreValidationError(f"{name} must be a JSON array, got {type(value).__name__}.")
    return value


@dataclass(frozen=True)
class BeatState:
    """Canonical beat-level structural state.

    The project README defines:
      St = (meter_id, beat_in_bar, boundary_lvl, key_id, chord_id, role_id, head_id, groove_id)
    """

    meter_id: int
    beat_in_bar: int
    boundary_lvl: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int

    def __post_init__(self) -> None:
        for name in (
            "meter_id",
            "beat_in_bar",
            "boundary_lvl",
            "key_id",
            "chord_id",
            "role_id",
            "head_id",
            "groove_id",
        ):
            _require_int(name, getattr(self, name), minimum=0)

    def token_labels(self, vocabularies: "Vocabularies") -> dict[str, str | None]:
        """Resolve human-readable labels for the state's structural token ids."""
        return {
            "meter": _safe_token_label(vocabularies.meters, self.meter_id),
            "beat": _safe_token_label(vocabularies.beat_positions, self.beat_in_bar),
            "boundary": _safe_token_label(vocabularies.boundaries, self.boundary_lvl),
            "key": _safe_token_label(vocabularies.keys, self.key_id),
            "chord": _safe_token_label(vocabularies.chords, self.chord_id),
            "role": _safe_token_label(vocabularies.roles, self.role_id),
            "head": _safe_token_label(vocabularies.heads, self.head_id),
            "groove": _safe_token_label(vocabularies.grooves, self.groove_id),
        }

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the structural state to a log/test-friendly mapping."""
        data: dict[str, object] = {
            "meter_id": self.meter_id,
            "beat_in_bar": self.beat_in_bar,
            "boundary_lvl": self.boundary_lvl,
            "key_id": self.key_id,
            "chord_id": self.chord_id,
            "role_id": self.role_id,
            "head_id": self.head_id,
            "groove_id": self.groove_id,
        }
        if vocabularies is not None:
            labels = self.token_labels(vocabularies)
            data.update(
                {
                    "meter_label": labels["meter"],
                    "beat_label": labels["beat"],
                    "boundary_label": labels["boundary"],
                    "key_label": labels["key"],
                    "chord_label": labels["chord"],
                    "role_label": labels["role"],
                    "head_label": labels["head"],
                    "groove_label": labels["groove"],
                }
            )
        return data

    def pretty(self, vocabularies: "Vocabularies | None" = None) -> str:
        """Return a compact human-readable representation for logs."""
        if vocabularies is None:
            return (
                "BeatState("
                f"meter_id={self.meter_id}, "
                f"beat_in_bar={self.beat_in_bar}, "
                f"boundary_lvl={self.boundary_lvl}, "
                f"key_id={self.key_id}, "
                f"chord_id={self.chord_id}, "
                f"role_id={self.role_id}, "
                f"head_id={self.head_id}, "
                f"groove_id={self.groove_id})"
            )

        labels = self.token_labels(vocabularies)
        rendered = ", ".join(
            (
                _format_token("meter", self.meter_id, labels["meter"]),
                _format_token("beat", self.beat_in_bar, labels["beat"]),
                _format_token("boundary", self.boundary_lvl, labels["boundary"]),
                _format_token("key", self.key_id, labels["key"]),
                _format_token("chord", self.chord_id, labels["chord"]),
                _format_token("role", self.role_id, labels["role"]),
                _format_token("head", self.head_id, labels["head"]),
                _format_token("groove", self.groove_id, labels["groove"]),
            )
        )
        return f"BeatState({rendered})"


@dataclass(frozen=True)
class NoteEvent:
    """Canonical score-level symbolic note event.

    Matches the design spec fields:
      (ton, toff, h, v, e, track)
    """

    ton: int
    toff: int
    h: int
    v: float
    e: ExpressiveControls = ()
    track: str = "default"

    def __post_init__(self) -> None:
        _require_int("ton", self.ton, minimum=0)
        _require_int("toff", self.toff, minimum=0)
        if self.toff <= self.ton:
            raise ValueError("toff must be > ton.")
        _require_int("h", self.h)
        _require_real("v", self.v, minimum=0.0)
        if self.v > 1.0:
            raise ValueError("v must be <= 1.0.")

        expressive_controls = tuple(self.e)
        for idx, value in enumerate(expressive_controls):
            _require_real(f"e[{idx}]", value)
        object.__setattr__(self, "e", expressive_controls)

        if not isinstance(self.track, str) or not self.track.strip():
            raise ValueError("track must be a non-empty string.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the note event to a JSON-friendly mapping."""
        return {
            "ton": self.ton,
            "toff": self.toff,
            "duration_ticks": self.toff - self.ton,
            "h": self.h,
            "v": self.v,
            "e": list(self.e),
            "track": self.track,
        }

    def pretty(self) -> str:
        """Return a compact, readable note-event summary."""
        return (
            "NoteEvent("
            f"track={self.track}, "
            f"ticks={self.ton}->{self.toff}, "
            f"h={self.h}, "
            f"v={self.v:.3f}, "
            f"e={list(self.e)})"
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NoteEvent":
        """Deserialize a note event, validating structure with actionable errors."""
        data = _require_mapping("NoteEvent payload", data)

        ton = _require_key(data, "ton", "NoteEvent payload")
        toff = _require_key(data, "toff", "NoteEvent payload")
        h = _require_key(data, "h", "NoteEvent payload")
        v = _require_key(data, "v", "NoteEvent payload")
        e = data.get("e", [])
        track = data.get("track", "default")

        if not isinstance(ton, int) or isinstance(ton, bool):
            raise ScoreValidationError(f"NoteEvent field 'ton' must be an int, got {type(ton).__name__}.")
        if not isinstance(toff, int) or isinstance(toff, bool):
            raise ScoreValidationError(f"NoteEvent field 'toff' must be an int, got {type(toff).__name__}.")
        if not isinstance(h, int) or isinstance(h, bool):
            raise ScoreValidationError(f"NoteEvent field 'h' must be an int, got {type(h).__name__}.")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ScoreValidationError(f"NoteEvent field 'v' must be a number, got {type(v).__name__}.")
        if not isinstance(e, list):
            raise ScoreValidationError(f"NoteEvent field 'e' must be a list, got {type(e).__name__}.")
        if not isinstance(track, str):
            raise ScoreValidationError(f"NoteEvent field 'track' must be a string, got {type(track).__name__}.")

        if "duration_ticks" in data:
            duration_ticks = data["duration_ticks"]
            if not isinstance(duration_ticks, int) or isinstance(duration_ticks, bool):
                raise ScoreValidationError(
                    f"NoteEvent field 'duration_ticks' must be an int, got {type(duration_ticks).__name__}."
                )
            if duration_ticks != toff - ton:
                raise ScoreValidationError(
                    "NoteEvent field 'duration_ticks' "
                    f"({duration_ticks}) does not match toff - ton ({toff - ton})."
                )

        try:
            return cls(ton=ton, toff=toff, h=h, v=float(v), e=tuple(float(x) for x in e), track=track)
        except (TypeError, ValueError) as exc:
            raise ScoreValidationError(f"NoteEvent payload failed validation: {exc}") from exc


@dataclass(frozen=True)
class Score:
    """Immutable symbolic score represented as note events."""

    note_events: Tuple[NoteEvent, ...] = ()
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0

    def __post_init__(self) -> None:
        note_events = tuple(self.note_events)
        if any(not isinstance(event, NoteEvent) for event in note_events):
            raise TypeError("note_events must contain only NoteEvent instances.")
        object.__setattr__(self, "note_events", note_events)

        _require_int("ticks_per_beat", self.ticks_per_beat, minimum=1)
        _require_real("tempo_bpm", self.tempo_bpm, minimum=0.0)
        if self.tempo_bpm == 0.0:
            raise ValueError("tempo_bpm must be > 0.")

    def __iter__(self) -> Iterator[NoteEvent]:
        return iter(self.note_events)

    def __len__(self) -> int:
        return len(self.note_events)

    def track_event_counts(self) -> dict[str, int]:
        """Return stable per-track event counts for diagnostics."""
        counts = Counter(event.track for event in self.note_events)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        """Serialize the score and all note events."""
        return {
            "event_count": len(self),
            "ticks_per_beat": self.ticks_per_beat,
            "tempo_bpm": self.tempo_bpm,
            "track_event_counts": self.track_event_counts(),
            "note_events": [event.to_dict() for event in self.note_events],
        }

    def pretty(self, *, max_events: int = 3) -> str:
        """Return a concise score summary with an event preview."""
        preview_events = ", ".join(event.pretty() for event in self.note_events[:max_events])
        if len(self.note_events) > max_events:
            preview_events = f"{preview_events}, ..."
        track_counts = ", ".join(
            f"{track}:{count}" for track, count in self.track_event_counts().items()
        )
        return (
            "Score("
            f"events={len(self)}, "
            f"tempo_bpm={self.tempo_bpm:.1f}, "
            f"ticks_per_beat={self.ticks_per_beat}, "
            f"tracks={{{track_counts}}}, "
            f"preview=[{preview_events}])"
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Score":
        """Deserialize a score, validating structure with actionable errors."""
        data = _require_mapping("Score payload", data)

        raw_events = _require_list("Score field 'note_events'", _require_key(data, "note_events", "Score payload"))
        note_events = tuple(
            NoteEvent.from_dict(_require_mapping(f"Score note_events[{index}]", item))
            for index, item in enumerate(raw_events)
        )

        kwargs: dict[str, object] = {"note_events": note_events}
        if "ticks_per_beat" in data:
            kwargs["ticks_per_beat"] = data["ticks_per_beat"]
        if "tempo_bpm" in data:
            kwargs["tempo_bpm"] = data["tempo_bpm"]

        try:
            score = cls(**kwargs)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ScoreValidationError(f"Score payload failed validation: {exc}") from exc

        if "event_count" in data and data["event_count"] != len(score):
            raise ScoreValidationError(
                f"Score field 'event_count' ({data['event_count']}) does not match "
                f"the number of note_events ({len(score)})."
            )
        if "track_event_counts" in data and data["track_event_counts"] != score.track_event_counts():
            raise ScoreValidationError(
                f"Score field 'track_event_counts' ({data['track_event_counts']}) does not match "
                f"the recomputed per-track counts ({score.track_event_counts()})."
            )

        return score


@dataclass(frozen=True)
class Layer:
    """Immutable collection of candidate BeatStates at a given beat index."""

    time_index: int
    states: Tuple[BeatState, ...]

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)

        states = tuple(self.states)
        if any(not isinstance(state, BeatState) for state in states):
            raise TypeError("states must contain only BeatState instances.")
        if len(states) != len(set(states)):
            raise ValueError("Layer states must be unique.")
        object.__setattr__(self, "states", states)

    def __iter__(self) -> Iterator[BeatState]:
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the layer and its candidate states."""
        return {
            "time_index": self.time_index,
            "size": len(self),
            "states": [state.to_dict(vocabularies) for state in self.states],
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None, *, max_states: int = 3) -> str:
        """Return a compact layer summary for logs."""
        preview = ", ".join(
            state.pretty(vocabularies) for state in self.states[:max_states]
        )
        if len(self.states) > max_states:
            preview = f"{preview}, ..."
        return f"Layer(t={self.time_index}, size={len(self)}, states=[{preview}])"


@dataclass(frozen=True)
class Edge:
    """Immutable transition edge between two BeatStates."""

    time_index: int
    source: BeatState
    target: BeatState
    log_weight: float

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.source, BeatState):
            raise TypeError("source must be a BeatState.")
        if not isinstance(self.target, BeatState):
            raise TypeError("target must be a BeatState.")
        _require_real("log_weight", self.log_weight)

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the edge and its endpoints."""
        return {
            "time_index": self.time_index,
            "log_weight": self.log_weight,
            "source": self.source.to_dict(vocabularies),
            "target": self.target.to_dict(vocabularies),
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None) -> str:
        """Return a compact human-readable edge description."""
        return (
            "Edge("
            f"t={self.time_index}, "
            f"log_weight={self.log_weight:.3f}, "
            f"source={self.source.pretty(vocabularies)}, "
            f"target={self.target.pretty(vocabularies)})"
        )


@dataclass(frozen=True)
class EndpointDistribution:
    """Normalized probability distribution over a specific graph layer."""

    layer: Layer
    probabilities: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layer, Layer):
            raise TypeError("layer must be a Layer.")
        if len(self.layer) == 0:
            raise ValueError("EndpointDistribution layer must not be empty.")

        probabilities = tuple(self.probabilities)
        if len(probabilities) != len(self.layer):
            raise ValueError("probabilities must align 1:1 with the layer states.")
        for idx, prob in enumerate(probabilities):
            _require_real(f"probabilities[{idx}]", prob, minimum=0.0)

        total = float(sum(probabilities))
        if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("EndpointDistribution probabilities must sum to 1.0.")

        object.__setattr__(self, "probabilities", probabilities)

    def probability_of(self, state: BeatState) -> float:
        """Return the probability mass of *state*, or 0.0 if absent."""
        for candidate, probability in zip(self.layer.states, self.probabilities):
            if candidate == state:
                return probability
        return 0.0

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the endpoint distribution and its support."""
        return {
            "time_index": self.layer.time_index,
            "support_size": len(self.layer),
            "support": [
                {
                    "state": state.to_dict(vocabularies),
                    "probability": probability,
                }
                for state, probability in zip(self.layer.states, self.probabilities)
            ],
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None, *, max_states: int = 3) -> str:
        """Return a compact endpoint-distribution summary."""
        support_preview = ", ".join(
            (
                f"{state.pretty(vocabularies)}@{probability:.3f}"
                for state, probability in zip(
                    self.layer.states[:max_states],
                    self.probabilities[:max_states],
                )
            )
        )
        if len(self.layer) > max_states:
            support_preview = f"{support_preview}, ..."
        return (
            "EndpointDistribution("
            f"t={self.layer.time_index}, "
            f"size={len(self.layer)}, "
            f"support=[{support_preview}])"
        )

# ==============================================================================
# 2. FILE: aimusic/core/config.py
# ==============================================================================
from enum import Enum
from typing import Optional, Sequence

RegisterRange = Tuple[int, int]
ValueRange = Tuple[float, float]

def _coerce_non_empty_str_tuple(name: str, values: Sequence[str]) -> Tuple[str, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{name} entries must be non-empty strings.")
    return items

def _coerce_positive_int_tuple(name: str, values: Sequence[int]) -> Tuple[int, ...]:
    items = tuple(values)
    if not items:
        raise ValueError(f"{name} must not be empty.")
    for item in items:
        _require_int(name, item, minimum=1)
    return items

def _coerce_register_range(name: str, values: Sequence[int]) -> RegisterRange:
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two integer bounds.")
    low, high = items
    _require_int(f"{name}[0]", low)
    _require_int(f"{name}[1]", high)
    if low >= high:
        raise ValueError(f"{name} lower bound must be < upper bound.")
    return int(low), int(high)

def _coerce_unit_range(name: str, values: Sequence[float]) -> ValueRange:
    items = tuple(values)
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two numeric bounds.")
    low, high = float(items[0]), float(items[1])
    _require_real(f"{name}[0]", low, minimum=0.0)
    _require_real(f"{name}[1]", high, minimum=0.0)
    if high > 1.0:
        raise ValueError(f"{name} upper bound must be <= 1.0.")
    if low > high:
        raise ValueError(f"{name} lower bound must be <= upper bound.")
    return low, high

class MicrotonalRendering(Enum):
    MPE = "mpe"
    MTS = "mts"

class SBBackend(Enum):
    NUMPY = "numpy"
    JAX = "jax"

class PlanMethod(Enum):
    METHOD_A = "method_a"
    METHOD_B = "method_b"

class SectioningStrategy(Enum):
    SINGLE_PASS = "single_pass"
    SECTION_WISE = "section_wise"

class PriorFactorization(Enum):
    WHOLE_STATE = "whole_state"
    FACTORIZED = "factorized"
    MIXED = "mixed"

class PlaceholderPriorMode(Enum):
    NEUTRAL = "neutral"
    STRUCTURED = "structured"

@dataclass(frozen=True)
class EDOConfig:
    n: int
    base_tuning: float = 60.0
    microtonal_rendering_method: MicrotonalRendering = MicrotonalRendering.MPE
    pitch_bend_range: int = 48

    def __post_init__(self) -> None:
        _require_int("n", self.n, minimum=1)
        _require_real("base_tuning", self.base_tuning)
        _require_int("pitch_bend_range", self.pitch_bend_range, minimum=1)
        if not isinstance(self.microtonal_rendering_method, MicrotonalRendering):
            raise TypeError("microtonal_rendering_method must be a MicrotonalRendering value.")

@dataclass(frozen=True)
class StyleConfig:
    allowed_meters: Tuple[str, ...] = ("4/4", "5/4", "7/4")
    subdivision_patterns: Tuple[int, ...] = (3, 4)
    groove_families: Tuple[str, ...] = ("straight", "syncopated")
    chord_vocabulary_size: Optional[int] = None
    key_vocabulary_size: Optional[int] = None
    bass_register: RegisterRange = (28, 52)
    comping_register: RegisterRange = (45, 72)
    lead_register: RegisterRange = (60, 88)
    typical_density_range: ValueRange = (0.25, 0.85)

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_meters", _coerce_non_empty_str_tuple("allowed_meters", self.allowed_meters))
        object.__setattr__(self, "subdivision_patterns", _coerce_positive_int_tuple("subdivision_patterns", self.subdivision_patterns))
        object.__setattr__(self, "groove_families", _coerce_non_empty_str_tuple("groove_families", self.groove_families))
        if self.chord_vocabulary_size is not None:
            _require_int("chord_vocabulary_size", self.chord_vocabulary_size, minimum=1)
        if self.key_vocabulary_size is not None:
            _require_int("key_vocabulary_size", self.key_vocabulary_size, minimum=1)
        object.__setattr__(self, "bass_register", _coerce_register_range("bass_register", self.bass_register))
        object.__setattr__(self, "comping_register", _coerce_register_range("comping_register", self.comping_register))
        object.__setattr__(self, "lead_register", _coerce_register_range("lead_register", self.lead_register))
        object.__setattr__(self, "typical_density_range", _coerce_unit_range("typical_density_range", self.typical_density_range))

@dataclass(frozen=True)
class PriorWeights:
    lambda_data: float = 1.0
    lambda_gttm: float = 1.0
    lambda_target_tension: float = 1.0
    lambda_section_style: float = 1.0
    meter: float = 1.0
    grouping: float = 1.0
    harmonic: float = 1.0
    prolongational_role: float = 1.0
    melodic_head: float = 1.0
    groove: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lambda_data", "lambda_gttm", "lambda_target_tension", "lambda_section_style", "meter", "grouping", "harmonic", "prolongational_role", "melodic_head", "groove"):
            _require_real(name, getattr(self, name), minimum=0.0)
        if self.lambda_data == 0.0 and self.lambda_gttm == 0.0:
            raise ValueError("At least one of lambda_data or lambda_gttm must be > 0.")

@dataclass(frozen=True)
class NeuralPriorConfig:
    model_family: str = "external_neural_prior"
    model_version: str = "placeholder-v1"
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    manifest_path: Optional[str] = None
    supports_batch_scoring: bool = True
    batch_size: int = 32
    placeholder_mode: PlaceholderPriorMode = PlaceholderPriorMode.STRUCTURED
    default_logp: float = 0.0

    def __post_init__(self) -> None:
        for name in ("model_family", "model_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization value.")
        for name in ("checkpoint_path", "tokenizer_path", "manifest_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be None or a non-empty string.")
        if not isinstance(self.supports_batch_scoring, bool):
            raise TypeError("supports_batch_scoring must be a bool.")
        _require_int("batch_size", self.batch_size, minimum=1)
        if not isinstance(self.placeholder_mode, PlaceholderPriorMode):
            raise TypeError("placeholder_mode must be a PlaceholderPriorMode value.")
        _require_real("default_logp", self.default_logp)

@dataclass(frozen=True)
class SBConfig:
    horizon_t: int = 64
    max_iterations: int = 200
    tolerance: float = 1e-6
    temperature: float = 1.0
    k_max: int = 64
    d_max: int = 8
    proposal_budget: int = 256
    prior_guided_proposals: bool = False
    log_underflow_floor: float = -745.0
    raise_on_non_convergence: bool = False
    backend_selection: SBBackend = SBBackend.NUMPY

    def __post_init__(self) -> None:
        _require_int("horizon_t", self.horizon_t, minimum=1)
        _require_int("max_iterations", self.max_iterations, minimum=1)
        _require_real("tolerance", self.tolerance, minimum=0.0)
        if self.tolerance == 0.0:
            raise ValueError("tolerance must be > 0.")
        _require_real("temperature", self.temperature, minimum=0.0)
        if self.temperature == 0.0:
            raise ValueError("temperature must be > 0.")
        _require_int("k_max", self.k_max, minimum=1)
        _require_int("d_max", self.d_max, minimum=1)
        _require_int("proposal_budget", self.proposal_budget, minimum=1)
        if not isinstance(self.prior_guided_proposals, bool):
            raise TypeError("prior_guided_proposals must be a bool.")
        _require_real("log_underflow_floor", self.log_underflow_floor)
        if self.log_underflow_floor > 0.0:
            raise ValueError("log_underflow_floor must be <= 0.0.")
        if not isinstance(self.raise_on_non_convergence, bool):
            raise TypeError("raise_on_non_convergence must be a bool.")
        if not isinstance(self.backend_selection, SBBackend):
            raise TypeError("backend_selection must be an SBBackend value.")

@dataclass(frozen=True)
class DecodeConfig:
    subbeats_per_beat: int = 4
    drum_density: float = 0.75
    bass_density: float = 0.60
    comping_density: float = 0.55
    lead_density: float = 0.45
    bass_register: RegisterRange = (28, 52)
    comping_register: RegisterRange = (45, 72)
    lead_register: RegisterRange = (60, 88)
    min_comping_voices: int = 3
    max_comping_voices: int = 5
    max_lead_leap_steps: int = 7
    tension_velocity_range: ValueRange = (0.55, 1.0)
    tension_expression_range: ValueRange = (0.0, 1.0)

    def __post_init__(self) -> None:
        _require_int("subbeats_per_beat", self.subbeats_per_beat, minimum=1)
        for name in ("drum_density", "bass_density", "comping_density", "lead_density"):
            value = getattr(self, name)
            _require_real(name, value, minimum=0.0)
            if value > 1.0:
                raise ValueError(f"{name} must be <= 1.0.")
        object.__setattr__(self, "bass_register", _coerce_register_range("bass_register", self.bass_register))
        object.__setattr__(self, "comping_register", _coerce_register_range("comping_register", self.comping_register))
        object.__setattr__(self, "lead_register", _coerce_register_range("lead_register", self.lead_register))
        _require_int("min_comping_voices", self.min_comping_voices, minimum=1)
        _require_int("max_comping_voices", self.max_comping_voices, minimum=1)
        if self.min_comping_voices > self.max_comping_voices:
            raise ValueError("min_comping_voices must be <= max_comping_voices.")
        _require_int("max_lead_leap_steps", self.max_lead_leap_steps, minimum=1)
        object.__setattr__(self, "tension_velocity_range", _coerce_unit_range("tension_velocity_range", self.tension_velocity_range))
        object.__setattr__(self, "tension_expression_range", _coerce_unit_range("tension_expression_range", self.tension_expression_range))

@dataclass(frozen=True)
class PlanConfig:
    method: PlanMethod = PlanMethod.METHOD_A
    sectioning_strategy: SectioningStrategy = SectioningStrategy.SINGLE_PASS
    loop_midpoint: Optional[int] = None
    endpoint_top_k: int = 8
    endpoint_temperature: float = 1.0
    start_anchor_weight: float = 1.0
    end_anchor_weight: float = 1.0
    section_names: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.method, PlanMethod):
            raise TypeError("method must be a PlanMethod value.")
        if not isinstance(self.sectioning_strategy, SectioningStrategy):
            raise TypeError("sectioning_strategy must be a SectioningStrategy value.")
        if self.loop_midpoint is not None:
            _require_int("loop_midpoint", self.loop_midpoint, minimum=1)
        _require_int("endpoint_top_k", self.endpoint_top_k, minimum=1)
        _require_real("endpoint_temperature", self.endpoint_temperature, minimum=0.0)
        if self.endpoint_temperature == 0.0:
            raise ValueError("endpoint_temperature must be > 0.")
        _require_real("start_anchor_weight", self.start_anchor_weight, minimum=0.0)
        _require_real("end_anchor_weight", self.end_anchor_weight, minimum=0.0)
        if self.start_anchor_weight == 0.0 and self.end_anchor_weight == 0.0:
            raise ValueError("At least one endpoint anchor weight must be > 0.")
        section_names = tuple(self.section_names)
        if any(not isinstance(name, str) or not name.strip() for name in section_names):
            raise ValueError("section_names entries must be non-empty strings.")
        object.__setattr__(self, "section_names", section_names)
        if self.method is PlanMethod.METHOD_B and self.loop_midpoint is None:
            raise ValueError("loop_midpoint is required when method is METHOD_B.")
        if self.method is PlanMethod.METHOD_A and self.loop_midpoint is not None:
            raise ValueError("loop_midpoint is only valid when method is METHOD_B.")
        if self.sectioning_strategy is SectioningStrategy.SECTION_WISE and not self.section_names:
            raise ValueError("section_names are required for SECTION_WISE planning.")

# ==============================================================================
# SUMMARY & ARTIFACT LOCATION:
# All modules are archived and viewable on disk at:
# file:///c:/Users/hp/musicGeneration-1/aimusic_complete_code.py
# ==============================================================================
