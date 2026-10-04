from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    NeuralPriorConfig,
    PlanConfig,
    PlanMethod,
    PriorWeights,
    SBConfig,
    SectioningStrategy,
    StyleConfig,
)
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer, Score
from aimusic.core.rng import RNGKey, allocate_named_keys, random_unit
from aimusic.core.vocab import TonalContext, Vocabularies, build_tonal_context
from aimusic.decode import decode_path_to_score
from aimusic.planning.graph import EdgeScoreDiagnostics, SparseGraph, build_sparse_graph
from aimusic.planning.sb import (
    SBProblem,
    SBSolution,
    SampledBridgePath,
    SolvedBridge,
    build_sb_problem,
    map_bridge_path,
    sample_bridge_path,
    solve_sb,
)
from aimusic.scoring.priors import NullPrior, Prior
from aimusic.theory.edo import EDO
from aimusic.theory.tonal import get_fifth_steps


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _state_sort_key(state: BeatState) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        state.meter_id,
        state.beat_in_bar,
        state.boundary_lvl,
        state.key_id,
        state.chord_id,
        state.role_id,
        state.head_id,
        state.groove_id,
    )


@dataclass(frozen=True)
class PlanningSection:
    """Single section descriptor for structural planning and generative section guidance."""

    name: str
    start_time: int
    end_time: int
    boundary_level: int
    target_tension_arc: Tuple[float, ...] = (0.2, 0.85, 0.25)
    allowed_meters: Optional[Tuple[str, ...]] = None
    groove_family: Optional[str] = None
    target_key_id: Optional[int] = None
    target_density: Optional[float] = None
    preferred_roles: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        _require_int("start_time", self.start_time, minimum=0)
        _require_int("end_time", self.end_time, minimum=1)
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be > start_time.")
        _require_int("boundary_level", self.boundary_level, minimum=0)
        arc = tuple(float(item) for item in self.target_tension_arc)
        if len(arc) < 2:
            raise ValueError("target_tension_arc must contain at least two values.")
        for idx, value in enumerate(arc):
            _require_real(f"target_tension_arc[{idx}]", value)
        object.__setattr__(self, "target_tension_arc", arc)

        if self.allowed_meters is not None:
            meters = tuple(self.allowed_meters)
            if not meters or any(not isinstance(m, str) or not m.strip() for m in meters):
                raise ValueError("allowed_meters must be a tuple of non-empty strings.")
            object.__setattr__(self, "allowed_meters", meters)
        if self.groove_family is not None:
            if not isinstance(self.groove_family, str) or not self.groove_family.strip():
                raise ValueError("groove_family must be a non-empty string.")
        if self.target_key_id is not None:
            _require_int("target_key_id", self.target_key_id, minimum=0)
        if self.target_density is not None:
            _require_real("target_density", self.target_density, minimum=0.0)
            if self.target_density > 1.0:
                raise ValueError("target_density must be <= 1.0.")
        if self.preferred_roles is not None:
            roles = tuple(self.preferred_roles)
            if not roles or any(not isinstance(r, str) or not r.strip() for r in roles):
                raise ValueError("preferred_roles must be a tuple of non-empty strings.")
            object.__setattr__(self, "preferred_roles", roles)


def get_section_at_time(
    sections: Sequence[PlanningSection], t: int
) -> Optional[PlanningSection]:
    """Find the PlanningSection governing beat time index t."""
    for section in sections:
        if section.start_time <= t < section.end_time:
            return section
    return None



@dataclass(frozen=True)
class MethodARunConfig:
    """Pure run configuration bundle for EPIC 6 Method A orchestration."""

    total_beats: int
    seed: int = 0
    use_sampling: bool = False
    style_config: StyleConfig = field(default_factory=StyleConfig)
    prior_weights: PriorWeights = field(default_factory=PriorWeights)
    sb_config: Optional[SBConfig] = None
    decode_config: DecodeConfig = field(default_factory=DecodeConfig)
    plan_config: PlanConfig = field(default_factory=PlanConfig)
    neural_prior_config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    edo: int = 12
    sections: Optional[Tuple[PlanningSection, ...]] = None

    def __post_init__(self) -> None:
        _require_int("total_beats", self.total_beats, minimum=1)
        _require_int("seed", self.seed, minimum=0)
        if not isinstance(self.use_sampling, bool):
            raise TypeError("use_sampling must be a bool.")
        if self.plan_config.method is not PlanMethod.METHOD_A:
            raise ValueError("MethodARunConfig requires plan_config.method == METHOD_A.")
        _require_int("edo", self.edo, minimum=1)
        if self.sb_config is not None and self.sb_config.horizon_t != self.total_beats:
            raise ValueError("sb_config.horizon_t must equal total_beats for Method A runs.")
        if (
            self.plan_config.sectioning_strategy is SectioningStrategy.SECTION_WISE
            and len(self.plan_config.section_names) > self.total_beats
        ):
            raise ValueError(
                "SECTION_WISE planning requires total_beats >= len(section_names)."
            )
        if self.sections is not None:
            secs = tuple(self.sections)
            if not secs:
                raise ValueError("sections tuple cannot be empty.")
            if any(not isinstance(s, PlanningSection) for s in secs):
                raise TypeError("sections must contain only PlanningSection instances.")
            object.__setattr__(self, "sections", secs)


@dataclass(frozen=True)
class EndpointChoice:
    """Explicit chosen endpoint state plus provenance within a candidate distribution."""

    state: BeatState
    source_distribution: EndpointDistribution
    selected_index: int
    selected_probability: float
    selection_mode: str

    def __post_init__(self) -> None:
        if self.state not in self.source_distribution.layer.states:
            raise ValueError("state must belong to source_distribution.layer.")
        _require_int("selected_index", self.selected_index, minimum=0)
        if self.selected_index >= len(self.source_distribution.layer.states):
            raise ValueError("selected_index must be within source_distribution support.")
        _require_real("selected_probability", self.selected_probability)
        if not isinstance(self.selection_mode, str) or not self.selection_mode.strip():
            raise ValueError("selection_mode must be a non-empty string.")


@dataclass(frozen=True)
class MethodAEndpoints:
    """Endpoint distributions and section metadata for a Method A run."""

    pi0: EndpointDistribution
    piT: EndpointDistribution
    start_choice: EndpointChoice
    end_choice: EndpointChoice
    sections: Tuple[PlanningSection, ...]


@dataclass(frozen=True)
class TransitionDiagnostic:
    """Per-transition diagnostic metrics for a selected path."""

    time_index: int
    source_state: BeatState
    target_state: BeatState
    data_logp: float
    gttm_energy: float
    gttm_family_breakdown: Dict[str, float]
    target_tension: float
    realized_tension: float
    tension_deviation: float
    target_tension_penalty: float
    section_style_penalty: float
    section_style_breakdown: Dict[str, float]
    total_log_weight: float



@dataclass(frozen=True)
class MethodAPlanDiagnostics:
    """Inspectable diagnostics emitted by Method A orchestration."""

    section_tags: Tuple[str, ...]
    target_tension_arcs: Tuple[Tuple[float, ...], ...]
    chosen_start_state: BeatState
    chosen_end_state: BeatState
    endpoint_selection_mode: str
    chosen_start_probability: float
    chosen_end_probability: float
    path_mode: str
    graph_layer_sizes: Tuple[int, ...]
    bridge_iterations: int
    bridge_converged: bool
    rng_stream_ids: Tuple[str, ...]
    transition_diagnostics: Tuple[TransitionDiagnostic, ...] = ()


@dataclass(frozen=True)
class MethodAPlanResult:
    """Full output of a Method A planning pass."""

    run_config: MethodARunConfig
    tonal_context: TonalContext
    vocabularies: Vocabularies
    endpoints: MethodAEndpoints
    graph: SparseGraph
    sb_problem: SBProblem
    sb_solution: SBSolution
    bridge: SolvedBridge
    path: Tuple[BeatState, ...]
    path_score: Optional[float]
    sampled_path: Optional[SampledBridgePath]
    diagnostics: MethodAPlanDiagnostics
    path_edge_diagnostics: Tuple[EdgeScoreDiagnostics, ...] = ()


@dataclass(frozen=True)
class ExactBridgeDemoResult:
    """Compatibility wrapper for legacy bridge-demo scripts."""

    plan_result: MethodAPlanResult
    score: Score
    output_path: str


def _resolved_vocabs(
    vocabularies: Optional[Vocabularies],
    style_config: StyleConfig,
    edo: int,
) -> Vocabularies:
    return build_tonal_context(
        edo,
        style_config,
        vocabularies=vocabularies,
    ).vocabularies


def _resolved_sb_config(run_config: MethodARunConfig) -> SBConfig:
    if run_config.sb_config is not None:
        return run_config.sb_config
    return SBConfig(horizon_t=run_config.total_beats)


def _softmax(scores: Sequence[float], temperature: float) -> Tuple[float, ...]:
    logits = np.asarray(tuple(float(score) for score in scores), dtype=float)
    if logits.ndim != 1 or logits.size == 0:
        raise ValueError("scores must be a non-empty 1D sequence.")
    scaled = logits / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    normalized = weights / np.sum(weights)
    return tuple(float(value) for value in normalized)


def _align_endpoint_distribution(
    endpoint: EndpointDistribution,
    layer: Layer,
) -> EndpointDistribution:
    masses = [endpoint.probability_of(state) for state in layer.states]
    total = float(sum(masses))
    if total <= 0.0:
        raise ValueError("Endpoint support vanished after graph construction.")
    return EndpointDistribution(
        layer=layer,
        probabilities=tuple(mass / total for mass in masses),
    )


def _singleton_endpoint_distribution(state: BeatState, *, time_index: int) -> EndpointDistribution:
    return EndpointDistribution(
        layer=Layer(time_index=time_index, states=(state,)),
        probabilities=(1.0,),
    )


def _sample_index_from_distribution(
    endpoint: EndpointDistribution,
    key: RNGKey,
) -> tuple[int, RNGKey]:
    threshold, next_key = random_unit(key)
    running = 0.0
    for idx, probability in enumerate(endpoint.probabilities):
        running += probability
        if threshold <= running:
            return idx, next_key
    return len(endpoint.probabilities) - 1, next_key


def _choose_endpoint_state(
    endpoint: EndpointDistribution,
    *,
    key: RNGKey,
    sample: bool,
) -> tuple[EndpointChoice, RNGKey]:
    if sample:
        selected_index, next_key = _sample_index_from_distribution(endpoint, key)
        selection_mode = "sample"
    else:
        selected_index = max(
            range(len(endpoint.probabilities)),
            key=lambda idx: (endpoint.probabilities[idx], -idx),
        )
        next_key = key
        selection_mode = "argmax"
    return (
        EndpointChoice(
            state=endpoint.layer.states[selected_index],
            source_distribution=endpoint,
            selected_index=selected_index,
            selected_probability=endpoint.probabilities[selected_index],
            selection_mode=selection_mode,
        ),
        next_key,
    )


def build_section_plan(run_config: MethodARunConfig) -> Tuple[PlanningSection, ...]:
    if run_config.sections is not None:
        return run_config.sections

    plan_config = run_config.plan_config
    if plan_config.sectioning_strategy is SectioningStrategy.SINGLE_PASS:
        name = (
            plan_config.section_names[0]
            if plan_config.section_names
            else "method_a_single_pass"
        )
        return (
            PlanningSection(
                name=name,
                start_time=0,
                end_time=run_config.total_beats,
                boundary_level=3,
                allowed_meters=run_config.style_config.allowed_meters,
                groove_family=run_config.style_config.groove_families[0] if run_config.style_config.groove_families else None,
            ),
        )

    section_names = plan_config.section_names
    section_count = len(section_names)
    if section_count > run_config.total_beats:
        raise ValueError(
            "SECTION_WISE planning requires total_beats >= len(section_names)."
        )
    chunk = run_config.total_beats // section_count
    remainder = run_config.total_beats % section_count
    sections = []
    cursor = 0
    groove_fams = run_config.style_config.groove_families
    for idx, name in enumerate(section_names):
        length = chunk + (1 if idx < remainder else 0)
        next_cursor = cursor + max(1, length)
        groove_fam = groove_fams[idx % len(groove_fams)] if groove_fams else None
        sections.append(
            PlanningSection(
                name=name,
                start_time=cursor,
                end_time=next_cursor,
                boundary_level=3 if idx == section_count - 1 else 2,
                target_tension_arc=(0.2 + (0.1 * idx), 0.8, 0.25),
                allowed_meters=run_config.style_config.allowed_meters,
                groove_family=groove_fam,
            )
        )
        cursor = next_cursor
    last = sections[-1]
    if last.end_time != run_config.total_beats:
        sections[-1] = PlanningSection(
            name=last.name,
            start_time=last.start_time,
            end_time=run_config.total_beats,
            boundary_level=last.boundary_level,
            target_tension_arc=last.target_tension_arc,
            allowed_meters=last.allowed_meters,
            groove_family=last.groove_family,
            target_key_id=last.target_key_id,
            target_density=last.target_density,
            preferred_roles=last.preferred_roles,
        )
    return tuple(sections)


def _meter_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for signature in style_config.allowed_meters:
        if signature in vocabularies.meters.label_map:
            ids.append(vocabularies.meters.token_for_label(signature).id)
    if not ids:
        ids.append(vocabularies.meters.token_for_id(0).id)
    return tuple(dict.fromkeys(ids))


def _key_anchor_ids(run_config: MethodARunConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    fifth = get_fifth_steps(run_config.edo) % len(vocabularies.keys)
    anchors = (0, fifth, len(vocabularies.keys) // 2)
    return tuple(dict.fromkeys(anchor % len(vocabularies.keys) for anchor in anchors))


def _chord_id_for(key_id: int, quality: str, vocabularies: Vocabularies) -> int:
    for chord in vocabularies.chords:
        if chord.root_pc == key_id and chord.quality == quality:
            return chord.id
    return vocabularies.chords.token_for_id(0).id


def _groove_anchor_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    ids = []
    for groove in vocabularies.grooves:
        if groove.family in style_config.groove_families:
            ids.append(groove.id)
    return tuple(dict.fromkeys(ids[: max(1, min(4, len(ids)))]))


def _endpoint_boundary_level(*, is_start: bool, beat_in_bar: int, strong_beats: Tuple[int, ...]) -> int:
    if beat_in_bar not in strong_beats:
        return 0
    if is_start:
        return 3
    return 2 if beat_in_bar == 0 else 1


def _candidate_score(
    state: BeatState,
    *,
    is_start: bool,
    boundary_level: int,
    primary_key_id: int,
    section: Optional[PlanningSection] = None,
    vocabularies: Optional[Vocabularies] = None,
    weights: Optional[PriorWeights] = None,
) -> float:
    score = 0.0
    score += 2.0 if state.beat_in_bar == 0 else 0.4
    score += 1.5 if state.boundary_lvl == boundary_level else 0.0
    score += 1.2 if state.key_id == primary_key_id else 0.5
    if is_start:
        score += 1.1 if state.role_id == 0 else 0.0
        score += 0.8 if state.head_id == 1 else 0.2
    else:
        score += 1.1 if state.role_id == 3 else 0.4
        score += 0.8 if state.head_id == 1 else 0.3

    if weights is not None and section is not None and vocabularies is not None:
        if weights.lambda_section_style > 0.0:
            if section.allowed_meters and vocabularies.meters.has_id(state.meter_id):
                meter_label = vocabularies.meters.token_for_id(state.meter_id).label
                if meter_label in section.allowed_meters:
                    score += 1.0 * weights.lambda_section_style
            if section.target_key_id is not None and state.key_id == section.target_key_id:
                score += 1.5 * weights.lambda_section_style
            if section.groove_family and vocabularies.grooves.has_id(state.groove_id):
                groove_tok = vocabularies.grooves.token_for_id(state.groove_id)
                if groove_tok.family == section.groove_family:
                    score += 1.0 * weights.lambda_section_style
            if section.preferred_roles and vocabularies.roles.has_id(state.role_id):
                role_label = vocabularies.roles.token_for_id(state.role_id).label
                if role_label in section.preferred_roles:
                    score += 1.0 * weights.lambda_section_style
        if weights.lambda_target_tension > 0.0:
            target_t = section.target_tension_arc[0] if is_start else section.target_tension_arc[-1]
            from aimusic.scoring.tension import beat_tension
            rel_t = beat_tension(None, state, vocabularies, len(vocabularies.keys))
            dev = abs(rel_t - target_t)
            score -= dev * 2.0 * weights.lambda_target_tension

    return score


def _build_endpoint_distribution(
    *,
    time_index: int,
    beat_in_bar_by_meter: dict[int, int],
    is_start: bool,
    run_config: MethodARunConfig,
    vocabularies: Vocabularies,
    sections: Optional[Sequence[PlanningSection]] = None,
) -> EndpointDistribution:
    plan_config = run_config.plan_config
    sec = (
        get_section_at_time(sections, time_index if is_start else max(0, time_index - 1))
        if sections
        else None
    )

    has_style_guidance = (
        sec is not None
        and run_config.prior_weights is not None
        and run_config.prior_weights.lambda_section_style > 0.0
    )

    groove_ids = list(_groove_anchor_ids(run_config.style_config, vocabularies))
    if has_style_guidance and sec is not None and sec.groove_family:
        sec_grooves = [g.id for g in vocabularies.grooves if g.family == sec.groove_family]
        if sec_grooves:
            groove_ids = list(dict.fromkeys(sec_grooves + groove_ids))

    key_ids = list(_key_anchor_ids(run_config, vocabularies))
    if has_style_guidance and sec is not None and sec.target_key_id is not None and vocabularies.keys.has_id(sec.target_key_id):
        key_ids = list(dict.fromkeys([sec.target_key_id] + key_ids))

    head_ids = (1, 2)
    chord_qualities = ("maj", "min")

    scored_candidates: list[tuple[float, BeatState]] = []
    allowed_meter_ids = _meter_ids(run_config.style_config, vocabularies)
    if has_style_guidance and sec is not None and sec.allowed_meters:
        sec_meters = [
            vocabularies.meters.token_for_label(m).id
            for m in sec.allowed_meters
            if m in vocabularies.meters.label_map
        ]
        if sec_meters:
            allowed_meter_ids = tuple(dict.fromkeys(sec_meters + list(allowed_meter_ids)))

    for meter_id in allowed_meter_ids:
        beat_in_bar = beat_in_bar_by_meter[meter_id]
        strong_beats = vocabularies.meters.token_for_id(meter_id).strong_beats
        boundary_level = _endpoint_boundary_level(is_start=is_start, beat_in_bar=beat_in_bar, strong_beats=strong_beats)
        if has_style_guidance and sec is not None and sec.boundary_level > 0 and beat_in_bar == 0:
            boundary_level = sec.boundary_level

        if is_start or boundary_level > 0:
            role_ids = (0, 1) if is_start else (3, 2)
        else:
            role_ids = (0, 1)

        if has_style_guidance and sec is not None and sec.preferred_roles:
            pref_roles = [
                vocabularies.roles.token_for_label(r).id
                for r in sec.preferred_roles
                if r in vocabularies.roles.label_map
            ]
            if pref_roles:
                role_ids = tuple(dict.fromkeys(pref_roles + list(role_ids)))

        for key_id in key_ids:
            for quality in chord_qualities:
                chord_id = _chord_id_for(key_id, quality, vocabularies)
                for role_id in role_ids:
                    for head_id in head_ids:
                        for groove_id in groove_ids:
                            state = BeatState(
                                meter_id=meter_id,
                                beat_in_bar=beat_in_bar,
                                boundary_lvl=boundary_level,
                                key_id=key_id,
                                chord_id=chord_id,
                                role_id=role_id,
                                head_id=head_id,
                                groove_id=groove_id,
                            )
                            score = _candidate_score(
                                state,
                                is_start=is_start,
                                boundary_level=boundary_level,
                                primary_key_id=key_ids[0],
                                section=sec,
                                vocabularies=vocabularies,
                                weights=run_config.prior_weights,
                            )
                            score += (
                                run_config.plan_config.start_anchor_weight
                                if is_start
                                else run_config.plan_config.end_anchor_weight
                            )
                            scored_candidates.append((score, state))

    scored_candidates.sort(key=lambda item: (-item[0], _state_sort_key(item[1])))
    unique_states: list[BeatState] = []
    unique_scores: list[float] = []
    seen = set()
    for score, state in scored_candidates:
        if state in seen:
            continue
        seen.add(state)
        unique_states.append(state)
        unique_scores.append(score)
        if len(unique_states) >= plan_config.endpoint_top_k:
            break

    layer = Layer(time_index=time_index, states=tuple(unique_states))
    return EndpointDistribution(
        layer=layer,
        probabilities=_softmax(unique_scores, plan_config.endpoint_temperature),
    )



def generate_start_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    sections: Optional[Sequence[PlanningSection]] = None,
) -> EndpointDistribution:
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
    beat_positions = {meter_id: 0 for meter_id in _meter_ids(run_config.style_config, resolved_vocabs)}
    return _build_endpoint_distribution(
        time_index=0,
        beat_in_bar_by_meter=beat_positions,
        is_start=True,
        run_config=run_config,
        vocabularies=resolved_vocabs,
        sections=sections,
    )


def generate_end_endpoint_distribution(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    sections: Optional[Sequence[PlanningSection]] = None,
) -> EndpointDistribution:
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
    beat_positions = {}
    for meter_id in _meter_ids(run_config.style_config, resolved_vocabs):
        beats_per_bar = resolved_vocabs.meters.token_for_id(meter_id).beats_per_bar
        beat_positions[meter_id] = run_config.total_beats % beats_per_bar
    return _build_endpoint_distribution(
        time_index=run_config.total_beats,
        beat_in_bar_by_meter=beat_positions,
        is_start=False,
        run_config=run_config,
        vocabularies=resolved_vocabs,
        sections=sections,
    )


def generate_method_a_endpoints(
    run_config: MethodARunConfig,
    *,
    vocabularies: Optional[Vocabularies] = None,
    key: RNGKey,
    sample_endpoints: bool = False,
) -> tuple[MethodAEndpoints, RNGKey]:
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    resolved_vocabs = _resolved_vocabs(
        vocabularies, run_config.style_config, run_config.edo
    )
    sections = build_section_plan(run_config)
    pi0 = generate_start_endpoint_distribution(run_config, vocabularies=resolved_vocabs, sections=sections)
    piT = generate_end_endpoint_distribution(run_config, vocabularies=resolved_vocabs, sections=sections)
    start_choice, next_key = _choose_endpoint_state(
        pi0,
        key=key,
        sample=sample_endpoints,
    )
    end_choice, next_key = _choose_endpoint_state(
        piT,
        key=next_key,
        sample=sample_endpoints,
    )
    return MethodAEndpoints(
        pi0=pi0,
        piT=piT,
        start_choice=start_choice,
        end_choice=end_choice,
        sections=sections,
    ), next_key


def run_method_a(
    run_config: MethodARunConfig,
    *,
    key: RNGKey,
    prior: Optional[Prior] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[MethodAPlanResult, RNGKey]:
    """Run Method A from endpoint planning through SB path extraction."""
    _logger.info(f"Method A: {run_config.total_beats} beats, seed={run_config.seed}")
    tonal_context = build_tonal_context(
        run_config.edo,
        run_config.style_config,
        vocabularies=vocabularies,
    )
    resolved_vocabs = tonal_context.vocabularies
    resolved_sb = _resolved_sb_config(run_config)
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    stream_ids = (
        "endpoint_choice", "candidate_proposal", "bridge_sampling",
        "decoder.comping", "decoder.bass", "decoder.lead", "decoder.drums",
    )
    streams, next_key = allocate_named_keys(key, stream_ids)
    endpoints, _ = generate_method_a_endpoints(
        run_config,
        vocabularies=resolved_vocabs,
        key=streams["endpoint_choice"],
        sample_endpoints=run_config.use_sampling,
    )
    _logger.info(f"Endpoints: start={endpoints.start_choice.state} end={endpoints.end_choice.state}")
    start_endpoint = _singleton_endpoint_distribution(
        endpoints.start_choice.state,
        time_index=0,
    )
    end_endpoint = _singleton_endpoint_distribution(
        endpoints.end_choice.state,
        time_index=run_config.total_beats,
    )
    graph, _ = build_sparse_graph(
        start_layer=start_endpoint.layer,
        end_layer=end_endpoint.layer,
        total_beats=run_config.total_beats,
        sb_config=resolved_sb,
        style_config=run_config.style_config,
        vocabularies=resolved_vocabs,
        prior=NullPrior() if prior is None else prior,
        weights=run_config.prior_weights,
        edo=run_config.edo,
        key=streams["candidate_proposal"],
        d_max=resolved_sb.d_max,
        sections=endpoints.sections,
    )
    _logger.info(f"Graph built: {len(graph.layers)} layers, {sum(len(l.states) for l in graph.layers)} states")
    aligned_endpoints = MethodAEndpoints(
        pi0=_align_endpoint_distribution(endpoints.pi0, graph.layers[0]),
        piT=_align_endpoint_distribution(endpoints.piT, graph.layers[-1]),
        start_choice=endpoints.start_choice,
        end_choice=endpoints.end_choice,
        sections=endpoints.sections,
    )
    problem = build_sb_problem(graph, start_endpoint, end_endpoint, sb_config=resolved_sb)
    solution = solve_sb(problem)
    bridge = solution.to_bridge()
    _logger.info(f"SB solved: converged={solution.trace.converged}, iterations={solution.trace.iterations}")

    if run_config.use_sampling:
        sampled_path, _ = sample_bridge_path(bridge, streams["bridge_sampling"], include_edges=True, include_debug=True)
        path = sampled_path.path
        path_score = None
        _logger.info(f"Sampled path: {len(path) - 1} beats")
    else:
        path, path_score = map_bridge_path(bridge)
        sampled_path = None
        _logger.info(f"MAP path: {len(path) - 1} beats")

    from aimusic.scoring.gttm_features import calculate_gttm_energy, transition_family_scores
    from aimusic.scoring.priors import PriorContext
    from aimusic.scoring.tension import (
        beat_tension,
        section_style_breakdown,
        section_style_energy,
        target_tension_at_time,
    )

    trans_diags: list[TransitionDiagnostic] = []
    prev_st: Optional[BeatState] = None
    for idx, st in enumerate(path):
        if idx > 0 and prev_st is not None:
            t = idx - 1
            t_tension = target_tension_at_time(endpoints.sections, t)
            r_tension = beat_tension(prev_st, st, resolved_vocabs, run_config.edo)
            dev = r_tension - t_tension
            sec = get_section_at_time(endpoints.sections, t)
            t_pen = run_config.prior_weights.lambda_target_tension * (dev ** 2)
            s_pen = run_config.prior_weights.lambda_section_style * section_style_energy(
                prev_st, st, sec, resolved_vocabs, run_config.edo
            )
            s_breakdown = section_style_breakdown(
                prev_st, st, sec, resolved_vocabs, run_config.edo
            )
            data_logp = (
                prior.logp_next(
                    prev_st,
                    st,
                    t,
                    PriorContext(history=(prev_st,), section_name=sec.name if sec else None),
                )
                if prior is not None
                else 0.0
            )
            gttm_e = calculate_gttm_energy(
                prev_st,
                st,
                t,
                vocabularies=resolved_vocabs,
                edo=run_config.edo,
                weights=run_config.prior_weights,
            )
            gttm_families = transition_family_scores(
                prev_st, st, t, vocabularies=resolved_vocabs, edo=run_config.edo
            )
            edge_weight = 0.0
            if t < len(graph.edges_by_time):
                for e in graph.edges_by_time[t]:
                    if e.source == prev_st and e.target == st:
                        edge_weight = e.log_weight
                        break
            trans_diags.append(
                TransitionDiagnostic(
                    time_index=t,
                    source_state=prev_st,
                    target_state=st,
                    data_logp=float(data_logp),
                    gttm_energy=float(gttm_e),
                    gttm_family_breakdown=gttm_families,
                    target_tension=t_tension,
                    realized_tension=r_tension,
                    tension_deviation=dev,
                    target_tension_penalty=t_pen,
                    section_style_penalty=s_pen,
                    section_style_breakdown=s_breakdown,
                    total_log_weight=edge_weight,
                )
            )
        prev_st = st


    diagnostics = MethodAPlanDiagnostics(
        section_tags=tuple(section.name for section in endpoints.sections),
        target_tension_arcs=tuple(section.target_tension_arc for section in endpoints.sections),
        chosen_start_state=endpoints.start_choice.state,
        chosen_end_state=endpoints.end_choice.state,
        endpoint_selection_mode=endpoints.start_choice.selection_mode,
        chosen_start_probability=endpoints.start_choice.selected_probability,
        chosen_end_probability=endpoints.end_choice.selected_probability,
        path_mode="sample" if run_config.use_sampling else "map",
        graph_layer_sizes=graph.diagnostics.layer_sizes,
        bridge_iterations=solution.trace.iterations,
        bridge_converged=solution.trace.converged,
        rng_stream_ids=stream_ids,
        transition_diagnostics=tuple(trans_diags),
    )
    return MethodAPlanResult(
        run_config=run_config,
        tonal_context=tonal_context,
        vocabularies=resolved_vocabs,
        endpoints=aligned_endpoints,
        graph=graph,
        sb_problem=problem,
        sb_solution=solution,
        bridge=bridge,
        path=path,
        path_score=path_score,
        sampled_path=sampled_path,
        diagnostics=diagnostics,
        path_edge_diagnostics=graph.diagnostics_for_path(path),
    ), next_key


def render_exact_bridge_demo(
    *,
    start_chord: str,
    end_chord: str,
    output_path: str,
    total_beats: int,
    seed: int = 0,
    meter: str = "4/4",
    groove: str = "straight_8ths",
    style_config: Optional[StyleConfig] = None,
    decode_config: Optional[DecodeConfig] = None,
    tempo_bpm: float = 120.0,
    edo: int = 12,
) -> ExactBridgeDemoResult:
    """Render a short bridge example using the current Method A pipeline.

    `start_chord`, `end_chord`, `meter`, and `groove` are accepted for compatibility with
    legacy scripts. The current implementation delegates endpoint selection to Method A.
    """
    del start_chord, end_chord, meter, groove

    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_decode = DecodeConfig() if decode_config is None else decode_config
    run_config = MethodARunConfig(
        total_beats=total_beats,
        seed=seed,
        style_config=resolved_style,
        decode_config=resolved_decode,
        edo=edo,
    )
    plan_result, next_key = run_method_a(run_config, key=RNGKey(seed=seed))
    score, _ = decode_path_to_score(
        plan_result.path,
        decode_config=resolved_decode,
        vocabularies=plan_result.vocabularies,
        edo=plan_result.tonal_context.n,
        tempo_bpm=tempo_bpm,
        key=next_key,
    )
    from aimusic.render import render_midi
    from aimusic.theory.tonal import EDO

    render_midi(
        score,
        EDO(EDOConfig(n=plan_result.tonal_context.n, base_tuning=0)),
        output_path,
    )
    return ExactBridgeDemoResult(
        plan_result=plan_result,
        score=score,
        output_path=output_path,
    )
