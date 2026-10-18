"""scripts/req04_reachable_range.py — Measure reachable tension distribution on legal transitions.

Samples realized tension over:
1. Random walks of 64 beats for seeds in SEEDS=(7, 11, 23, 42, 101)
2. All graph edges of named planning fixtures (RISE_16, FALL_16, PLATEAU_16, TWO_SECTION_16, THREE_SECTION_24)

Outputs results to tests/fixtures/req04_reachable_range.json and prints a summary table.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey, choice
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.candidates import get_valid_next_states
from aimusic.scoring.priors import NullPrior
from aimusic.scoring.tension import beat_tension
from tests.req04_support.fixtures import (
    FALL_16,
    PLATEAU_16,
    RISE_16,
    SEEDS,
    THREE_SECTION_24,
    TWO_SECTION_16,
)


def _compute_stats(values: List[float]) -> Dict[str, Any]:
    arr = np.array(values, dtype=np.float64)
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(np.max(arr)),
    }


def main() -> None:
    vocabs = DEFAULT_VOCABULARIES
    style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
    prior = NullPrior()

    overall_tensions: List[float] = []
    role_tensions: Dict[str, List[float]] = {"hold": [], "prep": [], "change": [], "cad": []}

    print("Sampling legal transitions from random walks (64 beats x 5 seeds)...")
    for seed in SEEDS:
        key = RNGKey(seed=seed)
        curr_state = BeatState(
            meter_id=0,
            beat_in_bar=0,
            boundary_lvl=2,
            key_id=0,
            chord_id=0,
            role_id=0,
            head_id=1,
            groove_id=0,
        )
        for t in range(64):
            gen_res, key = get_valid_next_states(
                curr_state,
                t,
                key,
                d_max=16,
                style_config=style,
                vocabularies=vocabs,
                prior=prior,
                edo=12,
            )
            candidates = gen_res.states
            if not candidates:
                break
            next_state, key = choice(key, candidates)
            tau = beat_tension(curr_state, next_state, vocabs, edo=12)
            role_label = vocabs.roles.token_for_id(next_state.role_id).label

            overall_tensions.append(tau)
            if role_label in role_tensions:
                role_tensions[role_label].append(tau)
            curr_state = next_state

    print("Sampling legal transitions from fixture graphs...")
    from aimusic.core.rng import allocate_named_keys
    from aimusic.planning.graph import build_sparse_graph
    from aimusic.planning.plans import (
        MethodARunConfig,
        _resolved_sb_config,
        _singleton_endpoint_distribution,
        build_tonal_context,
        generate_method_a_endpoints,
    )
    from aimusic.planning.sections import SectionTimeline

    fixtures = [RISE_16, FALL_16, PLATEAU_16, TWO_SECTION_16, THREE_SECTION_24]
    seen_edges: set[tuple[BeatState, BeatState]] = set()

    for fix in fixtures:
        for seed in SEEDS:
            cfg = MethodARunConfig(
                total_beats=fix.total_beats,
                seed=seed,
                edo=fix.edo,
                sections=fix.sections,
                prior_weights=PriorWeights(lambda_target_tension=0.0, lambda_section_style=0.0),
                style_config=fix.style_config or StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
                sb_config=fix.sb_config or SBConfig(horizon_t=fix.total_beats, k_max=16, d_max=4),
            )
            tonal_context = build_tonal_context(cfg.edo, cfg.style_config)
            fix_vocabs = tonal_context.vocabularies
            streams, _ = allocate_named_keys(RNGKey(seed=seed), ("endpoint_choice", "candidate_proposal"))
            endpoints, _ = generate_method_a_endpoints(cfg, vocabularies=fix_vocabs, key=streams["endpoint_choice"])
            start_ep = _singleton_endpoint_distribution(endpoints.start_choice.state, time_index=0)
            end_ep = _singleton_endpoint_distribution(endpoints.end_choice.state, time_index=cfg.total_beats)
            timeline = SectionTimeline(sections=endpoints.sections, total_beats=cfg.total_beats)
            graph, _ = build_sparse_graph(
                start_layer=start_ep.layer,
                end_layer=end_ep.layer,
                total_beats=cfg.total_beats,
                sb_config=_resolved_sb_config(cfg),
                style_config=cfg.style_config,
                vocabularies=fix_vocabs,
                prior=prior,
                key=streams["candidate_proposal"],
                edo=cfg.edo,
                weights=cfg.prior_weights,
                d_max=_resolved_sb_config(cfg).d_max,
                sections=endpoints.sections,
                timeline=timeline,
            )
            for layer in graph.edges_by_time:
                for edge in layer:
                    edge_key = (edge.source, edge.target)
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)

                    tau = beat_tension(edge.source, edge.target, fix_vocabs, edo=fix.edo)
                    role_label = fix_vocabs.roles.token_for_id(edge.target.role_id).label

                    overall_tensions.append(tau)
                    if role_label in role_tensions:
                        role_tensions[role_label].append(tau)

    # Git commit hash
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        commit_hash = "unknown"

    overall_stats = _compute_stats(overall_tensions)
    per_role_stats = {role: _compute_stats(vals) for role, vals in role_tensions.items() if vals}

    data = {
        "commit": commit_hash,
        "command": "python scripts/req04_reachable_range.py",
        "total_samples": len(overall_tensions),
        "overall": overall_stats,
        "per_role": per_role_stats,
    }

    out_path = Path("tests/fixtures/req04_reachable_range.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nWrote reachable range fixture to {out_path}")
    print("\n" + "=" * 80)
    print("REACHABLE TENSION DISTRIBUTION TABLE")
    print("=" * 80)
    headers = ["Category", "Count", "Min", "p5", "p25", "p50", "p75", "p95", "Max"]
    fmt = "{:<10} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format(*headers))
    print("-" * 80)

    row = ["overall", str(overall_stats["count"])] + [f"{overall_stats[k]:.4f}" for k in ["min", "p5", "p25", "p50", "p75", "p95", "max"]]
    print(fmt.format(*row))

    for role in ["hold", "prep", "change", "cad"]:
        if role in per_role_stats:
            s = per_role_stats[role]
            r_row = [role, str(s["count"])] + [f"{s[k]:.4f}" for k in ["min", "p5", "p25", "p50", "p75", "p95", "max"]]
            print(fmt.format(*r_row))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
