"""req04_make_baseline.py — Generate and record upstream baseline paths from commit 084c4a4."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM_COMMIT = "084c4a4d55eeff91b60d994b2ecf69b5cb90faae"
SEEDS = (7, 11, 23, 42, 101)
BEAT_COUNTS = (4, 16)
EDO19_CASE = {"seed": 11, "beats": 4, "edo": 19}
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "tests" / "fixtures" / "req04_baseline.json"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        worktree_path = Path(tmp_dir) / "baseline_tree"
        print(f"Creating git worktree at {worktree_path} for commit {UPSTREAM_COMMIT}...")
        subprocess.check_call(
            ["git", "worktree", "add", str(worktree_path), UPSTREAM_COMMIT],
            cwd=str(REPO_ROOT),
        )

        try:
            # Prepend worktree path so aimusic is loaded from upstream baseline
            sys.path.insert(0, str(worktree_path))

            from aimusic.core.config import StyleConfig  # type: ignore[import-not-found]
            from aimusic.core.rng import RNGKey  # type: ignore[import-not-found]
            from aimusic.planning.plans import MethodARunConfig, run_method_a  # type: ignore[import-not-found]

            style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
            results: dict[str, list[list[int]]] = {}

            for seed in SEEDS:
                for beats in BEAT_COUNTS:
                    key_name = f"seed_{seed}_beats_{beats}_edo_12"
                    print(f"Running baseline {key_name}...")
                    cfg = MethodARunConfig(
                        total_beats=beats,
                        seed=seed,
                        edo=12,
                        style_config=style,
                    )
                    res, _ = run_method_a(cfg, key=RNGKey(seed=seed))
                    raw_path = [
                        [
                            s.meter_id,
                            s.beat_in_bar,
                            s.boundary_lvl,
                            s.key_id,
                            s.chord_id,
                            s.role_id,
                            s.head_id,
                            s.groove_id,
                        ]
                        for s in res.path
                    ]
                    results[key_name] = raw_path

            # 19-EDO smoke case
            key_name_19 = f"seed_{EDO19_CASE['seed']}_beats_{EDO19_CASE['beats']}_edo_{EDO19_CASE['edo']}"
            print(f"Running baseline {key_name_19}...")
            cfg_19 = MethodARunConfig(
                total_beats=EDO19_CASE["beats"],
                seed=EDO19_CASE["seed"],
                edo=EDO19_CASE["edo"],
                style_config=style,
            )
            res_19, _ = run_method_a(cfg_19, key=RNGKey(seed=EDO19_CASE["seed"]))
            results[key_name_19] = [
                [
                    s.meter_id,
                    s.beat_in_bar,
                    s.boundary_lvl,
                    s.key_id,
                    s.chord_id,
                    s.role_id,
                    s.head_id,
                    s.groove_id,
                ]
                for s in res_19.path
            ]

            payload = {
                "metadata": {
                    "upstream_commit": UPSTREAM_COMMIT,
                    "python_version": sys.version,
                    "seeds": list(SEEDS),
                    "beat_counts": list(BEAT_COUNTS),
                    "edo_smoke": EDO19_CASE,
                    "command": "python scripts/req04_make_baseline.py",
                },
                "paths": results,
            }

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            with OUTPUT_FILE.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)

            print(f"Saved baseline paths to {OUTPUT_FILE}")

        finally:
            print("Cleaning up git worktree...")
            subprocess.check_call(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=str(REPO_ROOT),
            )


if __name__ == "__main__":
    main()
