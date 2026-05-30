import argparse
import json
import sys
from pathlib import Path

from aimusic.core.diagnostics import (
    RunManifest, StructuralDiagnostics, TimelineEvent, compute_tension_curve, SBDiagnostics
)

def handle_generate(args: argparse.Namespace) -> None:
    print(f"Starting generation with seed: {args.seed}")
    
    roles = [TimelineEvent(0.0, 4.0, "Tonic"), TimelineEvent(4.0, 8.0, "Dominant")]
    struct_stats = StructuralDiagnostics(
        key_timeline=[TimelineEvent(0.0, 8.0, "C Major")],
        role_timeline=roles,
        tension_curve=compute_tension_curve(roles)
    )
    
    # Mock SB Diagnostics until the full pipeline
    sb_stats = SBDiagnostics(
        iterations_run=45, converged=True, final_max_delta=1e-5,
        layer_sizes=[12, 24, 24, 12], pruned_nodes=3, effective_entropy=1.2
    )
    
    manifest = RunManifest(
        seed=args.seed, config_dump={"edo": args.edo}, 
        structural_stats=struct_stats, sb_stats=sb_stats
    )
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{manifest.run_id}_manifest.json"
    
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    print(f"Run manifest saved to: {manifest_path}")

def handle_inspect(args: argparse.Namespace) -> None:
    manifest_path = Path(args.file)
    if not manifest_path.exists():
        print(f"Error: Could not find manifest at {args.file}")
        sys.exit(1)
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    print(f"\n=== Inspection Report for Run: {data.get('run_id')} ===")
    
    # --- SB Math Diagnostics ---
    sb = data.get("sb_stats", {})
    print("\n--- Schrödinger Bridge Health ---")
    status = "🟢 Converged" if sb.get("converged") else "🔴 FAILED"
    print(f"Status:      {status} (in {sb.get('iterations_run')} iterations)")
    print(f"Max Delta:   {sb.get('final_max_delta')}")
    print(f"Entropy:     {sb.get('effective_entropy'):.4f} (Lower = More Confident)")
    print(f"Pruned dead: {sb.get('pruned_nodes')} nodes")
    print(f"Layer sizes: {sb.get('layer_sizes')}")

    # --- Structural Timelines ---
    structure = data.get("structure", {})
    print("\n--- Tension Arc ---")
    for time_val, tension in structure.get("tension_curve", []):
        bar = "█" * int(tension * 20)
        print(f"Beat {time_val:04.1f}: {bar} ({tension})")
    print("=========================================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="GTTM + SB Symbolic Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a new score")
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--edo", type=int, default=12)
    gen_parser.add_argument("--out", type=str, default="./outputs")
    gen_parser.set_defaults(func=handle_generate)

    ins_parser = subparsers.add_parser("inspect", help="Inspect diagnostics")
    ins_parser.add_argument("file", type=str)
    ins_parser.set_defaults(func=handle_inspect)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()