import argparse
import json
import sys
from pathlib import Path

from aimusic.core.diagnostics import RunManifest, StructuralDiagnostics, TimelineEvent, compute_tension_curve

def handle_generate(args: argparse.Namespace) -> None:
    print(f"Starting generation with seed: {args.seed}")
    
    # Simulate a generated timeline structure
    roles = [
        TimelineEvent(0.0, 4.0, "Tonic"),
        TimelineEvent(4.0, 8.0, "Subdominant"),
        TimelineEvent(8.0, 12.0, "Dominant"),
        TimelineEvent(12.0, 16.0, "Tonic"),
    ]
    tension = compute_tension_curve(roles)
    
    struct_stats = StructuralDiagnostics(
        key_timeline=[TimelineEvent(0.0, 16.0, "C Major")],
        chord_timeline=[TimelineEvent(0.0, 8.0, "Cmaj7"), TimelineEvent(8.0, 16.0, "G7")],
        role_timeline=roles,
        boundaries=[0.0, 8.0, 16.0],
        tension_curve=tension
    )
    
    manifest = RunManifest(
        seed=args.seed,
        config_dump={"edo": args.edo, "horizon": 16},
        structural_stats=struct_stats
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
    print(f"Seed: {data.get('seed')} | Config: {data.get('config')}")
    
    structure = data.get("structure", {})
    print("\n--- Structural Timelines ---")
    print(f"Keys:   {[k['label'] for k in structure.get('key_timeline', [])]}")
    print(f"Chords: {[c['label'] for c in structure.get('chord_timeline', [])]}")
    print(f"Roles:  {[r['label'] for r in structure.get('role_timeline', [])]}")
    
    print("\n--- Tension Arc ---")
    for time, tension in structure.get("tension_curve", []):
        bar = "█" * int(tension * 20)  # Simple ASCII bar chart
        print(f"Beat {time:04.1f}: {bar} ({tension})")
    print("=========================================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="GTTM + SB Symbolic Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a new symbolic score")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--edo", type=int, default=12, help="Equal division of the octave")
    gen_parser.add_argument("--out", type=str, default="./outputs", help="Output directory")
    gen_parser.set_defaults(func=handle_generate)

    ins_parser = subparsers.add_parser("inspect", help="Inspect diagnostics from a previous run")
    ins_parser.add_argument("file", type=str, help="Path to the manifest JSON file")
    ins_parser.set_defaults(func=handle_inspect)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()