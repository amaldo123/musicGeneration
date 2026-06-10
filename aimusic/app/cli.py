import argparse
import json
import sys
from pathlib import Path

from aimusic.core.config import EDOConfig
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render.midi_render import SymbolicNote, render_midi
from aimusic.theory.edo import EDO

from aimusic.core.diagnostics import (
    RunManifest, StructuralDiagnostics, TimelineEvent, compute_tension_curve, SBDiagnostics
)

def handle_generate(args: argparse.Namespace) -> None:
    """Handles the 'generate' CLI command."""
    print(f"Starting generation with seed: {args.seed}")
    result = run_method_a(MethodARunConfig(
        total_beats=args.beats, seed=args.seed, edo=args.edo,
    ))
    print(f"Pipeline: {len(result.path) - 1} beats, converged={result.sb_solution.trace.converged}")

    score = decode_path_to_score(result.path, vocabularies=result.vocabularies, edo=args.edo)
    print(f"Decoded: {len(score.note_events)} notes")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    midi_path = out_dir / f"output_{args.seed}.mid"
    tpb = score.ticks_per_beat
    edo_obj = EDO(EDOConfig(n=args.edo))
    notes_out = []
    for n in score.note_events:
        h = n.h
        midi_note, _ = edo_obj.to_midi(h)
        while midi_note > 127:
            h -= args.edo
            midi_note, _ = edo_obj.to_midi(h)
        while midi_note < 0:
            h += args.edo
            midi_note, _ = edo_obj.to_midi(h)
        notes_out.append(SymbolicNote(h, n.ton / tpb, n.toff / tpb, int(n.v * 127)))
    render_midi(notes_out, edo_obj, str(midi_path))
    print(f"MIDI file saved to: {midi_path}")
    
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
        seed=args.seed,
        config_dump={"beats": args.beats, "edo": args.edo},
        structural_stats=struct_stats, sb_stats=sb_stats
    )
    
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = out_dir / f"{manifest.run_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        
    print(f"Run manifest saved to: {manifest_path}")
    print("Generation complete. (Pipeline wiring pending...)")

def handle_inspect(args: argparse.Namespace) -> None:
    """Handles the 'inspect' CLI command."""
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
    structure = data.get("structural_stats", {})
    print("\n--- Tension Arc ---")
    for time_val, tension in structure.get("tension_curve", []):
        bar = "█" * int(tension * 20)
        print(f"Beat {time_val:04.1f}: {bar} ({tension})")
    print("=========================================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="GTTM + SB Symbolic Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a new symbolic score")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--edo", type=int, default=12, help="Equal division of the octave (e.g., 12, 19)")
    gen_parser.add_argument("--out", type=str, default="./outputs", help="Output directory")
    gen_parser.set_defaults(func=handle_generate)

    ins_parser = subparsers.add_parser("inspect", help="Inspect diagnostics")
    ins_parser.add_argument("file", type=str)
    ins_parser.set_defaults(func=handle_inspect)

    
    exp_parser = subparsers.add_parser("export", help="Export a generated score to MIDI")
    exp_parser.add_argument("file", type=str, help="Path to the score data")
    
    exp_parser.set_defaults(func=lambda args: print("Export command placeholder invoked."))

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()