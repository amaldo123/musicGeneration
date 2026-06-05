import argparse
import json
import sys
from pathlib import Path

from aimusic.core.config import EDOConfig
from aimusic.core.diagnostics import RunManifest
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render.midi_render import SymbolicNote, render_midi
from aimusic.theory.edo import EDO


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

    manifest = RunManifest(
        seed=args.seed,
        config_dump={"beats": args.beats, "edo": args.edo},
    )
    manifest_path = out_dir / f"{manifest.run_id}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
        
    print(f"Run manifest saved to: {manifest_path}")
    print("Generation complete. use 'inspect' cli command to view diagnostics.")

def handle_inspect(args: argparse.Namespace) -> None:
    """Handles the 'inspect' CLI command."""
    manifest_path = Path(args.file)
    if not manifest_path.exists():
        print(f"Error: Could not find manifest at {args.file}")
        sys.exit(1)
        
    with open(manifest_path, "r") as f:
        data = json.load(f)
        
    print(f"\n=== Inspection Report for Run: {data.get('run_id')} ===")
    print(f"Generated on: {data.get('timestamp')}")
    print(f"Seed:         {data.get('seed')}")
    print(f"Config:       {json.dumps(data.get('config'), indent=2)}")
    print("=========================================================\n")

def main() -> None:
    parser = argparse.ArgumentParser(description="GTTM + SB Symbolic Music Generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate a new symbolic score")
    gen_parser.add_argument("--beats", type=int, default=64, help="Number of beats to generate")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    gen_parser.add_argument("--edo", type=int, default=12, help="Equal division of the octave (e.g., 12, 19)")
    gen_parser.add_argument("--out", type=str, default="./outputs", help="Output directory")
    gen_parser.set_defaults(func=handle_generate)

    ins_parser = subparsers.add_parser("inspect", help="Inspect diagnostics from a previous run")
    ins_parser.add_argument("file", type=str, help="Path to the manifest JSON file")
    ins_parser.set_defaults(func=handle_inspect)

    
    exp_parser = subparsers.add_parser("export", help="Export a generated score to MIDI")
    exp_parser.add_argument("file", type=str, help="Path to the score data")
    
    exp_parser.set_defaults(func=lambda args: print("Export command placeholder invoked."))

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()