from __future__ import annotations

import argparse
from pathlib import Path

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import DEFAULT_GROOVE_FAMILIES, DEFAULT_METER_SIGNATURES
from aimusic.planning.plans import render_exact_bridge_demo


DEFAULT_GROOVE_LABELS = {
    "straight": "straight_8ths",
    "syncopated": "syncopated_8ths",
    "swing": "swing_8ths",
}


def _parse_csv(value: str) -> tuple[str, ...]:
    parts = tuple(item.strip() for item in value.split(",") if item.strip())
    if not parts:
        raise argparse.ArgumentTypeError("value must contain at least one non-empty item.")
    return parts


def _beats_for_meter(meter_label: str, bars: int) -> int:
    beats_text, _, denominator = meter_label.partition("/")
    if not beats_text or denominator != "4":
        raise ValueError(f"Unsupported meter label: {meter_label}")
    return int(beats_text) * bars


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a short SB bridge MIDI matrix across meters and groove families."
    )
    parser.add_argument("--start", required=True, help="Start chord label, e.g. C:maj")
    parser.add_argument("--end", required=True, help="End chord label, e.g. G:dom")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory where the generated MIDI files will be written",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed")
    parser.add_argument(
        "--meters",
        type=_parse_csv,
        default=DEFAULT_METER_SIGNATURES,
        help="Comma-separated meter labels",
    )
    parser.add_argument(
        "--families",
        type=_parse_csv,
        default=DEFAULT_GROOVE_FAMILIES,
        help="Comma-separated groove families",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=4,
        help="Bars per generated example; keeps outputs short and bar-aligned",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for meter in args.meters:
        for family in args.families:
            if family not in DEFAULT_GROOVE_LABELS:
                raise ValueError(
                    f"Unsupported groove family: {family}. "
                    f"Expected one of {tuple(DEFAULT_GROOVE_LABELS)}."
                )
            total_beats = _beats_for_meter(meter, args.bars)
            groove_label = DEFAULT_GROOVE_LABELS[family]
            output_path = out_dir / f"bridge_{meter.replace('/', '-')}_{family}.mid"
            result = render_exact_bridge_demo(
                start_chord=args.start,
                end_chord=args.end,
                output_path=str(output_path),
                total_beats=total_beats,
                seed=args.seed,
                meter=meter,
                groove=groove_label,
                style_config=StyleConfig(
                    allowed_meters=(meter,),
                    groove_families=(family,),
                ),
            )
            seconds = total_beats * 0.5
            print(
                f"{output_path}\t{meter}\t{family}\tbeats={total_beats}"
                f"\tseconds={seconds:.1f}\tevents={len(result.score.note_events)}"
            )


if __name__ == "__main__":
    main()
