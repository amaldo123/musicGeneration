from __future__ import annotations

from typing import Any

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import Vocabularies


def _style_config_snapshot(style_config: StyleConfig) -> dict[str, Any]:
    return {
        "allowed_meters": list(style_config.allowed_meters),
        "subdivision_patterns": list(style_config.subdivision_patterns),
        "groove_families": list(style_config.groove_families),
        "chord_vocabulary_size": style_config.chord_vocabulary_size,
        "key_vocabulary_size": style_config.key_vocabulary_size,
        "bass_register": list(style_config.bass_register),
        "comping_register": list(style_config.comping_register),
        "lead_register": list(style_config.lead_register),
        "typical_density_range": list(style_config.typical_density_range),
    }


def export_vocabularies_json(
    vocabularies: Vocabularies,
    style_config: StyleConfig,
) -> dict[str, Any]:
    """Export token enumerations and the StyleConfig snapshot for neural training."""
    if not isinstance(vocabularies, Vocabularies):
        raise TypeError("vocabularies must be a Vocabularies instance.")
    if not isinstance(style_config, StyleConfig):
        raise TypeError("style_config must be a StyleConfig instance.")

    return {
        "style_config": _style_config_snapshot(style_config),
        "meter": [
            {
                "id": token.id,
                "label": token.label,
                "beats_per_bar": token.beats_per_bar,
                "strong_beats": list(token.strong_beats),
            }
            for token in vocabularies.meters
        ],
        "beat_position": [
            {"id": token.id, "label": token.label, "index": token.index}
            for token in vocabularies.beat_positions
        ],
        "boundary": [
            {"id": token.id, "label": token.label, "level": token.level}
            for token in vocabularies.boundaries
        ],
        "key": [
            {"id": token.id, "label": token.label, "root_pc": token.root_pc}
            for token in vocabularies.keys
        ],
        "chord": [
            {
                "id": token.id,
                "label": token.label,
                "root_pc": token.root_pc,
                "quality": token.quality,
            }
            for token in vocabularies.chords
        ],
        "role": [
            {"id": token.id, "label": token.label, "description": token.description}
            for token in vocabularies.roles
        ],
        "head": [
            {"id": token.id, "label": token.label, "description": token.description}
            for token in vocabularies.heads
        ],
        "groove": [
            {
                "id": token.id,
                "label": token.label,
                "family": token.family,
                "subdivision": token.subdivision,
            }
            for token in vocabularies.grooves
        ],
    }


def vocabularies_from_export(data: dict[str, Any]) -> tuple[dict[str, Any], StyleConfig]:
    """Parse an exported vocabulary document back into a StyleConfig snapshot."""
    style_data = data.get("style_config")
    if not isinstance(style_data, dict):
        raise ValueError("vocabularies export must contain a style_config object.")
    style_config = StyleConfig(
        allowed_meters=tuple(style_data["allowed_meters"]),
        subdivision_patterns=tuple(style_data["subdivision_patterns"]),
        groove_families=tuple(style_data["groove_families"]),
        chord_vocabulary_size=int(style_data["chord_vocabulary_size"]),
        key_vocabulary_size=int(style_data["key_vocabulary_size"]),
        bass_register=tuple(style_data["bass_register"]),
        comping_register=tuple(style_data["comping_register"]),
        lead_register=tuple(style_data["lead_register"]),
        typical_density_range=tuple(style_data["typical_density_range"]),
    )
    return data, style_config
