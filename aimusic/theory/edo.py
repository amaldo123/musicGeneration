import math
from typing import Tuple
from aimusic.core.config import EDOConfig, MicrotonalRendering


class EDO:
    """
    Represents an Equal Division of the Octave (EDO) system
    and provides utilities for pitch math.
    """

    def __init__(self, config: EDOConfig):
        self.config = config

    def pitch_class(self, h: int) -> int:
        """
        Calculates the pitch class for a given pitch height.

        Args:
            h: The pitch height in EDO steps.

        Returns:
            The pitch class as an integer in Z_n.
        """
        return h % self.config.n

    def to_midi(self, h: int) -> Tuple[int, int]:
        """
        Converts an EDO pitch height to a MIDI note and pitch bend.

        Args:
            h: The pitch height in EDO steps.

        Returns:
            A tuple containing the MIDI note number and the pitch bend value.
            The pitch bend is an integer from -8192 to 8191.
        """
        if not isinstance(h, int) or isinstance(h, bool):
            raise TypeError("h must be an int measured in EDO steps.")

        if self.config.microtonal_rendering_method == MicrotonalRendering.MTS:
            midi_note_float = self.config.base_tuning + h * (12.0 / self.config.n)
            return (int(round(midi_note_float)), 0)

        # MPE rendering. Choose the nearest MIDI key and encode the remaining
        # fractional semitone exactly as a channel pitch bend.
        target_midi_pitch = self.config.base_tuning + h * (12.0 / self.config.n)
        nearest_midi_note = math.floor(target_midi_pitch + 0.5)
        if nearest_midi_note < 0 or nearest_midi_note > 127:
            raise ValueError(
                f"EDO pitch height {h} maps outside the MIDI note range: "
                f"{target_midi_pitch:.6f}."
            )

        semitone_offset = target_midi_pitch - nearest_midi_note
        bend_fraction = semitone_offset / self.config.pitch_bend_range
        if abs(bend_fraction) > 1.0:
            raise ValueError(
                f"Pitch-bend range {self.config.pitch_bend_range} cannot represent "
                f"EDO pitch height {h}."
            )

        # MIDI pitch bend is asymmetric: -8192 is full-scale down and
        # +8191 is full-scale up.
        scale = 8191 if bend_fraction >= 0.0 else 8192
        pitch_bend = max(-8192, min(8191, round(bend_fraction * scale)))
        return (nearest_midi_note, pitch_bend)

    def __repr__(self) -> str:
        return f"EDO(n={self.config.n})"
