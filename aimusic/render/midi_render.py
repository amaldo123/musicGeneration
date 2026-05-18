import mido
from dataclasses import dataclass
from typing import List, Tuple

from aimusic.theory.edo import EDO

@dataclass(frozen=True)
class SymbolicNote:
    """A renderer-agnostic representation of a musical note."""
    pitch_height: int
    start_time: float  
    end_time: float    
    velocity: int = 64

def render_midi(
    notes: List[SymbolicNote], 
    edo: EDO, 
    output_path: str, 
    ticks_per_beat: int = 480
) -> None:
    """
    Renders a symbolic score into a deterministic 12-EDO MIDI file.
    """
    if edo.config.n != 12:
        raise NotImplementedError("Currently, only 12-EDO rendering is fully supported.")

    events: List[Tuple[int, int, str, int, int, int]] = []
    
    for note in notes:
        midi_note, _ = edo.to_midi(note.pitch_height)
        
        start_tick = int(note.start_time * ticks_per_beat)
        end_tick = int(note.end_time * ticks_per_beat)
        
        events.append((end_tick, 0, 'note_off', midi_note, 0, 0))
        events.append((start_tick, 1, 'note_on', midi_note, note.velocity, 0))
        
    events.sort()
    
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track.append(mido.MetaMessage('track_name', name='12-EDO Export', time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    
    current_tick = 0
    for abs_tick, _, msg_type, note_num, vel, ch in events:
        delta_tick = abs_tick - current_tick
        
        track.append(mido.Message(
            msg_type, 
            note=note_num, 
            velocity=vel, 
            time=delta_tick, 
            channel=ch
        ))
        
        current_tick = abs_tick
        
    track.append(mido.MetaMessage('end_of_track', time=0))
    
    mid.save(output_path)