import mido
from dataclasses import dataclass
from typing import List, Tuple, Dict

from aimusic.theory.edo import EDO
from aimusic.core.config import MicrotonalRendering

@dataclass(frozen=True)
class SymbolicNote:
    """A renderer-agnostic representation of a musical note."""
    pitch_height: int
    start_time: float  
    end_time: float    
    velocity: int = 64  
    timbre: int | None = None      
    pressure: int | None = None    

def _allocate_channels(notes: List[SymbolicNote]) -> List[Tuple[SymbolicNote, int]]:
    """
    Allocates MIDI channels (1-15) for notes to support MPE-style polyphony.
    Channel 0 is reserved as the MPE master channel.
    """
    
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    allocated_notes = []
    channel_free_times: Dict[int, float] = {ch: 0.0 for ch in range(1, 16)}
    
    for note in sorted_notes:
        free_channels = [ch for ch, free_time in channel_free_times.items() if free_time <= note.start_time]
        
        if free_channels:
            chosen_ch = free_channels[0]
        else:
            chosen_ch = min(channel_free_times, key=lambda ch: channel_free_times[ch])
            
        channel_free_times[chosen_ch] = note.end_time
        allocated_notes.append((note, chosen_ch))
        
    return allocated_notes

def render_midi(
    notes: List[SymbolicNote], 
    edo: EDO, 
    output_path: str, 
    ticks_per_beat: int = 480
) -> None:
    """
    Renders a symbolic score into a deterministic MIDI file.
    Supports N-EDO tunings via MPE-style channel allocation and pitch bends.
    """
    
    if edo.config.microtonal_rendering_method == MicrotonalRendering.MTS:
        raise NotImplementedError(
            "MTS (MIDI Tuning Standard) rendering is currently deferred. "
            "Due to limited modern VST support, please use MicrotonalRendering.MPE."
        )

    allocated_notes = _allocate_channels(notes)

    events: List[Tuple[int, int, str, int, int, int]] = []
    
    for note, channel in allocated_notes:
        midi_note, pitch_bend = edo.to_midi(note.pitch_height)
        start_tick = int(note.start_time * ticks_per_beat)
        end_tick = int(note.end_time * ticks_per_beat)
        
        events.append((end_tick, 0, 'note_off', midi_note, 0, channel))
        
        if pitch_bend != 0:
            events.append((start_tick, 1, 'pitchwheel', pitch_bend, 0, channel))
            
        if note.timbre is not None:
            events.append((start_tick, 2, 'control_change', 74, note.timbre, channel))
            
        if note.pressure is not None:
            events.append((start_tick, 3, 'aftertouch', note.pressure, 0, channel))
        
        events.append((start_tick, 4, 'note_on', midi_note, note.velocity, channel))
        
    events.sort()
    
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    track_name = f"{edo.config.n}-EDO Export"
    track.append(mido.MetaMessage('track_name', name=track_name, time=0))
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
    
    current_tick = 0
    for abs_tick, _, msg_type, val1, val2, ch in events:
        delta_tick = abs_tick - current_tick
        
        if msg_type == 'pitchwheel':
            track.append(mido.Message('pitchwheel', pitch=val1, time=delta_tick, channel=ch))
        elif msg_type == 'control_change':
            track.append(mido.Message('control_change', control=val1, value=val2, time=delta_tick, channel=ch))
        elif msg_type == 'aftertouch':
            track.append(mido.Message('aftertouch', value=val1, time=delta_tick, channel=ch))
        else:
            track.append(mido.Message(msg_type, note=val1, velocity=val2, time=delta_tick, channel=ch))
            
        current_tick = abs_tick
        
    track.append(mido.MetaMessage('end_of_track', time=0))
    
    mid.save(output_path)