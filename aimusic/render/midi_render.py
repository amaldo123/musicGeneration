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
    Raises ValueError if more than 15 notes overlap simultaneously.
    """
    
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    allocated_notes = []
    channel_free_times: Dict[int, float] = {ch: 0.0 for ch in range(1, 16)}
    
    for note in sorted_notes:
        free_channels = [ch for ch, free_time in channel_free_times.items() if free_time <= note.start_time]
        
        # Explicit policy for MPE polyphony overflow.
        if not free_channels:
            raise ValueError(
                f"MPE polyphony limit exceeded: Attempted to play > 15 overlapping notes "
                f"at time {note.start_time}. No free channels available."
            )
            
        chosen_ch = free_channels[0]
        
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
    
    # Set Pitch Bend Range via RPN for each channel (except master)
    pb_range = edo.config.pitch_bend_range
    unique_channels = set(ch for _, ch in allocated_notes)
    
    for ch in unique_channels:
        events.append((0, -4, 'control_change', 101, 0, ch))        # RPN MSB
        events.append((0, -3, 'control_change', 100, 0, ch))        # RPN LSB (Pitch Bend Sensitivity)
        events.append((0, -2, 'control_change', 6, pb_range, ch))   # Data MSB (Semitones)
        events.append((0, -1, 'control_change', 38, 0, ch))         # Data LSB (Cents)
    
    for note, channel in allocated_notes:
        midi_note, pitch_bend = edo.to_midi(note.pitch_height)
        start_tick = int(note.start_time * ticks_per_beat)
        end_tick = int(note.end_time * ticks_per_beat)
        
        events.append((end_tick, 0, 'note_off', midi_note, 0, channel))
        
        # Explicit pitchwheel reset to clear stale MPE detunes
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

@dataclass(frozen=True)
class MidiSummary:
    """A statistical summary of a rendered MIDI file for quick inspection."""
    total_notes: int
    unique_channels: Tuple[int, ...]
    pitch_bend_events: int
    timbre_events: int
    pressure_events: int

    def print_report(self) -> None:
        """Prints a human-readable console report of the MIDI file."""
        print("\n=== MIDI Rendering Summary ===")
        print(f"Total Notes Played:   {self.total_notes}")
        print(f"Unique Channels Used: {len(self.unique_channels)} {self.unique_channels}")
        print(f"Pitch Bend Events:    {self.pitch_bend_events}")
        print(f"Timbre (CC74) Events: {self.timbre_events}")
        print(f"Pressure Events:      {self.pressure_events}")
        print("==============================\n")

def summarize_midi(filepath: str) -> MidiSummary:
    """Reads a .mid file from disk and tallies its expressive and structural contents."""
    mid = mido.MidiFile(filepath)
    
    note_count = 0
    channels = set()
    pb_count = 0
    timbre_count = 0
    pressure_count = 0
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count += 1
                channels.add(msg.channel)
            elif msg.type == 'pitchwheel':
                pb_count += 1
            elif msg.type == 'control_change' and msg.control == 74:
                timbre_count += 1
            elif msg.type == 'aftertouch':
                pressure_count += 1
                
    return MidiSummary(
        total_notes=note_count,
        unique_channels=tuple(sorted(channels)),
        pitch_bend_events=pb_count,
        timbre_events=timbre_count,
        pressure_events=pressure_count
    )