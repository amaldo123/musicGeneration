import time
import uuid
import dataclasses  
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

@dataclass(frozen=True)
class TimelineEvent:
    start_time: float
    end_time: float
    label: str

@dataclass
class StructuralDiagnostics:
    key_timeline: List[TimelineEvent] = field(default_factory=list)
    chord_timeline: List[TimelineEvent] = field(default_factory=list)
    role_timeline: List[TimelineEvent] = field(default_factory=list)
    groove_timeline: List[TimelineEvent] = field(default_factory=list)
    boundaries: List[float] = field(default_factory=list)
    tension_curve: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_timeline": [dataclasses.asdict(e) for e in self.key_timeline],
            "chord_timeline": [dataclasses.asdict(e) for e in self.chord_timeline],
            "role_timeline": [dataclasses.asdict(e) for e in self.role_timeline],
            "groove_timeline": [dataclasses.asdict(e) for e in self.groove_timeline],
            "boundaries": self.boundaries,
            "tension_curve": self.tension_curve
        }

@dataclass(frozen=True)
class RunManifest:
    """Captures all parameters required to perfectly reproduce a generation run."""
    seed: int
    config_dump: Dict[str, Any]
    structural_stats: StructuralDiagnostics = field(default_factory=StructuralDiagnostics)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    version: str = "0.1.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "seed": self.seed,
            "config": self.config_dump,
            "structure": self.structural_stats.to_dict()
        }

def compute_tension_curve(role_timeline: List[TimelineEvent]) -> List[Tuple[float, float]]:
    tension_map = {
        "Tonic": 0.1,
        "Subdominant": 0.5,
        "Dominant": 0.9,
        "Transition": 0.6
    }
    
    curve = []
    for event in role_timeline:
        tension = tension_map.get(event.label, 0.5)
        curve.append((event.start_time, tension))
        
    return curve