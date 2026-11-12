"""Shared data structures between modules and the abstract Detector interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Module names (fixed for alarm metadata)
MODULE_ANOMALY = "ANOMALI"
MODULE_WEAPON = "SILAH"
MODULE_ABANDONED = "TERK"
MODULE_FACE_DETECT = "YUZ"
MODULE_FACE_MATCH = "YUZ_ESL"


@dataclass
class Detection:
    """A single bbox detection."""
    module: str
    bbox: Tuple[int, int, int, int]      # x1, y1, x2, y2
    label: str = ""
    confidence: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)
    color: Tuple[int, int, int] = (0, 255, 0)  # BGR


@dataclass
class Alarm:
    """Alarm/warning produced by a module (for the banner and disk save)."""
    module: str
    severity: str = "ALARM"               # "WARNING" | "ALARM"
    message: str = ""
    detections: List[Detection] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleResult:
    """One module's output for a single frame."""
    module: str
    detections: List[Detection] = field(default_factory=list)
    alarms: List[Alarm] = field(default_factory=list)
    inference_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseModule:
    """Every module implements this interface."""

    name: str = "base"

    def __init__(self, settings: Any, device: str):
        self.settings = settings
        self.device = device
        self._loaded = False

    def load(self) -> None:
        """Lazy: call before loading heavy models."""
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def process(self, frame: np.ndarray, frame_idx: int, t_sec: float) -> ModuleResult:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset tracker / state machine."""
        pass

    def close(self) -> None:
        pass
