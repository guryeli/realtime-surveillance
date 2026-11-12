"""
Cross-platform alarm sound.

Plays WAV/OGG with `pygame.mixer`. On no-audio setups it fails silently —
the application flow is not interrupted. Cooldown prevents dozens of
sounds from overlapping in the same second.
"""
from __future__ import annotations

import math
import struct
import time
import wave
from pathlib import Path
from threading import Lock


_LOCK = Lock()
_LAST_PLAY: dict[str, float] = {}


def _ensure_pygame_init() -> bool:
    try:
        import pygame

        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
        return True
    except Exception as e:
        print(f"[audio] pygame init failed: {e}")
        return False


def generate_alarm_wav(out_path: Path, duration_s: float = 0.6,
                       freq_hz: int = 880, sample_rate: int = 22050) -> Path:
    """Generate a short synthetic alarm tone (WAV). ~6 KB."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = int(duration_s * sample_rate)
    amp = 16000
    with wave.open(str(out_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # Two-tone siren: 880 ↔ 660 Hz
        for i in range(n):
            t = i / sample_rate
            f = freq_hz if (i // (sample_rate // 6)) % 2 == 0 else int(freq_hz * 0.75)
            sample = int(amp * math.sin(2 * math.pi * f * t))
            # Amplitude envelope (attack/decay)
            env = min(i / 800, (n - i) / 800, 1.0)
            wf.writeframes(struct.pack("<h", int(sample * env)))
    return out_path


def play_sound(path: Path | str, cooldown_s: float = 0.5) -> bool:
    """Play a sound. A repeat call for the same file within cooldown_s is silently ignored."""
    p = str(path)
    now = time.time()
    with _LOCK:
        last = _LAST_PLAY.get(p, 0.0)
        if now - last < cooldown_s:
            return False
        _LAST_PLAY[p] = now

    if not _ensure_pygame_init():
        return False
    try:
        import pygame

        snd = pygame.mixer.Sound(p)
        snd.play()
        return True
    except Exception as e:
        print(f"[audio] play failed: {e}")
        return False


def shutdown() -> None:
    try:
        import pygame

        if pygame.mixer.get_init():
            pygame.mixer.quit()
    except Exception:
        pass
