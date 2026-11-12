"""CUDA / MPS / CPU autodetect and listing."""
from __future__ import annotations

from typing import List


def list_devices() -> List[str]:
    """Return every available device: ['cpu', 'mps', 'cuda:0', ...]."""
    devices = ["cpu"]
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f"cuda:{i}")
    except ImportError:
        pass
    return devices


def auto_select() -> str:
    devs = list_devices()
    for pref in ("cuda:0", "mps", "cpu"):
        if pref in devs:
            return pref
    return "cpu"


def describe(device: str) -> str:
    try:
        import torch
    except ImportError:
        return device
    if device.startswith("cuda"):
        idx = 0 if ":" not in device else int(device.split(":")[1])
        try:
            n = torch.cuda.get_device_name(idx)
            mem = torch.cuda.get_device_properties(idx).total_memory / 1024 ** 3
            return f"{device} → {n} ({mem:.1f} GB)"
        except Exception:
            return device
    if device == "mps":
        return "mps → Apple Silicon GPU"
    return f"{device} → CPU"
