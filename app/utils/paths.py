"""Path helpers based on the project root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
DETECTIONS_DIR = ROOT / "detections"
FACE_DB_DIR = ROOT / "face_database"
SOUNDS_DIR = ROOT / "sounds"
LOGS_DIR = ROOT / "logs"
CONFIG_PATH = ROOT / "settings.json"

for d in (DETECTIONS_DIR, FACE_DB_DIR, SOUNDS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def model_path(module: str, filename: str) -> Path:
    """models/<module>/<filename>"""
    return MODELS_DIR / module / filename
