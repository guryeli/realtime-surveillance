"""
Detection management:
  - save jpg + json (tespit_<MODULE>_<YYYY-MM-DD_HH-MM-SS>.{jpg,json})
  - play sound (with cooldown)
  - hold "active alarm" state for the top banner
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ..modules.base import Alarm
from ..utils.audio import play_sound
from ..utils.paths import DETECTIONS_DIR, SOUNDS_DIR


@dataclass
class BannerState:
    active: bool = False
    text: str = ""
    module: str = ""
    severity: str = "ALARM"
    started_at: float = 0.0
    expires_at: float = 0.0


@dataclass
class AlertEvent:
    timestamp: str
    module: str
    severity: str
    message: str
    image_path: str
    json_path: str
    extra: dict = field(default_factory=dict)


class AlertManager:
    def __init__(self, alert_settings, app_settings_dict_provider):
        """app_settings_dict_provider: callable → full settings.to_dict()"""
        self.settings = alert_settings
        self._get_settings_dict = app_settings_dict_provider
        self.banner = BannerState()
        self._sound_path = SOUNDS_DIR / self.settings.sound_file
        self._dets_dir = (
            Path(self.settings.detections_dir)
            if Path(self.settings.detections_dir).is_absolute()
            else DETECTIONS_DIR
        )
        self._dets_dir.mkdir(parents=True, exist_ok=True)
        self._history: List[AlertEvent] = []
        self._sound_cooldown_per_module: dict[str, float] = {}

    # --------------- public --------------- #
    def handle(self, frame: np.ndarray, alarms: List[Alarm], frame_idx: int,
               t_sec: float) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        if not alarms:
            self._tick_banner(t_sec)
            return events

        # Banner: show the latest alarm
        top = alarms[-1]
        self.banner = BannerState(
            active=True,
            text=top.message,
            module=top.module,
            severity=top.severity,
            started_at=t_sec,
            expires_at=t_sec + float(self.settings.banner_duration_seconds),
        )

        for alarm in alarms:
            ev = self._save(frame, alarm, frame_idx, t_sec)
            if ev:
                events.append(ev)
                self._history.append(ev)

        # Sound (1 s cooldown per module — avoids spam if 5 modules alarm at once)
        if self.settings.play_sound:
            now = time.time()
            last = self._sound_cooldown_per_module.get(top.module, 0.0)
            if now - last > 1.0:
                self._sound_cooldown_per_module[top.module] = now
                play_sound(self._sound_path, cooldown_s=0.0)

        return events

    def _tick_banner(self, t_sec: float) -> None:
        if self.banner.active and t_sec >= self.banner.expires_at:
            self.banner = BannerState()

    # --------------- internal --------------- #
    def _save(self, frame: np.ndarray, alarm: Alarm, frame_idx: int,
              t_sec: float) -> Optional[AlertEvent]:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # multiple alarms in the same second -> append microseconds
        ts_full = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
        base = f"tespit_{alarm.module}_{ts_full}"
        img_path = self._dets_dir / f"{base}.jpg"
        json_path = self._dets_dir / f"{base}.json"

        img_str, json_str = "", ""
        if self.settings.save_screenshots:
            try:
                vis = self._draw_for_snapshot(frame, alarm)
                cv2.imwrite(str(img_path), vis,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                img_str = str(img_path)
            except Exception as e:
                print(f"[alert] screenshot save error: {e}")

        if self.settings.save_json:
            payload = {
                "timestamp": ts,
                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                "module": alarm.module,
                "severity": alarm.severity,
                "message": alarm.message,
                "frame_index": int(frame_idx),
                "video_t_sec": float(t_sec),
                "detections": [
                    {
                        "module": d.module,
                        "bbox": list(d.bbox),
                        "label": d.label,
                        "confidence": d.confidence,
                        "extra": d.extra,
                    }
                    for d in alarm.detections
                ],
                "extra": alarm.extra,
                "image_path": img_str,
                "settings_snapshot": self._get_settings_dict(),
            }
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)
                json_str = str(json_path)
            except Exception as e:
                print(f"[alert] json save error: {e}")

        return AlertEvent(
            timestamp=ts,
            module=alarm.module,
            severity=alarm.severity,
            message=alarm.message,
            image_path=img_str,
            json_path=json_str,
            extra=alarm.extra,
        )

    @staticmethod
    def _draw_for_snapshot(frame: np.ndarray, alarm: Alarm) -> np.ndarray:
        vis = frame.copy()
        for d in alarm.detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), d.color, 2)
            txt = d.label
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)),
                          (x1 + tw + 6, y1), d.color, -1)
            cv2.putText(vis, txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        # banner
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w, 40), (0, 0, 200), -1)
        cv2.putText(vis, f"ALARM [{alarm.module}] {alarm.message}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return vis
