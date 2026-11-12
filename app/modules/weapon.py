"""
Weapon detection — fine-tuned YOLOv8L (87.47% mAP@50, ~114 FPS GPU).

Multi-frame persistence: a weapon that appears and disappears in one frame
may be a false positive. The alarm fires only if it is visible for N
consecutive frames. Cooldown prevents spam.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, List

import numpy as np

from .base import (
    Alarm,
    BaseModule,
    Detection,
    MODULE_WEAPON,
    ModuleResult,
)
from ..utils.paths import model_path


class WeaponModule(BaseModule):
    name = MODULE_WEAPON

    def __init__(self, settings, device: str):
        super().__init__(settings, device)
        self._yolo = None
        self._consecutive_hits: Deque[bool] = deque(maxlen=8)
        self._last_alarm_t: float = -1e9

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        path = model_path("weapon", self.settings.yolo_model)
        self._yolo = YOLO(str(path))
        self._loaded = True

    def reset(self) -> None:
        self._consecutive_hits.clear()
        self._last_alarm_t = -1e9

    def process(self, frame: np.ndarray, frame_idx: int, t_sec: float) -> ModuleResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        results = self._yolo.predict(
            frame,
            conf=self.settings.conf_threshold,
            iou=self.settings.iou_threshold,
            device=self.device,
            verbose=False,
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            cls = results.boxes.cls.cpu().numpy().astype(int)
            names = results.names if hasattr(results, "names") else self._yolo.names
            for (x1, y1, x2, y2), conf, c in zip(xyxy, confs, cls):
                label = names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                detections.append(
                    Detection(
                        module=self.name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=f"{label} {conf:.2f}",
                        confidence=float(conf),
                        color=(0, 0, 255),
                        extra={"class": label},
                    )
                )

        # consecutive frame logic
        self._consecutive_hits.append(len(detections) > 0)
        recent_n = self.settings.min_consecutive_frames
        active = (
            len(self._consecutive_hits) >= recent_n
            and all(list(self._consecutive_hits)[-recent_n:])
        )
        alarms: List[Alarm] = []
        if active and detections and (t_sec - self._last_alarm_t) > 2.0:
            self._last_alarm_t = t_sec
            best = max(detections, key=lambda d: d.confidence)
            alarms.append(
                Alarm(
                    module=self.name,
                    severity="ALARM",
                    message=f"Weapon detected ({best.label})",
                    detections=detections.copy(),
                    extra={"top_class": best.extra.get("class", "")},
                )
            )

        return ModuleResult(
            module=self.name,
            detections=detections,
            alarms=alarms,
            inference_ms=infer_ms,
        )
