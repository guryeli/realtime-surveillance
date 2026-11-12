"""
Face detection — YOLOv8L-Face (11.7 ms median, 0.685 AP@0.5 — the fastest option).

Model tested on WIDER Face. Bboxes are fed to the face-matching module
(when FaceMatch is enabled).
"""
from __future__ import annotations

import time
from typing import List

import numpy as np

from .base import (
    Alarm,
    BaseModule,
    Detection,
    MODULE_FACE_DETECT,
    ModuleResult,
)
from ..utils.paths import model_path


class FaceDetectModule(BaseModule):
    name = MODULE_FACE_DETECT

    def __init__(self, settings, device: str):
        super().__init__(settings, device)
        self._yolo = None

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        path = model_path("face_detect", self.settings.yolo_model)
        self._yolo = YOLO(str(path))
        self._loaded = True

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
            min_size = self.settings.min_face_size
            for (x1, y1, x2, y2), conf in zip(xyxy, confs):
                w = x2 - x1
                h = y2 - y1
                if w < min_size or h < min_size:
                    continue
                detections.append(
                    Detection(
                        module=self.name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=f"face {conf:.2f}",
                        confidence=float(conf),
                        color=(255, 200, 0),
                        extra={"size": (int(w), int(h))},
                    )
                )

        return ModuleResult(
            module=self.name,
            detections=detections,
            alarms=[],
            inference_ms=infer_ms,
            extra={"faces": len(detections)},
        )
