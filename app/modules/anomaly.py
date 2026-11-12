"""
Anomaly detection — crowd panic + density anomaly.

Compact integrated version of Method 1 (YOLO + ByteTrack + track-level
statistics) from the AnomaliDedection project.

Outputs:
  - density  : grid-based density (person count + hotspot)
  - panic    : ratio of running people + at least X runners
  - global   : scene-wide person count over the limit
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np

from .base import (
    Alarm,
    BaseModule,
    Detection,
    MODULE_ANOMALY,
    ModuleResult,
)
from ..utils.paths import model_path

# COCO person class id
PERSON_CLASS = 0


class AnomalyModule(BaseModule):
    name = MODULE_ANOMALY

    def __init__(self, settings, device: str):
        super().__init__(settings, device)
        self._yolo = None
        self._hist_maxlen = max(
            5, min(30, int(self.settings.speed_history_maxlen))
        )
        self._track_history: Dict[int, Deque[Tuple[float, float, float]]] = defaultdict(
            self._new_hist
        )
        # alarm state machine
        self._panic_start_t: float | None = None
        self._density_start_t: float | None = None
        self._last_panic_alarm_t: float = 0.0
        self._last_density_alarm_t: float = 0.0

    def _new_hist(self) -> Deque[Tuple[float, float, float]]:
        return deque(maxlen=self._hist_maxlen)

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        path = model_path("anomaly", self.settings.yolo_model)
        self._yolo = YOLO(str(path))
        self._loaded = True

    def reset(self) -> None:
        self._hist_maxlen = max(
            5, min(30, int(self.settings.speed_history_maxlen))
        )
        self._track_history = defaultdict(self._new_hist)
        self._panic_start_t = None
        self._density_start_t = None

    # --------------- core --------------- #
    def _track(self, frame: np.ndarray):
        s = self.settings
        imgsz = int(max(320, min(1280, s.imgsz)))
        # round to a multiple of 32 (YOLO expectation)
        imgsz = (imgsz // 32) * 32
        return self._yolo.track(
            frame,
            persist=True,
            classes=[PERSON_CLASS],
            conf=float(s.conf_threshold),
            iou=float(s.iou_threshold),
            imgsz=imgsz,
            max_det=int(max(1, min(1000, s.max_det))),
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )[0]

    def _compute_speeds(
        self, tracks: List[Tuple[int, float, float]], t_sec: float
    ) -> Dict[int, float]:
        """track_id → pixels/second speed. Sliding window (speed_history_maxlen)."""
        speeds: Dict[int, float] = {}
        for tid, cx, cy in tracks:
            hist = self._track_history[tid]
            hist.append((t_sec, cx, cy))
            if len(hist) >= 2:
                t0, x0, y0 = hist[0]
                dt = max(t_sec - t0, 1e-3)
                d = float(np.hypot(cx - x0, cy - y0))
                speeds[tid] = d / dt
            else:
                speeds[tid] = 0.0
        # Drop stale tracks
        active_ids = {t[0] for t in tracks}
        for k in list(self._track_history.keys()):
            if k not in active_ids:
                del self._track_history[k]
        return speeds

    def _density_signals(
        self, person_centers: List[Tuple[float, float]], h: int, w: int
    ) -> Tuple[bool, str, np.ndarray]:
        rows = max(1, self.settings.grid_rows)
        cols = max(1, self.settings.grid_cols)
        grid = np.zeros((rows, cols), dtype=int)
        if w == 0 or h == 0:
            return False, "", grid
        for cx, cy in person_centers:
            r = min(int(cy / h * rows), rows - 1)
            c = min(int(cx / w * cols), cols - 1)
            grid[r, c] += 1
        total = len(person_centers)
        local_max = int(grid.max()) if grid.size else 0

        global_alarm = total > self.settings.max_expected_people
        local_alarm = local_max > self.settings.max_people_per_cell
        triggered = global_alarm or local_alarm
        msg = ""
        if triggered:
            parts = []
            if global_alarm:
                parts.append(
                    f"total {total} > {self.settings.max_expected_people}"
                )
            if local_alarm:
                parts.append(
                    f"hotspot {local_max} > {self.settings.max_people_per_cell}"
                )
            msg = "Density anomaly: " + ", ".join(parts)
        return triggered, msg, grid

    def _panic_signals(
        self, speeds: Dict[int, float]
    ) -> Tuple[bool, str, int, int]:
        total = len(speeds)
        if total < self.settings.panic_min_total_people:
            return False, "", 0, total
        runners = sum(
            1 for s in speeds.values() if s > self.settings.running_speed_pixels_per_sec
        )
        if runners < self.settings.panic_min_runners:
            return False, "", runners, total
        ratio = runners / total
        if ratio < self.settings.panic_running_ratio:
            return False, "", runners, total
        msg = (
            f"Collective panic: {runners}/{total} people running "
            f"(ratio {ratio:.0%})"
        )
        return True, msg, runners, total

    # --------------- main entry --------------- #
    def process(self, frame: np.ndarray, frame_idx: int, t_sec: float) -> ModuleResult:
        if not self._loaded:
            self.load()
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        result = self._track(frame)
        infer_ms = (time.perf_counter() - t0) * 1000

        detections: List[Detection] = []
        tracks_xyc: List[Tuple[int, float, float]] = []  # (tid, cx, cy)
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for tid, (x1, y1, x2, y2), conf in zip(ids, xyxy, confs):
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                tracks_xyc.append((int(tid), float(cx), float(cy)))
                detections.append(
                    Detection(
                        module=self.name,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        label=f"P#{tid}",
                        confidence=float(conf),
                        color=(0, 200, 0),
                        extra={"track_id": int(tid)},
                    )
                )

        speeds = self._compute_speeds(tracks_xyc, t_sec)
        # attach speed info to detections
        sd = {tid: s for tid, s in speeds.items()}
        for d in detections:
            tid = d.extra.get("track_id")
            if tid in sd:
                d.extra["speed"] = round(sd[tid], 1)
                if sd[tid] > self.settings.running_speed_pixels_per_sec:
                    d.color = (0, 0, 255)
                    d.label = f"P#{tid} RUN {sd[tid]:.0f}px/s"
                else:
                    d.label = f"P#{tid} {sd[tid]:.0f}px/s"

        person_centers = [(t[1], t[2]) for t in tracks_xyc]
        density_alarm, density_msg, grid = self._density_signals(person_centers, h, w)
        panic_alarm, panic_msg, runners, total = self._panic_signals(speeds)

        # persistence (against noise)
        alarms: List[Alarm] = []
        now = t_sec
        if density_alarm:
            self._density_start_t = self._density_start_t or now
            elapsed = now - self._density_start_t
            if elapsed >= self.settings.alarm_persistence_seconds:
                if now - self._last_density_alarm_t > self.settings.alarm_cooldown_seconds:
                    self._last_density_alarm_t = now
                    alarms.append(
                        Alarm(
                            module=self.name,
                            severity="ALARM",
                            message=density_msg,
                            detections=detections.copy(),
                            extra={
                                "type": "density",
                                "people": total,
                                "grid": grid.tolist(),
                            },
                        )
                    )
        else:
            self._density_start_t = None

        if panic_alarm:
            self._panic_start_t = self._panic_start_t or now
            elapsed = now - self._panic_start_t
            if elapsed >= self.settings.alarm_persistence_seconds:
                if now - self._last_panic_alarm_t > self.settings.alarm_cooldown_seconds:
                    self._last_panic_alarm_t = now
                    alarms.append(
                        Alarm(
                            module=self.name,
                            severity="ALARM",
                            message=panic_msg,
                            detections=detections.copy(),
                            extra={
                                "type": "panic",
                                "runners": runners,
                                "total": total,
                            },
                        )
                    )
        else:
            self._panic_start_t = None

        return ModuleResult(
            module=self.name,
            detections=detections,
            alarms=alarms,
            inference_ms=infer_ms,
            extra={"people": total, "runners": runners, "grid": grid.tolist()},
        )
