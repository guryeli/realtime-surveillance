"""
Abandoned-object detection — compact integrated version of the TerkEdilmisNesne
project's Method 1 (Distance) approach.

Logic:
  1. Detect person + bag (backpack/handbag/suitcase) with YOLOv8m (COCO)
  2. Track both with Ultralytics ByteTrack
  3. When a bag track is first seen, it is assigned to the nearest person as owner
  4. If the owner moves farther than detach_distance and does not return for
     ≥ abandonment_seconds → ABANDONED → alarm

State machine:
  UNASSIGNED → OWNED → DETACHED → ABANDONED
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import (
    Alarm,
    BaseModule,
    Detection,
    MODULE_ABANDONED,
    ModuleResult,
)
from ..utils.paths import model_path


@dataclass
class BagState:
    track_id: int
    owner_track_id: Optional[int] = None
    state: str = "UNASSIGNED"      # UNASSIGNED | OWNED | DETACHED | ABANDONED
    last_seen_t: float = 0.0
    detached_since_t: Optional[float] = None
    last_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    label: str = ""
    alarm_fired: bool = False


def _center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2, (y1 + y2) / 2


class AbandonedModule(BaseModule):
    name = MODULE_ABANDONED

    def __init__(self, settings, device: str):
        super().__init__(settings, device)
        self._yolo = None
        self._bags: Dict[int, BagState] = {}
        self._last_alarm_t: float = -1e9

    def load(self) -> None:
        if self._loaded:
            return
        from ultralytics import YOLO

        path = model_path("abandoned", self.settings.yolo_model)
        self._yolo = YOLO(str(path))
        self._loaded = True

    def reset(self) -> None:
        self._bags.clear()
        self._last_alarm_t = -1e9

    def _track(self, frame: np.ndarray):
        classes = [self.settings.person_class_id] + list(self.settings.bag_class_ids)
        return self._yolo.track(
            frame,
            persist=True,
            classes=classes,
            conf=self.settings.conf_threshold,
            iou=self.settings.iou_threshold,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )[0]

    @staticmethod
    def _nearest_person(bag_center, persons: Dict[int, Tuple[float, float]]):
        if not persons:
            return None, float("inf")
        best_id, best_d = None, float("inf")
        bx, by = bag_center
        for pid, (px, py) in persons.items():
            d = float(np.hypot(bx - px, by - py))
            if d < best_d:
                best_id, best_d = pid, d
        return best_id, best_d

    def process(self, frame: np.ndarray, frame_idx: int, t_sec: float) -> ModuleResult:
        if not self._loaded:
            self.load()

        t0 = time.perf_counter()
        result = self._track(frame)
        infer_ms = (time.perf_counter() - t0) * 1000

        person_centers: Dict[int, Tuple[float, float]] = {}
        bag_records: List[Tuple[int, str, Tuple[int, int, int, int], float]] = []
        detections: List[Detection] = []
        names = result.names if hasattr(result, "names") else self._yolo.names

        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for tid, (x1, y1, x2, y2), c, conf in zip(ids, xyxy, cls, confs):
                bbox = (int(x1), int(y1), int(x2), int(y2))
                cx, cy = _center(bbox)
                label_name = (
                    names.get(int(c), str(c)) if isinstance(names, dict) else str(c)
                )
                if int(c) == self.settings.person_class_id:
                    person_centers[int(tid)] = (cx, cy)
                    detections.append(
                        Detection(
                            module=self.name,
                            bbox=bbox,
                            label=f"P#{tid}",
                            confidence=float(conf),
                            color=(0, 200, 0),
                            extra={"track_id": int(tid), "kind": "person"},
                        )
                    )
                elif int(c) in self.settings.bag_class_ids:
                    bag_records.append((int(tid), label_name, bbox, float(conf)))

        # Update bag state machines
        alarms: List[Alarm] = []
        seen_bag_ids = set()
        for tid, lbl, bbox, conf in bag_records:
            seen_bag_ids.add(tid)
            bs = self._bags.get(tid)
            if bs is None:
                bs = BagState(track_id=tid, label=lbl)
                self._bags[tid] = bs
            bs.last_seen_t = t_sec
            bs.last_bbox = bbox

            bag_center = _center(bbox)
            nearest_pid, nearest_d = self._nearest_person(bag_center, person_centers)

            # ByteTrack person IDs can change quickly; instead of binding ownership
            # to a fixed ID we bind it to geometry: nearest person within the
            # threshold of the bag = carrier.

            # 1) UNASSIGNED → OWNED
            if bs.state == "UNASSIGNED":
                if nearest_pid is not None and nearest_d <= self.settings.ownership_distance_thresh:
                    bs.owner_track_id = nearest_pid
                    bs.state = "OWNED"

            # 2) OWNED: carrier = nearest person to the bag (if within detach); updated even if the ID changes
            elif bs.state == "OWNED":
                if nearest_pid is None or nearest_d > self.settings.detach_distance_thresh:
                    bs.state = "DETACHED"
                    bs.detached_since_t = t_sec
                else:
                    bs.owner_track_id = nearest_pid

            # 3) DETACHED → OWNED (moved close again)
            elif bs.state == "DETACHED":
                if nearest_pid is not None and nearest_d <= self.settings.ownership_distance_thresh:
                    bs.owner_track_id = nearest_pid
                    bs.state = "OWNED"
                    bs.detached_since_t = None

            # 4) DETACHED → ABANDONED ?
            if bs.state == "DETACHED" and bs.detached_since_t is not None:
                elapsed = t_sec - bs.detached_since_t
                if elapsed >= self.settings.abandonment_seconds:
                    bs.state = "ABANDONED"

            # detection list
            color_map = {
                "UNASSIGNED": (200, 200, 200),
                "OWNED": (0, 255, 0),
                "DETACHED": (0, 165, 255),
                "ABANDONED": (0, 0, 255),
            }
            elapsed_str = ""
            if bs.state == "DETACHED" and bs.detached_since_t is not None:
                elapsed_str = f" ({int(t_sec - bs.detached_since_t)}s)"
            detections.append(
                Detection(
                    module=self.name,
                    bbox=bbox,
                    label=f"{lbl}#{tid} {bs.state}{elapsed_str}",
                    confidence=conf,
                    color=color_map.get(bs.state, (255, 255, 0)),
                    extra={
                        "track_id": tid,
                        "kind": "bag",
                        "state": bs.state,
                        "owner_id": bs.owner_track_id,
                    },
                )
            )

            # alarm
            if bs.state == "ABANDONED" and not bs.alarm_fired:
                bs.alarm_fired = True
                if (t_sec - self._last_alarm_t) > 2.0:
                    self._last_alarm_t = t_sec
                    alarms.append(
                        Alarm(
                            module=self.name,
                            severity="ALARM",
                            message=(
                                f"Abandoned {lbl} (#{tid}) — "
                                f"{int(self.settings.abandonment_seconds)}s"
                            ),
                            detections=[d for d in detections if d.extra.get("track_id") == tid],
                            extra={
                                "bag_id": tid,
                                "kind": lbl,
                                "owner_id": bs.owner_track_id,
                            },
                        )
                    )

        # forget bags not seen for a long time (60 s)
        stale = [tid for tid, bs in self._bags.items() if t_sec - bs.last_seen_t > 60]
        for tid in stale:
            del self._bags[tid]

        return ModuleResult(
            module=self.name,
            detections=detections,
            alarms=alarms,
            inference_ms=infer_ms,
            extra={"bags": len(self._bags), "persons": len(person_centers)},
        )
