"""
FrameProcessor — runs all active modules in sequence + frame skip + visualize.

Design decisions:
  - Modules run sequentially (same thread per frame). A simple, deterministic
    flow instead of carrying IPC overhead between 5 modules and the GUI thread.
  - Frame skip: if settings.performance.frame_skip > 0, processing runs on
    every (N+1)th frame; the frames in between are shown raw.
  - Face matching runs only when FaceDetect is active and consumes its bboxes
    (so the same face is not searched twice).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from .alert_manager import AlertManager
from .settings import Settings
from ..modules.base import Detection, ModuleResult, Alarm
from ..modules.anomaly import AnomalyModule
from ..modules.weapon import WeaponModule
from ..modules.abandoned import AbandonedModule
from ..modules.face_detect import FaceDetectModule
from ..modules.face_match import FaceMatchModule


@dataclass
class FrameOutput:
    frame: np.ndarray                    # final with overlay (BGR)
    raw: np.ndarray                       # processed but without overlay
    fps: float = 0.0
    inference_ms_per_module: Dict[str, float] = field(default_factory=dict)
    detections: List[Detection] = field(default_factory=list)
    alarms: List[Alarm] = field(default_factory=list)
    banner_active: bool = False
    banner_text: str = ""
    banner_module: str = ""
    counters: Dict[str, int] = field(default_factory=dict)


class FrameProcessor:
    def __init__(self, settings: Settings, device: str):
        self.settings = settings
        self.device = device
        self.alerts = AlertManager(settings.alerts, settings.to_dict)

        self.anomaly = AnomalyModule(settings.anomaly, device)
        self.weapon = WeaponModule(settings.weapon, device)
        self.abandoned = AbandonedModule(settings.abandoned, device)
        self.face_detect = FaceDetectModule(settings.face_detect, device)
        self.face_match = FaceMatchModule(settings.face_match, device)

        self._frame_idx = 0
        self._fps_buf = []
        self._last_full_output: FrameOutput | None = None

    # --------------- module loading --------------- #
    def preload(self) -> None:
        """Load heavy model files for active modules (call from the Worker
        thread so the GUI thread is not blocked)."""
        if self.settings.anomaly.enabled:
            self.anomaly.load()
        if self.settings.weapon.enabled:
            self.weapon.load()
        if self.settings.abandoned.enabled:
            self.abandoned.load()
        if self.settings.face_detect.enabled:
            self.face_detect.load()
        if self.settings.face_match.enabled and self.settings.face_detect.enabled:
            self.face_match.load()

    def reload_face_gallery(self) -> int:
        if self.face_match.is_loaded():
            return self.face_match.reload_gallery()
        return 0

    # --------------- main entry --------------- #
    def process(self, frame_bgr: np.ndarray, t_sec: float) -> FrameOutput:
        self._frame_idx += 1
        skip = max(0, int(self.settings.performance.frame_skip))
        do_inference = (self._frame_idx % (skip + 1) == 0)

        # downscale (optional)
        ow, oh = self.settings.output_size()
        if (frame_bgr.shape[1], frame_bgr.shape[0]) != (ow, oh):
            frame_bgr = cv2.resize(frame_bgr, (ow, oh), interpolation=cv2.INTER_AREA)

        if not do_inference and self._last_full_output is not None:
            # Skip frame: stamp the previous overlay onto the new raw frame
            out = self._draw_overlay(frame_bgr, self._last_full_output.detections,
                                     self._last_full_output.alarms, self._last_full_output)
            return out

        all_dets: List[Detection] = []
        all_alarms: List[Alarm] = []
        infer_ms: Dict[str, float] = {}
        counters: Dict[str, int] = {}

        # Order matters: heavy modules first, then face_match (depends on face_detect bboxes)
        if self.settings.anomaly.enabled:
            r = self.anomaly.process(frame_bgr, self._frame_idx, t_sec)
            all_dets += r.detections
            all_alarms += r.alarms
            infer_ms[r.module] = r.inference_ms
            counters["people"] = r.extra.get("people", 0)
        if self.settings.weapon.enabled:
            r = self.weapon.process(frame_bgr, self._frame_idx, t_sec)
            all_dets += r.detections
            all_alarms += r.alarms
            infer_ms[r.module] = r.inference_ms
        if self.settings.abandoned.enabled:
            r = self.abandoned.process(frame_bgr, self._frame_idx, t_sec)
            all_dets += r.detections
            all_alarms += r.alarms
            infer_ms[r.module] = r.inference_ms

        face_dets: List[Detection] = []
        if self.settings.face_detect.enabled:
            r = self.face_detect.process(frame_bgr, self._frame_idx, t_sec)
            face_dets = list(r.detections)
            all_dets += r.detections
            all_alarms += r.alarms
            infer_ms[r.module] = r.inference_ms
            counters["faces"] = r.extra.get("faces", 0)

            if self.settings.face_match.enabled:
                r2 = self.face_match.process(
                    frame_bgr, self._frame_idx, t_sec, face_dets=face_dets
                )
                # match labels occupy the same region as face_detect bboxes;
                # color override so matches stand out
                all_dets += r2.detections
                all_alarms += r2.alarms
                infer_ms[r2.module] = r2.inference_ms

        # Alert save + sound + banner
        self.alerts.handle(frame_bgr.copy(), all_alarms, self._frame_idx, t_sec)

        # FPS
        now = time.perf_counter()
        self._fps_buf.append(now)
        self._fps_buf = [t for t in self._fps_buf if now - t <= 1.0]
        fps = float(len(self._fps_buf))

        out = self._make_output(frame_bgr, all_dets, all_alarms, infer_ms, fps, counters)
        self._last_full_output = out
        return out

    # --------------- visualize --------------- #
    def _make_output(self, frame, dets, alarms, infer_ms, fps, counters) -> FrameOutput:
        out_obj = FrameOutput(
            frame=frame.copy(),
            raw=frame,
            fps=fps,
            inference_ms_per_module=infer_ms,
            detections=dets,
            alarms=alarms,
            counters=counters,
        )
        return self._draw_overlay(out_obj.frame, dets, alarms, out_obj)

    def _draw_overlay(self, frame, dets, alarms, out: FrameOutput) -> FrameOutput:
        vis = frame.copy()
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), d.color, 2)
            txt = d.label
            if not txt:
                continue
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, max(0, y1 - th - 6)),
                          (x1 + tw + 6, y1), d.color, -1)
            cv2.putText(vis, txt, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # bottom-left stat overlay
        h, w = vis.shape[:2]
        line = f"FPS {out.fps:>4.1f}"
        for m, ms in out.inference_ms_per_module.items():
            line += f"  {m}:{ms:.0f}ms"
        if out.counters:
            extra = " ".join(f"{k}:{v}" for k, v in out.counters.items())
            line += "  " + extra
        cv2.rectangle(vis, (0, h - 24), (min(len(line) * 9 + 10, w), h), (0, 0, 0), -1)
        cv2.putText(vis, line, (5, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

        # banner state (red strip at the top while an alarm is active)
        b = self.alerts.banner
        if b.active:
            cv2.rectangle(vis, (0, 0), (w, 44), (0, 0, 200), -1)
            cv2.putText(
                vis,
                f"ALARM [{b.module}] {b.text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            out.banner_active = True
            out.banner_text = b.text
            out.banner_module = b.module

        out.frame = vis
        return out

    def reset(self) -> None:
        for m in (self.anomaly, self.weapon, self.abandoned,
                  self.face_detect, self.face_match):
            try:
                m.reset()
            except Exception:
                pass
        self._frame_idx = 0
        self._fps_buf.clear()
        self._last_full_output = None
