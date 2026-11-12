"""
QThread-based capture + processing worker.

Structure:
  - Capture loop runs on a separate thread (so the GUI is not blocked)
  - FrameProcessor runs sequentially (modules in order)
  - When a new frame is ready it is sent to the UI via the `frame_ready` signal
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ..core.frame_processor import FrameProcessor, FrameOutput
from ..core.settings import Settings
from ..utils.device import auto_select


class ProcessingWorker(QThread):
    frame_ready = Signal(object)        # FrameOutput
    status = Signal(str)                # log/info text
    error = Signal(str)
    started_ok = Signal()

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self._running = False
        self._cap: Optional[cv2.VideoCapture] = None
        self.processor: Optional[FrameProcessor] = None

    # --------------- public --------------- #
    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def reload_face_gallery(self) -> None:
        if self.processor:
            n = self.processor.reload_face_gallery()
            self.status.emit(f"Face gallery rescanned: {n} people")

    # --------------- thread main --------------- #
    def run(self) -> None:
        try:
            device = self._resolve_device()
            self.status.emit(f"Device: {device}")
            self.processor = FrameProcessor(self.settings, device)
            self.status.emit("Loading models...")
            self.processor.preload()
            self.status.emit("Models ready.")
            self._open_capture()
            if self._cap is None or not self._cap.isOpened():
                if self.settings.is_debug_video_source():
                    self.error.emit(
                        "Could not open the debug video (check the file path or codec)."
                    )
                else:
                    self.error.emit("Could not open the camera / stream.")
                return
            if self.settings.is_debug_video_source():
                self.status.emit(
                    f"Debug: video source — {Path(self.settings.camera.debug_video_path.strip()).name}"
                )
            self.started_ok.emit()
            self._running = True
            self._loop()
        except Exception as e:
            import traceback
            self.error.emit(f"Worker error: {e}\n{traceback.format_exc()}")
        finally:
            self._cleanup()

    # --------------- internal --------------- #
    def _resolve_device(self) -> str:
        ps = self.settings.performance
        if ps.device != "auto":
            return ps.device
        # force CPU if use_gpu is False
        if not ps.use_gpu:
            return "cpu"
        return auto_select()

    def _open_capture(self) -> None:
        is_file = self.settings.is_debug_video_source()
        src = self.settings.resolved_source()
        if is_file:
            p = Path(str(src))
            if not p.is_file():
                self.error.emit(f"Debug video not found:\n{p}")
                self._cap = None
                return
        cap = cv2.VideoCapture(str(src) if isinstance(src, str) else src)
        if not is_file:
            # Webcam / RTSP: request resolution (can cause issues on some backends for a file source)
            ow, oh = self.settings.output_size()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ow)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, oh)
            cap.set(cv2.CAP_PROP_FPS, self.settings.camera.fps_cap)
        self._cap = cap

    def _loop(self) -> None:
        t_start = time.perf_counter()
        cam = self.settings.camera
        if self.settings.is_debug_video_source() and cam.debug_video_sync_realtime:
            self._loop_debug_video_realtime(t_start)
        else:
            self._loop_webcam_or_video_max_speed(t_start)

    def _loop_debug_video_realtime(self, t_start: float) -> None:
        """File source: play at the video's own speed; drop intermediate frames if inference is slow."""
        cap0 = self._cap
        assert cap0 is not None
        vf = float(cap0.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(vf) or vf < 1.0 or vf > 240.0:
            vf = 25.0
        cap_fps = max(1, int(self.settings.camera.fps_cap))
        playback_fps = min(vf, float(cap_fps))
        frame_period = 1.0 / playback_fps
        next_due = time.perf_counter()
        self.status.emit(
            f"Debug: real-time sync — source {vf:.1f} FPS, "
            f"target {playback_fps:.1f} FPS (frames are skipped if processing cannot keep up)."
        )

        while self._running:
            cap = self._cap
            if cap is None:
                return
            now = time.perf_counter()
            while next_due + frame_period < now and self._running:
                cap = self._cap
                if cap is None:
                    return
                if not cap.grab():
                    if not self._rewind_or_stop_debug_video():
                        return
                    next_due = time.perf_counter()
                    break
                next_due += frame_period

            cap = self._cap
            if cap is None:
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                if not self._rewind_or_stop_debug_video():
                    return
                next_due = time.perf_counter()
                continue

            t_video = time.perf_counter() - t_start
            try:
                out: FrameOutput = self.processor.process(frame, t_video)
            except Exception as e:
                import traceback
                self.error.emit(f"Processing error: {e}\n{traceback.format_exc()}")
                break
            self.frame_ready.emit(out)

            next_due += frame_period
            stall = next_due - time.perf_counter()
            if stall > 0:
                time.sleep(stall)

    def _loop_webcam_or_video_max_speed(self, t_start: float) -> None:
        """Webcam / RTSP or file (sync off): cap with fps_cap, or full speed."""
        assert self._cap is not None
        is_file = self.settings.is_debug_video_source()
        target_fps = max(1, int(self.settings.camera.fps_cap))
        target_dt = 1.0 / target_fps
        last_t = time.perf_counter()

        while self._running:
            cap = self._cap
            if cap is None:
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_file:
                    if self.settings.camera.debug_video_loop:
                        try:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        except Exception:
                            cap.release()
                            self._open_capture()
                            if self._cap is None or not self._cap.isOpened():
                                self.error.emit("Could not reopen the debug video.")
                                break
                        self.status.emit("Debug: end of video — rewound to the start (loop).")
                        continue
                    self.status.emit("Debug: video finished (loop off). Stream stopped.")
                    break
                time.sleep(0.05)
                continue

            t_video = time.perf_counter() - t_start
            try:
                out: FrameOutput = self.processor.process(frame, t_video)
            except Exception as e:
                import traceback
                self.error.emit(f"Processing error: {e}\n{traceback.format_exc()}")
                break
            self.frame_ready.emit(out)

            if not is_file:
                now = time.perf_counter()
                sleep_for = target_dt - (now - last_t)
                if sleep_for > 0:
                    time.sleep(sleep_for)
                last_t = time.perf_counter()

    def _rewind_or_stop_debug_video(self) -> bool:
        """True = keep looping, False = worker exits."""
        if not self.settings.is_debug_video_source():
            return False
        if self.settings.camera.debug_video_loop:
            try:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception:
                self._cap.release()
                self._open_capture()
                if self._cap is None or not self._cap.isOpened():
                    self.error.emit("Could not reopen the debug video.")
                    return False
            self.status.emit("Debug: end of video — rewound to the start (loop).")
            return True
        self.status.emit("Debug: video finished (loop off). Stream stopped.")
        return False

    def _cleanup(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.status.emit("Worker closed.")
