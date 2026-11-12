"""Right panel — video render + top alarm banner + bottom status strip."""
from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class VideoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._fps_buf = deque(maxlen=30)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # TOP: alarm banner
        self.banner = QLabel("")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setMinimumHeight(48)
        self.banner.setMaximumHeight(56)
        self.banner.setStyleSheet(
            "background:#0f172a;color:#94a3b8;font-weight:700;"
            "font-size:18px;padding:6px;"
        )
        self.banner.setText("System ready")
        layout.addWidget(self.banner)

        # MIDDLE: video area
        self.video = QLabel("Press Start")
        self.video.setAlignment(Qt.AlignCenter)
        self.video.setStyleSheet("background:#000;color:#475569;font-size:18px;")
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video, 1)

        # BOTTOM: status + recent detection log
        bottom = QFrame()
        bl = QHBoxLayout(bottom); bl.setContentsMargins(8, 4, 8, 6)
        self.status_label = QLabel("FPS — | waiting")
        self.status_label.setStyleSheet("color:#cbd5e1;font-family:Menlo,Monaco,monospace;")
        bl.addWidget(self.status_label, 2)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(80)
        self.log.setStyleSheet(
            "background:#020617;color:#a3e635;"
            "font-family:Menlo,Monaco,monospace;font-size:11px;"
        )
        bl.addWidget(self.log, 3)

        bottom.setMaximumHeight(96)
        bottom.setStyleSheet("background:#0b1220;")
        layout.addWidget(bottom)

    # --------------- public --------------- #
    def update_frame(self, out) -> None:
        img = out.frame
        if img is None or img.size == 0:
            return
        # BGR → RGB → QImage
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.video.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video.setPixmap(scaled)

        # Banner
        if out.banner_active:
            self.banner.setText(f"⚠  ALARM [{out.banner_module}]   {out.banner_text}")
            self.banner.setStyleSheet(
                "background:#dc2626;color:white;font-weight:800;"
                "font-size:20px;padding:6px;"
            )
        else:
            self.banner.setText("Monitoring active — waiting for detections")
            self.banner.setStyleSheet(
                "background:#064e3b;color:#a7f3d0;font-weight:600;"
                "font-size:16px;padding:6px;"
            )

        # Status
        parts = [f"FPS {out.fps:>4.1f}"]
        for m, ms in out.inference_ms_per_module.items():
            parts.append(f"{m}:{ms:.0f}ms")
        for k, v in out.counters.items():
            parts.append(f"{k}={v}")
        self.status_label.setText("  |  ".join(parts))

    def set_status(self, text: str) -> None:
        self.banner.setText(text)
        self.banner.setStyleSheet(
            "background:#0f172a;color:#94a3b8;font-weight:600;"
            "font-size:16px;padding:6px;"
        )

    def append_log(self, text: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def reset_video(self) -> None:
        self.video.setPixmap(QPixmap())
        self.video.setText("Press Start")
