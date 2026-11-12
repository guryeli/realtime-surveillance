"""Main window — full screen + settings on the left + video on the right."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QWidget,
)

from ..core.settings import Settings
from .settings_panel import SettingsPanel
from .video_panel import VideoPanel
from .worker import ProcessingWorker


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.worker: ProcessingWorker | None = None
        self.setWindowTitle("Realtime Surveillance — Integrated Security Monitoring")

        self._build()
        self._shortcuts()

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        h = QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        h.addWidget(splitter)

        self.settings_panel = SettingsPanel(self.settings, self)
        self.video_panel = VideoPanel(self)

        splitter.addWidget(self.settings_panel)
        splitter.addWidget(self.video_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 1460])

        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(
            "Ready — select modules in the left panel and press Start. "
            "F11: full screen, ESC: quit"
        )

        # signals
        self.settings_panel.start_clicked.connect(self.on_start)
        self.settings_panel.stop_clicked.connect(self.on_stop)
        self.settings_panel.reload_gallery_clicked.connect(self.on_reload_gallery)

    def _shortcuts(self) -> None:
        QShortcut(QKeySequence("F11"), self, activated=self.toggle_fullscreen)
        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+Q"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.on_reload_gallery)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    # --------------- start / stop --------------- #
    def on_start(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self.settings.any_module_active():
            QMessageBox.warning(
                self,
                "No module selected",
                "Enable at least one module (Anomaly, Weapon, Abandoned, Face, ...).",
            )
            return
        cam = self.settings.camera
        if cam.debug_video_enabled:
            if not cam.debug_video_path.strip():
                QMessageBox.warning(
                    self,
                    "Debug video",
                    "Select a video file while \"Use a local video file instead of webcam / RTSP\" "
                    "is on, or uncheck the box.",
                )
                return
            p = Path(cam.debug_video_path.strip()).expanduser()
            if not p.is_file():
                QMessageBox.warning(
                    self,
                    "Debug video",
                    f"File not found:\n{p}",
                )
                return
        self.video_panel.set_status("Loading models...")
        self.video_panel.append_log("Starting...")
        self.worker = ProcessingWorker(self.settings)
        self.worker.frame_ready.connect(self.video_panel.update_frame)
        self.worker.status.connect(self.video_panel.append_log)
        self.worker.error.connect(self._on_worker_error)
        self.worker.started_ok.connect(lambda: self.video_panel.append_log("Stream started."))
        self.worker.start()
        self.settings_panel.set_running(True)

    def on_stop(self) -> None:
        if not self.worker:
            return
        self.video_panel.append_log("Stopping...")
        self.worker.stop()
        self.worker = None
        self.video_panel.reset_video()
        self.video_panel.set_status("Stopped")
        self.settings_panel.set_running(False)

    def on_reload_gallery(self) -> None:
        if self.worker:
            self.worker.reload_face_gallery()
        else:
            self.video_panel.append_log(
                "The gallery is scanned automatically when the stream starts."
            )

    # --------------- error handlers --------------- #
    def _on_worker_error(self, msg: str) -> None:
        self.video_panel.append_log(f"[ERROR] {msg.splitlines()[0]}")
        QMessageBox.critical(self, "Error", msg)
        self.on_stop()

    def closeEvent(self, e) -> None:
        try:
            from ..utils.audio import shutdown
            shutdown()
        except Exception:
            pass
        if self.worker:
            self.worker.stop()
        super().closeEvent(e)
