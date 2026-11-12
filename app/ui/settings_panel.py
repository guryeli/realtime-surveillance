"""
Left panel — all settings (camera, performance, 5 modules).

Design: scrollable + accordion (QGroupBox collection).
"Start" button at the top; below it "Scan Gallery", "Save/Load Settings".
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.settings import Settings
from ..utils.device import auto_select, describe, list_devices


class SettingsPanel(QWidget):
    start_clicked = Signal()
    stop_clicked = Signal()
    reload_gallery_clicked = Signal()

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setMinimumWidth(440)
        self.setMaximumWidth(540)
        self._build()

    # --------------- build --------------- #
    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        title = QLabel("Realtime Surveillance — Settings")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        outer.addWidget(title)

        # Start/Stop
        row = QHBoxLayout()
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setStyleSheet(
            "background:#16a34a;color:white;font-weight:bold;padding:8px;"
        )
        self.btn_start.clicked.connect(self.start_clicked.emit)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setStyleSheet(
            "background:#dc2626;color:white;font-weight:bold;padding:8px;"
        )
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_start, 1)
        row.addWidget(self.btn_stop, 1)
        outer.addLayout(row)

        # Save/load
        row2 = QHBoxLayout()
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.clicked.connect(self._save_settings)
        self.btn_load = QPushButton("Load Settings")
        self.btn_load.clicked.connect(self._load_settings)
        row2.addWidget(self.btn_save, 1)
        row2.addWidget(self.btn_load, 1)
        outer.addLayout(row2)

        # Scrollable settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        v = QVBoxLayout(body)
        v.setSpacing(8)

        self._build_camera_group(v)
        self._build_debug_video_group(v)
        self._build_performance_group(v)
        self._build_anomaly_group(v)
        self._build_weapon_group(v)
        self._build_abandoned_group(v)
        self._build_face_detect_group(v)
        self._build_face_match_group(v)
        v.addStretch(1)

    # --------------- groups --------------- #
    def _build_camera_group(self, parent_layout) -> None:
        cs = self.settings.camera
        gb = QGroupBox("📷 Camera")
        f = QFormLayout(gb)

        self.cam_index = QSpinBox()
        self.cam_index.setRange(0, 16)
        self.cam_index.setValue(int(cs.source))
        self.cam_index.valueChanged.connect(lambda v: setattr(cs, "source", int(v)))
        f.addRow("Webcam index", self.cam_index)

        self.cam_url = QLineEdit(cs.source_str)
        self.cam_url.setPlaceholderText("RTSP/HTTP URL (optional, takes priority if set)")
        self.cam_url.textChanged.connect(lambda s: setattr(cs, "source_str", s))
        f.addRow("Or URL", self.cam_url)

        self.cam_res = QComboBox()
        self.cam_res.addItems(["480p", "720p", "1080p", "native"])
        self.cam_res.setCurrentText(cs.output_resolution)
        self.cam_res.currentTextChanged.connect(
            lambda s: setattr(cs, "output_resolution", s)
        )
        f.addRow("Processing resolution", self.cam_res)

        self.cam_w = QSpinBox(); self.cam_w.setRange(160, 4096); self.cam_w.setValue(cs.target_width)
        self.cam_h = QSpinBox(); self.cam_h.setRange(120, 2160); self.cam_h.setValue(cs.target_height)
        self.cam_w.valueChanged.connect(lambda v: setattr(cs, "target_width", int(v)))
        self.cam_h.valueChanged.connect(lambda v: setattr(cs, "target_height", int(v)))
        row = QHBoxLayout()
        row.addWidget(QLabel("W"))
        row.addWidget(self.cam_w)
        row.addWidget(QLabel("H"))
        row.addWidget(self.cam_h)
        wrap = QWidget(); wrap.setLayout(row)
        f.addRow("Native (if resolution=native)", wrap)

        self.cam_fps = QSpinBox(); self.cam_fps.setRange(1, 120); self.cam_fps.setValue(cs.fps_cap)
        self.cam_fps.valueChanged.connect(lambda v: setattr(cs, "fps_cap", int(v)))
        f.addRow("FPS cap", self.cam_fps)

        hint = QLabel(
            "<small>If a local video is selected in Debug, that file becomes the "
            "source instead of the webcam/URL (priority: debug video → URL → webcam).</small>"
        )
        hint.setWordWrap(True)
        f.addRow(hint)

        parent_layout.addWidget(gb)

    def _build_debug_video_group(self, parent_layout) -> None:
        """Feed a local video file into the pipeline like a webcam stream."""
        cs = self.settings.camera
        gb = QGroupBox("🐛  Debug — Video simulation (instead of webcam)")
        f = QFormLayout(gb)

        self.cb_debug_video = QCheckBox(
            "Use a local video file instead of webcam / RTSP"
        )
        self.cb_debug_video.setChecked(cs.debug_video_enabled)
        self.cb_debug_video.toggled.connect(
            lambda on: setattr(cs, "debug_video_enabled", bool(on))
        )
        f.addRow(self.cb_debug_video)

        self.debug_video_path = QLineEdit(cs.debug_video_path)
        self.debug_video_path.setPlaceholderText(
            "/path/to/video.mp4  or  …/AnomaliDedection/data/videos/…"
        )
        self.debug_video_path.textChanged.connect(
            lambda s: setattr(cs, "debug_video_path", s)
        )
        row = QHBoxLayout()
        row.addWidget(self.debug_video_path, 1)
        btn = QPushButton("Choose file…")
        btn.clicked.connect(self._browse_debug_video)
        row.addWidget(btn)
        wrap = QWidget()
        wrap.setLayout(row)
        f.addRow("Video file", wrap)

        self.cb_debug_loop = QCheckBox("Loop the video (rewind when it ends)")
        self.cb_debug_loop.setChecked(cs.debug_video_loop)
        self.cb_debug_loop.toggled.connect(
            lambda on: setattr(cs, "debug_video_loop", bool(on))
        )
        f.addRow(self.cb_debug_loop)

        self.cb_debug_sync = QCheckBox(
            "Play the video in real time (drop intermediate frames if processing is slow / heavy)"
        )
        self.cb_debug_sync.setChecked(cs.debug_video_sync_realtime)
        self.cb_debug_sync.setToolTip(
            "When on, playback stays in sync with the source video FPS; if inference "
            "takes too long, frames are skipped before decode — no heavy slow-motion effect. "
            "When off, the file is read as fast as possible (no upper bound)."
        )
        self.cb_debug_sync.toggled.connect(
            lambda on: setattr(cs, "debug_video_sync_realtime", bool(on))
        )
        f.addRow(self.cb_debug_sync)

        note = QLabel(
            "<small>OpenCV-supported formats: MP4, AVI, MOV … "
            "Processing resolution and frame skip are applied the same way as for a webcam.</small>"
        )
        note.setWordWrap(True)
        f.addRow(note)

        parent_layout.addWidget(gb)

    def _browse_debug_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select a video for debug",
            str(Path.home()),
            "Video (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)",
        )
        if path:
            self.settings.camera.debug_video_path = path
            self.debug_video_path.setText(path)
            self.settings.camera.debug_video_enabled = True
            self.cb_debug_video.setChecked(True)

    def _build_performance_group(self, parent_layout) -> None:
        ps = self.settings.performance
        gb = QGroupBox("⚙️  Performance / Device")
        f = QFormLayout(gb)

        avail = list_devices()
        cpu_ok = "cpu" in avail
        gpu_devs = [d for d in avail if d != "cpu"]
        gpu_ok = bool(gpu_devs)

        self.cb_cpu = QCheckBox(f"CPU available ({'yes' if cpu_ok else 'no'})")
        self.cb_cpu.setChecked(ps.use_cpu)
        self.cb_cpu.stateChanged.connect(lambda s: setattr(ps, "use_cpu", bool(s)))
        f.addRow(self.cb_cpu)

        gpu_label = (
            f"GPU available: {', '.join(describe(d) for d in gpu_devs)}"
            if gpu_ok
            else "GPU available: no"
        )
        self.cb_gpu = QCheckBox(gpu_label)
        self.cb_gpu.setChecked(ps.use_gpu and gpu_ok)
        self.cb_gpu.setEnabled(gpu_ok)
        self.cb_gpu.stateChanged.connect(lambda s: setattr(ps, "use_gpu", bool(s)))
        f.addRow(self.cb_gpu)

        self.cmb_device = QComboBox()
        self.cmb_device.addItem("auto")
        self.cmb_device.addItems(avail)
        if ps.device in (["auto"] + avail):
            self.cmb_device.setCurrentText(ps.device)
        else:
            self.cmb_device.setCurrentText("auto")
        self.cmb_device.currentTextChanged.connect(
            lambda s: setattr(ps, "device", s)
        )
        f.addRow("Active device", self.cmb_device)

        self.sp_skip = QSpinBox()
        self.sp_skip.setRange(0, 10)
        self.sp_skip.setValue(ps.frame_skip)
        self.sp_skip.setToolTip(
            "0=process every frame, 1=1 of every 2 frames, 2=1 of every 3 frames ..."
        )
        self.sp_skip.valueChanged.connect(lambda v: setattr(ps, "frame_skip", int(v)))
        f.addRow("Frame skip", self.sp_skip)

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 99999)
        self.sp_seed.setValue(ps.seed)
        self.sp_seed.valueChanged.connect(lambda v: setattr(ps, "seed", int(v)))
        f.addRow("Seed", self.sp_seed)

        # Alert group
        gb2 = QGroupBox("🔔  Detection Log / Alarm")
        f2 = QFormLayout(gb2)
        als = self.settings.alerts

        self.cb_save_jpg = QCheckBox("Save a snapshot of the detection moment (.jpg)")
        self.cb_save_jpg.setChecked(als.save_screenshots)
        self.cb_save_jpg.stateChanged.connect(
            lambda s: setattr(als, "save_screenshots", bool(s))
        )
        f2.addRow(self.cb_save_jpg)

        self.cb_save_json = QCheckBox("Save details as JSON (.json)")
        self.cb_save_json.setChecked(als.save_json)
        self.cb_save_json.stateChanged.connect(
            lambda s: setattr(als, "save_json", bool(s))
        )
        f2.addRow(self.cb_save_json)

        self.cb_sound = QCheckBox("Play alarm sound")
        self.cb_sound.setChecked(als.play_sound)
        self.cb_sound.stateChanged.connect(
            lambda s: setattr(als, "play_sound", bool(s))
        )
        f2.addRow(self.cb_sound)

        self.sp_banner_dur = QDoubleSpinBox()
        self.sp_banner_dur.setRange(1.0, 30.0)
        self.sp_banner_dur.setValue(als.banner_duration_seconds)
        self.sp_banner_dur.valueChanged.connect(
            lambda v: setattr(als, "banner_duration_seconds", float(v))
        )
        f2.addRow("Banner duration (s)", self.sp_banner_dur)

        parent_layout.addWidget(gb)
        parent_layout.addWidget(gb2)

    # ---- ANOMALY ---- #
    def _build_anomaly_group(self, parent_layout) -> None:
        s = self.settings.anomaly
        gb = QGroupBox("🚨  Anomaly Detection (crowd panic & density)")
        gb.setCheckable(True)
        gb.setChecked(s.enabled)
        gb.toggled.connect(lambda b: setattr(s, "enabled", bool(b)))
        f = QFormLayout(gb)

        self.an_max_people = QSpinBox(); self.an_max_people.setRange(1, 10000); self.an_max_people.setValue(s.max_expected_people)
        self.an_max_people.valueChanged.connect(lambda v: setattr(s, "max_expected_people", int(v)))
        f.addRow("Max density (total people)", self.an_max_people)

        self.an_per_cell = QSpinBox(); self.an_per_cell.setRange(1, 1000); self.an_per_cell.setValue(s.max_people_per_cell)
        self.an_per_cell.valueChanged.connect(lambda v: setattr(s, "max_people_per_cell", int(v)))
        f.addRow("Per-cell cap (hotspot)", self.an_per_cell)

        self.an_speed = QDoubleSpinBox(); self.an_speed.setRange(10.0, 2000.0); self.an_speed.setValue(s.running_speed_pixels_per_sec)
        self.an_speed.valueChanged.connect(lambda v: setattr(s, "running_speed_pixels_per_sec", float(v)))
        f.addRow("Running-speed threshold (px/s)", self.an_speed)

        self.an_ratio = QDoubleSpinBox(); self.an_ratio.setRange(0.05, 1.0); self.an_ratio.setSingleStep(0.05); self.an_ratio.setValue(s.panic_running_ratio)
        self.an_ratio.valueChanged.connect(lambda v: setattr(s, "panic_running_ratio", float(v)))
        f.addRow("Running ratio (for panic)", self.an_ratio)

        self.an_min_run = QSpinBox(); self.an_min_run.setRange(1, 100); self.an_min_run.setValue(s.panic_min_runners)
        self.an_min_run.valueChanged.connect(lambda v: setattr(s, "panic_min_runners", int(v)))
        f.addRow("Min running people", self.an_min_run)

        self.an_min_total = QSpinBox(); self.an_min_total.setRange(1, 200); self.an_min_total.setValue(s.panic_min_total_people)
        self.an_min_total.valueChanged.connect(lambda v: setattr(s, "panic_min_total_people", int(v)))
        f.addRow("Min total people (for panic)", self.an_min_total)

        self.an_persistence = QDoubleSpinBox(); self.an_persistence.setRange(0.1, 30.0); self.an_persistence.setValue(s.alarm_persistence_seconds)
        self.an_persistence.valueChanged.connect(lambda v: setattr(s, "alarm_persistence_seconds", float(v)))
        f.addRow("Alarm persistence (s)", self.an_persistence)

        self.an_cooldown = QDoubleSpinBox(); self.an_cooldown.setRange(0.5, 60.0); self.an_cooldown.setValue(s.alarm_cooldown_seconds)
        self.an_cooldown.valueChanged.connect(lambda v: setattr(s, "alarm_cooldown_seconds", float(v)))
        f.addRow("Cooldown (s)", self.an_cooldown)

        self.an_conf = QDoubleSpinBox(); self.an_conf.setRange(0.05, 0.95); self.an_conf.setSingleStep(0.05); self.an_conf.setValue(s.conf_threshold)
        self.an_conf.valueChanged.connect(lambda v: setattr(s, "conf_threshold", float(v)))
        f.addRow("YOLO confidence", self.an_conf)

        self.an_iou = QDoubleSpinBox(); self.an_iou.setRange(0.20, 0.95); self.an_iou.setSingleStep(0.05); self.an_iou.setValue(s.iou_threshold)
        self.an_iou.valueChanged.connect(lambda v: setattr(s, "iou_threshold", float(v)))
        f.addRow("YOLO IoU (NMS)", self.an_iou)

        self.an_imgsz = QSpinBox(); self.an_imgsz.setRange(320, 1280); self.an_imgsz.setSingleStep(32); self.an_imgsz.setValue(s.imgsz)
        self.an_imgsz.setToolTip(
            "Inference resolution. 960–1280 raises recall on fast/distant people; GPU and ms cost go up."
        )
        self.an_imgsz.valueChanged.connect(lambda v: setattr(s, "imgsz", int(v)))
        f.addRow("YOLO imgsz (inference)", self.an_imgsz)

        self.an_maxdet = QSpinBox(); self.an_maxdet.setRange(50, 1000); self.an_maxdet.setSingleStep(50); self.an_maxdet.setValue(s.max_det)
        self.an_maxdet.valueChanged.connect(lambda v: setattr(s, "max_det", int(v)))
        f.addRow("max_det (crowd)", self.an_maxdet)

        self.an_hist = QSpinBox(); self.an_hist.setRange(5, 30); self.an_hist.setValue(s.speed_history_maxlen)
        self.an_hist.valueChanged.connect(lambda v: setattr(s, "speed_history_maxlen", int(v)))
        f.addRow("Speed window (frames)", self.an_hist)

        self.an_model = QLineEdit(s.yolo_model)
        self.an_model.setPlaceholderText("yolov8m.pt or yolov8l.pt …")
        self.an_model.textChanged.connect(lambda t: setattr(s, "yolo_model", t.strip() or "yolov8m.pt"))
        f.addRow("Weights file (models/anomaly/)", self.an_model)

        an_hint = QLabel(
            "<small>Fast motion: raise <b>imgsz</b> and lower <b>conf</b> first; if needed "
            "copy <b>yolov8l.pt</b>. Ultralytics: <code>pip install -U ultralytics</code>. "
            "Frame skip lowers recall for anomaly.</small>"
        )
        an_hint.setWordWrap(True)
        f.addRow(an_hint)

        parent_layout.addWidget(gb)

    # ---- WEAPON ---- #
    def _build_weapon_group(self, parent_layout) -> None:
        s = self.settings.weapon
        gb = QGroupBox("🔫  Weapon Detection")
        gb.setCheckable(True)
        gb.setChecked(s.enabled)
        gb.toggled.connect(lambda b: setattr(s, "enabled", bool(b)))
        f = QFormLayout(gb)

        self.we_conf = QDoubleSpinBox(); self.we_conf.setRange(0.05, 0.95); self.we_conf.setSingleStep(0.05); self.we_conf.setValue(s.conf_threshold)
        self.we_conf.valueChanged.connect(lambda v: setattr(s, "conf_threshold", float(v)))
        f.addRow("Confidence threshold", self.we_conf)

        self.we_iou = QDoubleSpinBox(); self.we_iou.setRange(0.1, 0.95); self.we_iou.setValue(s.iou_threshold)
        self.we_iou.valueChanged.connect(lambda v: setattr(s, "iou_threshold", float(v)))
        f.addRow("IoU", self.we_iou)

        self.we_consec = QSpinBox(); self.we_consec.setRange(1, 10); self.we_consec.setValue(s.min_consecutive_frames)
        self.we_consec.valueChanged.connect(lambda v: setattr(s, "min_consecutive_frames", int(v)))
        f.addRow("Min consecutive frames (FP↓)", self.we_consec)

        info = QLabel("Model: <b>yolov8l_weapon_best.pt</b> "
                      "(fine-tuned, mAP@50 87.47%, ~114 FPS)")
        info.setWordWrap(True)
        f.addRow(info)

        parent_layout.addWidget(gb)

    # ---- ABANDONED OBJECT ---- #
    def _build_abandoned_group(self, parent_layout) -> None:
        s = self.settings.abandoned
        gb = QGroupBox("🎒  Abandoned Object")
        gb.setCheckable(True)
        gb.setChecked(s.enabled)
        gb.toggled.connect(lambda b: setattr(s, "enabled", bool(b)))
        f = QFormLayout(gb)

        self.ab_aban = QDoubleSpinBox(); self.ab_aban.setRange(1.0, 600.0); self.ab_aban.setValue(s.abandonment_seconds)
        self.ab_aban.valueChanged.connect(lambda v: setattr(s, "abandonment_seconds", float(v)))
        f.addRow("Abandonment time (s)", self.ab_aban)

        self.ab_own = QDoubleSpinBox(); self.ab_own.setRange(20.0, 1000.0); self.ab_own.setValue(s.ownership_distance_thresh)
        self.ab_own.valueChanged.connect(lambda v: setattr(s, "ownership_distance_thresh", float(v)))
        f.addRow("Owner match distance (px)", self.ab_own)

        self.ab_det = QDoubleSpinBox(); self.ab_det.setRange(20.0, 2000.0); self.ab_det.setValue(s.detach_distance_thresh)
        self.ab_det.valueChanged.connect(lambda v: setattr(s, "detach_distance_thresh", float(v)))
        f.addRow("Detach distance (px)", self.ab_det)

        self.ab_conf = QDoubleSpinBox(); self.ab_conf.setRange(0.05, 0.95); self.ab_conf.setSingleStep(0.05); self.ab_conf.setValue(s.conf_threshold)
        self.ab_conf.valueChanged.connect(lambda v: setattr(s, "conf_threshold", float(v)))
        f.addRow("YOLO confidence", self.ab_conf)

        info = QLabel("Model: <b>yolov8m.pt</b> (COCO: person, backpack, "
                      "handbag, suitcase) + ByteTrack")
        info.setWordWrap(True)
        f.addRow(info)
        parent_layout.addWidget(gb)

    # ---- FACE DETECT ---- #
    def _build_face_detect_group(self, parent_layout) -> None:
        s = self.settings.face_detect
        gb = QGroupBox("🙂  Face Detection")
        gb.setCheckable(True)
        gb.setChecked(s.enabled)
        gb.toggled.connect(self._on_face_detect_toggled)
        self._fd_groupbox = gb
        f = QFormLayout(gb)

        self.fd_conf = QDoubleSpinBox(); self.fd_conf.setRange(0.05, 0.95); self.fd_conf.setSingleStep(0.05); self.fd_conf.setValue(s.conf_threshold)
        self.fd_conf.valueChanged.connect(lambda v: setattr(s, "conf_threshold", float(v)))
        f.addRow("Confidence", self.fd_conf)

        self.fd_iou = QDoubleSpinBox(); self.fd_iou.setRange(0.1, 0.95); self.fd_iou.setValue(s.iou_threshold)
        self.fd_iou.valueChanged.connect(lambda v: setattr(s, "iou_threshold", float(v)))
        f.addRow("IoU", self.fd_iou)

        self.fd_min = QSpinBox(); self.fd_min.setRange(8, 1024); self.fd_min.setValue(s.min_face_size)
        self.fd_min.valueChanged.connect(lambda v: setattr(s, "min_face_size", int(v)))
        f.addRow("Min face size (px)", self.fd_min)

        info = QLabel("Model: <b>yolov8l-face.pt</b> "
                      "(WIDER Face AP@0.5 0.685, 11.7 ms median)")
        info.setWordWrap(True)
        f.addRow(info)
        parent_layout.addWidget(gb)

    def _on_face_detect_toggled(self, on: bool) -> None:
        self.settings.face_detect.enabled = bool(on)
        # face_match is meaningful only while face_detect is active
        if not on:
            self.settings.face_match.enabled = False
            self._fm_groupbox.setChecked(False)
            self._fm_groupbox.setEnabled(False)
        else:
            self._fm_groupbox.setEnabled(True)

    # ---- FACE MATCH ---- #
    def _build_face_match_group(self, parent_layout) -> None:
        s = self.settings.face_match
        gb = QGroupBox("🔎  Face Matching (1:N gallery)")
        gb.setCheckable(True)
        gb.setChecked(s.enabled and self.settings.face_detect.enabled)
        gb.setEnabled(self.settings.face_detect.enabled)
        gb.toggled.connect(lambda b: setattr(s, "enabled", bool(b)))
        self._fm_groupbox = gb
        f = QFormLayout(gb)

        warn = QLabel("⚠  Face Detection must be active for this module.")
        warn.setStyleSheet("color:#92400e;")
        warn.setWordWrap(True)
        f.addRow(warn)

        self.fm_thresh = QDoubleSpinBox(); self.fm_thresh.setRange(0.20, 0.95); self.fm_thresh.setSingleStep(0.05); self.fm_thresh.setValue(s.similarity_threshold)
        self.fm_thresh.valueChanged.connect(lambda v: setattr(s, "similarity_threshold", float(v)))
        f.addRow("Match threshold (cosine)", self.fm_thresh)

        # Bulk file picker
        self.lst = QListWidget()
        self.lst.setMaximumHeight(120)
        for p in s.gallery_paths:
            self.lst.addItem(p)
        f.addRow("Gallery images", self.lst)

        row = QHBoxLayout()
        b_add = QPushButton("Add Photos…")
        b_add.clicked.connect(self._add_gallery_files)
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self._clear_gallery_files)
        b_reload = QPushButton("Scan Gallery")
        b_reload.clicked.connect(self.reload_gallery_clicked.emit)
        row.addWidget(b_add)
        row.addWidget(b_clr)
        row.addWidget(b_reload)
        wrap = QWidget(); wrap.setLayout(row)
        f.addRow(wrap)

        info = QLabel("Model: <b>InsightFace ArcFace (buffalo_l)</b> "
                      "(LFW 99.57% acc, 0.70% EER)")
        info.setWordWrap(True)
        f.addRow(info)
        parent_layout.addWidget(gb)

    # --------------- handlers --------------- #
    def _add_gallery_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select photos for the gallery",
            str(Path.cwd()),
            "Images (*.jpg *.jpeg *.png *.bmp);;All (*)",
        )
        if not files:
            return
        existing = set(self.settings.face_match.gallery_paths)
        for f in files:
            if f in existing:
                continue
            self.settings.face_match.gallery_paths.append(f)
            self.lst.addItem(f)

    def _clear_gallery_files(self) -> None:
        self.settings.face_match.gallery_paths.clear()
        self.lst.clear()

    def _save_settings(self) -> None:
        from ..utils.paths import CONFIG_PATH
        self.settings.save(CONFIG_PATH)

    def _load_settings(self) -> None:
        from ..utils.paths import CONFIG_PATH
        new = Settings.load(CONFIG_PATH)
        # in-place update (keep the reference)
        for k in ("camera", "performance", "alerts", "anomaly", "weapon",
                  "abandoned", "face_detect", "face_match"):
            cur = getattr(self.settings, k)
            for f, v in vars(getattr(new, k)).items():
                setattr(cur, f, v)
        # rebuild form widgets — simple path: redraw the whole UI
        # (closer to production: re-read setting values → plain set)
        # redraw for the MVP:
        for i in reversed(range(self.layout().count())):
            item = self.layout().itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self._build()

    # --------------- public --------------- #
    def set_running(self, running: bool) -> None:
        self.btn_start.setEnabled(not running)
        self.btn_stop.setEnabled(running)
