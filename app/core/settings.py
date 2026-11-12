"""
All module settings (dataclass-based) + JSON persistence.

Each module has its own settings block; performance/device/camera settings
sit at the top level. Settings changed from the GUI are written to JSON
via `Settings.save()`.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------- Module settings blocks ---------------- #

@dataclass
class AnomalySettings:
    enabled: bool = False
    # yolov8s (speed) < yolov8m (balance) < yolov8l (better for distant/small + fast motion)
    yolo_model: str = "yolov8m.pt"           # under models/anomaly/; copy the file for yolov8l.pt
    method: str = "both"                      # "1" | "2" | "both"
    conf_threshold: float = 0.18             # lower = recall ↑ on motion blur / fast objects, FP risk ↑
    iou_threshold: float = 0.45              # NMS; in crowds 0.5→0.45 is sometimes a more stable track
    imgsz: int = 960                         # YOLO inference; 640→960 is a strong lever for distant/fast people (GPU recommended)
    max_det: int = 300                       # max detections in a crowded scene

    # Density
    max_expected_people: int = 30
    max_people_per_cell: int = 15
    grid_rows: int = 3
    grid_cols: int = 3

    # Speed / panic
    running_speed_pixels_per_sec: float = 200.0
    panic_running_ratio: float = 0.30
    panic_min_runners: int = 3
    panic_min_total_people: int = 5

    # Optical flow
    flow_high_magnitude: float = 3.0
    flow_panic_cell_ratio: float = 0.10
    flow_low_entropy_thresh: float = 1.5
    flow_baseline_frames: int = 60

    # Alarm timing
    alarm_persistence_seconds: float = 1.0
    alarm_cooldown_seconds: float = 3.0

    # Speed (px/s) window — a slightly longer window for fast motion + ID noise
    speed_history_maxlen: int = 12           # how many time samples per track (deque)


@dataclass
class WeaponSettings:
    enabled: bool = False
    yolo_model: str = "yolov8l_weapon_best.pt"
    conf_threshold: float = 0.40              # keep high for weapons, lower FP
    iou_threshold: float = 0.45
    min_consecutive_frames: int = 2           # alarm if visible for N consecutive frames


@dataclass
class AbandonedSettings:
    enabled: bool = False
    yolo_model: str = "yolov8m.pt"            # COCO pretrained, person+bag
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45

    # COCO classes: person=0, backpack=24, handbag=26, suitcase=28
    person_class_id: int = 0
    bag_class_ids: List[int] = field(default_factory=lambda: [24, 26, 28])

    # Ownership / detachment
    ownership_distance_thresh: float = 80.0   # pixels
    detach_distance_thresh: float = 150.0
    abandonment_seconds: float = 5.0          # unowned for this long → alarm


@dataclass
class FaceDetectSettings:
    enabled: bool = False
    yolo_model: str = "yolov8l-face.pt"
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    min_face_size: int = 24                   # drop faces smaller than this


@dataclass
class FaceMatchSettings:
    """ArcFace-based 1:N gallery match.

    Requires FaceDetectSettings.enabled = True to be active.
    Gallery: every jpg/png under face_database/. Each file is one person
    (filename = person name, e.g. enes.jpg).
    """
    enabled: bool = False
    arcface_model: str = "buffalo_l"          # insightface model_zoo
    similarity_threshold: float = 0.50        # cosine, above this = match
    gallery_paths: List[str] = field(default_factory=list)
    rebuild_gallery: bool = True              # scan on first launch


# ---------------- General settings ---------------- #

@dataclass
class CameraSettings:
    source: int = 0                           # 0/1/... or "rtsp://..." (str)
    source_str: str = ""                      # str source override (RTSP/HTTP)
    target_width: int = 1280
    target_height: int = 720
    output_resolution: str = "720p"           # "480p"|"720p"|"1080p"|"native"
    fps_cap: int = 30

    # Debug: use a local video file as the source, like a webcam (OpenCV file capture)
    debug_video_enabled: bool = False
    debug_video_path: str = ""                # .mp4 / .avi / .mov ...
    debug_video_loop: bool = True             # rewind to the start when finished, continuous test
    # True: play in real time at the video's own FPS; drop intermediate frames (grab) if processing cannot keep up
    # False: read as fast as possible (benchmark / heavy-processing mode)
    debug_video_sync_realtime: bool = True


@dataclass
class PerformanceSettings:
    device: str = "auto"                      # auto|cpu|mps|cuda:0
    use_cpu: bool = True
    use_gpu: bool = True                       # prefer MPS/CUDA
    frame_skip: int = 0                        # if >0, every (N+1)th frame is processed
    enable_half_precision: bool = False        # fp16 on CUDA
    seed: int = 42


@dataclass
class AlertSettings:
    save_screenshots: bool = True
    save_json: bool = True
    play_sound: bool = True
    sound_file: str = "alarm.wav"             # under sounds/
    detections_dir: str = "detections"
    banner_duration_seconds: float = 4.0


# ---------------- Top-level Settings ---------------- #

@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    anomaly: AnomalySettings = field(default_factory=AnomalySettings)
    weapon: WeaponSettings = field(default_factory=WeaponSettings)
    abandoned: AbandonedSettings = field(default_factory=AbandonedSettings)
    face_detect: FaceDetectSettings = field(default_factory=FaceDetectSettings)
    face_match: FaceMatchSettings = field(default_factory=FaceMatchSettings)

    # ---------------- Persistence ---------------- #
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        s = cls()
        for key, value in data.items():
            if not hasattr(s, key) or not isinstance(value, dict):
                continue
            block = getattr(s, key)
            for k, v in value.items():
                if hasattr(block, k):
                    setattr(block, k, v)
        return s

    # ---------------- Helpers ---------------- #
    def output_size(self) -> tuple[int, int]:
        m = {
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        if self.camera.output_resolution in m:
            return m[self.camera.output_resolution]
        return (self.camera.target_width, self.camera.target_height)

    def resolved_source(self):
        """int or str — for OpenCV VideoCapture.

        Priority: (1) Debug video file — fed into the pipeline like a webcam
                  (2) RTSP/HTTP URL or numeric string → camera index
                  (3) Webcam index
        """
        cam = self.camera
        if cam.debug_video_enabled and cam.debug_video_path.strip():
            p = Path(cam.debug_video_path.strip()).expanduser()
            return str(p.resolve()) if p.exists() else str(p)
        if cam.source_str.strip():
            s = cam.source_str.strip()
            return int(s) if s.isdigit() else s
        return int(cam.source)

    def is_debug_video_source(self) -> bool:
        """Worker: for file source + loop logic."""
        if not self.camera.debug_video_enabled:
            return False
        return bool(self.camera.debug_video_path.strip())

    def any_module_active(self) -> bool:
        return (
            self.anomaly.enabled
            or self.weapon.enabled
            or self.abandoned.enabled
            or self.face_detect.enabled
        )
