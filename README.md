# Realtime Surveillance — Integrated Security Monitoring

A **5-module real-time security monitoring application** developed as part of a
Master's thesis. Five separate components are combined under a single GUI so the
user can turn them on and off, tune them, and watch them live.

| Module | Task | Model | Benchmark reference |
|---|---|---|---|
| 🚨 **Anomaly Detection** | Crowd panic + density anomaly | YOLOv8m + ByteTrack, `imgsz` 960 (set from the GUI) | UMN: F1=1.0 (v2-production); for fast motion use L + imgsz↑ |
| 🔫 **Weapon Detection** | Handgun/knife detection | YOLOv8L fine-tuned (`yolov8l_weapon_best.pt`) | mAP@0.5 = **87.47%**, ~114 FPS |
| 🎒 **Abandoned Object** | Bag/suitcase separated from its owner | YOLOv8m (COCO) + ByteTrack + distance ownership | ABODA: TerkEdilmisNesne project metric |
| 🙂 **Face Detection** | Find faces in the scene | YOLOv8L-Face (`yolov8l-face.pt`) | WIDER Face AP@0.5 0.685, **11.7 ms** |
| 🔎 **Face Matching** | 1:N gallery match | InsightFace ArcFace `buffalo_l` | LFW **99.57% acc**, 0.70% EER |

Each module's backbone was benchmarked in detail in its own project (AnomaliDedection, SilahTespit,
TerkEdilmisNesne, YuzTespit, YuzEslestirme), and the "best" models here were
chosen directly from those results.

---

## Quick Start

```bash
# 1) Use the venv at the workspace root
cd "/path/to/PythonArastirmalar"
source venv/bin/activate

# 2) Install packages specific to RealtimeSurveillance (PySide6, pygame, lap)
pip install -r RealtimeSurveillance/requirements.txt

# 3) Run
cd RealtimeSurveillance
python run.py
```

The app opens full screen. From the **left panel**, turn on the modules you want
(Anomaly / Weapon / Abandoned / Face / Face Matching) with the checkboxes, set the
thresholds, and press **Start**.

Shortcuts:
- `F11` → full screen ↔ window
- `Esc` or `Ctrl+Q` → quit
- `Ctrl+R` → rescan the face gallery

---

## Architecture

```
RealtimeSurveillance/
├── run.py                          # Quick launcher (python run.py)
├── settings.json                   # Persisted settings (written by Save/Load)
│
├── app/
│   ├── main.py                     # QApplication entrypoint
│   ├── core/
│   │   ├── settings.py             # Dataclass-based settings schema + JSON
│   │   ├── frame_processor.py      # Orchestrates all modules in order
│   │   └── alert_manager.py        # Detection → JPG + JSON + sound + banner state
│   ├── modules/
│   │   ├── base.py                 # Detection / Alarm / ModuleResult dataclasses
│   │   ├── anomaly.py              # YOLOv8 + ByteTrack + speed/density anomaly
│   │   ├── weapon.py               # YOLOv8L weapon best
│   │   ├── abandoned.py            # YOLOv8m + COCO bag class + ownership state
│   │   ├── face_detect.py          # YOLOv8L-Face
│   │   └── face_match.py           # InsightFace ArcFace 1:N gallery
│   ├── ui/
│   │   ├── main_window.py          # Full-screen shell + signals
│   │   ├── settings_panel.py       # Left panel: all settings
│   │   ├── video_panel.py          # Right panel: video + alarm banner
│   │   └── worker.py               # QThread: capture + processing
│   └── utils/
│       ├── paths.py                # ROOT, MODELS_DIR, DETECTIONS_DIR, ...
│       ├── device.py               # CUDA / MPS / CPU autodetect
│       ├── seed.py                 # Reproducibility
│       └── audio.py                # Alarm sound via pygame (with cooldown)
│
├── models/                         # All weights are stored here, per module
│   ├── anomaly/yolov8s.pt
│   ├── weapon/yolov8l_weapon_best.pt
│   ├── abandoned/yolov8m.pt
│   ├── face_detect/yolov8l-face.pt
│   └── face_match/                 # InsightFace loads automatically from ~/.insightface
│
├── face_database/                  # Bulk photos for the 1:N gallery (.jpg/.png)
├── detections/                     # Detections: tespit_<MODULE>_<DATE_TIME>.{jpg,json}
├── sounds/alarm.wav                # Short synthetic siren (auto-generated)
└── logs/
```

---

## Left Panel — Settings

### 📷 Camera
| Setting | Description |
|---|---|
| Webcam index | OS camera index (0 = built-in, 1 = USB, ...) |
| URL | RTSP/HTTP IP camera URL (takes priority if set) |
| Processing resolution | `480p` / `720p` / `1080p` / `native` — if the camera is 4K, processing drops to 720p → large FPS gain |
| Native W/H | Target width/height when "native" is selected |
| FPS cap | Upper FPS bound (so CPU/GPU is not worked harder than needed) |

### 🐛 Debug — Video simulation (instead of webcam)
Feed UMN / ABODA and similar recordings into the pipeline **like a live webcam stream**:

| Setting | Description |
|---|---|
| Local video instead of webcam | When on, source = selected file (OpenCV `VideoCapture(path)`) |
| Video file | MP4/AVI/MOV/MKV … via `Choose file…` |
| Loop the video | When on, rewinds to the start at end of file; continuous benchmark/settings test |
| Play at real-time speed | When on (default), playback stays in sync with the FPS reported by the video; if inference lags, intermediate frames are **skipped** with `grab()` — it does not process every frame heavily and produce "slow motion". When off, the file is read as fast as possible. |

**Priority order:** debug video (on + valid path) → RTSP/HTTP URL → webcam index.

A warning is shown before `Start` if the path is empty or the file is missing. Processing resolution and frame skip are applied through `FrameProcessor` the same way as for a webcam. The playback-speed ceiling is the **smaller** of **FPS cap** and the video's own FPS (e.g. 25 FPS video + cap 60 → 25 FPS).

### ⚙️ Performance / Device
- **CPU available** ✅: always true
- **GPU available** ✅: MPS on Apple Silicon, CUDA on NVIDIA, detected automatically
- **Active device**: `auto` (recommended), `cpu`, `mps`, `cuda:0` ...
- **Frame skip**: `0` every frame, `1` 1 of every 2 frames, `2` 1 of every 3 frames ... (1-2 recommended for CPU)
- **Seed**: deterministic run (default 42)

### 🔔 Detection Log / Alarm
- ☑ Snapshot of the detection moment → `detections/tespit_<MODULE>_<YYYY-MM-DD_HH-MM-SS_ms>.jpg`
- ☑ Details as JSON → `.json` with the same name (frame_index, video_t_sec, bboxes, label, confidence, module-specific extra)
- ☑ Alarm sound (sounds/alarm.wav) — 1 s cooldown per module, no spam
- Banner duration (s): how long the red strip at the top stays visible

### 🚨 Anomaly Detection
- **Max density (total)**: alarm if the number of people in the scene exceeds this
- **Per-cell cap (hotspot)**: the scene is split into a 3x3 grid; alarm if any cell exceeds this count (even if the total is normal)
- **Running-speed threshold (px/s)**: a track is counted as "running" if its pixels/second speed is above this
- **Running ratio (for panic)**: "collective panic" if the runners/total ratio is above this threshold and the minimum counts are met
- **Min running people** + **Min total people**: noise filter (one person running should not alarm)
- **Persistence (s)** + **Cooldown (s)**: an alarm should not fire from a single noisy frame, and should not fire again immediately

> **Note**: The full tuning ablation for this module is in the
> "Three-run tuning ablation" section of `AnomaliDedection/README.md`. Defaults
> are taken from the **v2-production** profile there (F1=1.0 on UMN).

### 🔫 Weapon Detection
- **Confidence threshold**: default 0.40 (keep high for weapons → false positive ↓)
- **IoU**: for NMS
- **Min consecutive frames**: do not alarm unless it is visible for N consecutive frames (FP ↓)

> Model is fine-tuned (Ubuntu RTX 5090, 100 epoch) — `SilahTespit/README.md`

### 🎒 Abandoned Object
- **Abandonment time (s)**: if the bag does not return within this long after leaving its owner → ABANDONED → alarm
- **Owner match distance (px)**: if the bag is within this distance of a person, that person becomes the owner
- **Ownership state machine**: UNASSIGNED → OWNED → DETACHED → ABANDONED
- **Detach distance (px)**: counted as "DETACHED" if the nearest person to the bag exceeds this threshold
- **Owner ID while OWNED**: because the ByteTrack person ID changes often, the owner is **not locked to a fixed track ID**; each frame the **nearest person within the detach threshold** of the bag is treated as the carrier (backpack + walking person scenario).

> COCO classes are used: `person=0, backpack=24, handbag=26, suitcase=28`

### 🙂 Face Detection
- Confidence + IoU + min face size
- YOLOv8L-Face was chosen because it is the fastest option (RetinaFace is more accurate but 5×
  slower; in a live pipeline that trade-off means a 4-5× speed loss for a <5% difference
  in WIDER Face AP@0.5).

### 🔎 Face Matching
- **Prerequisite**: Face Detection must be active (enforced automatically)
- **Match threshold (cosine)**: 0.50 (default, a good operating point for ArcFace)
- **Gallery**:
  1. Put `face_database/<name>.jpg` → scanned automatically
  2. or bulk-select with **Add Photos** from the GUI
  3. Then recompute embeddings with **Scan Gallery**
- On a match the banner shows "Match: \<name\> (sim 0.78)" + save + sound

---

## Right Panel — Live View

```
┌──────────────────────────────────────────────────────────────────┐
│  ⚠ ALARM [SILAH] Weapon detected (knife 0.91)                    │  ← top banner
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│             ╔══════════════╗                                     │
│             ║              ║   ← live camera + bbox/labels       │
│             ║   knife 0.91 ║                                     │
│             ╚══════════════╝                                     │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│  FPS 28.4 | ANOMALI:18ms YUZ:11ms SILAH:32ms | people=4 faces=2  │
│  [15:35:21] Models ready.                                        │
│  [15:35:23] Stream started.                                      │
│  [15:35:47] Match: enes (sim 0.78)                               │
└──────────────────────────────────────────────────────────────────┘
```

- **Top banner**: alarm active → red + large; otherwise dark green "Monitoring active"
- **Bottom status**: instantaneous FPS, per-module inference times (ms), counters (people, faces, ...)
- **Bottom-right log**: system messages (model load, alarm, gallery rebuild)

---

## Detection Record Format

Two files are written for each alarm.

**`tespit_SILAH_2026-12-12_22-45-13_472.jpg`** — the alarm frame with banner + bboxes overlaid.

**`tespit_SILAH_2026-12-12_22-45-13_472.json`**:
```json
{
  "timestamp": "2026-12-12_22-45-13",
  "timestamp_iso": "2026-12-12T22:45:13.472",
  "module": "SILAH",
  "severity": "ALARM",
  "message": "Weapon detected (knife 0.91)",
  "frame_index": 1284,
  "video_t_sec": 42.83,
  "detections": [
    {
      "module": "SILAH",
      "bbox": [482, 217, 612, 388],
      "label": "knife 0.91",
      "confidence": 0.91,
      "extra": {"class": "knife"}
    }
  ],
  "extra": {"top_class": "knife"},
  "image_path": "/.../detections/tespit_SILAH_2026-12-12_22-45-13_472.jpg",
  "settings_snapshot": { ... all application settings at that moment ... }
}
```

`settings_snapshot` means every alarm carries a full trace of **which configuration**
produced it → critical for tabulating results in the thesis.

---

## Module Order and Performance

`FrameProcessor.process()` calls the active modules in this order (for every frame):

1. **Anomaly** (ByteTrack tracking + density grid + speed) — ~16-21 ms (MPS)
2. **Weapon** — ~30-40 ms (YOLOv8L; 11M params)
3. **Abandoned** (ByteTrack + COCO bag) — ~22-30 ms
4. **Face Detection** — ~12 ms
5. **Face Matching** (consumes the bboxes from 4) — ~20-50 ms per face (ArcFace)

With everything on, expected on Apple Silicon (M3+) MPS: **6-10 FPS**. For higher FPS:
- **Frame skip = 1 or 2** (most effective — modules process every 2nd/3rd frame instead of every frame)
- **Processing resolution 480p** (required for 4K cameras)
- Turn on only the modules you need (e.g. only **Anomaly + Weapon** → 25-35 FPS)
- If an NVIDIA GPU is available, select `cuda:0` → the whole pipeline comfortably exceeds 50 FPS

Best practice is to turn **Face Matching** on only at critical moments rather than
leaving it on continuously (an extra 20-50 ms per face).

---

## Settings File (Persistence)

- **Save**: `Save Settings` on the left panel → `RealtimeSurveillance/settings.json`
- **Load**: `Load Settings` from the same place (also loaded automatically after
  an application restart)

That way you do not have to re-enter tuning for the same camera/scene every time.

---

## Relationship to the Backed-Up Modules

This application is the **production version** of the 5 separate projects in the
parent directory (`AnomaliDedection`, `SilahTespit`, `TerkEdilmisNesne`, `YuzTespit`,
`YuzEslestirme`):
- Each module's best weights were taken as a **copy** under `models/<module>/`
  → this application still runs on its own if the parent projects change.
- Each module uses a simplified, real-time-friendly variant of its methodology
  (e.g. for anomaly, Method 1 / track-based; Method 2 / optical flow is heavier
  and does not add value to the live pipeline).
- Detailed benchmark + ablation results remain in each parent project's own
  `README.md`. This application can be kept as the "**product**" section of the
  thesis; the benchmarks can be kept separately as the "**experiment**" section.

---

## Known Notes

- **macOS pygame ↔ cv2 SDL2 warning**: both ship an SDL2 dylib; the output shows
  a "may cause spurious casting failures" warning, but behavior is unaffected.
- **Face Matching gallery**: if **face_database/** is empty on first launch, the
  module drops into the "empty_gallery" state; it becomes active after photos are
  added from the GUI and "Scan Gallery" is pressed.
- **MPS + large models**: on M1/M2, FPS with YOLOv8L can drop to 5-8 → frame skip 2 is recommended.

---

## License / Attribution

This application is for an academic thesis. Models used:
- Ultralytics YOLOv8 (AGPL-3.0)
- InsightFace ArcFace (Apache-2.0)
- PySide6 (LGPL)
