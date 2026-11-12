"""
Face matching — InsightFace ArcFace (buffalo_l, w600k_r50; LFW 99.57% acc).

Gallery: build ArcFace embeddings from user-selected images (face_database/
or gallery_paths) and match them to FaceDetect bboxes with cosine similarity.

Alarm: when a known person is detected in the scene (not whitelist semantics;
reports who matched; can be used like a "wanted person alert").
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .base import (
    Alarm,
    BaseModule,
    Detection,
    MODULE_FACE_MATCH,
    ModuleResult,
)
from ..utils.paths import FACE_DB_DIR


class FaceMatchModule(BaseModule):
    """1:N gallery match. Takes bboxes from the FaceDetect module; does not
    detect faces on its own (for efficiency).
    """

    name = MODULE_FACE_MATCH
    requires = (MODULE_FACE_MATCH,)  # runs together with face_detect

    def __init__(self, settings, device: str):
        super().__init__(settings, device)
        self._app = None  # insightface FaceAnalysis
        self._gallery: Dict[str, np.ndarray] = {}  # name -> embedding
        self._last_alarm_t: Dict[str, float] = {}

    def load(self) -> None:
        if self._loaded:
            return
        from insightface.app import FaceAnalysis

        ctx_id = -1
        if self.device.startswith("cuda"):
            try:
                ctx_id = int(self.device.split(":")[-1])
            except Exception:
                ctx_id = 0

        self._app = FaceAnalysis(
            name=self.settings.arcface_model,
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        self._build_gallery()
        self._loaded = True

    def _gallery_paths(self) -> List[Path]:
        paths = list(self.settings.gallery_paths or [])
        # also add every image under face_database/
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(str(p) for p in FACE_DB_DIR.rglob(ext))
        # uniq
        return [Path(p) for p in dict.fromkeys(paths)]

    def _build_gallery(self) -> None:
        self._gallery.clear()
        paths = self._gallery_paths()
        if not paths:
            print("[face_match] gallery empty — add images to face_database/ "
                  "or to the selected photos.")
            return
        print(f"[face_match] scanning {len(paths)} images for the gallery...")
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                faces = self._app.get(img)
            except Exception as e:
                print(f"[face_match] {p.name}: extraction error {e}")
                continue
            if not faces:
                continue
            # take the largest face
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = face.normed_embedding
            name = p.stem
            self._gallery[name] = emb
        print(f"[face_match] gallery ready: {len(self._gallery)} people "
              f"({list(self._gallery.keys())[:5]}{'...' if len(self._gallery) > 5 else ''})")

    def reload_gallery(self) -> int:
        """Called when the GUI requests a rescan."""
        self._build_gallery()
        return len(self._gallery)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def process(self, frame: np.ndarray, frame_idx: int, t_sec: float,
                face_dets: Optional[List[Detection]] = None) -> ModuleResult:
        """If face_dets is omitted, insightface runs its own detection."""
        if not self._loaded:
            self.load()
        if not self._gallery:
            return ModuleResult(module=self.name, inference_ms=0.0,
                                extra={"reason": "empty_gallery"})

        detections: List[Detection] = []
        alarms: List[Alarm] = []
        t0 = time.perf_counter()

        if face_dets:
            # Crop from FaceDetect bboxes, then recognize with insightface
            for fd in face_dets:
                x1, y1, x2, y2 = fd.bbox
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(frame.shape[1] - 1, x2); y2 = min(frame.shape[0] - 1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # insightface cannot handle small crops → resize
                if min(crop.shape[:2]) < 112:
                    scale = 112 / min(crop.shape[:2])
                    crop = cv2.resize(
                        crop,
                        (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                try:
                    faces = self._app.get(crop)
                except Exception:
                    continue
                if not faces:
                    continue
                emb = faces[0].normed_embedding
                self._match_and_collect(emb, fd.bbox, t_sec, detections, alarms)
        else:
            # insightface runs its own face detection on the full frame
            try:
                faces = self._app.get(frame)
            except Exception:
                faces = []
            for face in faces:
                bbox = tuple(map(int, face.bbox))
                emb = face.normed_embedding
                self._match_and_collect(emb, bbox, t_sec, detections, alarms)

        infer_ms = (time.perf_counter() - t0) * 1000
        return ModuleResult(
            module=self.name,
            detections=detections,
            alarms=alarms,
            inference_ms=infer_ms,
            extra={"gallery_size": len(self._gallery)},
        )

    def _match_and_collect(self, emb, bbox, t_sec, detections, alarms):
        best_name, best_sim = None, -1.0
        for name, gemb in self._gallery.items():
            sim = self._cosine(emb, gemb)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None:
            return
        if best_sim >= self.settings.similarity_threshold:
            label = f"{best_name} ({best_sim:.2f})"
            color = (0, 0, 255)
            extra = {"name": best_name, "similarity": round(best_sim, 4), "match": True}
            detections.append(
                Detection(module=self.name, bbox=bbox, label=label,
                          confidence=float(best_sim), color=color, extra=extra)
            )
            # cooldown per name (5 s)
            last = self._last_alarm_t.get(best_name, -1e9)
            if t_sec - last > 5.0:
                self._last_alarm_t[best_name] = t_sec
                alarms.append(
                    Alarm(
                        module=self.name,
                        severity="ALARM",
                        message=f"Match: {best_name} (sim {best_sim:.2f})",
                        detections=[detections[-1]],
                        extra=extra,
                    )
                )
        else:
            label = f"? ({best_sim:.2f})"
            detections.append(
                Detection(module=self.name, bbox=bbox, label=label,
                          confidence=float(best_sim), color=(150, 150, 150),
                          extra={"top_match": best_name, "similarity": round(best_sim, 4),
                                 "match": False})
            )
