from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FacePoseResult:
    has_face: bool
    is_frontal_or_profile: bool
    orientation: str  # "FRONTAL", "PROFILE", "BACKSIDE", "OVERHEAD"
    face_score: float  # 0.0 to 1.0 confidence
    skin_ratio_upper: float  # Fraction of skin-tone pixels in facial region
    face_box_crop: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2) in crop coordinates
    details: str = ""


class FacePoseVerifier:
    """
    High-speed, robust face and body orientation verifier.
    Ensures that persons are only registered as unique individuals when their
    face / frontal appearance is verified, preventing back-of-head phantom registrations.
    """

    def __init__(self) -> None:
        self._frontal_cascade: Optional[cv2.CascadeClassifier] = None
        self._profile_cascade: Optional[cv2.CascadeClassifier] = None
        self._alt_cascade: Optional[cv2.CascadeClassifier] = None
        self._init_cascades()

    def _init_cascades(self) -> None:
        try:
            cascade_dir = cv2.data.haarcascades
            alt2_path = os.path.join(cascade_dir, "haarcascade_frontalface_alt2.xml")
            default_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
            profile_path = os.path.join(cascade_dir, "haarcascade_profileface.xml")

            if os.path.exists(alt2_path):
                self._alt_cascade = cv2.CascadeClassifier(alt2_path)
            if os.path.exists(default_path):
                self._frontal_cascade = cv2.CascadeClassifier(default_path)
            if os.path.exists(profile_path):
                self._profile_cascade = cv2.CascadeClassifier(profile_path)

            logger.info("Face & orientation cascades initialized successfully.")
        except Exception as exc:
            logger.warning("Could not initialize face cascades: %s", exc)

    def detect_face_and_pose(self, crop: np.ndarray) -> FacePoseResult:
        """
        Analyzes a person crop to detect facial presence, skin tone distribution,
        and orientation (Frontal / Profile / Backside / Overhead).
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return FacePoseResult(
                has_face=False,
                is_frontal_or_profile=False,
                orientation="BACKSIDE",
                face_score=0.0,
                skin_ratio_upper=0.0,
                details="Empty or invalid crop",
            )

        h, w = crop.shape[:2]

        # Check aspect ratio for extreme overhead/top-down cuts
        aspect_ratio = h / max(1, w)
        if aspect_ratio < 0.85:
            # Very wide crop or top-down view
            is_overhead = True
        else:
            is_overhead = False

        # Isolate upper body / head region (top 45% of bounding box)
        head_h = max(16, int(h * 0.45))
        head_region = crop[0:head_h, :]

        # 1. Skin tone evaluation in facial quadrant (top 5% to 35% height, 20% to 80% width)
        face_y1, face_y2 = int(h * 0.04), int(h * 0.38)
        face_x1, face_x2 = int(w * 0.18), int(w * 0.82)
        face_roi = crop[face_y1:face_y2, face_x1:face_x2]

        skin_ratio = 0.0
        if face_roi.size > 0:
            ycrcb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2YCrCb)
            # Skin tone threshold in YCrCb: Cr [133..173], Cb [77..127]
            skin_mask = cv2.inRange(ycrcb, np.array([0, 133, 77], dtype=np.uint8), np.array([255, 173, 127], dtype=np.uint8))
            skin_pixels = cv2.countNonZero(skin_mask)
            total_pixels = face_roi.shape[0] * face_roi.shape[1]
            skin_ratio = float(skin_pixels) / max(1, total_pixels)

        # 2. Multi-scale Haar Face Cascade Detection
        gray_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2GRAY)
        gray_head = cv2.equalizeHist(gray_head)

        detected_faces = []
        # Try Alt2 (most precise)
        if self._alt_cascade and not self._alt_cascade.empty():
            faces = self._alt_cascade.detectMultiScale(
                gray_head,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.15), int(head_h * 0.20)),
            )
            if len(faces) > 0:
                detected_faces.extend(faces)

        # Try default frontal if none detected
        if len(detected_faces) == 0 and self._frontal_cascade and not self._frontal_cascade.empty():
            faces = self._frontal_cascade.detectMultiScale(
                gray_head,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(int(w * 0.15), int(head_h * 0.20)),
            )
            if len(faces) > 0:
                detected_faces.extend(faces)

        # Try profile face
        is_profile = False
        if len(detected_faces) == 0 and self._profile_cascade and not self._profile_cascade.empty():
            pfaces = self._profile_cascade.detectMultiScale(
                gray_head,
                scaleFactor=1.15,
                minNeighbors=3,
                minSize=(int(w * 0.15), int(head_h * 0.20)),
            )
            if len(pfaces) > 0:
                detected_faces.extend(pfaces)
                is_profile = True
            else:
                # Try flipped for other profile side
                flipped = cv2.flip(gray_head, 1)
                pfaces_flip = self._profile_cascade.detectMultiScale(
                    flipped,
                    scaleFactor=1.15,
                    minNeighbors=3,
                    minSize=(int(w * 0.15), int(head_h * 0.20)),
                )
                if len(pfaces_flip) > 0:
                    is_profile = True
                    # Unflip coordinates
                    fx, fy, fw, fh = pfaces_flip[0]
                    unflipped_x = gray_head.shape[1] - (fx + fw)
                    detected_faces.append((unflipped_x, fy, fw, fh))

        # 3. Formulate Orientation & Decision
        best_box = None
        if len(detected_faces) > 0:
            # Pick the largest detected face
            fx, fy, fw, fh = max(detected_faces, key=lambda b: b[2] * b[3])
            best_box = (int(fx), int(fy), int(fx + fw), int(fy + fh))
            face_score = min(1.0, 0.70 + (skin_ratio * 0.30))
            orientation = "PROFILE" if is_profile else "FRONTAL"
            return FacePoseResult(
                has_face=True,
                is_frontal_or_profile=True,
                orientation=orientation,
                face_score=face_score,
                skin_ratio_upper=skin_ratio,
                face_box_crop=best_box,
                details=f"Face cascade matched ({orientation}, skin ratio: {skin_ratio:.1%})",
            )

        # If cascade missed due to blur/lighting, but skin tone is prominent in facial center:
        if skin_ratio >= 0.24:
            # High frontal facial skin probability
            return FacePoseResult(
                has_face=True,
                is_frontal_or_profile=True,
                orientation="FRONTAL",
                face_score=min(0.85, 0.50 + (skin_ratio * 0.40)),
                skin_ratio_upper=skin_ratio,
                face_box_crop=(face_x1, face_y1, face_x2, face_y2),
                details=f"Facial skin distribution matched (skin ratio: {skin_ratio:.1%})",
            )

        # Otherwise, the view is from the BACKSIDE or OVERHEAD (Hair/Scalp/Collar only)
        orientation = "OVERHEAD" if is_overhead else "BACKSIDE"
        return FacePoseResult(
            has_face=False,
            is_frontal_or_profile=False,
            orientation=orientation,
            face_score=0.0,
            skin_ratio_upper=skin_ratio,
            face_box_crop=None,
            details=f"Backside / Overhead view detected (skin ratio: {skin_ratio:.1%}, no face)",
        )
