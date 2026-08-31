from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config, get_config
from tracking.face_verifier import FacePoseResult, FacePoseVerifier
from tracking.llm_person_profiler import (
    LLMPersonProfiler,
    LLMReIDDecision,
    PersonSemanticProfile,
)
from utils.logger import get_logger
from utils.timeutils import utcnow

logger = get_logger(__name__)


@dataclass
class ClothingSignature:
    upper_hist: np.ndarray  # HSV color histogram of upper body (shirt/top)
    lower_hist: np.ndarray  # HSV color histogram of lower body (pants/bottom)
    full_color: np.ndarray  # Full body color feature
    deep_embedding: np.ndarray  # Deep neural feature
    composite: np.ndarray  # Full combined normalized vector


@dataclass
class GlobalPerson:
    global_id: str
    person_number: int
    display_name: str
    role: str = "visitor"  # "visitor" or "sales_person"
    embedding: Optional[np.ndarray] = None  # Primary representative embedding
    embeddings: List[np.ndarray] = field(default_factory=list)  # Multi-view memory bank (up to 16 templates)
    clothing_signatures: List[ClothingSignature] = field(default_factory=list)  # Clothing memory bank
    semantic_profile: Optional[PersonSemanticProfile] = None  # LLM extracted persona and clothing profile
    llm_reasoning: str = ""  # Latest LLM decision reasoning
    first_seen: datetime = field(default_factory=utcnow)
    last_seen: datetime = field(default_factory=utcnow)
    total_dwell_seconds: float = 0.0
    visit_count: int = 1
    is_active: bool = True
    current_track_id: Optional[int] = None
    last_active_timestamp: Optional[datetime] = None
    thumbnail_path: Optional[str] = None
    thumbnail_base64: Optional[str] = None
    frames_tracked: int = 0
    face_score: float = 0.0  # Quality score of primary frontal face capture
    has_verified_face: bool = False  # True once verified from frontal/face perspective
    view_angles: List[str] = field(default_factory=list)

    def add_feature_template(self, signature: ClothingSignature, max_templates: int = 16) -> None:
        """Adds an appearance template to the person's multi-view memory bank."""
        self.clothing_signatures.append(signature)
        self.embeddings.append(signature.composite)
        if len(self.clothing_signatures) > max_templates:
            self.clothing_signatures.pop(0)
        if len(self.embeddings) > max_templates:
            self.embeddings.pop(0)

        # Update representative embedding with running average
        if self.embedding is None:
            self.embedding = signature.composite.copy()
        else:
            updated = 0.85 * self.embedding + 0.15 * signature.composite
            norm = np.linalg.norm(updated)
            if norm > 1e-6:
                self.embedding = updated / norm

    def update_dwell(self, current_ts: datetime) -> None:
        if self.last_active_timestamp is not None:
            delta = (current_ts - self.last_active_timestamp).total_seconds()
            if 0.0 < delta < 10.0:  # Valid continuous step
                self.total_dwell_seconds += delta
        self.last_active_timestamp = current_ts
        self.last_seen = current_ts
        self.is_active = True
        self.frames_tracked += 1

    def mark_left(self) -> None:
        self.is_active = False
        self.current_track_id = None
        self.last_active_timestamp = None

    def to_dict(self) -> dict:
        return {
            "global_id": self.global_id,
            "person_number": self.person_number,
            "display_name": self.display_name,
            "role": self.role,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "total_dwell_seconds": round(self.total_dwell_seconds, 1),
            "visit_count": self.visit_count,
            "is_active": self.is_active,
            "current_track_id": self.current_track_id,
            "thumbnail_path": self.thumbnail_path,
            "has_verified_face": self.has_verified_face,
            "face_score": round(self.face_score, 2),
            "semantic_profile": self.semantic_profile.to_dict() if self.semantic_profile else None,
            "llm_reasoning": self.llm_reasoning,
        }


class ReIDFeatureExtractor:
    """Extracts discriminative multi-region clothing & deep appearance embeddings from person image crops."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.reid_cfg = self.config.reid
        self._torch_model = None
        self._torch_transforms = None
        self._device = "cpu"
        self._init_model()

    def _init_model(self) -> None:
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms

            try:
                weights = models.MobileNet_V3_Large_Weights.DEFAULT
                model = models.mobilenet_v3_large(weights=weights)
                dim = 960
            except Exception:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                model = models.mobilenet_v3_small(weights=weights)
                dim = 576

            model.classifier = torch.nn.Identity()
            model.eval()
            self._torch_model = model
            self._feature_dim = dim
            self._torch_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            logger.info("Upgraded ReID PyTorch MobileNetV3-Large (%d-dim) feature extractor loaded.", dim)
        except Exception as exc:
            logger.warning(
                "Could not initialize PyTorch ReID model (%s). Using spatial clothing color descriptor.",
                exc,
            )
            self._torch_model = None

    def _calc_color_hist(self, region: np.ndarray) -> np.ndarray:
        """Extracts dual HSV + CIELAB histogram for robust lighting-invariant clothing matching."""
        if region is None or region.size == 0:
            return np.zeros(64, dtype=np.float32)
        # HSV (32 bins)
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
        hsv_part = np.concatenate([h_hist, s_hist, v_hist])

        # CIELAB for lighting & shadow invariance (32 bins)
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        l_hist = cv2.calcHist([lab], [0], None, [8], [0, 256]).flatten()
        a_hist = cv2.calcHist([lab], [1], None, [12], [0, 256]).flatten()
        b_hist = cv2.calcHist([lab], [2], None, [12], [0, 256]).flatten()
        lab_part = np.concatenate([l_hist, a_hist, b_hist])

        part = np.concatenate([hsv_part, lab_part])
        norm = np.linalg.norm(part)
        return (part / norm) if norm > 1e-6 else np.zeros(64, dtype=np.float32)

    def extract_signature(self, crop: np.ndarray) -> Optional[ClothingSignature]:
        """Extracts complete clothing and appearance signature from a person crop."""
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return None

        h, w = crop.shape[:2]

        # Center-weighted regions to minimize background noise
        x_min, x_max = int(w * 0.10), int(w * 0.90)
        if x_max <= x_min:
            x_min, x_max = 0, w

        # Upper Body (Shirt / Top): 12% to 52% height
        # Lower Body (Pants / Jeans / Bottom): 52% to 92% height
        upper_region = crop[int(h * 0.12) : int(h * 0.52), x_min:x_max]
        lower_region = crop[int(h * 0.52) : int(h * 0.92), x_min:x_max]
        full_region = crop[:, x_min:x_max]

        upper_hist = self._calc_color_hist(upper_region)
        lower_hist = self._calc_color_hist(lower_region)
        full_color = self._calc_color_hist(full_region)

        # Deep appearance feature from MobileNetV3 Large
        deep_dim = getattr(self, "_feature_dim", 960)
        deep_feat = np.zeros(deep_dim, dtype=np.float32)
        if self._torch_model is not None:
            try:
                import torch

                rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self._torch_transforms(rgb_crop).unsqueeze(0)
                with torch.no_grad():
                    deep_raw = self._torch_model(tensor).cpu().numpy().flatten()
                norm = np.linalg.norm(deep_raw)
                if norm > 1e-6:
                    deep_feat = deep_raw / norm
            except Exception as exc:
                logger.debug("Torch feature inference exception: %s", exc)

        # Build composite feature: Upper Clothing (35%) + Lower Clothing (35%) + Deep Features (30%)
        composite = np.concatenate([
            upper_hist * 0.6,
            lower_hist * 0.6,
            full_color * 0.3,
            deep_feat * 0.45,
        ]).astype(np.float32)
        norm = np.linalg.norm(composite)
        if norm > 1e-6:
            composite = composite / norm

        return ClothingSignature(
            upper_hist=upper_hist,
            lower_hist=lower_hist,
            full_color=full_color,
            deep_embedding=deep_feat,
            composite=composite,
        )

    def extract(self, crop: np.ndarray) -> np.ndarray:
        sig = self.extract_signature(crop)
        if sig is None:
            return np.zeros(1024, dtype=np.float32)
        return sig.composite

    @staticmethod
    def compute_similarity(sig1: ClothingSignature, sig2: ClothingSignature) -> float:
        """Computes a multi-parameter clothing and appearance similarity score."""
        upper_sim = float(np.dot(sig1.upper_hist, sig2.upper_hist))
        lower_sim = float(np.dot(sig1.lower_hist, sig2.lower_hist))
        full_color_sim = float(np.dot(sig1.full_color, sig2.full_color))
        deep_sim = float(np.dot(sig1.deep_embedding, sig2.deep_embedding)) if (
            np.linalg.norm(sig1.deep_embedding) > 0 and np.linalg.norm(sig2.deep_embedding) > 0
        ) else full_color_sim

        # Weighted multi-parameter score:
        # - Upper clothing match: 35%
        # - Lower clothing match: 35%
        # - Deep appearance / body shape match: 20%
        # - Full color palette match: 10%
        score = (0.35 * upper_sim) + (0.35 * lower_sim) + (0.20 * deep_sim) + (0.10 * full_color_sim)
        return max(0.0, min(1.0, score))


class PersonGallery:
    """
    Maintains unique registered persons with Face/Frontal orientation gating,
    multi-view appearance memory banks, and biometric ReID.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self.reid_cfg = self.config.reid
        self.extractor = ReIDFeatureExtractor(self.config)
        self.face_verifier = FacePoseVerifier()
        self.similarity_threshold = self.reid_cfg.similarity_threshold
        self.momentum = self.reid_cfg.momentum

        self._persons: Dict[str, GlobalPerson] = {}
        self._track_to_global: Dict[int, str] = {}
        self._next_person_idx: int = 1
        self._sales_persons: set[str] = set()

        self.profiler = LLMPersonProfiler(self.config)
        self.llm_decisions: List[LLMReIDDecision] = []
        self.thumbnails_dir = self.config.paths.thumbnails
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)

    def designate_sales_person(self, global_id: str, is_sales: bool = True, name: Optional[str] = None) -> None:
        person = self._persons.get(global_id)
        if person is not None:
            if is_sales:
                person.role = "sales_person"
                if name:
                    person.display_name = name
                self._sales_persons.add(global_id)
            else:
                person.role = "visitor"
                if name:
                    person.display_name = name
                self._sales_persons.discard(global_id)
            logger.info("Person %s role updated to %s (%s).", global_id, person.role, person.display_name)

    def set_person_name(self, global_id: str, name: str) -> None:
        person = self._persons.get(global_id)
        if person is not None:
            person.display_name = name

    def get_all_persons(self) -> List[GlobalPerson]:
        return list(self._persons.values())

    def get_person(self, global_id: str) -> Optional[GlobalPerson]:
        return self._persons.get(global_id)

    def get_by_track(self, track_id: int) -> Optional[GlobalPerson]:
        gid = self._track_to_global.get(track_id)
        return self._persons.get(gid) if gid else None

    def active_visitors_count(self) -> int:
        return sum(1 for p in self._persons.values() if p.is_active and p.role == "visitor")

    def active_sales_count(self) -> int:
        return sum(1 for p in self._persons.values() if p.is_active and p.role == "sales_person")

    def total_unique_visitors_count(self) -> int:
        return sum(1 for p in self._persons.values() if p.role == "visitor")

    def total_unique_sales_count(self) -> int:
        return sum(1 for p in self._persons.values() if p.role == "sales_person")

    def get_llm_decisions(self) -> List[LLMReIDDecision]:
        return list(self.llm_decisions)

    def _save_thumbnail(self, crop: np.ndarray, global_id: str) -> Tuple[Optional[str], Optional[str]]:
        if not self.reid_cfg.save_thumbnails or crop.size == 0:
            return None, None
        try:
            filename = f"{global_id}.jpg"
            filepath = self.thumbnails_dir / filename
            cv2.imwrite(str(filepath), crop)

            _, buf = cv2.imencode(".jpg", cv2.resize(crop, (96, 128)))
            b64_str = base64.b64encode(buf).decode("utf-8")
            return str(filepath), b64_str
        except Exception as exc:
            logger.warning("Could not save thumbnail for %s: %s", global_id, exc)
            return None, None

    def match_or_create(
        self,
        frame: np.ndarray,
        box: Tuple[float, float, float, float],
        track_id: int,
        timestamp: datetime,
    ) -> Tuple[Optional[GlobalPerson], bool, float]:
        """
        Matches a detected person crop against INACTIVE gallery persons.
        Gated by Facial / Frontal Verification: A NEW person is only created if their
        face / frontal view is clearly identified, preventing back-of-head phantom registrations.
        Returns: (Optional[GlobalPerson], is_new_visitor, similarity_score)
        """
        # If this track ID is already mapped and active, retrieve it directly
        if track_id in self._track_to_global:
            gid = self._track_to_global[track_id]
            person = self._persons.get(gid)
            if person is not None:
                person.update_dwell(timestamp)
                person.current_track_id = track_id
                return person, False, 1.0

        # Extract crop from frame
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        crop_h, crop_w = crop.shape[:2]
        if crop_h < self.reid_cfg.crop_min_size or crop_w < self.reid_cfg.crop_min_size:
            current_signature = None
        else:
            current_signature = self.extractor.extract_signature(crop)

        # 1. Evaluate Face & Pose Orientation
        face_info: FacePoseResult = self.face_verifier.detect_face_and_pose(crop)

        # Extract immediate visual persona profile (instant 2ms)
        current_profile = self.profiler._extract_local_vision_profile(crop, f"Track-{track_id}")

        # 2. Compare against INACTIVE persons (persons who previously left into interior rooms)
        candidate_matches: List[Tuple[GlobalPerson, float]] = []

        if current_signature is not None and len(self._persons) > 0:
            for person in self._persons.values():
                # PHYSICAL EXCLUSION: If person is currently visible on camera with another track ID, skip!
                if person.is_active and person.current_track_id is not None and person.current_track_id != track_id:
                    continue

                person_sim = 0.0
                if person.clothing_signatures:
                    for stored_sig in person.clothing_signatures:
                        s = self.extractor.compute_similarity(current_signature, stored_sig)
                        if s > person_sim:
                            person_sim = s
                elif person.embedding is not None:
                    person_sim = float(np.dot(current_signature.composite, person.embedding))

                if person_sim >= 0.50:  # Candidate match
                    candidate_matches.append((person, person_sim))

        # Sort candidate matches by highest similarity first
        candidate_matches.sort(key=lambda x: x[1], reverse=True)

        # 3. Decision Arbitration for Returning Visitors
        if candidate_matches:
            best_cand, best_sim = candidate_matches[0]
            if best_sim >= 0.56:
                # Re-ID Match Found for returning person
                person = best_cand
                was_inactive = not person.is_active

                if current_signature is not None:
                    person.add_feature_template(current_signature)

                # If face is now visible with higher quality, upgrade the primary face thumbnail
                if face_info.is_frontal_or_profile and face_info.face_score > person.face_score:
                    thumb_path, thumb_b64 = self._save_thumbnail(crop, person.global_id)
                    if thumb_path:
                        person.thumbnail_path = thumb_path
                        person.thumbnail_base64 = thumb_b64
                    person.face_score = face_info.face_score
                    person.has_verified_face = True

                person.semantic_profile = current_profile
                person.is_active = True
                person.current_track_id = track_id
                person.update_dwell(timestamp)
                self._track_to_global[track_id] = person.global_id

                if was_inactive:
                    person.visit_count += 1
                    logger.info(
                        "🔄 Re-identified returning %s (%s) with %d%% visual similarity (Orientation: %s).",
                        person.global_id,
                        person.display_name,
                        int(best_sim * 100),
                        face_info.orientation,
                    )

                return person, False, best_sim

        # 4. Strict Registration Gating:
        # A NEW person is ONLY registered if their face / frontal view is confirmed.
        if not face_info.is_frontal_or_profile:
            # The person is seen from the backside, extreme overhead angle, or head only without face
            logger.debug(
                "⏳ Track %d seen from %s (no face detected). Holding provisional status until front/face is shown.",
                track_id,
                face_info.orientation,
            )
            return None, False, 0.0

        # 5. Create & Register Confirmed New Person (Frontal Face Verified)
        gid = f"P-{self._next_person_idx:03d}"
        self._next_person_idx += 1

        thumb_path, thumb_b64 = self._save_thumbnail(crop, gid)
        person = GlobalPerson(
            global_id=gid,
            person_number=self._next_person_idx - 1,
            display_name=f"Visitor #{self._next_person_idx - 1}",
            role="visitor",
            embedding=current_signature.composite if current_signature else None,
            semantic_profile=current_profile,
            llm_reasoning=f"Frontal face confirmed ({face_info.details})",
            first_seen=timestamp,
            last_seen=timestamp,
            total_dwell_seconds=0.0,
            visit_count=1,
            is_active=True,
            current_track_id=track_id,
            last_active_timestamp=timestamp,
            thumbnail_path=thumb_path,
            thumbnail_base64=thumb_b64,
            face_score=face_info.face_score,
            has_verified_face=True,
            view_angles=[face_info.orientation],
        )
        if current_signature is not None:
            person.add_feature_template(current_signature)

        self._persons[gid] = person
        self._track_to_global[track_id] = gid
        if crop is not None and crop.size > 0:
            self.profiler.profile_person_async(crop, person)

        logger.info(
            "👤 NEW Person Verified & Registered: %s (Track ID: %d, Face Score: %.2f, Angle: %s)",
            gid,
            track_id,
            face_info.face_score,
            face_info.orientation,
        )
        return person, True, 1.0

    def update_active_person_appearance(
        self,
        frame: np.ndarray,
        box: Tuple[float, float, float, float],
        track_id: int,
        timestamp: Optional[datetime] = None,
    ) -> Optional[GlobalPerson]:
        """
        Periodically captures additional viewpoints/poses while person is moving in front of camera.
        If track was provisional (waiting for face), dynamically registers them once face is shown.
        """
        ts = timestamp or utcnow()
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]

        if crop.shape[0] < self.reid_cfg.crop_min_size or crop.shape[1] < self.reid_cfg.crop_min_size:
            return self.get_by_track(track_id)

        gid = self._track_to_global.get(track_id)
        
        # Case A: Track is already registered to a GlobalPerson
        if gid:
            person = self._persons.get(gid)
            if not person:
                return None

            # Check if a higher-quality frontal face is now visible
            face_info = self.face_verifier.detect_face_and_pose(crop)
            if face_info.is_frontal_or_profile and face_info.face_score > (person.face_score + 0.10):
                thumb_path, thumb_b64 = self._save_thumbnail(crop, person.global_id)
                if thumb_path:
                    person.thumbnail_path = thumb_path
                    person.thumbnail_base64 = thumb_b64
                person.face_score = face_info.face_score
                person.has_verified_face = True
                logger.debug("Updated primary face thumbnail for %s (Score: %.2f)", person.global_id, face_info.face_score)

            # Periodically record multi-angle clothing signatures (front, sides, and back)
            if person.frames_tracked % 15 == 0:
                sig = self.extractor.extract_signature(crop)
                if sig is not None:
                    person.add_feature_template(sig)
                    if face_info.orientation not in person.view_angles:
                        person.view_angles.append(face_info.orientation)

            return person

        # Case B: Track was PROVISIONAL (entered from backside/overhead without face)
        # Check if the person has now turned around and shows their face
        person, is_new, sim = self.match_or_create(frame, box, track_id, ts)
        return person

    def on_track_lost(self, track_id: int) -> None:
        gid = self._track_to_global.pop(track_id, None)
        if gid:
            person = self._persons.get(gid)
            if person is not None and (person.current_track_id == track_id or person.current_track_id is None):
                person.mark_left()
                logger.debug("Person %s marked inactive/left (Track %d lost).", gid, track_id)

    def reset(self) -> None:
        self._persons.clear()
        self._track_to_global.clear()
        self._next_person_idx = 1
        self._sales_persons.clear()
        logger.info("PersonGallery reset.")
