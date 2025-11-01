from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np  # type: ignore
import cv2  # type: ignore

# Lazy import mediapipe to avoid import cost when only training
mp = None  # type: ignore[var-annotated]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_holistic(min_det: float = 0.3, min_trk: float = 0.3):
    global mp
    if mp is None:
        import mediapipe as _mp  # type: ignore

        globals()["mp"] = _mp
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_trk,
    )
    return holistic


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(frame_rgb)


def extract_1662_from_holistic(frame_rgb: np.ndarray, holistic) -> np.ndarray:
    """Return a (1662,) float32 feature vector using the same scheme as backend.
    Order: pose(33*4), left hand(21*3), right hand(21*3), face(468*3).
    Pads/truncates to 1662 to be safe.
    """
    import numpy as _np

    res = holistic.process(frame_rgb)

    def _arr(vals, d):
        try:
            return _np.array(vals, dtype=_np.float32)
        except Exception:
            return _np.zeros(d, dtype=_np.float32)

    pose = _arr(
        [
            [lm.x, lm.y, lm.z, getattr(lm, "visibility", 0.0)]
            for lm in (res.pose_landmarks.landmark if res.pose_landmarks else [])
        ],
        33 * 4,
    ).flatten()
    lh = _arr(
        [
            [lm.x, lm.y, lm.z]
            for lm in (
                res.left_hand_landmarks.landmark if res.left_hand_landmarks else []
            )
        ],
        21 * 3,
    ).flatten()
    rh = _arr(
        [
            [lm.x, lm.y, lm.z]
            for lm in (
                res.right_hand_landmarks.landmark if res.right_hand_landmarks else []
            )
        ],
        21 * 3,
    ).flatten()
    face = _arr(
        [
            [lm.x, lm.y, lm.z]
            for lm in (res.face_landmarks.landmark if res.face_landmarks else [])
        ],
        468 * 3,
    ).flatten()

    feat = _np.concatenate([pose, lh, rh, face]).astype(_np.float32)
    if feat.shape[0] < 1662:
        feat = _np.pad(feat, (0, 1662 - feat.shape[0]))
    elif feat.shape[0] > 1662:
        feat = feat[:1662]
    return feat


@dataclass
class CaptureConfig:
    labels: Tuple[str, ...]
    seq_len: int = 30
    n_seqs_per_label: int = 30
    out_dir: str = "data/sign"
    cam_index: int = 0


def draw_hud(frame: np.ndarray, text: str, color=(0, 255, 0)) -> None:
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
