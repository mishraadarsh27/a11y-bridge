from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

# Lazy import mediapipe only when needed
mp = None  # type: ignore[var-annotated]


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


def extract_1662_from_frame(frame_bgr: np.ndarray, holistic) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    res = holistic.process(rgb)

    def _arr(vals, d):
        try:
            return np.array(vals, dtype=np.float32)
        except Exception:
            return np.zeros(d, dtype=np.float32)

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

    feat = np.concatenate([pose, lh, rh, face]).astype(np.float32)
    if feat.shape[0] < 1662:
        feat = np.pad(feat, (0, 1662 - feat.shape[0]))
    elif feat.shape[0] > 1662:
        feat = feat[:1662]
    return feat


def uniform_indices(n: int, k: int) -> List[int]:
    if n <= k:
        return list(range(n))
    return [int(i * (n - 1) / (k - 1)) for i in range(k)]


def build_video_label_map(
    wlasl_json: Path, split: str = "train", limit_classes: int | None = None
) -> Dict[str, str]:
    with wlasl_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    label_counts: Dict[str, int] = {}
    mapping: Dict[str, str] = {}
    for item in data:
        label = item.get("label") or item.get("gloss")
        if not label:
            continue
        if limit_classes is not None and label_counts.get(label, 0) >= 10**9:
            # placeholder; we'll trim later globally
            pass
        for inst in item.get("instances", []):
            if inst.get("split") != split:
                continue
            vid = inst.get("video_id")
            if not vid:
                continue
            mapping[str(vid)] = str(label)
            label_counts[label] = label_counts.get(label, 0) + 1
    if limit_classes is None:
        return mapping
    # keep only top-N frequent labels
    top = {
        k
        for k, _ in sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :limit_classes
        ]
    }
    return {vid: lab for vid, lab in mapping.items() if lab in top}


def process_videos(
    videos_dir: Path,
    mapping: Dict[str, str],
    out_dir: Path,
    max_clips_per_label: int | None = None,
    seq_len: int = 30,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    holo = get_holistic()
    written = 0
    skipped = 0
    per_label: Dict[str, int] = {}
    for p in sorted(videos_dir.glob("*.mp4")):
        vid = p.stem
        lab = mapping.get(vid)
        if not lab:
            skipped += 1
            continue
        if (
            max_clips_per_label is not None
            and per_label.get(lab, 0) >= max_clips_per_label
        ):
            continue
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            skipped += 1
            continue
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            skipped += 1
            continue
        idxs = uniform_indices(len(frames), seq_len)
        feats = [extract_1662_from_frame(frames[i], holo) for i in idxs]
        arr = np.stack(feats, axis=0)  # (T,1662)
        lab_dir = out_dir / lab
        lab_dir.mkdir(parents=True, exist_ok=True)
        np.save(lab_dir / f"{vid}.npy", arr)
        written += 1
        per_label[lab] = per_label.get(lab, 0) + 1
    return written, skipped


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Prepare WLASL videos -> data/sign sequences (30x1662)"
    )
    ap.add_argument(
        "--src",
        required=True,
        help="Path to folder containing videos/ and WLASL_v0.3.json",
    )
    ap.add_argument("--out", default="data/sign", help="Output dataset directory")
    ap.add_argument(
        "--limit_classes",
        type=int,
        default=None,
        help="Keep only top-N frequent labels (optional)",
    )
    ap.add_argument(
        "--max_clips", type=int, default=None, help="Limit clips per label (optional)"
    )
    ap.add_argument("--seq_len", type=int, default=30)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    videos = src / "videos"
    meta = src / "WLASL_v0.3.json"
    if not videos.is_dir() or not meta.exists():
        raise SystemExit("Expected videos/ directory and WLASL_v0.3.json in --src")
    mapping = build_video_label_map(
        meta, split="train", limit_classes=args.limit_classes
    )
    out_dir = Path(args.out)
    written, skipped = process_videos(
        videos,
        mapping,
        out_dir,
        max_clips_per_label=args.max_clips,
        seq_len=args.seq_len,
    )
    print(f"Prepared sequences: written={written}, skipped={skipped}, out={out_dir}")


if __name__ == "__main__":
    main()
