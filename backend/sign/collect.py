from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np  # type: ignore
import cv2  # type: ignore

from .common import (
    CaptureConfig,
    ensure_dir,
    get_holistic,
    bgr_to_rgb,
    extract_1662_from_holistic,
    draw_hud,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect sign language sequences using MediaPipe Holistic (saves .npy sequences)"
    )
    p.add_argument(
        "--labels",
        nargs="+",
        required=False,
        help="Class labels, e.g., hello thanks yes no (if omitted, read from --labels-file)",
    )
    p.add_argument(
        "--labels-file",
        default="data/sign/labels.txt",
        help="Path to labels.txt (one label per line) used when --labels not provided",
    )
    p.add_argument("--seq-len", type=int, default=30, help="Frames per sequence")
    p.add_argument(
        "--per-class", type=int, default=30, help="Number of sequences per label"
    )
    p.add_argument("--out", default="data/sign", help="Output dir to store sequences")
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    return p.parse_args()


def main():
    args = parse_args()
    labels: List[str]
    if args.labels and len(args.labels) > 0:
        labels = [label.strip() for label in args.labels if label.strip()]
    else:
        # Fallback to labels file
        lf = args.labels_file
        if not os.path.exists(lf):
            raise FileNotFoundError(
                f"No labels provided and labels file not found: {lf}. Pass --labels or create labels file."
            )
        with open(lf, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if not labels:
            raise RuntimeError("Labels file is empty.")
    cfg = CaptureConfig(
        labels=tuple(labels),
        seq_len=args.seq_len,
        n_seqs_per_label=args.per_class,
        out_dir=args.out,
        cam_index=args.cam,
    )

    # Prepare dirs
    for lab in cfg.labels:
        ensure_dir(os.path.join(cfg.out_dir, lab))

    cap = cv2.VideoCapture(cfg.cam_index)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    holistic = get_holistic()

    print("Controls: press SPACE to start/stop recording a sequence, q to quit")
    print(f"Labels: {cfg.labels}")

    cur_label_idx = 0
    rec = False
    frames: List[np.ndarray] = []
    seq_count_for_label = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_flipped = cv2.flip(frame, 1)
            hud_text = f"Label: {cfg.labels[cur_label_idx]} ({seq_count_for_label}/{cfg.n_seqs_per_label})"

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == 32:  # SPACE
                rec = not rec
                if rec:
                    frames = []
                    hud_text += " | REC"
                else:
                    hud_text += " | PAUSE"
            elif key == ord("n"):
                # Next label
                cur_label_idx = (cur_label_idx + 1) % len(cfg.labels)
                seq_count_for_label = 0
                rec = False
                frames = []
            elif key == ord("p"):
                # Prev label
                cur_label_idx = (cur_label_idx - 1) % len(cfg.labels)
                seq_count_for_label = 0
                rec = False
                frames = []

            if rec and len(frames) < cfg.seq_len:
                rgb = bgr_to_rgb(frame_flipped)
                feat = extract_1662_from_holistic(rgb, holistic)  # (1662,)
                frames.append(feat)
                hud_text += f" | {len(frames)}/{cfg.seq_len}"

            if rec and len(frames) >= cfg.seq_len:
                # Save sequence
                out_dir = os.path.join(cfg.out_dir, cfg.labels[cur_label_idx])
                ensure_dir(out_dir)
                existing = list(Path(out_dir).glob("*.npy"))
                idx = (
                    0
                    if not existing
                    else max([int(p.stem) for p in existing if p.stem.isdigit()] + [0])
                    + 1
                )
                out_path = os.path.join(out_dir, f"{idx}.npy")
                np.save(out_path, np.array(frames, dtype=np.float32))  # (T,1662)
                seq_count_for_label += 1
                print(f"Saved {out_path}")
                rec = False
                frames = []
                if seq_count_for_label >= cfg.n_seqs_per_label:
                    # auto move to next label
                    cur_label_idx = (cur_label_idx + 1) % len(cfg.labels)
                    seq_count_for_label = 0

            draw_hud(frame_flipped, hud_text)
            cv2.imshow(
                "Collect - SPACE to REC, n/p switch label, q quit", frame_flipped
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
