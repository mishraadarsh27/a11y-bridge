from __future__ import annotations

import argparse
import json
import os
from collections import deque

import numpy as np  # type: ignore
import cv2  # type: ignore
import tensorflow as tf  # type: ignore

from .common import get_holistic, bgr_to_rgb, extract_1662_from_holistic, draw_hud


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live inference using trained Keras model (action.h5)"
    )
    ap.add_argument(
        "--models",
        default="models",
        help="Models dir containing action.h5 and labels.json",
    )
    ap.add_argument("--cam", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    h5_path = os.path.join(args.models, "action.h5")
    labels_path = os.path.join(args.models, "labels.json")
    if not os.path.exists(h5_path):
        raise RuntimeError(f"Model not found: {h5_path}. Train first.")
    if not os.path.exists(labels_path):
        raise RuntimeError(f"Labels not found: {labels_path}.")

    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    model = tf.keras.models.load_model(h5_path)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    holistic = get_holistic()
    buf: deque[np.ndarray] = deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = bgr_to_rgb(frame)
            feat = extract_1662_from_holistic(rgb, holistic)
            buf.append(feat)

            if len(buf) == 30:
                x = np.asarray(list(buf), dtype=np.float32)[None, ...]  # (1,30,1662)
                probs = model.predict(x, verbose=0)[0].astype(float)
                ci = int(np.argmax(probs))
                label = labels[ci] if ci < len(labels) else str(ci)
                score = float(probs[ci])
                draw_hud(frame, f"{label} ({score:.2f})")
            else:
                draw_hud(frame, f"Warming up {len(buf)}/30", color=(0, 200, 200))

            cv2.imshow("Live Inference - q to quit", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
