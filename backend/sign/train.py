from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.utils.class_weight import compute_class_weight  # type: ignore
import tensorflow as tf  # type: ignore

from .common import ensure_dir


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    labels = sorted([d.name for d in Path(data_dir).iterdir() if d.is_dir()])
    X: List[np.ndarray] = []
    y: List[int] = []
    for li, lab in enumerate(labels):
        for p in sorted(Path(data_dir, lab).glob("*.npy")):
            arr = np.load(p)
            if arr.ndim != 2 or arr.shape[1] != 1662:
                continue
            # Ensure fixed length 30 (pad/trim)
            T = 30
            if arr.shape[0] < T:
                pad = np.zeros((T - arr.shape[0], 1662), dtype=np.float32)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > T:
                arr = arr[:T]
            X.append(arr.astype(np.float32))
            y.append(li)
    if not X:
        raise RuntimeError("No data found. Run sign/collect.py first.")
    Xn = np.stack(X, axis=0)  # (N, 30, 1662)
    yn = np.array(y, dtype=np.int64)
    return Xn, yn, labels


def build_model(n_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(30, 1662), name="seq")
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    ap = argparse.ArgumentParser(
        description="Train LSTM sign classifier and save to models/action.h5"
    )
    ap.add_argument(
        "--data",
        default="data/sign",
        help="Dataset dir (label subfolders with .npy sequences)",
    )
    ap.add_argument("--models", default="models", help="Output models dir")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    X, y, labels = load_dataset(args.data)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Optional class weights to handle imbalance
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

    model = build_model(n_classes=len(labels))

    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        Xtr,
        ytr,
        validation_data=(Xte, yte),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        verbose=1,
        callbacks=cb,
    )

    ensure_dir(args.models)
    h5_path = os.path.join(args.models, "action.h5")
    model.save(h5_path)
    with open(os.path.join(args.models, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    # Report test accuracy
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"Saved {h5_path}. Test acc: {acc:.3f}")


if __name__ == "__main__":
    main()
