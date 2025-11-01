from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore


LETTERS_24 = [
    ch for ch in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") if ch not in ("J", "Z")
]  # Sign-MNIST excludes J and Z due to motion


def load_signmnist_csv(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Loads Sign-MNIST CSV (label,pixel1..pixel784) -> (N,28,28,1) float32 in [0,1], labels int64.
    Uses pandas if available, otherwise falls back to Python csv module for lower memory usage.
    """
    try:
        import pandas as pd  # type: ignore

        df = pd.read_csv(csv_path)
        y = df["label"].to_numpy(dtype=np.int64)
        X = df.drop(columns=["label"]).to_numpy(dtype=np.uint8)
        X = X.reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        return X, y
    except Exception:
        # Fallback reader (slower, but avoids pandas dependency)
        import csv

        ys: list[int] = []
        xs: list[np.ndarray] = []
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if not header or header[0] != "label":
                raise RuntimeError("Invalid CSV: expected 'label' as first column")
            for row in reader:
                ys.append(int(row[0]))
                pixels = np.array(row[1:], dtype=np.uint8)
                xs.append(pixels)
        X = np.stack(xs, axis=0).reshape((-1, 28, 28, 1)).astype(np.float32) / 255.0
        y = np.array(ys, dtype=np.int64)
        return X, y


def build_cnn(n_classes: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation="relu")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CNN on Sign-MNIST CSV dataset")
    ap.add_argument(
        "--data_dir",
        required=True,
        help="Path containing sign_mnist_train.csv and sign_mnist_test.csv",
    )
    ap.add_argument("--models", default="models", help="Output models dir")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()

    train_csv = os.path.join(args.data_dir, "sign_mnist_train.csv")
    test_csv = os.path.join(args.data_dir, "sign_mnist_test.csv")
    if not os.path.exists(train_csv) or not os.path.exists(test_csv):
        raise FileNotFoundError(
            "Expected sign_mnist_train.csv and sign_mnist_test.csv in --data_dir"
        )

    Xtr, ytr = load_signmnist_csv(train_csv)
    Xte, yte = load_signmnist_csv(test_csv)

    # Derive class ids and label names (24 letters excluding J,Z)
    classes = sorted(list({int(v) for v in ytr} | {int(v) for v in yte}))
    n_classes = len(classes)
    if n_classes != 24:
        raise RuntimeError(
            f"Expected 24 classes, found {n_classes} -> unexpected Sign-MNIST variant"
        )

    # Map labels to 0..C-1 if needed (some dumps already are 0..23)
    id_to_idx = {cid: i for i, cid in enumerate(classes)}
    ytr_idx = np.array([id_to_idx[int(v)] for v in ytr], dtype=np.int64)
    yte_idx = np.array([id_to_idx[int(v)] for v in yte], dtype=np.int64)

    # Standard letter names in order 0..23
    labels = LETTERS_24

    model = build_cnn(n_classes=n_classes)

    cb = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=1
        ),
    ]

    model.fit(
        Xtr,
        ytr_idx,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
        callbacks=cb,
    )

    loss, acc = model.evaluate(Xte, yte_idx, verbose=0)
    os.makedirs(args.models, exist_ok=True)
    h5_path = os.path.join(args.models, "signmnist_cnn.h5")
    labels_path = os.path.join(args.models, "labels_signmnist.json")
    model.save(h5_path)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Saved {h5_path}. Test acc: {acc:.3f}")


if __name__ == "__main__":
    main()
