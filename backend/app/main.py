from __future__ import annotations

# ruff: noqa: E402

import json
import os
from typing import Any, Literal, Optional
from collections import deque

# Ensure project root is on sys.path when running as a script (e.g., "python app/main.py")
import sys
from pathlib import Path

_CURR_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURR_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from fastapi import FastAPI, WebSocket, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect
from translation.service import (
    TranslationRequest,
    TranslationResponse,
    get_translation_service,
)

# Optional, lazily-initialized processors for sign frames
HANDS = None
HOLISTIC = None
CLASSIFIER = None  # ONNX classifier session if loaded
TF_CLASSIFIER = None  # TensorFlow Keras model if loaded
CLASSIFIER_LABELS: Optional[list[str]] = None
# Sign-MNIST CNN (static image A–Y) integration
TF_SIGNMNIST = None  # TensorFlow Keras CNN for 28x28 grayscale
SIGNMNIST_LABELS: Optional[list[str]] = None
# Fast static-gesture classifier (per-frame 21-hand keypoints -> kNN)
STATIC_CLF = None  # scikit-learn classifier (e.g., KNN)
STATIC_LABELS: Optional[list[str]] = None

app = FastAPI(title="VaaniSetu (CommuniBridge) Backend", version="0.1.0")

# Allow local frontend dev server and configurable production origins via env
_DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
_ENV_ORIGINS = os.getenv("ALLOW_ORIGINS", "")
_ALLOW_ORIGINS = [
    o.strip() for o in _ENV_ORIGINS.split(",") if o.strip()
] or _DEFAULT_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TextPayload(BaseModel):
    text: str = Field(min_length=0, max_length=2000)


class SignFramePayload(BaseModel):
    # Data URL string: "data:image/jpeg;base64,..."
    image: str = Field(min_length=10)


class ClientMessage(BaseModel):
    type: Literal["health", "text", "stt_result", "tts_text", "sign_frame", "translate"]
    payload: Optional[dict[str, Any]] = None


class ServerMessage(BaseModel):
    type: Literal[
        "health",
        "text_echo",
        "stt_ack",
        "tts_ack",
        "sign_status",
        "sign_text",
        "translation",
        "error",
    ]
    payload: dict[str, Any]


@app.get("/health", response_model=HealthResponse)
def health():
    return JSONResponse(HealthResponse().model_dump())


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Return 204 No Content if no favicon exists
    from fastapi import Response

    return Response(status_code=204)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):

    await ws.accept()
    print("[WS] Connection accepted")
    # Simple output mode: ws://.../ws?mode=simple
    simple_mode = ws.query_params.get("mode") == "simple"  # type: ignore[attr-defined]
    # Select sign model: ws://.../ws?sign_model=signmnist to force Sign-MNIST CNN path
    sign_model = ws.query_params.get("sign_model")  # type: ignore[attr-defined]
    use_signmnist = sign_model == "signmnist"
    # Per-connection state (e.g., sliding window of keypoints)
    seq_buf: "deque[list[float]]" = deque(maxlen=30)
    # Temporal smoothing for predictions - store last N predictions with scores
    prediction_history: "deque[tuple[str, float]]" = deque(
        maxlen=15
    )  # Last 15 predictions
    last_stable_prediction: Optional[tuple[str, float]] = None
    prediction_stability_counter: int = 0
    try:
        while True:
            raw = await ws.receive_text()
            print(f"[WS] Received: {raw[:100]}...")
            # Back-compat: accept plain text -> echo
            try:
                print("[WS] Attempting to parse JSON...")
                msg_data = json.loads(raw)
                print(f"[WS] JSON parsed successfully: {msg_data}")
            except json.JSONDecodeError as e:
                print(f"[WS] JSON decode error: {e}")
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="text_echo", payload={"text": raw}
                        ).model_dump()
                    )
                )
                continue

            try:
                msg = ClientMessage(**msg_data)
                print(f"[WS] Parsed message type: {msg.type}")
            except Exception as e:  # pydantic validation error
                print(f"[WS] Validation error: {e}")
                err = ServerMessage(
                    type="error", payload={"message": f"invalid message: {e}"}
                )
                await ws.send_text(json.dumps(err.model_dump()))
                continue

            if msg.type == "health":
                print("[WS] Processing health check")
                response = json.dumps(
                    ServerMessage(
                        type="health", payload=HealthResponse().model_dump()
                    ).model_dump()
                )
                print(f"[WS] Sending response: {response}")
                await ws.send_text(response)
            elif msg.type == "text":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="text_echo", payload={"text": data.text}
                        ).model_dump()
                    )
                )
            elif msg.type == "stt_result":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="stt_ack", payload={"text": data.text}
                        ).model_dump()
                    )
                )
            elif msg.type == "stt_audio":
                # Transcribe base64 audio sent as data URL
                try:
                    import re
                    from base64 import b64decode
                    import tempfile
                    import os

                    payload = SttAudioPayload(**(msg.payload or {}))
                    m = re.match(r"^data:audio/[^;]+;base64,(.+)$", payload.audio)
                    if not m:
                        raise ValueError("invalid audio data URL")
                    audio_bytes = b64decode(m.group(1))
                    _ensure_stt_model(payload.model or "base")
                    ext = ".wav"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
                        tf.write(audio_bytes)
                        tmp_path = tf.name
                    try:
                        if STT_BACKEND == "faster-whisper":
                            segments_iter, info = STT_MODEL.transcribe(  # type: ignore[attr-defined]
                                tmp_path,
                                language=(
                                    None
                                    if (payload.lang or "auto") == "auto"
                                    else payload.lang
                                ),
                                beam_size=1,
                            )
                            text_parts = [s.text for s in segments_iter]
                            text = (" ".join(t.strip() for t in text_parts)).strip()
                        elif STT_BACKEND == "whisper":
                            result = STT_MODEL.transcribe(  # type: ignore[attr-defined]
                                tmp_path,
                                language=(
                                    None
                                    if (payload.lang or "auto") == "auto"
                                    else payload.lang
                                ),
                            )
                            text = (result.get("text") or "").strip()
                        else:
                            raise RuntimeError("STT backend not available")
                        await ws.send_text(
                            json.dumps(
                                ServerMessage(
                                    type="stt_ack", payload={"text": text}
                                ).model_dump()
                            )
                        )
                    finally:
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            ServerMessage(
                                type="error",
                                payload={"message": f"stt failed: {e}"},
                            ).model_dump()
                        )
                    )
            elif msg.type == "tts_text":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="tts_ack", payload={"text": data.text}
                        ).model_dump()
                    )
                )
            elif msg.type == "translate":
                # Handle real-time translation
                try:
                    from translation.service import get_translation_service

                    payload = msg.payload or {}
                    text = payload.get("text", "")
                    source_lang = payload.get("source_lang", "auto")
                    target_lang = payload.get("target_lang", "en")

                    service = get_translation_service()
                    result = await service.translate(text, source_lang, target_lang)

                    await ws.send_text(
                        json.dumps(
                            ServerMessage(
                                type="translation",
                                payload=result.model_dump(),
                            ).model_dump()
                        )
                    )
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            ServerMessage(
                                type="error",
                                payload={"message": f"translation failed: {e}"},
                            ).model_dump()
                        )
                    )
            elif msg.type == "sign_frame":
                # Lazy import heavy deps to avoid overhead during simple tests
                try:
                    import re
                    from base64 import b64decode
                    import numpy as np  # type: ignore
                    import cv2  # type: ignore
                    import math

                    # Lazy-init global processors once
                    global HANDS, HOLISTIC
                    import mediapipe as mp  # type: ignore

                    if HANDS is None:
                        HANDS = mp.solutions.hands.Hands(
                            static_image_mode=False,
                            max_num_hands=2,
                            min_detection_confidence=0.5,  # Increased from 0.2 for better accuracy
                            min_tracking_confidence=0.5,  # Increased from 0.2 for better accuracy
                        )
                    if HOLISTIC is None:
                        HOLISTIC = mp.solutions.holistic.Holistic(
                            static_image_mode=False,
                            model_complexity=1,
                            enable_segmentation=False,
                            refine_face_landmarks=False,
                            min_detection_confidence=0.5,  # Increased from 0.2 for better accuracy
                            min_tracking_confidence=0.5,  # Increased from 0.2 for better accuracy
                        )

                    data = SignFramePayload(**(msg.payload or {}))
                    # Extract base64 data from data URL
                    m = re.match(r"^data:image/[^;]+;base64,(.+)$", data.image)
                    if not m:
                        raise ValueError("invalid data URL")
                    img_bytes = b64decode(m.group(1))
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        raise ValueError("could not decode image")
                    h_, w_ = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb = np.ascontiguousarray(frame_rgb)
                    # Hint MediaPipe that we won't modify the frame to allow internal optimizations
                    try:
                        frame_rgb.flags.writeable = False  # type: ignore[attr-defined]
                    except Exception:
                        pass

                    def gesture_from_landmarks(lms) -> str:
                        try:
                            # Use normalized coordinates (0..1). Tips: 4,8,12,16,20; wrist: 0
                            wrist = lms.landmark[0]
                            tips = [lms.landmark[i] for i in (4, 8, 12, 16, 20)]
                            # Approximate hand scale by max distance between any two landmarks
                            xs = [p.x for p in lms.landmark]
                            ys = [p.y for p in lms.landmark]
                            scale = (
                                math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + 1e-6
                            )
                            dists = [
                                math.hypot(t.x - wrist.x, t.y - wrist.y) for t in tips
                            ]
                            avg = sum(dists) / len(dists)
                            r = avg / scale
                            if r > 0.55:
                                return "open_palm"
                            if r < 0.30:
                                return "fist"
                            return "unknown"
                        except Exception:
                            return "unknown"

                    num_hands = 0
                    error_msg: Optional[str] = None
                    hands_info: list[dict[str, Any]] = []
                    gestures: list[str] = []
                    first_lms = None
                    hand63: Optional[list[float]] = (
                        None  # 63-dim normalized features (21x3)
                    )
                    try:
                        res = HANDS.process(frame_rgb)  # type: ignore[attr-defined]
                        if getattr(res, "multi_hand_landmarks", None):
                            num_hands = len(res.multi_hand_landmarks)
                            # Handedness labels if available
                            labels = []
                            try:
                                for c in res.multi_handedness or []:  # type: ignore[attr-defined]
                                    item = c.classification[0]
                                    labels.append(
                                        {
                                            "label": item.label,
                                            "score": float(item.score),
                                        }
                                    )
                            except Exception:
                                pass
                            hands_info = labels
                            for idx, lms in enumerate(res.multi_hand_landmarks):
                                if first_lms is None:
                                    first_lms = lms
                                gestures.append(gesture_from_landmarks(lms))
                        else:
                            # Fallback to Holistic
                            hres = HOLISTIC.process(frame_rgb)  # type: ignore[attr-defined]
                            lh = getattr(hres, "left_hand_landmarks", None)
                            rh = getattr(hres, "right_hand_landmarks", None)
                            if lh is not None:
                                num_hands += 1
                                hands_info.append({"label": "Left", "score": None})
                                gestures.append(gesture_from_landmarks(lh))
                                if first_lms is None:
                                    first_lms = lh
                            if rh is not None:
                                num_hands += 1
                                hands_info.append({"label": "Right", "score": None})
                                gestures.append(gesture_from_landmarks(rh))
                                if first_lms is None:
                                    first_lms = rh
                    except Exception as inner_e:
                        error_msg = str(inner_e)

                    # Compute 63-dim normalized hand features if we have landmarks
                    if first_lms is not None:
                        try:
                            import numpy as _np  # type: ignore

                            xs = _np.array(
                                [lm.x for lm in first_lms.landmark], dtype=_np.float32
                            )
                            ys = _np.array(
                                [lm.y for lm in first_lms.landmark], dtype=_np.float32
                            )
                            zs = _np.array(
                                [lm.z for lm in first_lms.landmark], dtype=_np.float32
                            )
                            # Translate to wrist (idx 0)
                            xs -= xs[0]
                            ys -= ys[0]
                            zs -= zs[0]
                            # Scale by hand size (max range)
                            scale = float(
                                max(xs.max() - xs.min(), ys.max() - ys.min(), 1e-6)
                            )
                            xs /= scale
                            ys /= scale
                            zs /= max(scale, 1e-6)
                            feat = _np.stack([xs, ys, zs], axis=1).reshape(-1)
                            hand63 = feat.astype(_np.float32).tolist()
                        except Exception:
                            hand63 = None

                    # Build keypoint features for ML classification
                    sign_pred: Optional[dict[str, Any]] = None
                    try:
                        import numpy as np  # type: ignore
                        import os
                        import json as _json

                        # If client requested Sign-MNIST classifier, try that first (single-frame A–Y)
                        if use_signmnist:
                            global TF_SIGNMNIST, SIGNMNIST_LABELS
                            try:
                                if TF_SIGNMNIST is None:
                                    h5p = os.path.join("models", "signmnist_cnn.h5")
                                    if os.path.exists(h5p):
                                        import tensorflow as tf  # type: ignore

                                        TF_SIGNMNIST = tf.keras.models.load_model(h5p)
                                        try:
                                            with open(
                                                os.path.join(
                                                    "models", "labels_signmnist.json"
                                                ),
                                                "r",
                                                encoding="utf-8",
                                            ) as f:
                                                SIGNMNIST_LABELS = _json.load(f)
                                        except Exception:
                                            SIGNMNIST_LABELS = None
                            except Exception:
                                TF_SIGNMNIST = None

                            # If we have a model and landmarks to crop a hand ROI
                            if TF_SIGNMNIST is not None and first_lms is not None:
                                try:
                                    # Compute tight bbox around landmarks in pixel coords
                                    xs = [
                                        int(max(0, min(w_ - 1, lm.x * w_)))
                                        for lm in first_lms.landmark
                                    ]
                                    ys = [
                                        int(max(0, min(h_ - 1, lm.y * h_)))
                                        for lm in first_lms.landmark
                                    ]
                                    x0, x1 = max(0, min(xs)), min(w_ - 1, max(xs))
                                    y0, y1 = max(0, min(ys)), min(h_ - 1, max(ys))
                                    # Padding
                                    pad = int(0.2 * max(x1 - x0 + 1, y1 - y0 + 1))
                                    x0 = max(0, x0 - pad)
                                    y0 = max(0, y0 - pad)
                                    x1 = min(w_ - 1, x1 + pad)
                                    y1 = min(h_ - 1, y1 + pad)
                                    if x1 > x0 and y1 > y0:
                                        roi_bgr = frame_bgr[y0:y1, x0:x1]
                                        import cv2  # type: ignore

                                        roi_gray = cv2.cvtColor(
                                            roi_bgr, cv2.COLOR_BGR2GRAY
                                        )
                                        roi_28 = cv2.resize(
                                            roi_gray,
                                            (28, 28),
                                            interpolation=cv2.INTER_AREA,
                                        )
                                        x = (roi_28.astype(np.float32) / 255.0)[
                                            None, ..., None
                                        ]
                                        probs = TF_SIGNMNIST.predict(x, verbose=0)[0].astype(float).tolist()  # type: ignore[attr-defined]
                                        ci = int(
                                            max(
                                                range(len(probs)),
                                                key=lambda i: probs[i],
                                            )
                                        )
                                        label = (
                                            SIGNMNIST_LABELS[ci]
                                            if (
                                                SIGNMNIST_LABELS
                                                and ci < len(SIGNMNIST_LABELS)
                                            )
                                            else str(ci)
                                        )
                                        score = (
                                            float(probs[ci]) if ci < len(probs) else 0.0
                                        )
                                        # Only accept predictions with confidence >= 0.6
                                        if score >= 0.6:
                                            sign_pred = {"label": label, "score": score}
                                except Exception:
                                    pass

                        # If collecting samples for static classifier
                        try:
                            collect = (
                                (msg.payload or {}).get("collect")
                                if isinstance((msg.payload or {}), dict)
                                else None
                            )
                            if (
                                collect
                                and isinstance(collect, dict)
                                and hand63 is not None
                            ):
                                import numpy as _np  # type: ignore

                                lbl = str(collect.get("label", "")).strip()
                                if lbl:
                                    os.makedirs(
                                        os.path.join("data", "static"), exist_ok=True
                                    )
                                    fpath = os.path.join("data", "static", f"{lbl}.npy")
                                    if os.path.exists(fpath):
                                        try:
                                            arr = _np.load(fpath)
                                            arr = _np.vstack(
                                                [arr, _np.asarray(hand63)[None, :]]
                                            )
                                        except Exception:
                                            arr = _np.asarray(hand63)[None, :]
                                    else:
                                        arr = _np.asarray(hand63)[None, :]
                                    _np.save(fpath, arr)
                        except Exception:
                            pass

                        # If we have a trained static classifier, try that first
                        if sign_pred is None and hand63 is not None:
                            global STATIC_CLF, STATIC_LABELS
                            try:
                                if STATIC_CLF is not None and STATIC_LABELS:
                                    import numpy as _np  # type: ignore

                                    x = _np.asarray(hand63, dtype=_np.float32)[None, :]
                                    if hasattr(STATIC_CLF, "predict_proba"):
                                        probs = STATIC_CLF.predict_proba(x)[0]
                                        ci = int(probs.argmax())
                                        score = float(probs[ci])
                                    else:
                                        pred = STATIC_CLF.predict(x)[0]
                                        # try to synthesize a pseudo-score
                                        ci = int(pred)
                                        score = 1.0
                                    label = (
                                        STATIC_LABELS[ci]
                                        if (ci < len(STATIC_LABELS))
                                        else str(ci)
                                    )
                                    # only accept if confident
                                    if score >= 0.6:
                                        sign_pred = {"label": label, "score": score}
                            except Exception:
                                pass

                        # If no Sign-MNIST prediction made, fall back to sequence LSTM/ONNX pipeline
                        if sign_pred is None:
                            # Always extract 1662-dim features using Holistic to match the provided model
                            hres_feat = HOLISTIC.process(frame_rgb)  # type: ignore[attr-defined]

                            def _kp_or_zeros(arr, d):
                                try:
                                    return np.array(arr, dtype=np.float32)
                                except Exception:
                                    return np.zeros(d, dtype=np.float32)

                            pose = _kp_or_zeros(
                                [
                                    [
                                        res.x,
                                        res.y,
                                        res.z,
                                        getattr(res, "visibility", 0.0),
                                    ]
                                    for res in (
                                        getattr(
                                            hres_feat, "pose_landmarks", None
                                        ).landmark
                                        if getattr(hres_feat, "pose_landmarks", None)
                                        else []
                                    )
                                ],
                                33 * 4,
                            ).flatten()
                            lh = _kp_or_zeros(
                                [
                                    [res.x, res.y, res.z]
                                    for res in (
                                        getattr(
                                            hres_feat, "left_hand_landmarks", None
                                        ).landmark
                                        if getattr(
                                            hres_feat, "left_hand_landmarks", None
                                        )
                                        else []
                                    )
                                ],
                                21 * 3,
                            ).flatten()
                            rh = _kp_or_zeros(
                                [
                                    [res.x, res.y, res.z]
                                    for res in (
                                        getattr(
                                            hres_feat, "right_hand_landmarks", None
                                        ).landmark
                                        if getattr(
                                            hres_feat, "right_hand_landmarks", None
                                        )
                                        else []
                                    )
                                ],
                                21 * 3,
                            ).flatten()
                            face = _kp_or_zeros(
                                [
                                    [res.x, res.y, res.z]
                                    for res in (
                                        getattr(
                                            hres_feat, "face_landmarks", None
                                        ).landmark
                                        if getattr(hres_feat, "face_landmarks", None)
                                        else []
                                    )
                                ],
                                468 * 3,
                            ).flatten()
                            feat = np.concatenate([pose, lh, rh, face]).astype(
                                np.float32
                            )
                            # Ensure correct size by padding/truncating
                            if feat.shape[0] < 1662:
                                feat = np.pad(feat, (0, 1662 - feat.shape[0]))
                            elif feat.shape[0] > 1662:
                                feat = feat[:1662]
                            seq_buf.append(feat.tolist())

                            # Try ONNX model if available; else try TensorFlow Keras .h5; else heuristic
                            global CLASSIFIER, TF_CLASSIFIER, CLASSIFIER_LABELS
                            if CLASSIFIER is None and TF_CLASSIFIER is None:
                                # First try ONNX
                                try:
                                    import onnxruntime as ort  # type: ignore

                                    onnx_path = os.path.join(
                                        "models", "sign_r18lstm.onnx"
                                    )
                                    labels_path = os.path.join("models", "labels.json")
                                    if os.path.exists(onnx_path):
                                        CLASSIFIER = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
                                        try:
                                            with open(
                                                labels_path, "r", encoding="utf-8"
                                            ) as f:
                                                CLASSIFIER_LABELS = json.load(f)
                                        except Exception:
                                            CLASSIFIER_LABELS = None
                                except Exception:
                                    CLASSIFIER = None
                                # If ONNX not loaded, try TF .h5
                                if CLASSIFIER is None:
                                    try:
                                        import tensorflow as tf  # type: ignore

                                        h5_path = os.path.join("models", "action.h5")
                                        labels_path = os.path.join(
                                            "models", "labels.json"
                                        )
                                        if os.path.exists(h5_path):
                                            TF_CLASSIFIER = tf.keras.models.load_model(
                                                h5_path
                                            )
                                            try:
                                                with open(
                                                    labels_path, "r", encoding="utf-8"
                                                ) as f:
                                                    CLASSIFIER_LABELS = json.load(f)
                                            except Exception:
                                                CLASSIFIER_LABELS = None
                                    except Exception:
                                        TF_CLASSIFIER = None

                            if CLASSIFIER is not None and len(seq_buf) >= 8:
                                # ONNX inference expects (1, T, F)
                                inp_name = CLASSIFIER.get_inputs()[0].name  # type: ignore[attr-defined]
                                x = np.asarray(list(seq_buf), dtype=np.float32)[
                                    None, ...
                                ]  # (1, T, F)
                                out = CLASSIFIER.run(None, {inp_name: x})  # type: ignore[attr-defined]
                                logits = out[0][0]  # assume (1, C)
                                import math

                                exps = [math.exp(float(v)) for v in logits]
                                s = sum(exps) + 1e-9
                                probs = [v / s for v in exps]
                                ci = int(max(range(len(probs)), key=lambda i: probs[i]))
                                score = float(probs[ci])
                                label = (
                                    CLASSIFIER_LABELS[ci]
                                    if (
                                        CLASSIFIER_LABELS
                                        and ci < len(CLASSIFIER_LABELS)
                                    )
                                    else str(ci)
                                )
                                # Only accept predictions with confidence >= 0.65
                                if score >= 0.65 and label != "unknown":
                                    sign_pred = {"label": label, "score": score}
                                    # Add to prediction history for temporal smoothing
                                    prediction_history.append((label, score))
                            elif TF_CLASSIFIER is not None and len(seq_buf) >= 30:
                                # TF model expects (None, 30, 1662)
                                x = np.asarray(list(seq_buf)[-30:], dtype=np.float32)[
                                    None, ...
                                ]  # (1, 30, 1662)
                                probs = TF_CLASSIFIER.predict(x, verbose=0)[0].astype(float).tolist()  # type: ignore[attr-defined]
                                ci = int(max(range(len(probs)), key=lambda i: probs[i]))
                                score = float(probs[ci]) if ci < len(probs) else 0.0
                                label = (
                                    CLASSIFIER_LABELS[ci]
                                    if (
                                        CLASSIFIER_LABELS
                                        and ci < len(CLASSIFIER_LABELS)
                                    )
                                    else str(ci)
                                )
                                # Only accept predictions with confidence >= 0.65
                                if score >= 0.65 and label != "unknown":
                                    sign_pred = {"label": label, "score": score}
                                    # Add to prediction history for temporal smoothing
                                    prediction_history.append((label, score))
                            else:
                                # Heuristic mapping using current frame gesture
                                g = gestures[0] if gestures else "unknown"
                                if g == "open_palm":
                                    sign_pred = {"label": "5", "score": 0.7}
                                elif g == "fist":
                                    sign_pred = {"label": "0", "score": 0.7}
                                else:
                                    sign_pred = {"label": "unknown", "score": 0.0}
                    except Exception as _:
                        pass

                    # Apply temporal smoothing/filtering before sending
                    if sign_pred is not None and len(prediction_history) >= 5:
                        # Use majority vote from recent predictions for stability
                        from collections import Counter

                        recent_labels = [p[0] for p in list(prediction_history)[-5:]]
                        label_counts = Counter(recent_labels)
                        most_common_label, count = label_counts.most_common(1)[0]

                        # Only update if same label appears at least 3 times in last 5 predictions
                        if count >= 3:
                            # Calculate average score for this label
                            label_scores = [
                                p[1]
                                for p in prediction_history
                                if p[0] == most_common_label
                            ]
                            avg_score = (
                                sum(label_scores) / len(label_scores)
                                if label_scores
                                else sign_pred["score"]
                            )

                            # Update prediction with smoothed result
                            if (
                                most_common_label != last_stable_prediction[0]
                                if last_stable_prediction
                                else True
                            ):
                                prediction_stability_counter = 0
                                last_stable_prediction = (most_common_label, avg_score)
                                sign_pred = {
                                    "label": most_common_label,
                                    "score": avg_score,
                                }
                            else:
                                prediction_stability_counter += 1
                                # Only update if stable for at least 2 frames
                                if prediction_stability_counter >= 2:
                                    sign_pred = {
                                        "label": most_common_label,
                                        "score": avg_score,
                                    }
                                else:
                                    # Keep using last stable prediction
                                    if last_stable_prediction:
                                        sign_pred = {
                                            "label": last_stable_prediction[0],
                                            "score": last_stable_prediction[1],
                                        }
                        else:
                            # Not stable enough, use last stable prediction if available
                            if last_stable_prediction:
                                sign_pred = {
                                    "label": last_stable_prediction[0],
                                    "score": last_stable_prediction[1],
                                }
                            else:
                                sign_pred = None
                    elif sign_pred is not None and len(prediction_history) < 5:
                        # Not enough history yet, but still use it if confident enough
                        if sign_pred["score"] >= 0.7:
                            last_stable_prediction = (
                                sign_pred["label"],
                                sign_pred["score"],
                            )
                        else:
                            sign_pred = None

                    # Send minimal or detailed payload depending on mode
                    if simple_mode and sign_pred is not None:
                        label = str(sign_pred.get("label", "unknown"))
                        score = float(sign_pred.get("score", 0.0))
                        # Only send confident, non-unknown predictions (increased threshold)
                        if label != "unknown" and score >= 0.6:
                            await ws.send_text(
                                json.dumps(
                                    ServerMessage(
                                        type="sign_text",
                                        payload={"text": label, "score": score},
                                    ).model_dump()
                                )
                            )
                    else:
                        payload: dict[str, Any] = {
                            "hand_detected": num_hands > 0,
                            "num_hands": num_hands,
                            "hands": hands_info,
                            "gestures": gestures,
                            "frame": {"w": w_, "h": h_},
                        }
                        if sign_pred is not None:
                            payload["sign"] = sign_pred
                        if error_msg:
                            payload["error"] = error_msg

                        await ws.send_text(
                            json.dumps(
                                ServerMessage(
                                    type="sign_status",
                                    payload=payload,
                                ).model_dump()
                            )
                        )
                except Exception as e:
                    await ws.send_text(
                        json.dumps(
                            ServerMessage(
                                type="error",
                                payload={"message": f"sign processing failed: {e}"},
                            ).model_dump()
                        )
                    )
    except WebSocketDisconnect:
        # Client disconnected
        print("[WS] Client disconnected")
        pass
    except Exception as e:
        print(f"[WS] Unexpected exception: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        try:
            await ws.close()
        except Exception:
            pass


# ---- Sign labels management ----
class LabelsPayload(BaseModel):
    labels: list[str]
    data_dir: str = "data/sign"


@app.get("/sign/labels")
def get_sign_labels(data_dir: str = "data/sign"):
    labels_file = os.path.join(data_dir, "labels.txt")
    labels: list[str] = []
    if os.path.exists(labels_file):
        try:
            with open(labels_file, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
        except Exception:
            labels = []
    if not labels and os.path.isdir(data_dir):
        # infer from subdirs if any
        try:
            labels = [
                d
                for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))
            ]
            labels.sort()
        except Exception:
            labels = []
    return {"labels": labels}


@app.post("/sign/labels")
def set_sign_labels(payload: LabelsPayload):
    os.makedirs(payload.data_dir, exist_ok=True)
    # write labels.txt
    labels_file = os.path.join(payload.data_dir, "labels.txt")
    with open(labels_file, "w", encoding="utf-8") as f:
        for lab in payload.labels:
            clean_label = lab.strip()
            if clean_label:
                f.write(clean_label + "\n")
                os.makedirs(os.path.join(payload.data_dir, clean_label), exist_ok=True)
    return {"ok": True, "labels": payload.labels}


# ---- Sign training integration endpoints ----

TRAINING_RUNNING: bool = False
TRAINING_MSG: Optional[str] = None
TRAINING_ACC: Optional[float] = None


class TrainRequest(BaseModel):
    data_dir: str = "data/sign"
    models_dir: str = "models"
    epochs: int = 25
    batch: int = 32


class TrainStatus(BaseModel):
    running: bool
    message: Optional[str] = None
    acc: Optional[float] = None


@app.post("/sign/train", response_model=TrainStatus)
def sign_train(req: TrainRequest, bg: BackgroundTasks):
    global TRAINING_RUNNING, TRAINING_MSG, TRAINING_ACC
    if TRAINING_RUNNING:
        return TrainStatus(running=True, message="already running")
    TRAINING_RUNNING = True
    TRAINING_MSG = "starting"

    def _job():
        global TRAINING_RUNNING, TRAINING_MSG, TRAINING_ACC, TF_CLASSIFIER, CLASSIFIER_LABELS
        try:
            TRAINING_MSG = "loading data"
            import sign.train as ST  # type: ignore
            import tensorflow as tf  # type: ignore
            import json as _json
            import os as _os
            import numpy as np  # type: ignore
            from sklearn.model_selection import train_test_split  # type: ignore
            from sklearn.utils.class_weight import compute_class_weight  # type: ignore
            from sign.common import ensure_dir as _ensure_dir  # type: ignore

            X, y, labels = ST.load_dataset(req.data_dir)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=42
            )
            classes = np.unique(y)
            class_weights = compute_class_weight(
                class_weight="balanced", classes=classes, y=y
            )
            cw = {int(c): float(w) for c, w in zip(classes, class_weights)}

            TRAINING_MSG = "building model"
            model = ST.build_model(n_classes=len(labels))

            TRAINING_MSG = "training"
            cb = [
                tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=0),
                tf.keras.callbacks.EarlyStopping(
                    patience=6, restore_best_weights=True, verbose=0
                ),
            ]
            model.fit(
                Xtr,
                ytr,
                validation_data=(Xte, yte),
                epochs=req.epochs,
                batch_size=req.batch,
                class_weight=cw,
                verbose=0,
                callbacks=cb,
            )

            _ensure_dir(req.models_dir)
            h5_path = _os.path.join(req.models_dir, "action.h5")
            model.save(h5_path)
            with open(
                _os.path.join(req.models_dir, "labels.json"), "w", encoding="utf-8"
            ) as f:
                _json.dump(labels, f, ensure_ascii=False)

            loss, acc = model.evaluate(Xte, yte, verbose=0)
            TRAINING_ACC = float(acc)
            TRAINING_MSG = f"done acc={acc:.3f}"

            # hot-reload model for WS inference
            TF_CLASSIFIER = tf.keras.models.load_model(h5_path)
            CLASSIFIER_LABELS = labels
        except Exception as e:  # noqa: BLE001
            TRAINING_MSG = f"failed: {e}"
        finally:
            TRAINING_RUNNING = False

    bg.add_task(_job)
    return TrainStatus(running=True, message="started")


@app.get("/sign/train_status", response_model=TrainStatus)
def sign_train_status():
    return TrainStatus(running=TRAINING_RUNNING, message=TRAINING_MSG, acc=TRAINING_ACC)


@app.post("/sign/reload_model", response_model=TrainStatus)
def sign_reload_model(models_dir: str = "models"):
    global TF_CLASSIFIER, CLASSIFIER_LABELS, TRAINING_MSG
    try:
        import tensorflow as tf  # type: ignore
        import os
        import json  # noqa: ICN001

        h5_path = os.path.join(models_dir, "action.h5")
        labels_path = os.path.join(models_dir, "labels.json")
        if not os.path.exists(h5_path):
            raise FileNotFoundError("action.h5 not found")
        TF_CLASSIFIER = tf.keras.models.load_model(h5_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            CLASSIFIER_LABELS = json.load(f)
        TRAINING_MSG = "model reloaded"
        return TrainStatus(running=False, message="reloaded")
    except Exception as e:  # noqa: BLE001
        return TrainStatus(running=False, message=f"reload failed: {e}")


# ---- Train Sign-MNIST CNN from CSVs ----
class TrainSignMNISTRequest(BaseModel):
    data_dir: str = (
        "data/signmnist"  # must contain sign_mnist_train.csv and sign_mnist_test.csv
    )
    models_dir: str = "models"
    epochs: int = 10
    batch: int = 128


@app.post("/sign/train_signmnist")
def train_signmnist(req: TrainSignMNISTRequest):
    global TF_SIGNMNIST, SIGNMNIST_LABELS
    try:
        import os
        import json  # noqa: ICN001
        import numpy as np  # type: ignore
        import tensorflow as tf  # type: ignore

        tr_csv = os.path.join(req.data_dir, "sign_mnist_train.csv")
        te_csv = os.path.join(req.data_dir, "sign_mnist_test.csv")
        if not (os.path.exists(tr_csv) and os.path.exists(te_csv)):
            raise FileNotFoundError(
                "sign_mnist_train.csv or sign_mnist_test.csv missing"
            )

        def load_csv(p: str):
            arr = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.int32)
            y = arr[:, 0].astype(np.int32)
            X = arr[:, 1:].astype(np.float32) / 255.0
            X = X.reshape((-1, 28, 28, 1))
            return X, y

        Xtr, ytr = load_csv(tr_csv)
        Xte, yte = load_csv(te_csv)
        n_classes = int(max(ytr.max(), yte.max())) + 1

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, 3, activation="relu", input_shape=(28, 28, 1)
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            Xtr,
            ytr,
            validation_split=0.1,
            epochs=req.epochs,
            batch_size=req.batch,
            verbose=0,
        )
        loss, acc = model.evaluate(Xte, yte, verbose=0)

        os.makedirs(req.models_dir, exist_ok=True)
        out_path = os.path.join(req.models_dir, "signmnist_cnn.h5")
        model.save(out_path)
        labels = [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
        ]
        with open(
            os.path.join(req.models_dir, "labels_signmnist.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(labels, f, ensure_ascii=False)
        TF_SIGNMNIST = tf.keras.models.load_model(out_path)
        SIGNMNIST_LABELS = labels
        return {"ok": True, "acc": float(acc)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


# ---- Static gesture classifier (kNN) ----
class StaticTrainRequest(BaseModel):
    data_dir: str = "data/static"
    models_dir: str = "models"
    n_neighbors: int = 5


@app.get("/sign/static_status")
def static_status(data_dir: str = "data/static"):
    import os
    import glob  # noqa: ICN001

    stats: dict[str, int] = {}
    if os.path.isdir(data_dir):
        for p in glob.glob(os.path.join(data_dir, "*.npy")):
            try:
                import numpy as _np  # type: ignore

                arr = _np.load(p)
                stats[os.path.splitext(os.path.basename(p))[0]] = int(arr.shape[0])
            except Exception:
                stats[os.path.splitext(os.path.basename(p))[0]] = 0
    return {"labels": stats}


@app.post("/sign/static_train")
def static_train(req: StaticTrainRequest):
    global STATIC_CLF, STATIC_LABELS
    try:
        import os
        import glob
        import json  # noqa: ICN001
        import numpy as _np  # type: ignore
        from sklearn.neighbors import KNeighborsClassifier  # type: ignore
        from joblib import dump  # type: ignore

        X_list: list[_np.ndarray] = []
        y_list: list[int] = []
        labels: list[str] = []
        if not os.path.isdir(req.data_dir):
            raise FileNotFoundError(f"data dir not found: {req.data_dir}")
        for i, f in enumerate(sorted(glob.glob(os.path.join(req.data_dir, "*.npy")))):
            arr = _np.load(f)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.size == 0:
                continue
            X_list.append(arr.astype(_np.float32))
            y_list.extend([i] * arr.shape[0])
            labels.append(os.path.splitext(os.path.basename(f))[0])
        if not X_list:
            raise RuntimeError("no samples to train")
        X = _np.vstack(X_list)
        y = _np.asarray(y_list, dtype=_np.int64)
        clf = KNeighborsClassifier(n_neighbors=max(1, req.n_neighbors))
        clf.fit(X, y)
        os.makedirs(req.models_dir, exist_ok=True)
        dump(clf, os.path.join(req.models_dir, "static_knn.pkl"))
        with open(
            os.path.join(req.models_dir, "static_labels.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(labels, f, ensure_ascii=False)
        STATIC_CLF = clf
        STATIC_LABELS = labels
        return {"ok": True, "labels": labels, "samples": int(X.shape[0])}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}


@app.post("/sign/reset_models")
def sign_reset_models():
    global TF_CLASSIFIER, CLASSIFIER_LABELS, TF_SIGNMNIST, SIGNMNIST_LABELS, STATIC_CLF, STATIC_LABELS
    TF_CLASSIFIER = None
    CLASSIFIER_LABELS = None
    TF_SIGNMNIST = None
    SIGNMNIST_LABELS = None
    STATIC_CLF = None
    STATIC_LABELS = None
    return {"ok": True}


# ---- Speech-to-Text (STT) ----
STT_MODEL = None  # cached model instance
STT_BACKEND: Optional[str] = None  # "faster-whisper" or "whisper"
STT_MODEL_NAME: Optional[str] = None


def _pick_device_for_stt() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_stt_model(model_name: str = "base") -> None:
    global STT_MODEL, STT_BACKEND, STT_MODEL_NAME
    if STT_MODEL is not None and STT_MODEL_NAME == model_name:
        return
    # Try faster-whisper first (fast, memory-efficient)
    try:
        from faster_whisper import WhisperModel  # type: ignore

        device = _pick_device_for_stt()
        compute_type = "float16" if device == "cuda" else "int8"
        STT_MODEL = WhisperModel(model_name, device=device, compute_type=compute_type)
        STT_BACKEND = "faster-whisper"
        STT_MODEL_NAME = model_name
        return
    except Exception:
        STT_MODEL = None
    # Fallback to openai-whisper
    try:
        import whisper  # type: ignore

        STT_MODEL = whisper.load_model(model_name)
        STT_BACKEND = "whisper"
        STT_MODEL_NAME = model_name
    except Exception as e:  # noqa: BLE001
        STT_MODEL = None
        STT_BACKEND = None
        STT_MODEL_NAME = None
        raise RuntimeError(f"Failed to load STT model '{model_name}': {e}")


@app.post("/stt")
async def stt_endpoint(
    file: UploadFile = File(...), lang: str = "auto", model: str = "base"
):
    import tempfile
    import os

    data = await file.read()
    _ensure_stt_model(model)
    # Write to a temp file to support both engines
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tf:
        tf.write(data)
        tmp_path = tf.name
    try:
        if STT_BACKEND == "faster-whisper":
            # faster-whisper API
            segments_iter, info = STT_MODEL.transcribe(  # type: ignore[attr-defined]
                tmp_path,
                language=None if lang == "auto" else lang,
                beam_size=1,
            )
            text_parts = []
            segs_out = []
            for s in segments_iter:
                text_parts.append(s.text)
                segs_out.append({"start": s.start, "end": s.end, "text": s.text})
            text = (" ".join(t.strip() for t in text_parts)).strip()
            return {
                "text": text,
                "language": getattr(info, "language", lang),
                "duration": getattr(info, "duration", None),
                "segments": segs_out,
                "backend": STT_BACKEND,
                "model": STT_MODEL_NAME,
            }
        elif STT_BACKEND == "whisper":

            result = STT_MODEL.transcribe(  # type: ignore[attr-defined]
                tmp_path,
                language=None if lang == "auto" else lang,
            )
            return {
                "text": (result.get("text") or "").strip(),
                "language": result.get("language", lang),
                "segments": result.get("segments", []),
                "backend": STT_BACKEND,
                "model": STT_MODEL_NAME,
            }
        else:
            raise RuntimeError("STT backend not available")
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


class SttAudioPayload(BaseModel):
    audio: str  # data URL: data:audio/xxx;base64,...
    lang: Optional[str] = "auto"
    model: Optional[str] = "base"


# ---- Translation API using LLM ----


@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(req: TranslationRequest):
    """
    Translate text using LLM (Ollama).

    Example:
        POST /translate
        {
            "text": "Hello, how are you?",
            "source_lang": "en",
            "target_lang": "hi"
        }
    """
    service = get_translation_service()
    return await service.translate(
        text=req.text,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
        model=req.model,
    )


class BatchTranslateRequest(BaseModel):
    texts: list[str]
    source_lang: str = "auto"
    target_lang: str = "en"


@app.post("/translate/batch")
async def batch_translate_endpoint(req: BatchTranslateRequest):
    """
    Translate multiple texts at once.

    Example:
        POST /translate/batch
        {
            "texts": ["Hello", "Goodbye"],
            "source_lang": "en",
            "target_lang": "hi"
        }
    """
    service = get_translation_service()
    results = await service.batch_translate(
        texts=req.texts,
        source_lang=req.source_lang,
        target_lang=req.target_lang,
    )
    return {"translations": results}


class DetectLanguageRequest(BaseModel):
    text: str


@app.post("/translate/detect")
async def detect_language_endpoint(req: DetectLanguageRequest):
    """
    Detect language of text.

    Example:
        POST /translate/detect
        {"text": "Namaste"}
    """
    service = get_translation_service()
    lang_code = await service.detect_language(req.text)
    return {"text": req.text, "detected_language": lang_code}


@app.get("/translate/languages")
async def supported_languages():
    """
    Get list of supported languages.
    """
    languages = [
        {"code": "en", "name": "English"},
        {"code": "hi", "name": "Hindi"},
        {"code": "ur", "name": "Urdu"},
        {"code": "pa", "name": "Punjabi"},
        {"code": "bn", "name": "Bengali"},
        {"code": "ta", "name": "Tamil"},
        {"code": "te", "name": "Telugu"},
        {"code": "mr", "name": "Marathi"},
        {"code": "gu", "name": "Gujarati"},
        {"code": "kn", "name": "Kannada"},
        {"code": "ml", "name": "Malayalam"},
        {"code": "es", "name": "Spanish"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "ar", "name": "Arabic"},
        {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
        {"code": "ko", "name": "Korean"},
    ]
    return {"languages": languages}
