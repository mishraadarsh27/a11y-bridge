from __future__ import annotations

import sys
from unittest.mock import MagicMock

# 🛡️ Local broken TensorFlow se bachav (Render par TF hai hi nahi, harmless)
sys.modules.setdefault('tensorflow', MagicMock())
sys.modules.setdefault('tensorflow.tools', MagicMock())
sys.modules.setdefault('tensorflow.tools.docs', MagicMock())

import json
import os
import math
import logging
from typing import Any, Literal, Optional
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from translation.service import (
    TranslationRequest,
    TranslationResponse,
    get_translation_service,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CommuniBridge Backend", version="1.0.0")

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
    image: str = Field(min_length=10)

class ClientMessage(BaseModel):
    type: Literal["health", "text", "stt_result", "tts_text", "sign_frame", "translate"]
    payload: Optional[dict[str, Any]] = None

class ServerMessage(BaseModel):
    type: Literal["health", "text_echo", "stt_ack", "tts_ack", "sign_status", "sign_text", "translation", "error"]
    payload: dict[str, Any]

@app.get("/health", response_model=HealthResponse)
def health():
    return JSONResponse(HealthResponse().model_dump())

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

HANDS = None

def get_gesture(landmarks) -> tuple[str, float]:
    """Rule-based gesture recognition using MediaPipe Hand Landmarks."""
    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]
    
    extended = []
    for tip, pip in zip(tips, pips):
        if tip == 4:
            dist_tip = math.hypot(landmarks[tip].x - landmarks[0].x, landmarks[tip].y - landmarks[0].y)
            dist_pip = math.hypot(landmarks[pip].x - landmarks[0].x, landmarks[pip].y - landmarks[0].y)
            if dist_tip > dist_pip * 1.1: extended.append(1)
            else: extended.append(0)
        else:
            if landmarks[tip].y < landmarks[pip].y: extended.append(1)
            else: extended.append(0)
                
    if extended == [0, 0, 0, 0, 0]: return "Fist", 0.9
    elif extended == [1, 1, 1, 1, 1]: return "Hello", 0.9
    elif extended == [0, 1, 0, 0, 0]: return "You", 0.9
    elif extended == [0, 1, 1, 0, 0]: return "Peace", 0.9
    elif extended == [1, 0, 0, 0, 0]: return "Thumbs Up", 0.9
    elif sum(extended) >= 4: return "Open Hand", 0.7
    else: return "Unknown", 0.5

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("[WS] Connection accepted")
    
    global HANDS
    if HANDS is None:
        try:
            from mediapipe.python.solutions.hands import Hands
            HANDS = Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_complexity=0,
            )
            logger.info("[WS] MediaPipe Hands loaded successfully!")
        except Exception as e:
            logger.error(f"MediaPipe load failed: {type(e).__name__}: {e}")
            await ws.close()
            return

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg_data = json.loads(raw)
                msg = ClientMessage(**msg_data)
            except Exception as e:
                await ws.send_text(json.dumps(ServerMessage(type="error", payload={"message": f"invalid msg: {e}"}).model_dump()))
                continue

            if msg.type == "health":
                await ws.send_text(json.dumps(ServerMessage(type="health", payload=HealthResponse().model_dump()).model_dump()))
            
            elif msg.type == "text":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(json.dumps(ServerMessage(type="text_echo", payload={"text": data.text}).model_dump()))
                
            elif msg.type == "stt_result":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(json.dumps(ServerMessage(type="stt_ack", payload={"text": data.text}).model_dump()))
                
            elif msg.type == "tts_text":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(json.dumps(ServerMessage(type="tts_ack", payload={"text": data.text}).model_dump()))
                
            elif msg.type == "translate":
                try:
                    payload = msg.payload or {}
                    service = get_translation_service()
                    result = await service.translate(
                        payload.get("text", ""), 
                        payload.get("source_lang", "auto"), 
                        payload.get("target_lang", "en")
                    )
                    await ws.send_text(json.dumps(ServerMessage(type="translation", payload=result.model_dump()).model_dump()))
                except Exception as e:
                    await ws.send_text(json.dumps(ServerMessage(type="error", payload={"message": f"translation failed: {e}"}).model_dump()))
                    
            elif msg.type == "sign_frame":
                try:
                    import re, cv2
                    from base64 import b64decode
                    import numpy as np

                    data = SignFramePayload(**(msg.payload or {}))
                    m = re.match(r"^data:image/[^;]+;base64,(.+)$", data.image)
                    if not m: raise ValueError("invalid data URL")
                    
                    img_bytes = b64decode(m.group(1))
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame_bgr is None: raise ValueError("could not decode image")
                        
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    
                    res = HANDS.process(frame_rgb)
                    num_hands, gesture, score = 0, "No Hand", 0.0
                    
                    if res.multi_hand_landmarks:
                        num_hands = len(res.multi_hand_landmarks)
                        gesture, score = get_gesture(res.multi_hand_landmarks[0].landmark)
                        
                    payload = {"hand_detected": num_hands > 0, "num_hands": num_hands, "sign": {"label": gesture, "score": score}}
                    await ws.send_text(json.dumps(ServerMessage(type="sign_status", payload=payload).model_dump()))
                    
                except Exception as e:
                    logger.error(f"Sign frame error: {e}")
                    await ws.send_text(json.dumps(ServerMessage(type="error", payload={"message": f"sign processing failed: {e}"}).model_dump()))

    except WebSocketDisconnect:
        logger.info("[WS] Client disconnected")
    except Exception as e:
        logger.error(f"[WS] Unexpected exception: {e}")
        try: await ws.close()
        except Exception: pass

@app.post("/translate", response_model=TranslationResponse)
async def translate_endpoint(req: TranslationRequest):
    service = get_translation_service()
    return await service.translate(text=req.text, source_lang=req.source_lang, target_lang=req.target_lang, model=req.model)

@app.get("/translate/languages")
async def supported_languages():
    languages = [
        {"code": "en", "name": "English"}, {"code": "hi", "name": "Hindi"},
        {"code": "es", "name": "Spanish"}, {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"}, {"code": "zh", "name": "Chinese"},
        {"code": "ja", "name": "Japanese"},
    ]
    return {"languages": languages}