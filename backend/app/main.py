from __future__ import annotations

import json
from typing import Any, Literal, Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="A11y Bridge Backend", version="0.1.0")

# Allow local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class TextPayload(BaseModel):
    text: str = Field(min_length=0, max_length=2000)


class ClientMessage(BaseModel):
    type: Literal["health", "text", "stt_result", "tts_text"]
    payload: Optional[dict[str, Any]] = None


class ServerMessage(BaseModel):
    type: Literal["health", "text_echo", "stt_ack", "tts_ack", "error"]
    payload: dict[str, Any]


@app.get("/health", response_model=HealthResponse)
def health():
    return JSONResponse(HealthResponse().model_dump())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    # Structured JSON protocol with simple handlers; extend with stt/tts/sign later
    try:
        while True:
            raw = await ws.receive_text()
            # Back-compat: accept plain text -> echo
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="text_echo", payload={"text": raw}
                        ).model_dump()
                    )
                )
                continue

            try:
                msg = ClientMessage(**parsed)
            except Exception as e:  # pydantic validation error
                err = ServerMessage(
                    type="error", payload={"message": f"invalid message: {e}"}
                )
                await ws.send_text(json.dumps(err.model_dump()))
                continue

            if msg.type == "health":
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="health", payload=HealthResponse().model_dump()
                        ).model_dump()
                    )
                )
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
            elif msg.type == "tts_text":
                data = TextPayload(**(msg.payload or {}))
                await ws.send_text(
                    json.dumps(
                        ServerMessage(
                            type="tts_ack", payload={"text": data.text}
                        ).model_dump()
                    )
                )
    except Exception:
        await ws.close()
