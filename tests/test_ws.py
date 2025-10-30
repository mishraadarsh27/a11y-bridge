import json
from fastapi.testclient import TestClient
from backend.app.main import app


def test_ws_text_echo_json():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "text", "payload": {"text": "hi"}}))
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["type"] == "text_echo"
        assert data["payload"]["text"] == "hi"


def test_ws_plain_text_backcompat():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text("hello")
        msg = ws.receive_text()
        data = json.loads(msg)
        assert data["type"] == "text_echo"
        assert data["payload"]["text"] == "hello"


def test_ws_stt_ack():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "stt_result", "payload": {"text": "speech"}}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "stt_ack"
        assert data["payload"]["text"] == "speech"


def test_ws_tts_ack():
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "tts_text", "payload": {"text": "hello"}}))
        data = json.loads(ws.receive_text())
        assert data["type"] == "tts_ack"
        assert data["payload"]["text"] == "hello"
