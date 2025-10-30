# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

Project scope and current state
- Goal: Multilingual, real-time, offline-first communication bridge across Sign/Text/Voice.
- Current code: Minimal FastAPI backend with a WebSocket echo and a /health endpoint. Frontend is not yet scaffolded (planned: React + TypeScript via Vite). See docs/ROADMAP.md for phased plan.

Common commands
- Dev setup (Windows PowerShell):
```powershell path=null start=null
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt -r requirements-dev.txt
```
- Start backend (dev, auto-reload on code changes):
```powershell path=null start=null
python -m uvicorn backend.app.main:app --reload --port 8000
```
- Frontend (Vite React + TS):
```powershell path=null start=null
cd frontend
npm install
npm run dev
```
Then open http://localhost:5173; the app auto-connects to ws://localhost:8000/ws. Type a message and click Send.
- Health check:
```powershell path=null start=null
Invoke-WebRequest http://127.0.0.1:8000/health | Select-Object -ExpandProperty Content
```
- Tests (pytest configured via pyproject.toml):
  - Run all tests:
```powershell path=null start=null
pytest
```
  - Run a single test file / function:
```powershell path=null start=null
pytest tests/test_health.py::test_health
```
  - Filter by keyword:
```powershell path=null start=null
pytest -k "keyword"
```
- Lint and format (ruff/black configured via pyproject.toml):
```powershell path=null start=null
ruff check backend tests
black backend tests
```
- Git hooks (pre-commit):
```powershell path=null start=null
pip install -r requirements-dev.txt
pre-commit install
pre-commit run --all-files
```

High-level architecture and how pieces fit
- Frontend (planned): React + TypeScript (Vite). Uses Web APIs for camera/microphone capture, connects to backend over WebSockets for low-latency streaming, renders a 3D avatar (Three.js) for sign synthesis, and uses i18next for i18n.
- Backend (present): FastAPI with WebSockets. Entry point is backend/app/main.py.
  - Today: /health returns {"status":"ok"}; /ws speaks a simple JSON protocol (also accepts plain text for back-compat):
    - Client ➜ `{ "type": "text", "payload": { "text": "hello" } }` → Server ➜ `{ "type": "text_echo", "payload": { "text": "hello" } }`
    - Client ➜ `{ "type": "stt_result", "payload": { "text": "hi" } }` → Server ➜ `{ "type": "stt_ack", "payload": { "text": "hi" } }`
    - Client ➜ `{ "type": "tts_text", "payload": { "text": "hello" } }` → Server ➜ `{ "type": "tts_ack", "payload": { "text": "hello" } }`
    - Client ➜ `{ "type": "health" }` → Server ➜ `{ "type": "health", "payload": { "status": "ok" } }`
  - Target pipelines (from README):
    - Sign recognition via MediaPipe keypoints + sequence model (e.g., ResNet-18 features + BiLSTM) and/or raw-frame CNN features.
    - Sign synthesis via avatar animation driven by gloss/pose sequences; rule-based glossing + optional LLM-assisted paraphrase.
    - STT via Vosk (offline), TTS via Piper/pyttsx3 (offline), translation via MarianMT models with Transformers.
    - Local LLM integration via llama.cpp/ctransformers for paraphrase/clarification.
  - Optimization: ONNX/TorchScript export and int8 quantization with onnxruntime for CPU-only inference.
- Data and models:
  - data/: indexes/metadata only; keep raw datasets outside the repo or manage via DVC/LFS.
  - models/: small configs/README; large binaries are git-ignored.
  - scripts/: utility scripts for data prep/training/eval (to be added per roadmap).

Repo-specific guidance for future agents
- Backend changes start at backend/app/main.py; add routers or WebSocket handlers here or via FastAPI routers as the backend grows.
- Python dependencies live in backend/requirements.txt. Dev-only tools (pytest, ruff, black) are not yet declared; add them before relying on lint/test commands.
- Tests should live under tests/ and use pytest patterns; prefer explicit test node selection when running a single test.
- Frontend is a placeholder. When scaffolding, use Vite (React + TS) under frontend/ and wire to the backend WebSocket at /ws.
- docs/ROADMAP.md encodes the phased delivery plan; align new features with the appropriate phase and CPU-only/offline constraints.
