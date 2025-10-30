# A11y Bridge

A multilingual, real-time, bidirectional communication bridge enabling deaf, hard-of-hearing, non-verbal, and blind users to communicate with everyone. It supports:
- Sign ➜ Text, Sign ➜ Voice
- Text ➜ Sign, Text ➜ Voice
- Voice ➜ Sign, Voice ➜ Text

Design goals: CPU-only (no special hardware), fully offline-capable with free/open-source components, privacy-first, accessible UI, and LLM-assisted language mediation.

## Architecture (high-level)
- Frontend (Web): React + TypeScript (Vite) with WebRTC/Web APIs for camera/mic capture, WebSockets for low-latency streaming, Three.js (avatar) for sign synthesis.
- Backend (Python): FastAPI + WebSockets for real-time pipelines.
  - Sign recognition: ResNet-18 + BiLSTM over pose/keypoint sequences (MediaPipe Holistic for efficient keypoint extraction) and/or raw-frame CNN features.
  - Sign synthesis: avatar animation from gloss/pose sequences; rule-based glossing + LLM-assisted paraphrase where appropriate.
  - STT (Speech-to-Text): Vosk (offline, multilingual) as default.
  - TTS (Text-to-Speech): Piper or pyttsx3 (offline) as default.
  - Translation: MarianMT (Helsinki-NLP) models via HuggingFace Transformers (offline-capable).
  - LLM integration: local models via llama.cpp or ctransformers for paraphrase/clarification/context bridging.
- Transport: WebSockets (low latency), optional WebRTC for P2P experiments.
- Optimization: ONNX/TorchScript export + quantization (int8) with onnxruntime for CPU.

## Technology stack (free/open-source)
- Backend: Python 3.11, FastAPI, Uvicorn, PyTorch, onnxruntime, numpy, opencv-python, mediapipe, transformers, sentencepiece, ctransformers or llama-cpp-python, vosk, pyttsx3/piper-phonemizer.
- Frontend: React + TypeScript, Vite, WebSockets, Three.js, TailwindCSS (optional), i18next.
- Data/Training: PyTorch Lightning (optional), scikit-learn, W&B alternative: MLflow local.
- Packaging: Tauri/Electron (later), Docker (dev), pre-commit + ruff/black (Python), eslint/prettier (JS).

## Roadmap
See docs/ROADMAP.md for a detailed, phase-by-phase plan, milestones, datasets, evaluation metrics, risks, and timelines.

## Run locally (placeholder)
- Backend: `python -m uvicorn backend.app.main:app --reload`
- Frontend: Will be scaffolded later with Vite (React + TS).

## License
MIT (see LICENSE).
