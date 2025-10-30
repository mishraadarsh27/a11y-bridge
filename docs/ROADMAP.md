# A11y Bridge – Detailed Roadmap

Goal: A CPU-only, multilingual, bidirectional communication bridge (Sign/Text/Voice in any direction), real-time, free/open-source, privacy-first.

## Phase 0 – Foundations (Week 0-1)
- Project scaffolding, repo standards, contribution guide.
- Accessibility and privacy principles; consent for media processing.
- Define supported languages for MVP (e.g., English + one more) and sign language (ASL subset to start).

## Phase 1 – MVP (Weeks 1-4)
Deliver an offline demo running on CPU:
- Sign ➜ Text (subset): Fingerspelling/digits using MediaPipe Holistic keypoints + ResNet18 features + BiLSTM classifier. Latency target: <200ms per frame on laptop CPU.
- Voice ➜ Text: Vosk STT (EN) streaming.
- Text ➜ Voice: Piper/pyttsx3 offline voice.
- Real-time transport: WebSocket streaming between frontend and backend.
- UI: Minimal web app with camera/mic toggle, three panels: Sign/Voice/Text streams.
- Metrics: Top-1 accuracy for sign subset; WER for STT; latency percentiles (p50/p95).

## Phase 2 – Sign Recognition v1 (Weeks 5-9)
- Data: Integrate WLASL/MS-ASL subsets (license-compliant). Create `scripts/prepare_data.py` to generate keypoint sequences and train/val/test splits.
- Model: ResNet-18 backbone for frame features (or direct keypoint MLP), BiLSTM/TemporalConv for sequence modeling.
- Training: PyTorch Lightning or native PyTorch. Early stopping, mixed precision (CPU bf16 where available), augmentations.
- Export: TorchScript + ONNX; quantize int8 with onnxruntime. Benchmark on CPU.
- Evaluate: Top-1/Top-5, confusion matrices, latency budget on commodity CPU.

## Phase 3 – Sign Synthesis v1 (Weeks 10-13)
- Approach: 3D avatar with a sign dictionary (common phrases). Render via Three.js; store animations as JSON (pose sequences).
- Text ➜ Sign: rule-based glossing for common phrases + mapping to animations; fall back to Text ➜ Voice if phrase unknown.
- Authoring: Build a small tool to create/edit animations. Store under `models/sign_animations/`.
- Accessibility: captions, adjustable avatar speed/size/contrast.

## Phase 4 – Multilingual Packs (Weeks 14-17)
- STT: Add Vosk models for chosen languages (offline-capable) with dynamic selection.
- TTS: Add Piper voices per language; select by locale. Fallback to OS TTS (pyttsx3) when voices unavailable.
- Translation: MarianMT pipelines for en↔xx; cache models locally.
- i18n: Frontend strings via i18next.

## Phase 5 – LLM Integration (Weeks 18-20)
- Local LLM (7B) via llama.cpp/ctransformers for:
  - Paraphrase/clarify noisy STT output.
  - Simplify/expand text to improve sign synthesis mapping.
  - Safety filter and context-aware rephrasing, fully offline where possible.
- Add prompt presets; allow user opt-in.

## Phase 6 – Real-time Optimization (Weeks 21-22)
- Streaming inference windows (sliding buffers) to reduce latency.
- Frame skipping/adaptive FPS; keypoint-only mode for ultra-low compute.
- ONNXRuntime threading/tuning; quantization-aware training for smaller models.

## Phase 7 – Accessibility & UX (Weeks 23-24)
- Full keyboard navigation, screen-reader support, high-contrast themes, scalable fonts.
- Conversation transcript with timestamps; export/share locally.
- Error recovery and offline-first UX.

## Phase 8 – Packaging & Deployment (Weeks 25-26)
- Desktop app via Tauri/Electron bundling backend + frontend.
- Optional Docker for dev; no cloud dependency required.
- Auto-updater (optional), model download manager with checksum verification.

## Phase 9 – Security & Privacy
- Local-only processing by default. No telemetry unless opt-in.
- Media never leaves device except if user explicitly enables cloud features.
- Model licenses vetted; dataset usage respects licenses.

## Datasets (examples)
- WLASL, MS-ASL, Phoenix-2014T (for SLT research, license permitting), custom curated phrase sets.

## Evaluation
- Sign recognition: Top-1/Top-5 accuracy, latency p50/p95.
- STT: WER/CER; TTS: MOS-like subjective tests.
- E2E: Task success rate in simulated dialogues.

## Risks & Mitigations
- Real-time on CPU: prefer keypoints over raw frames; quantization and ONNXRuntime.
- Sign synthesis quality: start with dictionary-based animations; expand gradually.
- Multilingual coverage: ship language packs incrementally; allow user downloads.

## Deliverables per Milestone
- Phase 1: runnable demo, docs for setup, short video.
- Phase 2: trained quantized model + benchmarks.
- Phase 3: avatar demo with phrase dictionary.
- Phase 4-6: multilingual packs + LLM-enabled mediation.

## Timeline Notes
Assumes ~1-2 contributors part-time. Adjust as needed.
