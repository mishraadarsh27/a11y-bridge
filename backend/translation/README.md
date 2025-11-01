# Translation Module

Modern LLM-based translation service for VaaniSetu (CommuniBridge) project.

## Features

- ✅ **LLM-powered translation** using Ollama
- ✅ **Multi-language support** (18+ languages including Hindi, Urdu, Tamil, etc.)
- ✅ **Auto language detection**
- ✅ **Batch translation** for efficiency
- ✅ **WebSocket integration** for real-time translation
- ✅ **REST API endpoints**
- ✅ **Privacy-first** (runs locally, no external API calls)

## Setup

### 1. Install Ollama

```bash
# Download from https://ollama.ai
# Or using package manager:
# Windows: winget install Ollama.Ollama
# Mac: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a FAST model (Recommended)

**Option A: Auto-setup (Easiest)**
```bash
python translation/setup_fast_model.py
```

**Option B: Manual setup**
```bash
# Recommended: Gemma2 2B (5-10x faster than llama3.2, only ~1.6GB)
ollama pull gemma2:2b

# Alternative: Qwen2.5 3B (balanced speed/quality)
ollama pull qwen2.5:3b

# Slower but higher quality:
ollama pull llama3.2:latest
```

### 3. Install Python dependencies

```bash
pip install httpx pydantic
```

### 4. Start Ollama server

```bash
ollama serve
```

## Usage

### REST API

#### Translate text

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "hi",
    "model": "llama3.2:latest"
  }'
```

Response:
```json
{
  "original": "Hello, how are you?",
  "translated": "नमस्ते, आप कैसे हैं?",
  "source_lang": "en",
  "target_lang": "hi",
  "confidence": 0.9
}
```

#### Batch translate

```bash
curl -X POST http://localhost:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Thank you", "Goodbye"],
    "source_lang": "en",
    "target_lang": "hi"
  }'
```

#### Detect language

```bash
curl -X POST http://localhost:8000/translate/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Bonjour"}'
```

Response:
```json
{
  "text": "Bonjour",
  "detected_language": "fr"
}
```

#### List supported languages

```bash
curl http://localhost:8000/translate/languages
```

### WebSocket

Connect to `ws://localhost:8000/ws` and send:

```json
{
  "type": "translate",
  "payload": {
    "text": "Hello world",
    "source_lang": "en",
    "target_lang": "hi"
  }
}
```

Response:
```json
{
  "type": "translation",
  "payload": {
    "original": "Hello world",
    "translated": "नमस्ते दुनिया",
    "source_lang": "en",
    "target_lang": "hi",
    "confidence": 0.9
  }
}
```

### Python API

```python
from translation import TranslationService

service = TranslationService()

# Simple translation
result = await service.translate(
    text="Hello",
    source_lang="en",
    target_lang="hi"
)
print(result.translated)

# Batch translation
results = await service.batch_translate(
    texts=["Hello", "Goodbye"],
    source_lang="en",
    target_lang="hi"
)

# Language detection
lang = await service.detect_language("Bonjour")
print(lang)  # "fr"
```

## Supported Languages

| Code | Language   | Code | Language   |
|------|-----------|------|-----------|
| en   | English   | hi   | Hindi     |
| ur   | Urdu      | pa   | Punjabi   |
| bn   | Bengali   | ta   | Tamil     |
| te   | Telugu    | mr   | Marathi   |
| gu   | Gujarati  | kn   | Kannada   |
| ml   | Malayalam | es   | Spanish   |
| fr   | French    | de   | German    |
| ar   | Arabic    | zh   | Chinese   |
| ja   | Japanese  | ko   | Korean    |

## Configuration

Set environment variables:

```bash
export OLLAMA_HOST=http://localhost:11434
export OLLAMA_MODEL=gemma2:2b  # Fast model (recommended)
export TRANSLATION_TIMEOUT=10.0  # Timeout in seconds
export TRANSLATION_CACHE=true    # Enable caching
export CACHE_MAX_SIZE=1000       # Max cached translations
```

Or configure programmatically:

```python
service = TranslationService(
    ollama_host="http://localhost:11434",
    model="llama3.2:latest",
    timeout=30.0
)
```

## Use Cases

1. **Sign Language Recognition → Translation**
   - Recognize sign → text → translate to user's language

2. **Real-time Communication Bridge**
   - Person A (Hindi) ↔ Person B (English)

3. **Accessibility**
   - Deaf users can communicate in their preferred language
   - Translation happens in real-time

4. **Multi-modal Communication**
   - Sign → Text → Translation → TTS (any language)

## Performance

### ⚡ Speed Optimizations (NEW)

- **Smart Caching**: Repeated translations are instant (cached in memory)
- **Fast Models**: Using `gemma2:2b` (~1.6GB) instead of `llama3.2` (5-10x faster)
- **Optimized Parameters**: Smaller context window (512 tokens) for faster inference
- **Reduced Timeout**: 10s timeout (down from 30s) for quicker failures
- **Batch Processing**: Translate multiple texts efficiently with `batch_translate()`
- **Async Operations**: Non-blocking, handles concurrent requests

### Benchmarks (approx.)

| Model | Load Time | Translation Time | Size |
|-------|-----------|-----------------|------|
| gemma2:2b (fastest) | ~500ms | ~1-2s | 1.6GB |
| qwen2.5:3b (balanced) | ~800ms | ~2-3s | 2GB |
| llama3.2 (quality) | ~1.5s | ~5-10s | 3GB |

### Other Benefits

- **Local inference**: No network latency
- **Privacy**: All data stays on device
- **No API costs**: Free and unlimited

## Alternative Models

You can use different Ollama models:

```bash
# Faster, smaller models
ollama pull llama3.2:1b

# Multilingual models
ollama pull aya:8b

# Specialized translation models
ollama pull command-r:latest
```

Then specify in the request:

```json
{
  "text": "Hello",
  "model": "aya:8b",
  "source_lang": "en",
  "target_lang": "hi"
}
```

## Error Handling

The service gracefully handles errors:
- If Ollama is not running, returns original text
- If translation fails, returns original text with confidence=0.0
- Network errors are caught and handled

## Future Enhancements

- [x] Add caching for frequently translated phrases ✅
- [x] Optimize for speed (fast models, reduced context) ✅
- [ ] Support for custom glossaries (sign language specific)
- [ ] Fine-tuned models for sign language text
- [ ] Streaming translations for long texts
- [ ] Translation memory/history
- [ ] Confidence scoring based on LLM uncertainty
- [ ] Persistent cache (Redis/SQLite)
