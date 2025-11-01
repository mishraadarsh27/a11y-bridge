# Translation Module - Quick Setup Guide

## 🚀 Quick Start (5 minutes)

### Step 1: Install Ollama

**Windows:**
```powershell
winget install Ollama.Ollama
```

**Mac:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Or download from: https://ollama.ai

### Step 2: Start Ollama and Pull Model

```bash
# Start Ollama server (in a separate terminal)
ollama serve

# Pull the translation model
ollama pull llama3.2:latest
```

### Step 3: Install Dependencies

```bash
pip install httpx pydantic
```

Or install from requirements file:
```bash
pip install -r requirements-translation.txt
```

### Step 4: Start Your Backend

```bash
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

## ✅ Test Translation

### Using curl:

```bash
# Basic translation
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "source_lang": "en",
    "target_lang": "hi"
  }'

# Auto-detect source language
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour",
    "source_lang": "auto",
    "target_lang": "en"
  }'

# Detect language
curl -X POST http://localhost:8000/translate/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते"}'

# Get supported languages
curl http://localhost:8000/translate/languages
```

### Using Python:

```python
import httpx
import asyncio

async def test_translation():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/translate",
            json={
                "text": "Hello world",
                "source_lang": "en",
                "target_lang": "hi"
            }
        )
        print(response.json())

asyncio.run(test_translation())
```

### Using WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'translate',
    payload: {
      text: 'Hello world',
      source_lang: 'en',
      target_lang: 'hi'
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'translation') {
    console.log('Translation:', data.payload.translated);
  }
};
```

## 🎯 Use Cases in Your Project

### 1. Sign → Text → Translation

```python
# User performs sign language gesture
sign_text = "Hello"  # From sign recognition

# Translate to user's preferred language
result = await translate_text(
    text=sign_text,
    source_lang="en",
    target_lang="hi"
)

# Display: "नमस्ते"
```

### 2. Real-time Communication Bridge

```python
# Person A speaks Hindi
hindi_text = "आप कैसे हैं?"

# Translate to English for Person B
result = await translate_text(
    text=hindi_text,
    source_lang="hi",
    target_lang="en"
)

# Person B sees: "How are you?"
```

### 3. Multi-lingual Sign Recognition

```python
# Detect sign → Translate → Speak in any language
sign_text = await recognize_sign(video_frame)
translated = await translate_text(sign_text, "en", "es")
await text_to_speech(translated.translated, lang="es")
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```bash
# .env
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
```

### Alternative Models

For different use cases:

```bash
# Faster (smaller model, ~1GB)
ollama pull llama3.2:1b

# Better multilingual (specialized for Indian languages)
ollama pull aya:8b

# Best quality (larger, slower)
ollama pull llama3.1:8b
```

Update your request:
```json
{
  "text": "Hello",
  "model": "aya:8b",
  "source_lang": "en",
  "target_lang": "hi"
}
```

## 📊 Performance Tips

1. **First request is slow**: Ollama loads model on first use (~2-5 seconds)
2. **Subsequent requests are fast**: ~200-500ms per translation
3. **Batch translations**: Use `/translate/batch` for multiple texts
4. **Keep Ollama running**: Don't restart between requests

## 🐛 Troubleshooting

### Ollama not responding

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
# Windows: Restart from Task Manager or System Tray
# Mac/Linux: pkill ollama && ollama serve
```

### Model not found

```bash
# List available models
ollama list

# Pull the required model
ollama pull llama3.2:latest
```

### Translation returns original text

This means either:
- Ollama is not running
- Model is not loaded
- Network/timeout issue

Check backend logs for specific error.

### Slow translations

- Use smaller model: `llama3.2:1b`
- Check CPU/RAM usage
- Consider GPU support if available

## 📚 API Reference

### POST `/translate`

**Request:**
```json
{
  "text": "string",
  "source_lang": "en",     // or "auto"
  "target_lang": "hi",
  "model": "llama3.2:latest"  // optional
}
```

**Response:**
```json
{
  "original": "Hello",
  "translated": "नमस्ते",
  "source_lang": "en",
  "target_lang": "hi",
  "confidence": 0.9
}
```

### POST `/translate/batch`

**Request:**
```json
{
  "texts": ["Hello", "Goodbye"],
  "source_lang": "en",
  "target_lang": "hi"
}
```

**Response:**
```json
{
  "translations": [
    {
      "original": "Hello",
      "translated": "नमस्ते",
      "source_lang": "en",
      "target_lang": "hi",
      "confidence": 0.9
    },
    {
      "original": "Goodbye",
      "translated": "अलविदा",
      "source_lang": "en",
      "target_lang": "hi",
      "confidence": 0.9
    }
  ]
}
```

### POST `/translate/detect`

**Request:**
```json
{
  "text": "Bonjour"
}
```

**Response:**
```json
{
  "text": "Bonjour",
  "detected_language": "fr"
}
```

### GET `/translate/languages`

**Response:**
```json
{
  "languages": [
    {"code": "en", "name": "English"},
    {"code": "hi", "name": "Hindi"},
    ...
  ]
}
```

## 🌟 Supported Languages

English (en), Hindi (hi), Urdu (ur), Punjabi (pa), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), Gujarati (gu), Kannada (kn), Malayalam (ml), Spanish (es), French (fr), German (de), Arabic (ar), Chinese (zh), Japanese (ja), Korean (ko)

## 🔐 Privacy & Security

✅ **All processing happens locally** - No data sent to external servers
✅ **Offline-capable** - Works without internet
✅ **Open source** - Fully auditable code
✅ **No API keys needed** - Free to use

## 📝 Next Steps

1. ✅ Basic translation working
2. 🔄 Integrate with sign recognition pipeline
3. 🔄 Add caching for common phrases
4. 🔄 Fine-tune models for sign language text
5. 🔄 Add streaming support for long texts

## 💡 Tips

- Start Ollama at system startup for instant translations
- Use `aya:8b` model for better Indian language support
- Cache translations for frequently used phrases
- Use batch endpoint for multiple translations
- Monitor Ollama resource usage

## 🆘 Support

For issues:
1. Check Ollama logs: `ollama logs`
2. Check FastAPI logs in terminal
3. Test with curl first before integrating
4. See `backend/translation/README.md` for detailed docs
