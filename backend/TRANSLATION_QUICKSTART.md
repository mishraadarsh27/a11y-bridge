# 🚀 Translation Quick Start (FAST Setup)

Yeh guide aapko **5 minutes** mein fast translation setup karne mein help karegi.

## Step 1: Ollama Install karein

### Windows
```powershell
winget install Ollama.Ollama
```

### Mac
```bash
brew install ollama
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Step 2: Fast Model Download karein

**Option A: Automatic (Recommended)**
```bash
cd backend
python translation/setup_fast_model.py
```

Yeh script automatically:
- ✅ Check karega ki Ollama chal raha hai
- ✅ Fast model (`gemma2:2b` - only 1.6GB) download karega
- ✅ Test karega translation ko

**Option B: Manual**
```bash
# Fast model (5-10x faster than llama3.2)
ollama pull gemma2:2b
```

## Step 3: Backend Start karein

```bash
cd backend
python -m app.main
```

Ya:
```bash
uvicorn app.main:app --reload --port 8000
```

## Step 4: Test karein

### WebSocket se test (Frontend se)
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "translate",
    payload: {
      text: "Hello, how are you?",
      source_lang: "en",
      target_lang: "hi"
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data.payload.translated); // "नमस्ते, आप कैसे हैं?"
};
```

### Python se test
```python
import asyncio
from translation import translate_text

async def test():
    result = await translate_text(
        text="Hello, how are you?",
        source_lang="en",
        target_lang="hi"
    )
    print(f"Original: {result.original}")
    print(f"Translation: {result.translated}")
    print(f"Confidence: {result.confidence}")

asyncio.run(test())
```

## Performance Tips ⚡

### 1. **Cache automatically enabled hai**
Repeated translations instantly return hoti hain (no LLM call).

### 2. **Model comparison**
| Model | Speed | Quality | Size |
|-------|-------|---------|------|
| `gemma2:2b` | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 1.6GB |
| `qwen2.5:3b` | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 2GB |
| `llama3.2` | ⚡⚡ | ⭐⭐⭐⭐⭐ | 3GB |

### 3. **Environment variables** (Optional)
```bash
# .env file mein add karein
OLLAMA_MODEL=gemma2:2b
TRANSLATION_TIMEOUT=10.0
TRANSLATION_CACHE=true
CACHE_MAX_SIZE=1000
```

### 4. **Model ko warm-up karein**
Pehli translation 2-3s leti hai (model loading), baad mein <1s.

```bash
# Quick warm-up
ollama run gemma2:2b "hello"
```

## Troubleshooting 🔧

### "Connection refused" error
```bash
# Ollama start karein
ollama serve
```

### Translation bahut slow hai
```bash
# Fast model use karein
export OLLAMA_MODEL=gemma2:2b

# Ya config file mein change karein
```

### Model download nahi ho raha
```bash
# Internet check karein, phir:
ollama pull gemma2:2b --insecure
```

## Auto-Translation Integration

### Sign Recognition → Auto Translate

Aap sign recognition output ko automatically translate kar sakte hain:

```python
# app/main.py mein already integrated hai
# sign_frame processing ke baad:

# Auto-translate detected sign text
if sign_text:
    result = await translate_text(
        text=sign_text,
        source_lang="en",  # Sign recognition output usually English
        target_lang="hi"   # User's preferred language
    )
    translated_text = result.translated
```

### Save with Translation

Database/file mein save karte waqt dono save karein:

```python
# Save both original and translated
output = {
    "original": sign_text,
    "original_lang": "en",
    "translated": translated_text,
    "target_lang": user_preferred_lang,
    "timestamp": datetime.now()
}
```

## Next Steps

✅ Translation module ready hai!

Ab aap:
1. Frontend se integrate kar sakte hain
2. Real-time translation use kar sakte hain
3. Multiple languages support kar sakte hain
4. Offline bhi kaam karega (local Ollama)

## Questions?

Translation code: `backend/translation/service.py`
Config: `backend/translation/config.py`
Examples: `backend/translation/example_usage.py`
