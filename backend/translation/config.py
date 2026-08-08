"""
Optimized translation configuration for faster performance.
"""

import os

# Recommended lightweight models for translation (fastest to slowest):
TRANSLATION_MODELS = {
    "fastest": "gemma2:2b",  # 2B params, ~5-10x faster than llama3.2
    "balanced": "qwen2.5:3b",  # 3B params, good speed + quality
    "quality": "llama3.2:latest",  # 3B params, better quality, slower
}

# Default to fastest for real-time use
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", TRANSLATION_MODELS["fastest"])

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
TIMEOUT = float(os.getenv("TRANSLATION_TIMEOUT", "10.0"))  # 10s timeout

# Cache settings
CACHE_ENABLED = os.getenv("TRANSLATION_CACHE", "true").lower() == "true"
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))

# Performance tuning
OLLAMA_OPTIONS = {
    "num_ctx": 512,  # Reduced context window (faster)
    "temperature": 0.2,  # Low temp for consistent translation
    "top_p": 0.8,
    "top_k": 20,
    "num_predict": 256,  # Max output tokens
}
