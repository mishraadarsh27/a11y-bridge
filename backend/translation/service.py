from __future__ import annotations

import os
import hashlib
from typing import Optional
import httpx
from pydantic import BaseModel


class TranslationRequest(BaseModel):
    text: str
    source_lang: str = "auto"
    target_lang: str = "en"
    model: str = "llama3.2:latest"


class TranslationResponse(BaseModel):
    original: str
    translated: str
    source_lang: str
    target_lang: str
    confidence: Optional[float] = None


class TranslationService:
    """Modern LLM-based translation service using Ollama with caching."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model: str = "llama3.2:latest",
        timeout: float = 15.0,  # Reduced from 30s for faster response
        cache_enabled: bool = True,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        self._cache: dict[str, TranslationResponse] = {}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def _build_prompt(self, text: str, source_lang: str, target_lang: str) -> str:
        """Build translation prompt for LLM."""
        lang_names = {
            "en": "English",
            "hi": "Hindi",
            "ur": "Urdu",
            "pa": "Punjabi",
            "bn": "Bengali",
            "ta": "Tamil",
            "te": "Telugu",
            "mr": "Marathi",
            "gu": "Gujarati",
            "kn": "Kannada",
            "ml": "Malayalam",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ar": "Arabic",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "auto": "auto-detect",
        }

        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)

        if source_lang == "auto":
            prompt = f"""Translate the following text to {target_name}.
Respond ONLY with the translation, nothing else. No explanations or additional text.

Text to translate:
{text}

Translation:"""
        else:
            prompt = f"""Translate the following text from {source_name} to {target_name}.
Respond ONLY with the translation, nothing else. No explanations or additional text.

Text to translate:
{text}

Translation:"""

        return prompt

    def _get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{text}:{source_lang}:{target_lang}"
        return hashlib.md5(content.encode()).hexdigest()

    async def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
        model: Optional[str] = None,
    ) -> TranslationResponse:
        """
        Translate text using LLM with caching.

        Args:
            text: Text to translate
            source_lang: Source language code (or "auto" for detection)
            target_lang: Target language code
            model: Ollama model name (overrides default)

        Returns:
            TranslationResponse with translated text
        """
        if not text or not text.strip():
            return TranslationResponse(
                original=text,
                translated=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=1.0,
            )

        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(text, source_lang, target_lang)
            if cache_key in self._cache:
                return self._cache[cache_key]

        model_name = model or self.model
        prompt = self._build_prompt(text, source_lang, target_lang)

        try:
            response = await self._client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 512,  # Smaller context = faster
                        "temperature": 0.2,  # Low temp for consistent translation
                        "top_p": 0.8,
                        "top_k": 20,
                        "num_predict": 256,  # Limit output length
                    },
                },
            )
            response.raise_for_status()
            data = response.json()
            translated = data.get("response", "").strip()

            # Basic confidence heuristic based on response quality
            confidence = 0.9 if len(translated) > 0 else 0.0

            result = TranslationResponse(
                original=text,
                translated=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=confidence,
            )

            # Cache successful translation
            if self.cache_enabled and len(self._cache) < 1000:  # Limit cache size
                cache_key = self._get_cache_key(text, source_lang, target_lang)
                self._cache[cache_key] = result

            return result

        except httpx.HTTPError:
            # Fallback: return original text on error
            return TranslationResponse(
                original=text,
                translated=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
            )

    async def detect_language(self, text: str) -> str:
        """
        Detect language of text using LLM.

        Returns:
            Language code (e.g., "en", "hi", "es")
        """
        if not text or not text.strip():
            return "en"

        prompt = f"""Detect the language of the following text and respond with ONLY the ISO 639-1 language code (e.g., en, hi, es, fr, ar, etc.).
No explanations, just the 2-letter code.

Text:
{text}

Language code:"""

        try:
            response = await self._client.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
            data = response.json()
            lang_code = data.get("response", "en").strip().lower()[:2]
            return lang_code if len(lang_code) == 2 else "en"
        except Exception:
            return "en"

    async def batch_translate(
        self,
        texts: list[str],
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> list[TranslationResponse]:
        """Translate multiple texts efficiently."""
        import asyncio

        tasks = [self.translate(text, source_lang, target_lang) for text in texts]
        return await asyncio.gather(*tasks)


# Global singleton instance
_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """Get or create global translation service instance."""
    global _service
    if _service is None:
        try:
            from .config import OLLAMA_HOST, DEFAULT_MODEL, TIMEOUT, CACHE_ENABLED

            _service = TranslationService(
                ollama_host=OLLAMA_HOST,
                model=DEFAULT_MODEL,
                timeout=TIMEOUT,
                cache_enabled=CACHE_ENABLED,
            )
        except ImportError:
            # Fallback if config not available
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            model = os.getenv("OLLAMA_MODEL", "gemma2:2b")  # Use fast model by default
            _service = TranslationService(
                ollama_host=ollama_host, model=model, timeout=10.0
            )
    return _service


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "en",
) -> TranslationResponse:
    """Convenience function for quick translation."""
    service = get_translation_service()
    return await service.translate(text, source_lang, target_lang)
