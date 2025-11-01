"""
Tests for translation service.

Run with: pytest tests/test_translation.py -v
"""

import pytest
from backend.translation.service import TranslationService, TranslationResponse


@pytest.fixture
async def service():
    """Create a translation service instance."""
    svc = TranslationService(
        ollama_host="http://localhost:11434", model="llama3.2:latest"
    )
    yield svc
    await svc.close()


@pytest.mark.asyncio
async def test_basic_translation(service):
    """Test basic translation functionality."""
    result = await service.translate(text="Hello", source_lang="en", target_lang="hi")

    assert isinstance(result, TranslationResponse)
    assert result.original == "Hello"
    assert len(result.translated) > 0
    assert result.source_lang == "en"
    assert result.target_lang == "hi"


@pytest.mark.asyncio
async def test_empty_text(service):
    """Test translation with empty text."""
    result = await service.translate(text="", source_lang="en", target_lang="hi")

    assert result.original == ""
    assert result.translated == ""
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_auto_detect(service):
    """Test auto language detection."""
    result = await service.translate(
        text="Bonjour", source_lang="auto", target_lang="en"
    )

    assert result.original == "Bonjour"
    assert len(result.translated) > 0


@pytest.mark.asyncio
async def test_language_detection(service):
    """Test language detection."""
    lang = await service.detect_language("Hello world")
    assert lang == "en"

    lang = await service.detect_language("Bonjour")
    assert lang in ["fr", "en"]  # May vary depending on model


@pytest.mark.asyncio
async def test_batch_translation(service):
    """Test batch translation."""
    texts = ["Hello", "Goodbye", "Thank you"]
    results = await service.batch_translate(
        texts=texts, source_lang="en", target_lang="es"
    )

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, TranslationResponse)
        assert len(result.translated) > 0


@pytest.mark.asyncio
async def test_multiple_languages(service):
    """Test translation across multiple languages."""
    test_cases = [
        ("Hello", "en", "hi"),
        ("Hello", "en", "es"),
        ("Hello", "en", "fr"),
    ]

    for text, source, target in test_cases:
        result = await service.translate(text, source, target)
        assert len(result.translated) > 0
        assert result.source_lang == source
        assert result.target_lang == target


def test_prompt_building(service):
    """Test prompt building logic."""
    prompt = service._build_prompt("Hello", "en", "hi")
    assert "Hello" in prompt
    assert "English" in prompt or "en" in prompt
    assert "Hindi" in prompt or "hi" in prompt


@pytest.mark.asyncio
async def test_error_handling(service):
    """Test error handling with invalid input."""
    # Should not raise exception, just return original text
    result = await service.translate(
        text="Test", source_lang="invalid", target_lang="invalid"
    )
    assert isinstance(result, TranslationResponse)
