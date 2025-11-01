"""
Example usage of the translation service.

Make sure Ollama is running locally:
    ollama serve
    ollama pull llama3.2:latest

Then run this script:
    python -m backend.translation.example_usage
"""

import asyncio
from service import TranslationService


async def main():
    # Initialize the service
    service = TranslationService(
        ollama_host="http://localhost:11434", model="llama3.2:latest"
    )

    print("=" * 60)
    print("Translation Service Examples")
    print("=" * 60)

    # Example 1: English to Hindi
    print("\n1. English → Hindi")
    result = await service.translate(
        text="Hello, how are you?", source_lang="en", target_lang="hi"
    )
    print(f"Original: {result.original}")
    print(f"Translated: {result.translated}")
    print(f"Confidence: {result.confidence}")

    # Example 2: Auto-detect language and translate to English
    print("\n2. Auto-detect → English")
    result = await service.translate(
        text="नमस्ते, आप कैसे हैं?", source_lang="auto", target_lang="en"
    )
    print(f"Original: {result.original}")
    print(f"Translated: {result.translated}")

    # Example 3: Language detection
    print("\n3. Language Detection")
    lang = await service.detect_language("Bonjour, comment allez-vous?")
    print(f"Detected language: {lang}")

    # Example 4: Batch translation
    print("\n4. Batch Translation (English → Spanish)")
    texts = ["Hello", "Thank you", "Goodbye"]
    results = await service.batch_translate(
        texts=texts, source_lang="en", target_lang="es"
    )
    for i, res in enumerate(results):
        print(f"{texts[i]} → {res.translated}")

    # Example 5: Sign language text translation
    print("\n5. Sign Language Context (English → Hindi)")
    result = await service.translate(
        text="I love you", source_lang="en", target_lang="hi"
    )
    print(f"Sign text: {result.original}")
    print(f"Translation: {result.translated}")

    print("\n" + "=" * 60)

    # Close the service
    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
