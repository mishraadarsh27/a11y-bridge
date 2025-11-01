"""
Example: Integrating Translation with Sign Recognition

This shows how to integrate the new translation module with your existing
sign language recognition pipeline.
"""

import asyncio
from typing import Optional
from translation.service import get_translation_service


# Mock sign recognition (replace with your actual implementation)
async def mock_sign_recognition(frame) -> Optional[str]:
    """
    Replace this with your actual sign recognition logic.
    Should return the recognized text from sign language.
    """
    # Simulating sign recognition
    await asyncio.sleep(0.1)  # Simulating processing time
    return "Hello"  # Example recognized text


# Integration Example 1: Sign → Translation → Display
async def sign_to_translated_text(frame, target_language: str = "hi"):
    """
    Complete pipeline: Sign language → Text → Translation

    Args:
        frame: Video frame containing sign language
        target_language: Target language code (e.g., "hi" for Hindi)

    Returns:
        Translated text
    """
    # Step 1: Recognize sign language
    recognized_text = await mock_sign_recognition(frame)

    if not recognized_text:
        return None

    print(f"Recognized: {recognized_text}")

    # Step 2: Translate to target language
    service = get_translation_service()
    result = await service.translate(
        text=recognized_text,
        source_lang="en",  # Assuming signs are mapped to English
        target_lang=target_language,
    )

    print(f"Translated ({target_language}): {result.translated}")

    return result.translated


# Integration Example 2: Bidirectional Communication
async def bidirectional_communication_bridge(
    person_a_input: str, person_a_lang: str, person_b_lang: str
) -> dict:
    """
    Enable communication between two people speaking different languages.

    Args:
        person_a_input: Text/sign from Person A
        person_a_lang: Person A's language
        person_b_lang: Person B's language

    Returns:
        Dictionary with original and translated text
    """
    service = get_translation_service()

    # Translate A → B
    a_to_b = await service.translate(
        text=person_a_input, source_lang=person_a_lang, target_lang=person_b_lang
    )

    return {
        "person_a_original": person_a_input,
        "person_a_lang": person_a_lang,
        "person_b_translation": a_to_b.translated,
        "person_b_lang": person_b_lang,
        "confidence": a_to_b.confidence,
    }


# Integration Example 3: Multi-language Sign Recognition
async def multilingual_sign_recognition(frame, target_languages: list[str]):
    """
    Recognize sign and translate to multiple languages simultaneously.

    Args:
        frame: Video frame
        target_languages: List of language codes to translate to

    Returns:
        Dictionary mapping language codes to translations
    """
    # Recognize sign
    recognized_text = await mock_sign_recognition(frame)

    if not recognized_text:
        return {}

    # Translate to all target languages in parallel
    service = get_translation_service()
    results = await service.batch_translate(
        texts=[recognized_text] * len(target_languages),
        source_lang="en",
        target_lang=target_languages[0],  # We'll do individual calls for simplicity
    )

    # Better approach: individual calls with different target languages
    translations = {}
    tasks = [
        service.translate(recognized_text, "en", lang) for lang in target_languages
    ]
    results = await asyncio.gather(*tasks)

    for lang, result in zip(target_languages, results):
        translations[lang] = result.translated

    return {"original": recognized_text, "translations": translations}


# Integration Example 4: WebSocket Handler with Translation
async def websocket_sign_translation_handler(websocket, message):
    """
    Handle WebSocket messages that include both sign recognition and translation.

    This can be integrated into your existing WebSocket handler in main.py
    """
    import json

    msg_type = message.get("type")
    payload = message.get("payload", {})

    if msg_type == "sign_frame_with_translation":
        # Extract parameters
        target_lang = payload.get("target_lang", "hi")
        frame_data = payload.get("frame")

        # Recognize sign (replace with actual implementation)
        recognized = await mock_sign_recognition(frame_data)

        if recognized:
            # Translate
            service = get_translation_service()
            translated = await service.translate(
                text=recognized, source_lang="en", target_lang=target_lang
            )

            # Send response
            response = {
                "type": "sign_translation_result",
                "payload": {
                    "original": recognized,
                    "translated": translated.translated,
                    "source_lang": "en",
                    "target_lang": target_lang,
                    "confidence": translated.confidence,
                },
            }
            await websocket.send_text(json.dumps(response))


# Integration Example 5: Real-time Translation Pipeline
class RealTimeTranslationPipeline:
    """
    A pipeline that continuously processes sign language frames
    and provides real-time translations.
    """

    def __init__(self, target_language: str = "hi"):
        self.target_language = target_language
        self.service = get_translation_service()
        self.last_translation = None
        self.translation_buffer = []

    async def process_frame(self, frame):
        """Process a single frame."""
        # Recognize sign
        recognized = await mock_sign_recognition(frame)

        if not recognized:
            return None

        # Avoid duplicate translations
        if recognized == self.last_translation:
            return None

        # Translate
        result = await self.service.translate(
            text=recognized, source_lang="en", target_lang=self.target_language
        )

        self.last_translation = recognized
        self.translation_buffer.append(
            {
                "original": recognized,
                "translated": result.translated,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

        # Keep only last 10 translations
        if len(self.translation_buffer) > 10:
            self.translation_buffer.pop(0)

        return result

    def get_history(self):
        """Get translation history."""
        return self.translation_buffer

    def clear_history(self):
        """Clear translation history."""
        self.translation_buffer.clear()
        self.last_translation = None


# Main demo
async def main():
    print("=" * 60)
    print("Translation Integration Examples")
    print("=" * 60)

    # Example 1: Basic sign → translation
    print("\n1. Sign → Translation")
    translated = await sign_to_translated_text(
        frame=None, target_language="hi"  # Mock frame
    )
    print(f"Result: {translated}")

    # Example 2: Bidirectional communication
    print("\n2. Bidirectional Communication")
    result = await bidirectional_communication_bridge(
        person_a_input="How are you?", person_a_lang="en", person_b_lang="hi"
    )
    print(f"A says (en): {result['person_a_original']}")
    print(f"B receives (hi): {result['person_b_translation']}")

    # Example 3: Multi-language translation
    print("\n3. Multi-language Sign Recognition")
    multi_result = await multilingual_sign_recognition(
        frame=None, target_languages=["hi", "es", "fr"]
    )
    print(f"Original: {multi_result.get('original')}")
    for lang, translation in multi_result.get("translations", {}).items():
        print(f"  {lang}: {translation}")

    # Example 4: Real-time pipeline
    print("\n4. Real-time Translation Pipeline")
    pipeline = RealTimeTranslationPipeline(target_language="hi")

    # Simulate processing multiple frames
    frames = [None, None, None]  # Mock frames
    for i, frame in enumerate(frames):
        result = await pipeline.process_frame(frame)
        if result:
            print(f"  Frame {i+1}: {result.original} → {result.translated}")

    print(f"\nHistory: {len(pipeline.get_history())} translations")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
