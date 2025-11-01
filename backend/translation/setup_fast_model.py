#!/usr/bin/env python3
"""
Setup script to download and configure fast translation model.
Usage: python translation/setup_fast_model.py
"""
import subprocess
import sys
import time

RECOMMENDED_MODEL = "gemma2:2b"  # Only ~1.6GB, very fast
FALLBACK_MODEL = "qwen2.5:3b"  # ~2GB, good alternative


def check_ollama_running():
    """Check if Ollama is running."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def pull_model(model_name: str) -> bool:
    """Pull Ollama model."""
    print(f"📥 Pulling {model_name}... (this may take a few minutes)")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name], capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"✅ Successfully pulled {model_name}")
            return True
        else:
            print(f"❌ Failed to pull {model_name}: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Download from: https://ollama.ai/download")
        return False
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout pulling {model_name}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_translation(model_name: str):
    """Test translation with the model."""
    print(f"\n🧪 Testing translation with {model_name}...")
    try:
        import httpx

        response = httpx.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Translate 'Hello, how are you?' to Hindi. Respond ONLY with the translation:\n\nTranslation:",
                "stream": False,
                "options": {
                    "num_ctx": 512,
                    "temperature": 0.2,
                    "num_predict": 50,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            translation = result.get("response", "").strip()
            print("   Original: Hello, how are you?")
            print(f"   Translation: {translation}")
            print("✅ Translation test passed!")
            return True
        else:
            print(f"❌ Test failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def main():
    print("🚀 Setting up fast translation model for a11y-bridge\n")

    # Check if Ollama is running
    if not check_ollama_running():
        print("⚠️  Ollama doesn't seem to be running.")
        print("   Please start Ollama and try again.")
        print("   Run: ollama serve")
        sys.exit(1)

    print("✅ Ollama is running\n")

    # Try to pull recommended model
    success = pull_model(RECOMMENDED_MODEL)

    if not success:
        print(f"\n⚠️  Trying fallback model {FALLBACK_MODEL}...")
        success = pull_model(FALLBACK_MODEL)
        model = FALLBACK_MODEL if success else None
    else:
        model = RECOMMENDED_MODEL

    if not success:
        print("\n❌ Failed to setup translation model.")
        print("   Please check your internet connection and try again.")
        sys.exit(1)

    # Test the model
    print()
    time.sleep(1)  # Give model time to load
    test_translation(model)

    print("\n✅ Setup complete!")
    print(f"   Model: {model}")
    print("   Size: ~1.6GB (fast!)")
    print("\n💡 To use this model, it's already configured as default.")
    print(f"   Or set environment variable: OLLAMA_MODEL={model}")


if __name__ == "__main__":
    main()
