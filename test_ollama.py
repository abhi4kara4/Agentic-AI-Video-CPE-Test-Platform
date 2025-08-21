#!/usr/bin/env python3
"""
Test script to check if Ollama and LLaVA are working properly
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.ollama_client import OllamaClient
from src.utils.logger import log
import requests

async def test_ollama():
    """Test Ollama connectivity and model availability"""
    log.info("🔍 Testing Ollama connectivity...")
    
    try:
        # Test basic connectivity
        response = requests.get("http://ollama:11434/api/tags", timeout=10)
        if response.status_code == 200:
            log.info("✓ Ollama service is accessible")
            models = response.json().get('models', [])
            log.info(f"Available models: {[m['name'] for m in models]}")
        else:
            log.error(f"✗ Ollama service error: {response.status_code}")
            return False
    except Exception as e:
        log.error(f"✗ Cannot connect to Ollama: {e}")
        return False
    
    # Test OllamaClient
    log.info("🧠 Testing Ollama client...")
    client = OllamaClient()
    
    # Check if model is available
    model_available = await client.check_model()
    if not model_available:
        log.warning("LLaVA model not found, attempting to pull...")
        if await client.pull_model():
            log.info("✓ LLaVA model pulled successfully")
        else:
            log.error("✗ Failed to pull LLaVA model")
            return False
    else:
        log.info("✓ LLaVA model is available")
    
    # Test simple image analysis with a minimal test image
    log.info("🖼️ Testing image analysis with dummy data...")
    try:
        # Create a simple 1x1 white pixel image in base64
        import base64
        from PIL import Image
        import io
        
        # Create a tiny test image
        test_image = Image.new('RGB', (1, 1), color='white')
        buffer = io.BytesIO()
        test_image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        log.info("Sending test image to LLaVA (this may take 10-60 seconds)...")
        result = await client.analyze_image(image_base64, "What do you see in this image?")
        
        if result and result.get('response'):
            log.info(f"✓ Image analysis working: {result['response'][:100]}...")
            log.info(f"Analysis took: {result.get('total_duration', 0):.1f} seconds")
        else:
            log.error("✗ Image analysis failed - no response")
            return False
    except Exception as e:
        log.error(f"✗ Image analysis error: {e}")
        return False
    
    log.info("🎉 Ollama tests completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    if success:
        print("\n✅ Ollama is working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Ollama has issues!")
        sys.exit(1)