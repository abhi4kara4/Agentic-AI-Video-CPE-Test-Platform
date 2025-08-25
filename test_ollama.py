#!/usr/bin/env python3
"""
Test script to check if Ollama and LLaVA are working properly
Supports testing different models and measuring performance
"""
import asyncio
import sys
import os
import time
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.agent.ollama_client import OllamaClient
from src.utils.logger import log
import requests

# Common smaller/faster models to test
MODELS_TO_TEST = {
    "llava:7b": "Default LLaVA 7B model",
    "llava:7b-v1.6-mistral-q4_0": "Quantized 4-bit version (faster, less accurate)",
    "llava:7b-v1.6-mistral-q2_K": "Heavily quantized 2-bit (fastest, least accurate)", 
    "bakllava:7b": "BakLLaVA - alternative vision model",
    "llava:13b": "Larger LLaVA model (slower, more accurate)",
    "llava-phi3": "Phi-3 based LLaVA (compact model)"
}

async def pull_model(model_name: str):
    """Pull a model if not available"""
    log.info(f"📥 Pulling model {model_name}...")
    try:
        async with requests.post(
            "http://ollama:11434/api/pull",
            json={"name": model_name},
            stream=True
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    if 'status' in data:
                        log.info(f"  {data['status']}")
        return True
    except Exception as e:
        log.error(f"Failed to pull model: {e}")
        return False

async def test_model_performance(model_name: str, test_images: list):
    """Test a specific model's performance"""
    log.info(f"\n{'='*60}")
    log.info(f"🧪 Testing model: {model_name}")
    log.info(f"   Description: {MODELS_TO_TEST.get(model_name, 'Custom model')}")
    log.info(f"{'='*60}")
    
    # Create custom client for this model
    from src.config import settings
    original_model = settings.ollama_model
    settings.ollama_model = model_name
    
    client = OllamaClient()
    
    # Check if model exists
    if not await client.check_model():
        log.warning(f"Model {model_name} not found. Attempting to pull...")
        if not await client.pull_model():
            log.error(f"Failed to pull model {model_name}")
            settings.ollama_model = original_model
            return None
    
    results = {
        "model": model_name,
        "tests": []
    }
    
    for image_path in test_images:
        log.info(f"\n📸 Testing with image: {image_path}")
        
        try:
            # Load image
            import base64
            from PIL import Image
            import io
            
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
                image_name = os.path.basename(image_path)
            else:
                # Create dummy image
                test_image = Image.new('RGB', (1920, 1080), color='white')
                buffer = io.BytesIO()
                test_image.save(buffer, format='PNG')
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                image_name = "dummy_1920x1080.png"
            
            # Test with structured prompt
            prompt = """Analyze this TV/STB screen and identify:
1. Screen type (home, app_rail, app_content, error, etc.)
2. Current app name if visible
3. Any text visible on screen
4. UI elements present (buttons, menus, etc.)
5. Any anomalies (black screen, buffering, errors)

Respond in JSON format."""
            
            start_time = time.time()
            result = await client.analyze_image(image_base64, prompt)
            end_time = time.time()
            
            analysis_time = end_time - start_time
            
            if result and result.get('response'):
                log.info(f"✓ Analysis completed in {analysis_time:.1f} seconds")
                log.info(f"Response preview: {result['response'][:200]}...")
                
                results["tests"].append({
                    "image": image_name,
                    "success": True,
                    "time": analysis_time,
                    "response": result['response']
                })
            else:
                log.error(f"✗ Analysis failed")
                results["tests"].append({
                    "image": image_name,
                    "success": False,
                    "time": analysis_time,
                    "error": "No response"
                })
                
        except Exception as e:
            log.error(f"✗ Test failed: {e}")
            results["tests"].append({
                "image": image_name,
                "success": False,
                "error": str(e)
            })
    
    # Calculate average time
    successful_tests = [t for t in results["tests"] if t.get("success")]
    if successful_tests:
        avg_time = sum(t["time"] for t in successful_tests) / len(successful_tests)
        results["average_time"] = avg_time
        log.info(f"\n📊 Average analysis time: {avg_time:.1f} seconds")
    
    settings.ollama_model = original_model
    return results

async def test_ollama():
    """Main test function"""
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
    
    # Get test images from command line or use defaults
    test_images = sys.argv[1:] if len(sys.argv) > 1 else ["dummy_image.png"]
    
    # Ask user which models to test
    print("\n🤖 Available models to test:")
    for i, (model, desc) in enumerate(MODELS_TO_TEST.items(), 1):
        print(f"{i}. {model} - {desc}")
    print(f"{len(MODELS_TO_TEST)+1}. Test all models")
    print(f"{len(MODELS_TO_TEST)+2}. Test only current model (llava:7b)")
    
    try:
        choice = input("\nSelect option (default: test current model): ").strip()
        if not choice or choice == str(len(MODELS_TO_TEST)+2):
            models_to_test = ["llava:7b"]
        elif choice == str(len(MODELS_TO_TEST)+1):
            models_to_test = list(MODELS_TO_TEST.keys())
        else:
            idx = int(choice) - 1
            models_to_test = [list(MODELS_TO_TEST.keys())[idx]]
    except:
        models_to_test = ["llava:7b"]
    
    # Test selected models
    all_results = []
    for model in models_to_test:
        result = await test_model_performance(model, test_images)
        if result:
            all_results.append(result)
    
    # Summary
    if all_results:
        log.info("\n" + "="*60)
        log.info("📊 PERFORMANCE SUMMARY")
        log.info("="*60)
        
        for result in all_results:
            avg_time = result.get("average_time", 0)
            log.info(f"{result['model']:30s} - Avg: {avg_time:5.1f}s")
        
        # Find fastest model
        fastest = min(all_results, key=lambda x: x.get("average_time", float('inf')))
        log.info(f"\n🏆 Fastest model: {fastest['model']} ({fastest['average_time']:.1f}s average)")
        
        # Save results to reports directory (accessible from host)
        report_path = "reports/model_performance_results.json"
        os.makedirs("reports", exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log.info(f"\n💾 Results saved to {report_path}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_ollama())
    if success:
        print("\n✅ Ollama tests completed!")
        sys.exit(0)
    else:
        print("\n❌ Ollama tests failed!")
        sys.exit(1)