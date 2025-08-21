#!/usr/bin/env python3
"""
Test lightweight vision models optimized for CPU performance
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

# Lightweight models optimized for speed
FAST_MODELS = {
    # Quantized LLaVA models (progressively smaller/faster)
    "llava:7b-v1.6-mistral-q4_0": {
        "desc": "4-bit quantized LLaVA (balanced)",
        "size": "~4GB",
        "speed": "medium"
    },
    "llava:7b-v1.6-mistral-q3_K_M": {
        "desc": "3-bit quantized LLaVA (faster)",
        "size": "~3GB", 
        "speed": "fast"
    },
    "llava:7b-v1.6-mistral-q2_K": {
        "desc": "2-bit quantized LLaVA (fastest)",
        "size": "~2.5GB",
        "speed": "very fast"
    },
    
    # Alternative lightweight models
    "moondream:latest": {
        "desc": "Moondream - 1.6B param vision model",
        "size": "~2GB",
        "speed": "very fast"
    },
    "llava-phi3:latest": {
        "desc": "Phi-3 based LLaVA (3.8B params)",
        "size": "~2.5GB",
        "speed": "fast"
    },
    "bakllava:7b-q4_0": {
        "desc": "BakLLaVA quantized version",
        "size": "~4GB",
        "speed": "medium"
    }
}

# Simplified test prompt for faster inference
FAST_TEST_PROMPT = """Quickly identify:
1. Screen type (home/app/error)
2. App name if visible
3. Main UI elements
Answer in 1-2 sentences."""

async def download_model(model_name: str):
    """Download model with progress tracking"""
    log.info(f"📥 Downloading {model_name}...")
    try:
        response = requests.post(
            "http://ollama:11434/api/pull",
            json={"name": model_name},
            stream=True,
            timeout=3600
        )
        
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    status = data.get('status', '')
                    
                    # Show download progress
                    if 'total' in data and 'completed' in data:
                        total = data['total']
                        completed = data['completed']
                        percent = (completed / total) * 100 if total > 0 else 0
                        log.info(f"  Progress: {percent:.1f}% - {status}")
                    else:
                        log.info(f"  {status}")
                        
                except json.JSONDecodeError:
                    pass
                    
        return True
    except Exception as e:
        log.error(f"Download failed: {e}")
        return False

async def benchmark_model(model_name: str, model_info: dict, test_images: list):
    """Benchmark a single model"""
    log.info(f"\n{'='*70}")
    log.info(f"🚀 Testing: {model_name}")
    log.info(f"   {model_info['desc']}")
    log.info(f"   Size: {model_info['size']} | Expected speed: {model_info['speed']}")
    log.info(f"{'='*70}")
    
    # Check if model exists
    try:
        response = requests.get("http://ollama:11434/api/tags")
        models = [m['name'] for m in response.json().get('models', [])]
        
        if model_name not in models:
            log.warning(f"Model not found locally. Downloading...")
            if not await download_model(model_name):
                return None
    except Exception as e:
        log.error(f"Failed to check models: {e}")
        return None
    
    # Configure client
    from src.config import settings
    original_model = settings.ollama_model
    original_timeout = settings.ollama_timeout
    
    # Use shorter timeout for fast models
    settings.ollama_model = model_name
    settings.ollama_timeout = 30  # 30 seconds max
    
    client = OllamaClient()
    
    results = {
        "model": model_name,
        "info": model_info,
        "tests": [],
        "metrics": {}
    }
    
    # Run multiple iterations for better timing
    iterations = 3
    
    for image_path in test_images:
        log.info(f"\n📸 Testing: {os.path.basename(image_path) if os.path.exists(image_path) else 'dummy image'}")
        
        times = []
        responses = []
        
        for i in range(iterations):
            try:
                # Load image
                import base64
                from PIL import Image
                import io
                
                if os.path.exists(image_path):
                    # Resize image for faster processing
                    img = Image.open(image_path)
                    # Downscale to 640x480 for speed
                    img.thumbnail((640, 480), Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG', quality=85)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                else:
                    # Small test image
                    img = Image.new('RGB', (640, 480), color='blue')
                    buffer = io.BytesIO()
                    img.save(buffer, format='JPEG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Time the analysis
                start = time.time()
                result = await client.analyze_image(image_base64, FAST_TEST_PROMPT)
                end = time.time()
                
                elapsed = end - start
                times.append(elapsed)
                
                if result and result.get('response'):
                    responses.append(result['response'])
                    log.info(f"  Run {i+1}: {elapsed:.2f}s - {result['response'][:80]}...")
                else:
                    log.error(f"  Run {i+1}: Failed")
                    
            except Exception as e:
                log.error(f"  Run {i+1}: Error - {str(e)[:50]}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            results["tests"].append({
                "image": os.path.basename(image_path) if os.path.exists(image_path) else "dummy",
                "iterations": len(times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "responses": responses
            })
            
            log.info(f"  ⏱️  Avg: {avg_time:.2f}s | Min: {min_time:.2f}s | Max: {max_time:.2f}s")
    
    # Calculate overall metrics
    if results["tests"]:
        all_times = []
        for test in results["tests"]:
            all_times.append(test["avg_time"])
        
        results["metrics"] = {
            "overall_avg": sum(all_times) / len(all_times),
            "fps_potential": 1.0 / (sum(all_times) / len(all_times)) if all_times else 0,
            "success_rate": len([t for t in results["tests"] if t["iterations"] > 0]) / len(results["tests"]) * 100
        }
        
        log.info(f"\n📊 Model Performance:")
        log.info(f"   Average speed: {results['metrics']['overall_avg']:.2f}s per frame")
        log.info(f"   Potential FPS: {results['metrics']['fps_potential']:.2f}")
        log.info(f"   Success rate: {results['metrics']['success_rate']:.0f}%")
    
    # Restore settings
    settings.ollama_model = original_model
    settings.ollama_timeout = original_timeout
    
    return results

async def main():
    """Main benchmark function"""
    log.info("🏃 Fast Model Benchmark for CPU Performance")
    log.info("=" * 70)
    
    # Check Ollama connectivity
    try:
        response = requests.get("http://ollama:11434/api/tags", timeout=5)
        if response.status_code != 200:
            log.error("Cannot connect to Ollama service")
            return False
    except Exception as e:
        log.error(f"Ollama connection failed: {e}")
        return False
    
    # Get test images
    test_images = sys.argv[1:] if len(sys.argv) > 1 else []
    if not test_images:
        log.info("No test images provided, using dummy image")
        test_images = ["dummy.jpg"]
    
    # Select models to test
    print("\n🤖 Select models to benchmark:")
    print("1. Test all fast models")
    print("2. Test quantized LLaVA only") 
    print("3. Test ultra-light models only (Moondream, Phi3)")
    print("4. Custom selection")
    
    choice = input("\nYour choice (default: 2): ").strip() or "2"
    
    if choice == "1":
        models_to_test = list(FAST_MODELS.keys())
    elif choice == "2":
        models_to_test = [m for m in FAST_MODELS.keys() if "llava" in m and "q" in m]
    elif choice == "3":
        models_to_test = ["moondream:latest", "llava-phi3:latest"]
    else:
        print("\nAvailable models:")
        for i, (model, info) in enumerate(FAST_MODELS.items(), 1):
            print(f"{i}. {model} - {info['desc']}")
        
        selected = input("Enter model numbers (comma-separated): ").strip()
        try:
            indices = [int(x.strip())-1 for x in selected.split(",")]
            models_to_test = [list(FAST_MODELS.keys())[i] for i in indices]
        except:
            models_to_test = ["llava:7b-v1.6-mistral-q4_0"]
    
    # Run benchmarks
    all_results = []
    
    for model in models_to_test:
        if model in FAST_MODELS:
            result = await benchmark_model(model, FAST_MODELS[model], test_images)
            if result:
                all_results.append(result)
    
    # Summary and recommendations
    if all_results:
        log.info("\n" + "="*70)
        log.info("🏁 BENCHMARK SUMMARY")
        log.info("="*70)
        
        # Sort by speed
        all_results.sort(key=lambda x: x["metrics"].get("overall_avg", float('inf')))
        
        for result in all_results:
            metrics = result["metrics"]
            log.info(f"\n{result['model']}")
            log.info(f"  Speed: {metrics['overall_avg']:.2f}s/frame ({metrics['fps_potential']:.2f} FPS)")
            log.info(f"  Success: {metrics['success_rate']:.0f}%")
        
        # Recommendations
        log.info("\n💡 RECOMMENDATIONS:")
        
        fastest = all_results[0]
        if fastest["metrics"]["overall_avg"] < 5:
            log.info(f"✅ {fastest['model']} is fast enough for real-time testing!")
            log.info(f"   Can process ~{fastest['metrics']['fps_potential']:.1f} frames per second")
        elif fastest["metrics"]["overall_avg"] < 10:
            log.info(f"⚡ {fastest['model']} is acceptable for testing with caching")
            log.info(f"   Consider implementing frame skipping or result caching")
        else:
            log.info(f"⚠️  Even the fastest model ({fastest['model']}) is too slow")
            log.info(f"   Consider using cloud GPU or dedicated inference server")
        
        # Save results
        with open("fast_model_benchmarks.json", "w") as f:
            json.dump(all_results, f, indent=2)
        log.info("\n💾 Detailed results saved to fast_model_benchmarks.json")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)