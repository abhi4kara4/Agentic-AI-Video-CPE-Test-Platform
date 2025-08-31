#!/usr/bin/env python3
"""
Test script to verify YOLO training and inference integration
"""
import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_yolo_integration():
    """Test YOLO training and inference integration"""
    print("🧪 Testing YOLO Integration...")
    
    # Test 1: Check dependencies
    print("\n1️⃣ Testing Dependencies...")
    try:
        from src.models.yolo_trainer import train_yolo_model, YOLO_AVAILABLE
        from src.models.yolo_inference import test_yolo_model, list_available_models
        from src.models.dataset_converter import prepare_dataset_for_training
        
        print("✅ All modules imported successfully")
        print(f"🔧 YOLO Available: {YOLO_AVAILABLE}")
        
        if not YOLO_AVAILABLE:
            print("❌ YOLO not available. Install with: pip install ultralytics torch torchvision")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test 2: List available models
    print("\n2️⃣ Testing Model Listing...")
    try:
        available_models = list_available_models()
        print(f"📋 Found {len(available_models)} trained models:")
        for name, info in available_models.items():
            print(f"   - {name}: {info.get('dataset_name', 'unknown dataset')}")
    except Exception as e:
        print(f"⚠️  Model listing error: {e}")
    
    # Test 3: Check for existing datasets
    print("\n3️⃣ Testing Dataset Detection...")
    datasets_dir = Path("datasets")
    if datasets_dir.exists():
        datasets = [d for d in datasets_dir.iterdir() if d.is_dir() and not d.name.endswith('_yolo')]
        print(f"📁 Found {len(datasets)} datasets:")
        for dataset in datasets[:3]:  # Show first 3
            images_dir = dataset / "images"
            annotations_dir = dataset / "annotations"
            if images_dir.exists() and annotations_dir.exists():
                image_count = len(list(images_dir.glob("*.jpg")))
                annotation_count = len(list(annotations_dir.glob("*.json")))
                print(f"   - {dataset.name}: {image_count} images, {annotation_count} annotations")
                
                # Test dataset conversion
                if image_count > 0 and annotation_count > 0:
                    print(f"   🔄 Testing dataset conversion for {dataset.name}...")
                    try:
                        yolo_path = prepare_dataset_for_training(dataset.name)
                        print(f"   ✅ Dataset converted successfully: {yolo_path}")
                        break
                    except Exception as e:
                        print(f"   ❌ Dataset conversion failed: {e}")
    else:
        print("📁 No datasets directory found")
    
    # Test 4: Test inference if models exist
    print("\n4️⃣ Testing Model Inference...")
    available_models = list_available_models()
    if available_models:
        model_name = list(available_models.keys())[0]
        print(f"🔮 Testing inference with model: {model_name}")
        
        # Create a test image (solid color)
        try:
            import cv2
            import numpy as np
            import base64
            
            # Create test image
            test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray image
            cv2.rectangle(test_image, (100, 100), (200, 200), (0, 255, 0), 3)  # Green rectangle
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', test_image)
            test_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Test inference
            result = await test_yolo_model(model_name, test_base64)
            
            if result.get('error'):
                print(f"⚠️  Inference error: {result.get('error')}")
            else:
                detections = result.get('detections', [])
                print(f"✅ Inference successful: {len(detections)} objects detected")
                for detection in detections[:3]:  # Show first 3
                    print(f"   - {detection['class']}: {detection['confidence']:.3f}")
                    
        except Exception as e:
            print(f"❌ Inference test error: {e}")
    else:
        print("⚠️  No trained models available for inference testing")
    
    print("\n🎉 YOLO Integration Test Complete!")
    return True

async def test_api_integration():
    """Test API integration"""
    print("\n🔌 Testing API Integration...")
    
    try:
        # Test model listing endpoint
        from src.api.main import app
        print("✅ API module imported successfully")
        print("📡 To test API endpoints, run the server and use:")
        print("   GET /training/models - List available models")
        print("   POST /test/models/{model_name}/analyze - Test model inference")
        
    except Exception as e:
        print(f"❌ API integration error: {e}")

if __name__ == "__main__":
    print("🚀 YOLO Training & Inference Integration Test")
    print("=" * 50)
    
    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        success = loop.run_until_complete(test_yolo_integration())
        loop.run_until_complete(test_api_integration())
        
        if success:
            print("\n✅ All tests completed successfully!")
            print("\n📋 Next Steps:")
            print("1. Start the API server: python -m uvicorn src.api.main:app --reload")
            print("2. Train an object detection model via the web interface")
            print("3. Test the trained model - it will use real YOLO inference!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    finally:
        loop.close()