"""
Real YOLO Inference Implementation
Adapted from ML_Training_Platform_Reference for integration with main API
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import base64
from PIL import Image
import io
import os

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOInference:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize YOLO inference with a trained model
        
        Args:
            model_path (str): Path to the trained model weights (.pt file)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO inference requires: pip install ultralytics torch torchvision")
        
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.class_names = None
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the YOLO model with PyTorch compatibility"""
        try:
            # Handle PyTorch compatibility
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals([
                    'ultralytics.nn.tasks.DetectionModel',
                    'ultralytics.nn.tasks.SegmentationModel', 
                    'ultralytics.nn.tasks.ClassificationModel',
                    'ultralytics.nn.tasks.PoseModel',
                    'ultralytics.nn.modules.block.C2f',
                    'ultralytics.nn.modules.conv.Conv',
                    'ultralytics.nn.modules.head.Detect'
                ])
            
            # Load model
            self.model = YOLO(str(self.model_path))
            self.class_names = self.model.names
            print(f"✅ YOLO model loaded successfully: {self.model_path}")
            print(f"📋 Classes: {list(self.class_names.values())}")
            
        except Exception as e:
            # Try alternative loading method
            try:
                print(f"First attempt failed: {str(e)}")
                print("Trying alternative loading method...")
                
                # Temporarily modify torch.load behavior
                original_load = torch.load
                
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                
                # Try loading again
                self.model = YOLO(str(self.model_path))
                self.class_names = self.model.names
                
                # Restore original torch.load
                torch.load = original_load
                
                print(f"✅ YOLO model loaded with alternative method: {self.model_path}")
                
            except Exception as e2:
                raise Exception(f"Failed to load model with both methods: {str(e)} | {str(e2)}")
    
    def predict_image(self, image: np.ndarray, return_visualization: bool = False) -> Dict[str, Any]:
        """
        Run inference on a single image
        
        Args:
            image (np.ndarray): Input image in BGR format
            return_visualization (bool): Whether to return annotated image
            
        Returns:
            Dict containing detections and optional visualization
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Run inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        inference_time = time.time() - start_time
        
        # Process results
        detections = []
        annotated_image = None
        
        if results and len(results) > 0:
            result = results[0]  # First (and only) result
            
            # Extract detections
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    confidence = float(confidences[i])
                    class_id = int(class_ids[i])
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    detection = {
                        "class": class_name,
                        "class_id": class_id,
                        "confidence": round(confidence, 3),
                        "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],  # x, y, width, height
                        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)]  # x1, y1, x2, y2
                    }
                    detections.append(detection)
            
            # Create annotated image if requested
            if return_visualization and result.plot is not None:
                annotated_image = result.plot()
        
        return {
            "detections": detections,
            "inference_time": round(inference_time, 3),
            "model_path": str(self.model_path),
            "num_detections": len(detections),
            "annotated_image": annotated_image,
            "image_shape": image.shape
        }
    
    def predict_base64(self, image_base64: str, return_visualization: bool = False) -> Dict[str, Any]:
        """
        Run inference on a base64 encoded image
        
        Args:
            image_base64 (str): Base64 encoded image
            return_visualization (bool): Whether to return annotated image
            
        Returns:
            Dict containing detections and optional visualization
        """
        # Decode base64 image
        try:
            # Remove data URL prefix if present
            if "," in image_base64:
                image_base64 = image_base64.split(",")[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array (RGB -> BGR for OpenCV)
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return self.predict_image(image_np, return_visualization)
            
        except Exception as e:
            return {
                "error": f"Failed to decode base64 image: {str(e)}",
                "detections": [],
                "inference_time": 0,
                "num_detections": 0
            }
    
    def predict_file(self, image_path: str, return_visualization: bool = False) -> Dict[str, Any]:
        """
        Run inference on an image file
        
        Args:
            image_path (str): Path to image file
            return_visualization (bool): Whether to return annotated image
            
        Returns:
            Dict containing detections and optional visualization
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            return self.predict_image(image, return_visualization)
            
        except Exception as e:
            return {
                "error": f"Failed to load image file: {str(e)}",
                "detections": [],
                "inference_time": 0,
                "num_detections": 0
            }
    
    def benchmark(self, image: np.ndarray, iterations: int = 5) -> Dict[str, Any]:
        """
        Benchmark inference performance
        
        Args:
            image (np.ndarray): Test image
            iterations (int): Number of iterations
            
        Returns:
            Performance metrics
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        times = []
        detection_counts = []
        
        # Warmup
        self.predict_image(image)
        
        # Benchmark
        for i in range(iterations):
            start_time = time.time()
            result = self.predict_image(image)
            inference_time = time.time() - start_time
            
            times.append(inference_time)
            detection_counts.append(result["num_detections"])
        
        return {
            "iterations": iterations,
            "avg_inference_time": round(np.mean(times), 3),
            "min_inference_time": round(np.min(times), 3),
            "max_inference_time": round(np.max(times), 3),
            "std_inference_time": round(np.std(times), 3),
            "avg_detections": round(np.mean(detection_counts), 1),
            "fps": round(1.0 / np.mean(times), 1)
        }


class ModelManager:
    """Manage multiple YOLO models with caching"""
    
    def __init__(self):
        self.loaded_models: Dict[str, YOLOInference] = {}
        self.model_metadata: Dict[str, Dict] = {}
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available trained models"""
        models_dir = Path("training/models")
        available_models = {}
        
        if not models_dir.exists():
            return available_models
        
        for model_dir in models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                weights_file = model_dir / "weights" / "best.pt"
                
                if metadata_file.exists() and weights_file.exists():
                    try:
                        import json
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        
                        available_models[model_dir.name] = {
                            "name": metadata.get("name", model_dir.name),
                            "dataset_type": metadata.get("dataset_type", "object_detection"),
                            "dataset_name": metadata.get("dataset_name", "unknown"),
                            "created_at": metadata.get("created_at", "unknown"),
                            "status": metadata.get("status", "unknown"),
                            "weights_path": str(weights_file),
                            "has_weights": True
                        }
                        
                    except Exception as e:
                        print(f"Warning: Could not read metadata for model {model_dir.name}: {e}")
        
        return available_models
    
    def load_model(self, model_name: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> YOLOInference:
        """Load a model (with caching)"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Find model weights
        models_dir = Path("training/models")
        model_dir = models_dir / model_name
        weights_file = model_dir / "weights" / "best.pt"
        
        if not weights_file.exists():
            raise FileNotFoundError(f"Model weights not found: {weights_file}")
        
        # Load model
        model = YOLOInference(
            model_path=str(weights_file),
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Cache model
        self.loaded_models[model_name] = model
        
        return model
    
    def predict(self, model_name: str, image_data, **kwargs) -> Dict[str, Any]:
        """Run inference using a specific model"""
        try:
            model = self.load_model(model_name)
            
            # Handle different input types
            if isinstance(image_data, str):
                # Check if it's a file path first (most common case)
                if os.path.exists(image_data):
                    # It's a valid file path
                    return model.predict_file(image_data, **kwargs)
                elif image_data.startswith("data:") or len(image_data) > 100:
                    # Base64 - either has data: prefix or is a long string
                    return model.predict_base64(image_data, **kwargs)
                else:
                    # Short string that's not a file - try as file path anyway
                    return model.predict_file(image_data, **kwargs)
            elif isinstance(image_data, np.ndarray):
                # NumPy array
                return model.predict_image(image_data, **kwargs)
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
                
        except Exception as e:
            return {
                "error": str(e),
                "detections": [],
                "inference_time": 0,
                "num_detections": 0
            }


# Global model manager instance
model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    return model_manager


# Convenience functions for API integration
async def test_yolo_model(model_name: str, image_data, return_visualization: bool = False) -> Dict[str, Any]:
    """
    Test a YOLO model with given image data
    
    Args:
        model_name: Name of the trained model
        image_data: Image data (base64, file path, or numpy array)
        return_visualization: Whether to return annotated image
        
    Returns:
        Inference results
    """
    try:
        manager = get_model_manager()
        result = manager.predict(
            model_name=model_name,
            image_data=image_data,
            return_visualization=return_visualization
        )
        
        # Add model info
        result["model_name"] = model_name
        result["model_type"] = "object_detection"
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "model_name": model_name,
            "detections": [],
            "inference_time": 0,
            "num_detections": 0
        }


def list_available_models() -> Dict[str, Dict]:
    """List all available trained YOLO models"""
    manager = get_model_manager()
    return manager.get_available_models()