"""
Real YOLO Training Implementation
Adapted from ML_Training_Platform_Reference for integration with main API
"""
import os
import yaml
import json
import asyncio
from typing import Dict, Any, Callable, Optional
from pathlib import Path
import time
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class YOLOTrainer:
    def __init__(self, dataset_path: str, model_name: str = 'yolov8n.pt', 
                 output_dir: str = 'training/models', project_name: str = 'custom_training'):
        """
        Initialize YOLO trainer for real model training
        
        Args:
            dataset_path (str): Path to dataset directory (must have YOLO format)
            model_name (str): Base model name or path
            output_dir (str): Output directory for trained models
            project_name (str): Project name for organizing runs
        """
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO training requires: pip install ultralytics torch torchvision")
        
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.project_name = project_name
        self.model = None
        self.training_results = None
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset(self) -> bool:
        """Validate dataset structure and files for YOLO training"""
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            raise Exception(f"Dataset YAML not found: {yaml_path}")
        
        # Load and validate YAML content
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        required_keys = ['nc', 'names']
        for key in required_keys:
            if key not in data_config:
                raise Exception(f"Missing required key in data.yaml: {key}")
        
        # Check dataset path in YAML
        dataset_root = data_config.get('path', str(self.dataset_path))
        if not os.path.isabs(dataset_root):
            dataset_root = os.path.join(self.dataset_path, dataset_root)
        
        # Update the YAML file to use absolute paths if needed
        if data_config.get('path') != str(self.dataset_path.absolute()):
            print(f"Updating dataset path in data.yaml to: {self.dataset_path.absolute()}")
            data_config['path'] = str(self.dataset_path.absolute())
            with open(yaml_path, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
        
        # Check if train and val directories exist
        train_images = self.dataset_path / 'images' / 'train'
        val_images = self.dataset_path / 'images' / 'val'
        
        if not train_images.exists():
            raise Exception(f"Training images directory not found: {train_images}")
        
        if not val_images.exists():
            raise Exception(f"Validation images directory not found: {val_images}")
        
        # Count images
        train_count = len([f for f in train_images.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        val_count = len([f for f in val_images.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        if train_count == 0:
            raise Exception(f"No training images found in: {train_images}")
        
        if val_count == 0:
            raise Exception(f"No validation images found in: {val_images}")
        
        print(f"Dataset validated: {data_config['nc']} classes, {len(data_config['names'])} class names")
        print(f"Training images: {train_count}, Validation images: {val_count}")
        print(f"Dataset path: {self.dataset_path.absolute()}")
        
        return True
    
    def initialize_model(self):
        """Initialize YOLO model with PyTorch compatibility"""
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
            
            # Check if it's a custom model path
            if os.path.exists(self.model_name):
                self.model = YOLO(self.model_name)
                print(f"Loaded custom model: {self.model_name}")
            else:
                # Use pretrained model
                self.model = YOLO(self.model_name)
                print(f"Loaded pretrained model: {self.model_name}")
                
        except Exception as e:
            # If the above fails, try alternative loading method
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
                self.model = YOLO(self.model_name)
                
                # Restore original torch.load
                torch.load = original_load
                
                print(f"Successfully loaded model with alternative method: {self.model_name}")
                
            except Exception as e2:
                raise Exception(f"Failed to initialize model with both methods: {str(e)} | {str(e2)}")
    
    async def train_async(self, epochs: int = 50, batch_size: int = 16, img_size: int = 640, 
                          learning_rate: float = 0.01, device: str = 'auto', workers: int = 8,
                          patience: int = 50, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Async wrapper for YOLO training with progress updates
        """
        # Run training in thread pool to avoid blocking
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, 
                self._train_sync, 
                epochs, batch_size, img_size, learning_rate, device, workers, patience, progress_callback
            )
        return result
    
    def _train_sync(self, epochs: int, batch_size: int, img_size: int, learning_rate: float,
                    device: str, workers: int, patience: int, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """
        Synchronous YOLO training implementation with progress tracking
        """
        try:
            print(f"Starting YOLO training with {epochs} epochs...")
            
            # Initialize model if not already done
            if self.model is None:
                self.initialize_model()
            
            # Validate dataset
            self.validate_dataset()
            
            # Smart device selection
            if device == 'auto':
                if torch.cuda.is_available():
                    device = '0'  # Use first CUDA device
                    print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    device = 'cpu'
                    print("💻 Using CPU (CUDA not available)")
            
            # Aggressive CPU optimizations to prevent hanging
            if device == 'cpu':
                actual_workers = 0  # Use single-threaded for CPU to avoid hanging
                actual_batch_size = min(batch_size, 4)  # Very small batch size for CPU
                # Force smaller image size for CPU training
                if img_size > 640:
                    img_size = 640
                    print(f"🔧 Reduced image size to {img_size} for CPU training")
                print(f"⚡ CPU training optimizations: batch_size={actual_batch_size}, workers={actual_workers}, img_size={img_size}")
                print("⚠️  CPU training will be slow. Consider using a smaller model like yolo11n for faster training.")
            else:
                actual_workers = workers
                actual_batch_size = batch_size
            
            # Progress tracking via output monitoring
            import threading
            import sys
            import io
            import re
            import time
            
            # Create a progress tracker that monitors training output
            class OutputProgressTracker:
                def __init__(self, callback, total_epochs):
                    self.callback = callback
                    self.total_epochs = total_epochs
                    self.current_epoch = 0
                    self.is_running = True
                    
                def parse_progress_from_output(self, output_line):
                    """Parse YOLO training output to extract epoch information"""
                    try:
                        # Look for patterns like "1/100" or "Epoch 1/100"
                        epoch_pattern = r'(\d+)/(\d+)'
                        matches = re.findall(epoch_pattern, output_line)
                        if matches:
                            current, total = matches[0]
                            current_epoch = int(current)
                            total_epochs = int(total)
                            
                            if current_epoch != self.current_epoch and current_epoch <= self.total_epochs:
                                self.current_epoch = current_epoch
                                
                                # Try to extract metrics from the line
                                metrics = {}
                                
                                # Look for loss values
                                loss_pattern = r'(\d+\.\d+)'
                                losses = re.findall(loss_pattern, output_line)
                                if len(losses) >= 3:  # box_loss, cls_loss, dfl_loss typically
                                    metrics['train_loss'] = float(losses[0])
                                
                                # Call progress callback
                                if self.callback:
                                    try:
                                        if asyncio.iscoroutinefunction(self.callback):
                                            # Create new event loop for sync context
                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                            loop.run_until_complete(self.callback(self.current_epoch, self.total_epochs, metrics))
                                            loop.close()
                                        else:
                                            self.callback(self.current_epoch, self.total_epochs, metrics)
                                    except Exception as e:
                                        print(f"Progress callback error: {e}")
                                
                                return True
                    except Exception as e:
                        pass
                    return False
                
                def stop(self):
                    self.is_running = False
            
            # Create progress tracker
            progress_tracker = OutputProgressTracker(progress_callback, epochs)
            
            # Capture training output to monitor progress
            class OutputCapture:
                def __init__(self, progress_tracker):
                    self.progress_tracker = progress_tracker
                    self.original_stdout = sys.stdout
                    self.original_stderr = sys.stderr
                    
                def write(self, text):
                    # Write to original output
                    self.original_stdout.write(text)
                    self.original_stdout.flush()
                    
                    # Parse for progress
                    if self.progress_tracker.is_running:
                        self.progress_tracker.parse_progress_from_output(text)
                
                def flush(self):
                    self.original_stdout.flush()
                
                def start_capture(self):
                    sys.stdout = self
                    # Don't capture stderr to avoid breaking error reporting
                
                def stop_capture(self):
                    sys.stdout = self.original_stdout
            
            # Start output capture for progress tracking
            output_capture = OutputCapture(progress_tracker)
            output_capture.start_capture()
            
            # Prepare training arguments
            train_args = {
                'data': str(self.dataset_path / 'data.yaml'),
                'epochs': epochs,
                'batch': actual_batch_size,
                'imgsz': img_size,
                'lr0': learning_rate,
                'device': device,
                'workers': actual_workers,
                'patience': patience,
                'project': str(self.output_dir),
                'name': self.project_name,
                'exist_ok': True,
                'verbose': True,
                'seed': 42,
                'val': True,
                'save': True,
                'cache': False,
                'resume': False
            }
            
            print(f"Training arguments: {train_args}")
            
            try:
                # Start training
                start_time = time.time()
                results = self.model.train(**train_args)
                end_time = time.time()
            finally:
                # Always restore output capture and stop progress tracking
                output_capture.stop_capture()
                progress_tracker.stop()
            
            # Get training results
            training_time = end_time - start_time
            best_model_path = results.save_dir / 'weights' / 'best.pt'
            last_model_path = results.save_dir / 'weights' / 'last.pt'
            
            # Extract metrics from results
            metrics = {
                'training_time': training_time,
                'epochs_completed': progress_tracker.current_epoch,
                'best_model_path': str(best_model_path),
                'last_model_path': str(last_model_path),
                'results_dir': str(results.save_dir),
            }
            
            # Try to get final metrics if available
            if hasattr(results, 'results_dict'):
                metrics.update(results.results_dict)
            
            self.training_results = metrics
            
            print(f"✅ Training completed successfully in {training_time:.2f}s")
            print(f"📁 Results saved to: {results.save_dir}")
            print(f"🏆 Best model: {best_model_path}")
            print(f"📊 Last model: {last_model_path}")
            
            return metrics
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                'error': error_msg,
                'success': False,
                'training_time': 0
            }
    
    def save_model_metadata(self, model_name: str, training_results: Dict[str, Any], 
                           dataset_name: str, dataset_type: str = "object_detection") -> str:
        """
        Save trained model metadata for API integration
        """
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            "name": model_name,
            "base_model": self.model_name,
            "dataset_type": dataset_type,
            "dataset_name": dataset_name,
            "created_at": datetime.now().isoformat(),
            "training_results": training_results,
            "model_files": {
                "best_weights": training_results.get('best_model_path'),
                "last_weights": training_results.get('last_model_path'),
                "results_dir": training_results.get('results_dir')
            },
            "status": "completed" if not training_results.get('error') else "failed"
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Model metadata saved to: {metadata_path}")
        return str(metadata_path)


def create_yolo_dataset_yaml(dataset_path: str, class_names: list, train_split: float = 0.8) -> str:
    """
    Create a YOLO-format data.yaml file for the dataset
    """
    dataset_path = Path(dataset_path)
    yaml_content = {
        'path': str(dataset_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"📄 YOLO dataset YAML created: {yaml_path}")
    return str(yaml_path)


# Integration helper for API
async def train_yolo_model(dataset_name: str, model_name: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    High-level function to train YOLO model for API integration
    """
    try:
        print(f"\nStarting REAL YOLO training:")
        print(f"  Dataset: {dataset_name}")
        print(f"  Model: {model_name}")
        print(f"  Config: {training_config}")
        
        # Setup paths - check multiple possible locations
        datasets_dir = Path("datasets")
        if not datasets_dir.exists():
            datasets_dir = Path("/app/datasets")
        
        dataset_path = datasets_dir / dataset_name
        print(f"Looking for dataset at: {dataset_path}")
        
        # Check if dataset exists as directory or ZIP file
        zip_locations = [
            Path(f"/app/training/{dataset_name}_training_yolo.zip"),  # Correct container path
            Path(f"/training/{dataset_name}_training_yolo.zip"),     # Fallback
            dataset_path / f"{dataset_name}.zip",
            dataset_path.with_suffix('.zip')
        ]
        
        found_zip = None
        for zip_path in zip_locations:
            print(f"Checking for ZIP at: {zip_path}")
            if zip_path.exists():
                found_zip = zip_path
                print(f"Found ZIP file: {zip_path}")
                break
        
        # If ZIP found, extract it to dataset directory
        if found_zip:
            print(f"Extracting ZIP: {found_zip}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            import zipfile
            with zipfile.ZipFile(found_zip, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
            print(f"Extracted to: {dataset_path}")
        
        # Check if dataset directory exists now
        if not dataset_path.exists():
            raise Exception(f"Dataset not found at {dataset_path} and no ZIP found at {zip_locations}")
        
        # Handle additional ZIP files in dataset directory
        print(f"Dataset directory contents:")
        for item in dataset_path.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Extract any additional ZIP files in dataset directory
        zip_files = list(dataset_path.glob("*.zip"))
        if zip_files:
            print(f"Found additional ZIP files: {[z.name for z in zip_files]}")
            import zipfile
            for zip_file in zip_files:
                print(f"Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
                print(f"Extracted {zip_file.name}")
        
        # Check and create YOLO format structure
        images_train = dataset_path / "images" / "train"
        images_val = dataset_path / "images" / "val"
        labels_train = dataset_path / "labels" / "train"
        labels_val = dataset_path / "labels" / "val"
        
        print(f"Checking YOLO structure:")
        print(f"  images/train: {images_train.exists()}")
        print(f"  images/val: {images_val.exists()}")
        print(f"  labels/train: {labels_train.exists()}")
        print(f"  labels/val: {labels_val.exists()}")
        
        # Create validation set if missing
        if not images_val.exists() and images_train.exists():
            print("Creating validation set...")
            import shutil
            images_val.mkdir(parents=True, exist_ok=True)
            labels_val.mkdir(parents=True, exist_ok=True)
            
            train_images = list(images_train.glob("*"))
            val_count = max(1, len(train_images) // 5)
            
            for img_file in train_images[:val_count]:
                shutil.copy2(img_file, images_val / img_file.name)
                label_file = labels_train / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy2(label_file, labels_val / label_file.name)
            print(f"Created validation set with {val_count} images")
        
        # Create data.yaml
        yaml_file = dataset_path / "data.yaml"
        if not yaml_file.exists():
            print(f"Creating data.yaml for {dataset_name}")
            yaml_content = {
                'path': str(dataset_path.absolute()),
                'train': 'images/train',
                'val': 'images/val',
                'nc': 4,
                'names': ['ui_element', 'button', 'text', 'image']
            }
            with open(yaml_file, 'w') as f:
                yaml.dump(yaml_content, f)
        
        # Extract training parameters
        epochs = training_config.get('epochs', 50)
        batch_size = training_config.get('batch_size', 16)
        learning_rate = training_config.get('learning_rate', 0.01)
        image_size = training_config.get('image_size', 640)
        patience = training_config.get('patience', 50)
        device = training_config.get('device', 'auto')
        
        # Get base model from training config
        selected_model = training_config.get('base_model', 'yolo11n')
        if not selected_model.endswith('.pt'):
            selected_model += '.pt'
        
        print(f"Using base model: {selected_model}")
        
        # Initialize trainer
        trainer = YOLOTrainer(
            dataset_path=str(dataset_path),
            model_name=selected_model,
            output_dir="training/models",
            project_name=model_name
        )
        
        # Start training with progress callback
        print(f"Starting YOLO training for model: {model_name}")
        
        # Create progress callback function that updates job metadata
        def create_progress_callback(job_dir, job_metadata, dataset_name, model_name):
            async def progress_callback(epoch, total_epochs, metrics=None):
                try:
                    print(f"Training progress: Epoch {epoch}/{total_epochs}")
                    
                    # Update job metadata with current progress
                    percentage = round((epoch / total_epochs) * 100, 1) if total_epochs > 0 else 0
                    
                    # Re-read current metadata in case other updates occurred
                    metadata_file = job_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file) as f:
                            current_metadata = json.load(f)
                    else:
                        current_metadata = job_metadata.copy()
                    
                    # Update progress
                    current_metadata["progress"] = {
                        "current_epoch": epoch,
                        "total_epochs": total_epochs,
                        "percentage": percentage,
                        "loss": metrics.get("train_loss", None) if metrics else None,
                        "mAP": metrics.get("mAP50", None) if metrics else None
                    }
                    current_metadata["status"] = "training"
                    current_metadata["updated_at"] = datetime.now().isoformat()
                    
                    # Save updated metadata
                    with open(metadata_file, "w") as f:
                        json.dump(current_metadata, f, indent=2)
                    
                    print(f"  Updated progress: {percentage}% ({epoch}/{total_epochs} epochs)")
                    if metrics:
                        print(f"  Metrics: {metrics}")
                        
                except Exception as e:
                    print(f"Progress callback error: {e}")
            
            return progress_callback
        
        # Get the job directory and metadata for progress updates
        # This is a bit hacky but necessary to access the job context
        import os
        import tempfile
        job_context_file = Path(tempfile.gettempdir()) / f"yolo_job_context_{model_name}.json"
        
        # Check if job context was saved by main API
        job_dir = None
        job_metadata = {}
        if job_context_file.exists():
            try:
                with open(job_context_file) as f:
                    context = json.load(f)
                    job_dir = Path(context["job_dir"])
                    job_metadata = context["job_metadata"]
            except:
                pass
        
        # Create progress callback
        if job_dir:
            progress_callback = create_progress_callback(job_dir, job_metadata, dataset_name, model_name)
        else:
            # Fallback progress callback
            async def progress_callback(epoch, total_epochs, metrics=None):
                print(f"Training progress: Epoch {epoch}/{total_epochs}")
                if metrics:
                    print(f"  Metrics: {metrics}")
        
        training_results = await trainer.train_async(
            epochs=epochs,
            batch_size=batch_size,
            img_size=image_size,
            learning_rate=learning_rate,
            device=device,
            patience=patience,
            progress_callback=progress_callback
        )
        
        # Clean up context file
        if job_context_file.exists():
            try:
                job_context_file.unlink()
            except:
                pass
        
        # Save metadata
        if not training_results.get('error'):
            trainer.save_model_metadata(
                model_name=model_name,
                training_results=training_results,
                dataset_name=dataset_name,
                dataset_type="object_detection"
            )
        
        return training_results
        
    except Exception as e:
        return {
            'error': str(e),
            'success': False,
            'training_time': 0
        }