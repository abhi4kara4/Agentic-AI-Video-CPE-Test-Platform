"""
YOLO import helper - safe version without emojis
"""
import os
import importlib.util
from src.utils.logger import log


def get_yolo_trainer():
    """Import YOLO trainer with robust error handling and detailed logging"""
    train_yolo_model = None
    
    # Method 1: Direct file import (most reliable for Docker)
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        trainer_path = os.path.join(src_dir, 'models', 'yolo_trainer.py')
        
        log.info(f"Attempting YOLO import from: {trainer_path}")
        log.info(f"File exists: {os.path.exists(trainer_path)}")
        log.info(f"Working dir: {os.getcwd()}")
        log.info(f"Script dir: {current_dir}")
        log.info(f"Src dir: {src_dir}")
        
        if os.path.exists(trainer_path):
            spec = importlib.util.spec_from_file_location("yolo_trainer", trainer_path)
            yolo_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolo_trainer)
            train_yolo_model = yolo_trainer.train_yolo_model
            log.info("Successfully imported YOLO trainer via direct file import")
            return train_yolo_model
        else:
            log.error(f"YOLO trainer file not found at: {trainer_path}")
            
            # Debug: List what files ARE available
            models_dir = os.path.join(src_dir, 'models')
            if os.path.exists(models_dir):
                available_files = os.listdir(models_dir)
                log.error(f"Available files in models dir: {available_files}")
            else:
                log.error(f"Models directory not found: {models_dir}")
            
            # Check src directory contents
            if os.path.exists(src_dir):
                src_contents = os.listdir(src_dir)
                log.error(f"Contents of src dir: {src_contents}")
    
    except Exception as e:
        log.error(f"Direct file import failed: {e}")
    
    # Method 2: Standard import (fallback)
    if not train_yolo_model:
        try:
            from src.models.yolo_trainer import train_yolo_model
            log.info("Successfully imported YOLO trainer via standard import")
            return train_yolo_model
        except ImportError as e:
            log.error(f"Standard import failed: {e}")
    
    raise ImportError("Failed to import YOLO trainer. Check /debug/files and /debug/import-test endpoints for details")