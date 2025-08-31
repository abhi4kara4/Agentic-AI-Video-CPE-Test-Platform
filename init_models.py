#!/usr/bin/env python3
"""
Initialize models directory in container
This ensures the models are available even after volume mount
"""
import os
import shutil
from pathlib import Path

def ensure_models_directory():
    """Ensure models directory and files exist"""
    models_dir = Path("/app/src/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py
    init_file = models_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Models package\n")
    
    # Check if we have model files in a backup location
    backup_models = Path("/app/models_backup")
    if backup_models.exists():
        print("Restoring models from backup...")
        for file in backup_models.glob("*.py"):
            dest = models_dir / file.name
            if not dest.exists():
                shutil.copy2(file, dest)
                print(f"Restored {file.name}")
    
    # If yolo_trainer.py doesn't exist, create a basic one
    trainer_file = models_dir / "yolo_trainer.py"
    if not trainer_file.exists():
        print("Creating temporary yolo_trainer.py...")
        trainer_file.write_text('''"""
Temporary YOLO Training Module
Replace this with the real implementation
"""
import asyncio
from typing import Dict, Any
from pathlib import Path

async def train_yolo_model(dataset_name: str, model_name: str, training_config: Dict[str, Any]) -> Dict[str, Any]:
    """Temporary YOLO training implementation"""
    print(f"YOLO Training - Dataset: {dataset_name}, Model: {model_name}")
    print(f"Config: {training_config}")
    
    # Simulate training for 5 seconds
    await asyncio.sleep(5)
    
    return {
        "success": True,
        "model_path": f"training/models/{model_name}.pt",
        "epochs_completed": training_config.get("epochs", 10),
        "message": "Training completed (temporary implementation)"
    }
''')
    
    # List contents
    print("\nModels directory contents:")
    for file in sorted(models_dir.iterdir()):
        print(f"  - {file.name} ({file.stat().st_size} bytes)")

if __name__ == "__main__":
    ensure_models_directory()
    print("\nModels initialization complete!")