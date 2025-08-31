#!/usr/bin/env python3
"""
Quick verification script to check if all required files exist
Run this before committing to ensure all files are in place
"""
import os
from pathlib import Path

def check_file(path, description):
    """Check if a file exists and show its status"""
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"✅ {description}: {path} ({size} bytes)")
        return True
    else:
        print(f"❌ {description}: {path} - MISSING")
        return False

def main():
    print("🔍 Verifying YOLO training files...")
    
    base_dir = Path(__file__).parent
    src_dir = base_dir / "src"
    models_dir = src_dir / "models"
    
    all_good = True
    
    # Check main files
    files_to_check = [
        (models_dir / "yolo_trainer.py", "YOLO Trainer"),
        (models_dir / "yolo_inference.py", "YOLO Inference"),
        (models_dir / "dataset_converter.py", "Dataset Converter"),
        (models_dir / "__init__.py", "Models __init__.py"),
        (src_dir / "utils" / "image_augmentation.py", "Image Augmentation"),
        (src_dir / "api" / "main.py", "Main API"),
        (base_dir / "requirements.txt", "Requirements"),
        (base_dir / "docker-compose.yml", "Docker Compose"),
        (base_dir / "Dockerfile", "Dockerfile")
    ]
    
    for file_path, description in files_to_check:
        if not check_file(file_path, description):
            all_good = False
    
    print("\n📋 Directory structure:")
    print(f"Base directory: {base_dir}")
    print(f"Src directory: {src_dir}")
    print(f"Models directory: {models_dir}")
    
    if models_dir.exists():
        print(f"\n📁 Contents of models directory:")
        for item in models_dir.iterdir():
            print(f"  - {item.name}")
    
    print(f"\n{'🎉 All files present!' if all_good else '⚠️  Some files are missing!'}")
    
    if all_good:
        print("\n🚀 Ready to commit and deploy!")
        print("\nNext steps:")
        print("1. git add .")
        print("2. git commit -m 'Add real YOLO training with Docker fixes'")
        print("3. git push")
        print("4. Pull on target machine and run: docker-compose up --build")
        print("5. Test at: http://localhost:8000/debug/files")
    
    return all_good

if __name__ == "__main__":
    main()