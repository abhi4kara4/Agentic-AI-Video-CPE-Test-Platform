#!/usr/bin/env python3
"""
Debug script to test imports from different paths
"""
import sys
import os

print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("Python path:")
for path in sys.path:
    print(f"  - {path}")

print("\nTesting imports...")

# Test 1: Direct src import
try:
    from src.models.yolo_trainer import train_yolo_model
    print("✅ Direct src import successful")
except ImportError as e:
    print(f"❌ Direct src import failed: {e}")

# Test 2: Relative import
try:
    from models.yolo_trainer import train_yolo_model
    print("✅ Relative import successful")  
except ImportError as e:
    print(f"❌ Relative import failed: {e}")

# Test 3: Check if files exist
files_to_check = [
    'src/models/yolo_trainer.py',
    'src/models/__init__.py',
    'src/__init__.py'
]

print("\nChecking file existence:")
for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{'✅' if exists else '❌'} {file_path} - {'exists' if exists else 'missing'}")

# Test 4: Manual path adjustment
print("\nTesting manual path adjustment...")
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(current_dir, 'src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    print(f"Added to path: {src_dir}")
    
    from models.yolo_trainer import train_yolo_model
    print("✅ Manual path adjustment successful")
except ImportError as e:
    print(f"❌ Manual path adjustment failed: {e}")

print("\nDone.")