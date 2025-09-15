#!/usr/bin/env python3
"""
Utility to find and fix corrupted JSON files in the training jobs directory
"""

import json
import os
from pathlib import Path


def check_and_fix_json_files(base_dir):
    """Check and fix corrupted JSON files"""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory {base_dir} does not exist")
        return

    corrupted_files = []
    fixed_files = []

    # Find all JSON files
    for json_file in base_path.rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                json.load(f)
            print(f"✅ {json_file} - OK")
        except json.JSONDecodeError as e:
            print(f"❌ {json_file} - Corrupted: {e}")
            corrupted_files.append((json_file, e))
            
            # Try to fix by creating basic metadata
            if "metadata.json" in str(json_file):
                try:
                    job_id = json_file.parent.name
                    basic_metadata = {
                        "job_id": job_id,
                        "job_name": job_id,
                        "status": "error",
                        "created_at": "unknown",
                        "config": {},
                        "error": f"Corrupted file recovered: {str(e)}"
                    }
                    
                    # Backup original file
                    backup_file = json_file.with_suffix('.json.backup')
                    json_file.rename(backup_file)
                    print(f"📄 Backed up to: {backup_file}")
                    
                    # Write fixed metadata
                    with open(json_file, 'w') as f:
                        json.dump(basic_metadata, f, indent=2)
                    print(f"🔧 Fixed: {json_file}")
                    fixed_files.append(json_file)
                    
                except Exception as fix_error:
                    print(f"❌ Could not fix {json_file}: {fix_error}")
        except Exception as e:
            print(f"❌ {json_file} - Error: {e}")

    print(f"\n📊 Summary:")
    print(f"   Corrupted files found: {len(corrupted_files)}")
    print(f"   Files fixed: {len(fixed_files)}")
    
    if corrupted_files:
        print(f"\n🔍 Corrupted files:")
        for file_path, error in corrupted_files:
            print(f"   - {file_path}: {error}")


if __name__ == "__main__":
    # Check training jobs directory
    training_dir = "training/jobs"
    if os.path.exists(training_dir):
        print(f"🔍 Checking JSON files in {training_dir}...")
        check_and_fix_json_files(training_dir)
    else:
        print(f"⚠️  Training directory {training_dir} not found")
        
    # Also check other common directories
    for check_dir in ["datasets", "models"]:
        if os.path.exists(check_dir):
            print(f"\n🔍 Checking JSON files in {check_dir}...")
            check_and_fix_json_files(check_dir)