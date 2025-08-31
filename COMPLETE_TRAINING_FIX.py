"""
Complete fix for Docker YOLO training issues.
Copy and paste the relevant parts into your main.py file.
"""

# 1. ADD THESE DEBUG ENDPOINTS TO main.py (before WebSocket endpoints)

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check file existence in container"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(src_dir, 'models')
    
    result = {
        "working_dir": os.getcwd(),
        "script_dir": current_dir,
        "src_dir": src_dir,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir),
        "python_path": sys.path[:5]
    }
    
    if os.path.exists(models_dir):
        try:
            files = os.listdir(models_dir)
            result["models_dir_contents"] = files
            result["file_details"] = {}
            for f in files:
                file_path = os.path.join(models_dir, f)
                if os.path.isfile(file_path):
                    result["file_details"][f] = {
                        "size": os.path.getsize(file_path),
                        "readable": os.access(file_path, os.R_OK),
                        "is_python": f.endswith('.py')
                    }
        except Exception as e:
            result["models_dir_error"] = str(e)
    else:
        if os.path.exists(src_dir):
            try:
                result["src_dir_contents"] = os.listdir(src_dir)
            except Exception as e:
                result["src_dir_error"] = str(e)
        else:
            result["src_dir_missing"] = True
    
    test_paths = [
        "/app/src/models/yolo_trainer.py",
        "./src/models/yolo_trainer.py", 
        os.path.join(models_dir, "yolo_trainer.py")
    ]
    
    import_tests = {}
    for path in test_paths:
        import_tests[path] = {
            "exists": os.path.exists(path),
            "absolute": os.path.abspath(path)
        }
    
    result["import_path_tests"] = import_tests
    return result


@app.get("/debug/import-test")
async def debug_import_test():
    """Test YOLO imports directly"""
    results = {"attempts": [], "success": False, "final_error": None}
    
    # Test 1: Direct import
    try:
        from src.models.yolo_trainer import train_yolo_model
        results["attempts"].append({"method": "direct_import", "success": True})
        results["success"] = True
        return results
    except Exception as e:
        results["attempts"].append({"method": "direct_import", "error": str(e)})
    
    # Test 2: File-based import
    try:
        import importlib.util
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        trainer_path = os.path.join(src_dir, 'models', 'yolo_trainer.py')
        
        if os.path.exists(trainer_path):
            spec = importlib.util.spec_from_file_location("yolo_trainer", trainer_path)
            yolo_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolo_trainer)
            train_yolo_model = yolo_trainer.train_yolo_model
            results["attempts"].append({"method": "file_import", "success": True, "path": trainer_path})
            results["success"] = True
            return results
        else:
            results["attempts"].append({"method": "file_import", "error": f"File not found: {trainer_path}"})
    except Exception as e:
        results["attempts"].append({"method": "file_import", "error": str(e)})
    
    results["final_error"] = "All import methods failed"
    return results


# 2. REPLACE THE TRAINING_STARTED BROADCAST (around line 1778)
# Change this:
#         await broadcast_update("training_started", {
#             "job_name": job_name,
#             "model_name": job_metadata["model_name"],
#             "dataset_name": job_metadata["dataset_name"]
#         })

# To this:
        await broadcast_update("training_started", {
            "job_name": job_name,
            "model_name": job_metadata.get("model_name", "unknown"),
            "dataset_name": job_metadata.get("dataset_name", "unknown"),
            "status": "started",
            "dataset_type": job_metadata.get("dataset_type", "object_detection")
        })


# 3. REPLACE THE COMPLEX YOLO IMPORT LOGIC
# Find the training function around line 1790 and replace the entire complex import section with:

def get_yolo_trainer():
    """Import YOLO trainer with robust error handling"""
    train_yolo_model = None
    
    # Method 1: Direct file import (most reliable for Docker)
    try:
        import importlib.util
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        trainer_path = os.path.join(src_dir, 'models', 'yolo_trainer.py')
        
        log.info(f"🔍 Attempting YOLO import from: {trainer_path}")
        log.info(f"📁 File exists: {os.path.exists(trainer_path)}")
        log.info(f"📂 Working dir: {os.getcwd()}")
        log.info(f"📂 Script dir: {current_dir}")
        log.info(f"📂 Src dir: {src_dir}")
        
        if os.path.exists(trainer_path):
            spec = importlib.util.spec_from_file_location("yolo_trainer", trainer_path)
            yolo_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolo_trainer)
            train_yolo_model = yolo_trainer.train_yolo_model
            log.info("✅ Successfully imported YOLO trainer via direct file import")
        else:
            log.error(f"❌ YOLO trainer file not found at: {trainer_path}")
            
            # Debug: List what files ARE available
            models_dir = os.path.join(src_dir, 'models')
            if os.path.exists(models_dir):
                available_files = os.listdir(models_dir)
                log.error(f"📋 Available files in models dir: {available_files}")
            else:
                log.error(f"❌ Models directory not found: {models_dir}")
            
            # Check src directory contents
            if os.path.exists(src_dir):
                src_contents = os.listdir(src_dir)
                log.error(f"📋 Contents of src dir: {src_contents}")
    
    except Exception as e:
        log.error(f"❌ Direct file import failed: {e}")
    
    # Method 2: Standard import (fallback)
    if not train_yolo_model:
        try:
            from src.models.yolo_trainer import train_yolo_model
            log.info("✅ Successfully imported YOLO trainer via standard import")
        except ImportError as e:
            log.error(f"❌ Standard import failed: {e}")
    
    if not train_yolo_model:
        raise ImportError("Failed to import YOLO trainer. Check /debug/files and /debug/import-test endpoints for details")
    
    return train_yolo_model

# Then in the training function, replace the complex import with:
                train_yolo_model = get_yolo_trainer()
                log.info(f"🚀 Starting real YOLO training for job {job_name}")


# 4. ALSO ADD THIS HELPER FUNCTION AT THE TOP LEVEL (after imports, before app creation)

def get_yolo_trainer():
    """Import YOLO trainer with robust error handling and detailed logging"""
    train_yolo_model = None
    
    # Method 1: Direct file import (most reliable for Docker)
    try:
        import importlib.util
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_dir = os.path.dirname(current_dir)
        trainer_path = os.path.join(src_dir, 'models', 'yolo_trainer.py')
        
        log.info(f"🔍 Attempting YOLO import from: {trainer_path}")
        log.info(f"📁 File exists: {os.path.exists(trainer_path)}")
        log.info(f"📂 Working dir: {os.getcwd()}")
        log.info(f"📂 Script dir: {current_dir}")
        log.info(f"📂 Src dir: {src_dir}")
        
        if os.path.exists(trainer_path):
            spec = importlib.util.spec_from_file_location("yolo_trainer", trainer_path)
            yolo_trainer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(yolo_trainer)
            train_yolo_model = yolo_trainer.train_yolo_model
            log.info("✅ Successfully imported YOLO trainer via direct file import")
            return train_yolo_model
        else:
            log.error(f"❌ YOLO trainer file not found at: {trainer_path}")
            
            # Debug: List what files ARE available
            models_dir = os.path.join(src_dir, 'models')
            if os.path.exists(models_dir):
                available_files = os.listdir(models_dir)
                log.error(f"📋 Available files in models dir: {available_files}")
            else:
                log.error(f"❌ Models directory not found: {models_dir}")
            
            # Check src directory contents
            if os.path.exists(src_dir):
                src_contents = os.listdir(src_dir)
                log.error(f"📋 Contents of src dir: {src_contents}")
    
    except Exception as e:
        log.error(f"❌ Direct file import failed: {e}")
    
    # Method 2: Standard import (fallback)
    if not train_yolo_model:
        try:
            from src.models.yolo_trainer import train_yolo_model
            log.info("✅ Successfully imported YOLO trainer via standard import")
            return train_yolo_model
        except ImportError as e:
            log.error(f"❌ Standard import failed: {e}")
    
    raise ImportError("Failed to import YOLO trainer. Check /debug/files and /debug/import-test endpoints for details")


# 5. THEN IN THE TRAINING FUNCTION, REPLACE THE ENTIRE COMPLEX IMPORT SECTION WITH:

            try:
                train_yolo_model = get_yolo_trainer()
                log.info(f"🚀 Starting real YOLO training for job {job_name}")
                
                # Update status to training
                job_metadata.update({
                    "status": "training",
                    "current_epoch": 0,
                    "total_epochs": total_epochs,
                    "updated_at": datetime.now().isoformat()
                })
                with open(job_dir / "metadata.json", "w") as f:
                    json.dump(job_metadata, f, indent=2)
                
                # Start real YOLO training
                training_results = await train_yolo_model(
                    dataset_name=job_metadata["dataset_name"],
                    model_name=job_metadata["model_name"],
                    training_config=job_metadata["config"]
                )
                
                # ... rest of training logic stays the same