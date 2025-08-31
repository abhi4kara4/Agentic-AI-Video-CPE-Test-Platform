# Docker Training Fixes

## 1. Add Debug Endpoints to main.py

Add these two endpoints to `src/api/main.py` before the `# WebSocket endpoints` comment:

```python
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
```

## 2. Fix WebSocket Message Structure

Replace this line (around line 1778):
```python
        await broadcast_update("training_started", {
            "job_name": job_name,
            "model_name": job_metadata["model_name"],
            "dataset_name": job_metadata["dataset_name"]
        })
```

With:
```python
        await broadcast_update("training_started", {
            "job_name": job_name,
            "model_name": job_metadata.get("model_name", "unknown"),
            "dataset_name": job_metadata.get("dataset_name", "unknown"),
            "status": "started",
            "dataset_type": job_metadata.get("dataset_type", "object_detection")
        })
```

## 3. Replace Complex YOLO Import Logic

Find the complex import logic in the training function (around line 1790) and replace it with this simpler, more robust version:

```python
            try:
                # Robust YOLO trainer import with multiple fallback methods
                train_yolo_model = None
                
                # Method 1: Direct file import (most reliable)
                try:
                    import importlib.util
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    src_dir = os.path.dirname(current_dir)
                    trainer_path = os.path.join(src_dir, 'models', 'yolo_trainer.py')
                    
                    log.info(f"Attempting to import YOLO trainer from: {trainer_path}")
                    log.info(f"File exists: {os.path.exists(trainer_path)}")
                    
                    if os.path.exists(trainer_path):
                        spec = importlib.util.spec_from_file_location("yolo_trainer", trainer_path)
                        yolo_trainer = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(yolo_trainer)
                        train_yolo_model = yolo_trainer.train_yolo_model
                        log.info("✅ Successfully imported YOLO trainer via direct file import")
                    else:
                        log.error(f"❌ YOLO trainer file not found at: {trainer_path}")
                        
                        # List what files ARE available
                        if os.path.exists(os.path.join(src_dir, 'models')):
                            available_files = os.listdir(os.path.join(src_dir, 'models'))
                            log.error(f"Available files in models dir: {available_files}")
                        else:
                            log.error(f"Models directory not found: {os.path.join(src_dir, 'models')}")
                
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
                    raise ImportError("Failed to import YOLO trainer using all available methods")
                
                log.info(f"🚀 Starting real YOLO training for job {job_name}")
```

## 4. Test After Deployment

After pushing and pulling the code:

1. **Check file mounting**: Visit `http://localhost:8000/debug/files`
2. **Test imports**: Visit `http://localhost:8000/debug/import-test` 
3. **Start training**: Try training and check both endpoints for debugging info

## 5. Common Issues and Solutions

**If files are missing in container:**
- Check Docker volume mount: `./src:/app/src`
- Ensure files exist on host before Docker build
- Rebuild Docker image: `docker-compose build --no-cache app`

**If imports fail:**
- The debug endpoints will show exact paths and errors
- Check file permissions: files should be readable
- Verify Python path includes correct directories

**If WebSocket errors persist:**
- Check browser dev console for exact error messages
- The new message structure includes all required fields

This comprehensive fix should resolve both the Docker mounting issues and the WebSocket parsing errors!