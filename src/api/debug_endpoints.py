"""
Debug endpoints for diagnosing Docker file mounting and import issues.
Add these to main.py before the WebSocket endpoints.
"""

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check file existence in container"""
    import os
    import sys
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(src_dir, 'models')
    
    result = {
        "working_dir": os.getcwd(),
        "script_dir": current_dir,
        "src_dir": src_dir,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir),
        "python_path": sys.path[:5]  # First 5 entries
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
        # Check what's in src dir
        if os.path.exists(src_dir):
            try:
                result["src_dir_contents"] = os.listdir(src_dir)
            except Exception as e:
                result["src_dir_error"] = str(e)
        else:
            result["src_dir_missing"] = True
    
    # Test direct import paths
    import_tests = {}
    test_paths = [
        "/app/src/models/yolo_trainer.py",
        "./src/models/yolo_trainer.py",
        os.path.join(models_dir, "yolo_trainer.py")
    ]
    
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
    import os
    import importlib.util
    
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