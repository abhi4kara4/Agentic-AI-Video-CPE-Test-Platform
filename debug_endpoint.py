"""
Add this debug endpoint to main.py to check file existence
"""

@app.get("/debug/files")
async def debug_files():
    """Debug endpoint to check file existence in container"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    models_dir = os.path.join(src_dir, 'models')
    
    result = {
        "working_dir": os.getcwd(),
        "script_dir": current_dir,
        "src_dir": src_dir,
        "models_dir": models_dir,
        "models_dir_exists": os.path.exists(models_dir)
    }
    
    if os.path.exists(models_dir):
        try:
            files = os.listdir(models_dir)
            result["models_dir_contents"] = files
            # Check file sizes
            result["file_details"] = {}
            for f in files:
                file_path = os.path.join(models_dir, f)
                result["file_details"][f] = {
                    "size": os.path.getsize(file_path),
                    "exists": os.path.exists(file_path)
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
    
    return result