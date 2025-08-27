from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import io
import cv2
import os
import json
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime
from pathlib import Path

from src.control.test_orchestrator import PlatformOrchestrator
from src.control.key_commands import KeyCommand
from src.utils.logger import log
from src.config import settings


# Global orchestrator instance
orchestrator: Optional[PlatformOrchestrator] = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                # Connection is broken, remove it
                self.active_connections.remove(connection)

manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global orchestrator
    
    log.info("Starting AI Test Platform API")
    
    # Initialize orchestrator
    orchestrator = PlatformOrchestrator()
    if not await orchestrator.initialize(require_device_lock=False):
        log.error("Failed to initialize orchestrator")
        # Continue without video capture for demo purposes
        log.warning("Running in demo mode without video capture")
    
    log.info("API started successfully")
    
    yield
    
    # Cleanup
    if orchestrator:
        await orchestrator.cleanup()
    log.info("API shutdown complete")


app = FastAPI(
    title="AI Video Test Platform",
    description="Agentic AI-powered testing platform for Set-Top Boxes and Smart TVs",
    version="0.1.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "video_capture": orchestrator.video_capture.is_running,
            "device_controller": orchestrator.device_controller.is_locked,
            "vision_agent": True  # Always true if orchestrator exists
        }
    }


# Video streaming endpoints
@app.get("/video/stream")
async def video_stream(device: Optional[str] = None, outlet: Optional[str] = None, resolution: Optional[str] = None):
    """Stream video frames"""
    log.info(f"Video stream requested - device: {device}, outlet: {outlet}, resolution: {resolution}")
    
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Parse resolution parameter
    resolution_w = settings.video_resolution_w
    resolution_h = settings.video_resolution_h
    
    if resolution:
        try:
            parts = resolution.split('x')
            if len(parts) == 2:
                resolution_w = int(parts[0])
                resolution_h = int(parts[1])
                log.info(f"Using resolution: {resolution_w}x{resolution_h}")
        except (ValueError, IndexError):
            log.warning(f"Invalid resolution format: {resolution}, using default")
    
    # Build dynamic stream URL
    device_id = device or settings.video_device_id
    outlet_id = outlet or settings.video_outlet
    
    dynamic_stream_url = (
        f"{settings.video_capture_base_url}/rack/cats-rack-sn-557.rack.abc.net:443"
        f"/magiq/video/device/{device_id}/stream"
        f"?outlet={outlet_id}"
        f"&resolution_w={resolution_w}"
        f"&resolution_h={resolution_h}"
    )
    
    log.info(f"Using dynamic stream URL: {dynamic_stream_url}")
    
    # Update the video capture URL dynamically
    orchestrator.video_capture.update_stream_url(dynamic_stream_url)
    
    if not orchestrator.video_capture.is_running:
        raise HTTPException(status_code=503, detail="Video capture not available")
    
    def generate_frames():
        while True:
            frame = orchestrator.video_capture.get_frame()
            if frame is None:
                # Sleep briefly if no frame available
                import time
                time.sleep(0.033)  # ~30 fps
                continue
                
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video/screenshot")
async def capture_screenshot():
    """Capture current screenshot"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    screenshot_path = orchestrator.video_capture.capture_screenshot()
    if not screenshot_path:
        raise HTTPException(status_code=500, detail="Failed to capture screenshot")
    
    # Also get the current frame and encode as base64 for frontend thumbnail
    base64_image = None
    try:
        frame = orchestrator.video_capture.get_frame()
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            import base64
            base64_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
            log.info("Screenshot captured with base64 thumbnail")
    except Exception as e:
        log.warning(f"Failed to generate base64 thumbnail: {e}")
    
    return {
        "screenshot_path": screenshot_path, 
        "base64_image": base64_image,
        "filename": screenshot_path.split('/')[-1] if screenshot_path else None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/image/{filename}")
async def get_screenshot_image(filename: str):
    """Serve a screenshot image file"""
    from fastapi.responses import FileResponse
    import os
    
    # Look for the file in screenshots directory
    screenshots_dir = Path("screenshots")
    file_path = screenshots_dir / filename
    
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(
        path=str(file_path),
        media_type="image/jpeg",
        filename=filename
    )


@app.get("/video/info")
async def video_info():
    """Get video capture information"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    info = orchestrator.video_capture.get_frame_info()
    log.info(f"Video info requested: {info}")
    return info


# Device control endpoints
@app.post("/device/lock")
async def lock_device():
    """Lock the device for exclusive control"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # First try to unlock any existing lock
    log.info("Attempting to unlock device before locking")
    unlock_success = await orchestrator.device_controller.unlock_device()
    log.info(f"Unlock attempt result: {unlock_success}")
    
    # Add a small delay to allow the unlock to propagate
    import asyncio
    await asyncio.sleep(1)
    
    # Now try to lock
    success = await orchestrator.device_controller.lock_device()
    
    if not success:
        # Try one more time after a brief delay
        log.info("First lock attempt failed, retrying after delay")
        await asyncio.sleep(2)
        success = await orchestrator.device_controller.lock_device()
        
        if not success:
            raise HTTPException(
                status_code=409, 
                detail="Device lock failed - may be in use by another session. Please try again in a few moments."
            )
    
    return {"status": "success", "action": "device_locked", "timestamp": datetime.now().isoformat()}


@app.post("/device/unlock")
async def unlock_device():
    """Unlock the device"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.device_controller.unlock_device()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to unlock device")
    
    return {"status": "success", "action": "device_unlocked", "timestamp": datetime.now().isoformat()}


@app.post("/device/key/{key_name}")
async def press_key(key_name: str):
    """Press a key on the device"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        key = KeyCommand.from_string(key_name)
        success = await orchestrator.device_controller.press_key(key)
        
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to press key: {key_name}")
        
        return {"status": "success", "key": key_name, "timestamp": datetime.now().isoformat()}
        
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid key name: {key_name}")


@app.post("/device/keys")
async def press_key_sequence(keys: List[str], delay_ms: int = 500):
    """Press a sequence of keys"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        key_commands = [KeyCommand.from_string(key) for key in keys]
        success = await orchestrator.device_controller.press_key_sequence(key_commands, delay_ms)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to execute key sequence")
        
        return {
            "status": "success",
            "keys": keys,
            "delay_ms": delay_ms,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/device/power/on")
async def power_on():
    """Power on the device"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.device_controller.power_on()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to power on device")
    
    return {"status": "success", "action": "power_on", "timestamp": datetime.now().isoformat()}


@app.post("/device/power/off")
async def power_off():
    """Power off the device"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.device_controller.power_off()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to power off device")
    
    return {"status": "success", "action": "power_off", "timestamp": datetime.now().isoformat()}


@app.post("/device/reboot")
async def reboot_device():
    """Reboot the device"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.device_controller.reboot()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reboot device")
    
    return {"status": "success", "action": "reboot", "timestamp": datetime.now().isoformat()}


@app.get("/device/status")
async def device_status():
    """Get device controller status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return orchestrator.device_controller.get_status()


# Screen analysis endpoints
@app.get("/screen/analyze")
async def analyze_current_screen():
    """Analyze current screen with AI"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return await orchestrator.get_current_screen_info()


@app.post("/screen/validate")
async def validate_screen(expected_state: str):
    """Validate if current screen matches expected state"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    frame = orchestrator.video_capture.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No video frame available")
    
    result = await orchestrator.vision_agent.validate_screen(frame, expected_state)
    return result


@app.get("/screen/history")
async def get_analysis_history():
    """Get screen analysis history"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    history = orchestrator.vision_agent.get_analysis_history()
    
    return {
        "count": len(history),
        "history": [
            {
                "timestamp": analysis.timestamp.isoformat(),
                "screen_type": analysis.screen_type,
                "app_name": analysis.app_name,
                "confidence": analysis.confidence,
                "anomalies": analysis.anomalies
            }
            for analysis in history
        ]
    }


# Navigation endpoints
@app.post("/navigation/home")
async def go_home():
    """Navigate to home screen"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.go_to_home()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to navigate to home")
    
    return {"status": "success", "action": "go_home", "timestamp": datetime.now().isoformat()}


@app.post("/navigation/app/{app_name}")
async def launch_app(app_name: str):
    """Launch an app from home screen"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    success = await orchestrator.launch_app_from_home(app_name)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to launch app: {app_name}")
    
    return {
        "status": "success",
        "action": "launch_app",
        "app_name": app_name,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/navigation/wait")
async def wait_for_screen(expected_screens: List[str], timeout: int = 30):
    """Wait for one of the expected screens"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    screen = await orchestrator.wait_for_screen(expected_screens, timeout)
    
    if not screen:
        raise HTTPException(
            status_code=408,
            detail=f"Timeout waiting for screens: {expected_screens}"
        )
    
    return {
        "status": "success",
        "found_screen": screen,
        "expected_screens": expected_screens,
        "timestamp": datetime.now().isoformat()
    }


# Test execution endpoints
@app.post("/test/run")
async def run_test_scenario(background_tasks: BackgroundTasks):
    """Run a test scenario"""
    # This would integrate with pytest to run specific test files
    # For now, return a placeholder
    test_id = str(uuid.uuid4())
    
    return {
        "test_id": test_id,
        "status": "started",
        "message": "Test execution not yet implemented",
        "timestamp": datetime.now().isoformat()
    }


# Dataset management endpoints
DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

SCREEN_STATES = {
    "home": "Main home screen with app rail",
    "app_rail": "App selection rail/ribbon visible",
    "app_loading": "App is loading/launching",
    "netflix_home": "Netflix home screen",
    "netflix_browse": "Netflix browsing content",
    "youtube_home": "YouTube home screen", 
    "youtube_player": "YouTube video player",
    "settings": "Settings or configuration screen",
    "error": "Error or problem screen",
    "loading": "Generic loading screen",
    "other": "Other screen type"
}

# Pydantic models for API requests
class LabelImageRequest(BaseModel):
    image_name: str
    screen_type: str
    notes: Optional[str] = ""
    label_data: Optional[Dict[str, Any]] = None


@app.get("/dataset/list")
async def list_datasets():
    """Get list of all datasets"""
    datasets = []
    for dataset_dir in DATASETS_DIR.iterdir():
        if dataset_dir.is_dir():
            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                datasets.append(metadata)
    
    return {"datasets": datasets}


@app.post("/dataset/create")
async def create_dataset(name: str, description: str = ""):
    """Create a new dataset"""
    dataset_id = str(uuid.uuid4())
    dataset_dir = DATASETS_DIR / name
    
    if dataset_dir.exists():
        raise HTTPException(status_code=400, detail="Dataset already exists")
    
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "images").mkdir()
    (dataset_dir / "annotations").mkdir()
    
    metadata = {
        "id": dataset_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
        "image_count": 0,
        "screen_states": SCREEN_STATES
    }
    
    with open(dataset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Broadcast dataset creation
    await broadcast_update("dataset_created", metadata)
    
    return metadata


@app.get("/dataset/{dataset_name}")
async def get_dataset(dataset_name: str):
    """Get dataset information"""
    dataset_dir = DATASETS_DIR / dataset_name
    metadata_file = dataset_dir / "metadata.json"
    
    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    # Update image count
    image_dir = dataset_dir / "images"
    if image_dir.exists():
        metadata["image_count"] = len(list(image_dir.glob("*.jpg")))
    
    return metadata


@app.post("/dataset/{dataset_name}/capture")
async def capture_to_dataset(dataset_name: str):
    """Capture current frame to dataset"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Capture screenshot
    frame = orchestrator.video_capture.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No video frame available")
    
    # Save image
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    image_filename = f"capture_{timestamp}.jpg"
    image_path = dataset_dir / "images" / image_filename
    
    cv2.imwrite(str(image_path), frame)
    
    result = {
        "status": "success",
        "image_filename": image_filename,
        "image_path": str(image_path),
        "timestamp": datetime.now().isoformat()
    }
    
    # Broadcast image capture
    await broadcast_update("image_captured", {
        "dataset_name": dataset_name,
        "image_filename": image_filename
    })
    
    return result


@app.post("/dataset/{dataset_name}/label")
async def label_image(dataset_name: str, request: LabelImageRequest):
    """Label an image in the dataset with extended support for different model types"""
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create annotation based on label data type
    annotation = {
        "image_filename": request.image_name,
        "labeled_at": datetime.now().isoformat(),
        "notes": request.notes
    }
    
    if request.label_data:
        dataset_type = request.label_data.get('datasetType')
        labels = request.label_data.get('labels', {})
        
        if dataset_type == 'object_detection':
            # Object detection format
            annotation.update({
                "dataset_type": "object_detection",
                "bounding_boxes": labels.get('boundingBoxes', []),
                "augmentation_options": request.label_data.get('augmentationOptions', {})
            })
        elif dataset_type == 'image_classification':
            # Image classification format
            annotation.update({
                "dataset_type": "image_classification", 
                "class_name": labels.get('className'),
                "confidence": labels.get('confidence', 100),
                "augmentation_options": request.label_data.get('augmentationOptions', {})
            })
        elif dataset_type == 'vision_llm':
            # Vision LLM format
            if request.screen_type not in SCREEN_STATES:
                raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of: {list(SCREEN_STATES.keys())}")
            
            annotation.update({
                "dataset_type": "vision_llm",
                "state": request.screen_type,
                "state_description": SCREEN_STATES.get(request.screen_type),
                "app_name": labels.get('app_name'),
                "ui_elements": labels.get('ui_elements', []),
                "visible_text": labels.get('visible_text', ''),
                "anomalies": labels.get('anomalies', []),
                "navigation": labels.get('navigation', {}),
                "augmentation_options": request.label_data.get('augmentationOptions', {})
            })
    else:
        # Legacy format for backward compatibility
        if request.screen_type not in SCREEN_STATES:
            raise HTTPException(status_code=400, detail=f"Invalid state. Must be one of: {list(SCREEN_STATES.keys())}")
        
        annotation.update({
            "dataset_type": "vision_llm",
            "state": request.screen_type,
            "state_description": SCREEN_STATES.get(request.screen_type)
        })
    
    # Save annotation
    annotation_file = dataset_dir / "annotations" / f"{Path(request.image_name).stem}.json"
    annotation_file.parent.mkdir(exist_ok=True)
    
    with open(annotation_file, "w") as f:
        json.dump(annotation, f, indent=2)
    
    # Broadcast image labeling
    await broadcast_update("image_labeled", {
        "dataset_name": dataset_name,
        "annotation": annotation
    })
    
    return annotation


@app.get("/dataset/{dataset_name}/images")
async def list_dataset_images(dataset_name: str):
    """List all images in a dataset"""
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    images_dir = dataset_dir / "images"
    annotations_dir = dataset_dir / "annotations"
    
    images = []
    for image_file in images_dir.glob("*.jpg"):
        annotation_file = annotations_dir / f"{image_file.stem}.json"
        annotation = None
        
        if annotation_file.exists():
            with open(annotation_file) as f:
                annotation = json.load(f)
        
        images.append({
            "filename": image_file.name,
            "path": str(image_file),
            "annotation": annotation
        })
    
    return {"images": images}


@app.delete("/dataset/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """Delete a dataset"""
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    import shutil
    shutil.rmtree(dataset_dir)
    
    return {"status": "success", "message": f"Dataset {dataset_name} deleted"}


@app.get("/dataset/states")
async def get_screen_states():
    """Get available screen states for labeling"""
    return {"states": SCREEN_STATES}


# Training management endpoints
TRAINING_DIR = Path("training")
TRAINING_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    "llava:7b": {"name": "LLaVA 7B", "size": "7B", "type": "vision-language"},
    "llava:7b-v1.6-mistral-q4_0": {"name": "LLaVA 7B Quantized", "size": "7B", "type": "vision-language"},
    "moondream:latest": {"name": "Moondream", "size": "1.6B", "type": "vision-language"},
    "phi3:mini": {"name": "Phi-3 Mini", "size": "3.8B", "type": "vision-language"}
}


@app.get("/training/models")
async def list_available_models():
    """Get list of available models for training"""
    return {"models": AVAILABLE_MODELS}


@app.post("/training/jobs")
async def create_training_job(
    job_name: str,
    dataset_name: str,
    base_model: str,
    epochs: int = 3,
    learning_rate: float = 0.0001,
    batch_size: int = 4
):
    """Create a new training job"""
    
    # Validate dataset exists
    dataset_dir = DATASETS_DIR / dataset_name
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate model
    if base_model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Available: {list(AVAILABLE_MODELS.keys())}")
    
    job_id = str(uuid.uuid4())
    job_dir = TRAINING_DIR / job_name
    
    if job_dir.exists():
        raise HTTPException(status_code=400, detail="Training job already exists")
    
    job_dir.mkdir(parents=True)
    
    training_config = {
        "job_id": job_id,
        "job_name": job_name,
        "dataset_name": dataset_name,
        "base_model": base_model,
        "parameters": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size
        },
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "started_at": None,
        "completed_at": None,
        "progress": {
            "current_epoch": 0,
            "total_epochs": epochs,
            "loss": None,
            "accuracy": None
        }
    }
    
    with open(job_dir / "config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    
    return training_config


@app.get("/training/jobs")
async def list_training_jobs():
    """Get list of all training jobs"""
    jobs = []
    for job_dir in TRAINING_DIR.iterdir():
        if job_dir.is_dir():
            config_file = job_dir / "config.json"
            if config_file.exists():
                with open(config_file) as f:
                    config = json.load(f)
                jobs.append(config)
    
    return {"jobs": jobs}


@app.get("/training/jobs/{job_name}")
async def get_training_job(job_name: str):
    """Get training job details"""
    job_dir = TRAINING_DIR / job_name
    config_file = job_dir / "config.json"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    with open(config_file) as f:
        config = json.load(f)
    
    # Check for logs
    log_file = job_dir / "training.log"
    logs = []
    if log_file.exists():
        with open(log_file) as f:
            logs = f.readlines()[-50:]  # Last 50 lines
    
    config["logs"] = logs
    return config


@app.post("/training/jobs/{job_name}/start")
async def start_training_job(job_name: str, background_tasks: BackgroundTasks):
    """Start a training job"""
    job_dir = TRAINING_DIR / job_name
    config_file = job_dir / "config.json"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    with open(config_file) as f:
        config = json.load(f)
    
    if config["status"] == "running":
        raise HTTPException(status_code=400, detail="Training job is already running")
    
    # Update status
    config["status"] = "running"
    config["started_at"] = datetime.now().isoformat()
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    # Broadcast training started
    await broadcast_update("training_started", {
        "job_name": job_name,
        "status": "running"
    })
    
    # Add to background tasks (simulated training)
    background_tasks.add_task(simulate_training, job_name)
    
    return {"status": "started", "job_name": job_name}


@app.post("/training/jobs/{job_name}/stop")
async def stop_training_job(job_name: str):
    """Stop a training job"""
    job_dir = TRAINING_DIR / job_name
    config_file = job_dir / "config.json"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    with open(config_file) as f:
        config = json.load(f)
    
    config["status"] = "stopped"
    config["completed_at"] = datetime.now().isoformat()
    
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    return {"status": "stopped", "job_name": job_name}


@app.delete("/training/jobs/{job_name}")
async def delete_training_job(job_name: str):
    """Delete a training job"""
    job_dir = TRAINING_DIR / job_name
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Training job not found")
    
    import shutil
    shutil.rmtree(job_dir)
    
    return {"status": "success", "message": f"Training job {job_name} deleted"}


async def simulate_training(job_name: str):
    """Simulate training progress (placeholder implementation)"""
    job_dir = TRAINING_DIR / job_name
    config_file = job_dir / "config.json"
    log_file = job_dir / "training.log"
    
    if not config_file.exists():
        return
    
    with open(config_file) as f:
        config = json.load(f)
    
    total_epochs = config["parameters"]["epochs"]
    
    with open(log_file, "w") as f:
        f.write(f"Starting training for {job_name}\n")
        f.write(f"Base model: {config['base_model']}\n")
        f.write(f"Dataset: {config['dataset_name']}\n")
        f.write(f"Epochs: {total_epochs}\n\n")
    
    for epoch in range(1, total_epochs + 1):
        await asyncio.sleep(10)  # Simulate training time
        
        # Update progress
        config["progress"]["current_epoch"] = epoch
        config["progress"]["loss"] = round(1.0 - (epoch / total_epochs) * 0.8, 4)  # Simulated decreasing loss
        config["progress"]["accuracy"] = round((epoch / total_epochs) * 0.9, 4)  # Simulated increasing accuracy
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Update log
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}/{total_epochs} - Loss: {config['progress']['loss']:.4f}, Accuracy: {config['progress']['accuracy']:.4f}\n")
        
        # Broadcast training progress
        await broadcast_update("training_progress", {
            "job_name": job_name,
            "progress": config["progress"]
        })
        
        # Check if stopped
        with open(config_file) as f:
            current_config = json.load(f)
        
        if current_config["status"] == "stopped":
            break
    
    # Mark as completed if not stopped
    with open(config_file) as f:
        final_config = json.load(f)
    
    if final_config["status"] == "running":
        final_config["status"] = "completed"
        final_config["completed_at"] = datetime.now().isoformat()
        
        with open(config_file, "w") as f:
            json.dump(final_config, f, indent=2)
        
        with open(log_file, "a") as f:
            f.write(f"\nTraining completed at {final_config['completed_at']}\n")
        
        # Broadcast training completion
        await broadcast_update("training_completed", {
            "job_name": job_name,
            "status": "completed",
            "completed_at": final_config['completed_at']
        })


# Model testing endpoints  
TESTING_DIR = Path("testing")
TESTING_DIR.mkdir(exist_ok=True)


@app.post("/test/models/compare")
async def compare_models(models: List[str], prompt: str = "Describe what you see on this TV screen"):
    """Compare multiple models on current frame"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    frame = orchestrator.video_capture.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No video frame available")
    
    test_id = str(uuid.uuid4())
    test_dir = TESTING_DIR / f"comparison_{test_id}"
    test_dir.mkdir(parents=True)
    
    # Save test frame
    frame_path = test_dir / "test_frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    
    results = []
    for model in models:
        if model not in AVAILABLE_MODELS:
            continue
            
        start_time = datetime.now()
        
        # Simulate model inference (placeholder)
        await asyncio.sleep(2)  # Simulate processing time
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Simulated response
        response = f"Analysis from {model}: This appears to be a TV interface showing various applications and menu options."
        
        result = {
            "model": model,
            "model_info": AVAILABLE_MODELS[model],
            "prompt": prompt,
            "response": response,
            "processing_time_seconds": processing_time,
            "timestamp": end_time.isoformat()
        }
        results.append(result)
    
    # Save comparison results
    comparison_data = {
        "test_id": test_id,
        "frame_path": str(frame_path),
        "prompt": prompt,
        "models_tested": models,
        "results": results,
        "created_at": datetime.now().isoformat()
    }
    
    with open(test_dir / "comparison.json", "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    return comparison_data


@app.post("/test/models/{model_name}/analyze")
async def test_single_model(model_name: str, prompt: str = "Describe what you see on this TV screen"):
    """Test a single model on current frame"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Available: {list(AVAILABLE_MODELS.keys())}")
    
    frame = orchestrator.video_capture.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="No video frame available")
    
    test_id = str(uuid.uuid4())
    test_dir = TESTING_DIR / f"single_test_{test_id}"
    test_dir.mkdir(parents=True)
    
    # Save test frame
    frame_path = test_dir / "test_frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    
    start_time = datetime.now()
    
    # Simulate model inference (placeholder)
    await asyncio.sleep(3)  # Simulate processing time
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # Simulated response
    response = f"Analysis from {model_name}: This appears to be a TV interface with navigation elements visible."
    confidence = 0.85  # Simulated confidence
    
    result = {
        "test_id": test_id,
        "model": model_name,
        "model_info": AVAILABLE_MODELS[model_name],
        "frame_path": str(frame_path),
        "prompt": prompt,
        "response": response,
        "confidence": confidence,
        "processing_time_seconds": processing_time,
        "timestamp": end_time.isoformat()
    }
    
    with open(test_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    return result


@app.post("/test/benchmark/{model_name}")
async def benchmark_model(
    model_name: str, 
    iterations: int = 5,
    prompt: str = "Describe what you see on this TV screen"
):
    """Benchmark a model with multiple iterations"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    if model_name not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model. Available: {list(AVAILABLE_MODELS.keys())}")
    
    benchmark_id = str(uuid.uuid4())
    benchmark_dir = TESTING_DIR / f"benchmark_{benchmark_id}"
    benchmark_dir.mkdir(parents=True)
    
    results = []
    total_time = 0
    
    for i in range(iterations):
        frame = orchestrator.video_capture.get_frame()
        if frame is None:
            continue
        
        # Save frame
        frame_path = benchmark_dir / f"frame_{i+1}.jpg"
        cv2.imwrite(str(frame_path), frame)
        
        start_time = datetime.now()
        
        # Simulate model inference (placeholder)
        await asyncio.sleep(2.5)  # Simulate processing time
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        total_time += processing_time
        
        # Simulated response
        response = f"Iteration {i+1}: TV interface analysis from {model_name}"
        confidence = 0.80 + (i * 0.02)  # Simulated varying confidence
        
        result = {
            "iteration": i + 1,
            "frame_path": str(frame_path),
            "response": response,
            "confidence": confidence,
            "processing_time_seconds": processing_time,
            "timestamp": end_time.isoformat()
        }
        results.append(result)
    
    # Calculate statistics
    processing_times = [r["processing_time_seconds"] for r in results]
    avg_time = sum(processing_times) / len(processing_times)
    min_time = min(processing_times)
    max_time = max(processing_times)
    
    confidences = [r["confidence"] for r in results]
    avg_confidence = sum(confidences) / len(confidences)
    
    benchmark_data = {
        "benchmark_id": benchmark_id,
        "model": model_name,
        "model_info": AVAILABLE_MODELS[model_name],
        "prompt": prompt,
        "iterations": iterations,
        "results": results,
        "statistics": {
            "average_processing_time": avg_time,
            "min_processing_time": min_time,
            "max_processing_time": max_time,
            "total_time": total_time,
            "average_confidence": avg_confidence
        },
        "created_at": datetime.now().isoformat()
    }
    
    with open(benchmark_dir / "benchmark.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
    
    return benchmark_data


@app.get("/test/history")
async def get_test_history():
    """Get history of all model tests"""
    tests = []
    
    for test_dir in TESTING_DIR.iterdir():
        if not test_dir.is_dir():
            continue
        
        # Check for different test types
        if (test_dir / "comparison.json").exists():
            with open(test_dir / "comparison.json") as f:
                test_data = json.load(f)
                test_data["test_type"] = "comparison"
                tests.append(test_data)
        elif (test_dir / "result.json").exists():
            with open(test_dir / "result.json") as f:
                test_data = json.load(f)
                test_data["test_type"] = "single"
                tests.append(test_data)
        elif (test_dir / "benchmark.json").exists():
            with open(test_dir / "benchmark.json") as f:
                test_data = json.load(f)
                test_data["test_type"] = "benchmark"
                tests.append(test_data)
    
    # Sort by creation time (newest first)
    tests.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {"tests": tests}


@app.delete("/test/history/clear")
async def clear_test_history():
    """Clear all test history"""
    import shutil
    shutil.rmtree(TESTING_DIR)
    TESTING_DIR.mkdir(exist_ok=True)
    
    return {"status": "success", "message": "Test history cleared"}


# WebSocket endpoints
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for messages
            data = await websocket.receive_text()
            
            # Handle ping/pong for connection health
            if data == "ping":
                await websocket.send_text("pong")
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Helper function to broadcast updates
async def broadcast_update(update_type: str, data: dict):
    """Broadcast updates to all connected WebSocket clients"""
    message = {
        "type": update_type,
        "timestamp": datetime.now().isoformat(),
        "data": data
    }
    await manager.broadcast(message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )