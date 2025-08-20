from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from contextlib import asynccontextmanager
import asyncio
import io
import cv2
from typing import Dict, Any, Optional, List
import uuid
from datetime import datetime

from src.control.test_orchestrator import TestOrchestrator
from src.control.key_commands import KeyCommand
from src.utils.logger import log
from src.config import settings


# Global orchestrator instance
orchestrator: Optional[TestOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global orchestrator
    
    log.info("Starting AI Test Platform API")
    
    # Initialize orchestrator
    orchestrator = TestOrchestrator()
    if not await orchestrator.initialize(require_device_lock=settings.require_device_lock):
        log.error("Failed to initialize orchestrator")
        raise RuntimeError("Failed to initialize test platform")
    
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
async def video_stream():
    """Stream video frames"""
    if not orchestrator or not orchestrator.video_capture.is_running:
        raise HTTPException(status_code=503, detail="Video capture not available")
    
    def generate_frames():
        for frame in orchestrator.video_capture.get_frame_stream():
            if frame is None:
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
    
    return {"screenshot_path": screenshot_path, "timestamp": datetime.now().isoformat()}


@app.get("/video/info")
async def video_info():
    """Get video capture information"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return orchestrator.video_capture.get_frame_info()


# Device control endpoints
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )