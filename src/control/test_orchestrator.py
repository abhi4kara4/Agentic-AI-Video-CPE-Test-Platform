import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from src.control.device_controller import DeviceController
from src.control.key_commands import KeyCommand
from src.capture.video_capture import VideoCapture
from src.capture.http_video_capture import HttpVideoCapture
from src.agent.vision_agent import VisionAgent
from src.utils.logger import log
from src.utils.stream_tester import test_stream_connectivity
from src.config import settings


class TestOrchestrator:
    """Orchestrates test execution by coordinating video capture, AI analysis, and device control"""
    
    def __init__(self, mac_address: Optional[str] = None):
        self.device_controller = DeviceController(mac_address)
        
        # Always try OpenCV first (better for video streams)
        log.info("Using OpenCV video capture")
        self.video_capture = VideoCapture()
            
        self.vision_agent = VisionAgent()
        self.current_state = "unknown"
        self.test_running = False
        
    async def initialize(self, require_device_lock: bool = True) -> bool:
        """Initialize all components"""
        try:
            # Initialize device controller
            await self.device_controller.initialize()
            
            # Try to lock device for exclusive control
            if require_device_lock:
                if not await self.device_controller.lock_device():
                    log.error("Failed to lock device")
                    return False
            else:
                # Try to lock but don't fail if it doesn't work
                lock_success = await self.device_controller.lock_device()
                if lock_success:
                    log.info("Device locked successfully")
                else:
                    log.warning("Could not lock device, continuing without lock (development mode)")
                
            # Start video capture with fallback (unless skipped)
            if settings.skip_video_capture:
                log.warning("Video capture skipped per configuration")
            else:
                # Test stream connectivity first
                log.info("Testing video stream connectivity...")
                stream_accessible = await test_stream_connectivity()
                
                if not stream_accessible:
                    log.warning("Stream connectivity test failed, but continuing anyway")
                
                if not self.video_capture.start():
                    log.warning("Primary video capture failed, trying fallback method")
                    
                    # Try alternative capture method
                    if isinstance(self.video_capture, HttpVideoCapture):
                        log.info("Falling back to OpenCV video capture")
                        self.video_capture = VideoCapture()
                    else:
                        log.info("Falling back to HTTP video capture")
                        self.video_capture = HttpVideoCapture()
                    
                    if not self.video_capture.start():
                        log.error("All video capture methods failed")
                        return False
                
            # Initialize vision agent
            if not await self.vision_agent.initialize():
                log.error("Failed to initialize vision agent")
                return False
                
            log.info("Test orchestrator initialized successfully")
            return True
            
        except Exception as e:
            log.error(f"Error initializing orchestrator: {e}")
            return False
            
    async def cleanup(self):
        """Cleanup all resources"""
        self.test_running = False
        
        # Stop video capture
        self.video_capture.stop()
        
        # Cleanup device controller
        await self.device_controller.cleanup()
        
        log.info("Test orchestrator cleaned up")
        
    async def navigate_to_app(self, app_name: str, max_attempts: int = 20) -> bool:
        """Navigate to a specific app from current screen"""
        log.info(f"Navigating to app: {app_name}")
        
        for attempt in range(max_attempts):
            # Capture and analyze current screen
            frame = self.video_capture.get_frame()
            if frame is None:
                log.error("No frame available")
                await asyncio.sleep(1)
                continue
                
            # Analyze screen
            context = {
                "expected_screen": "app_rail",
                "last_action": self.current_state,
                "test_step": f"navigate_to_{app_name}"
            }
            
            analysis = await self.vision_agent.analyze_screen(frame, context)
            if not analysis:
                log.error("Failed to analyze screen")
                await asyncio.sleep(1)
                continue
                
            # Get navigation decision
            decision = self.vision_agent.get_navigation_decision(app_name, analysis)
            
            log.info(f"Attempt {attempt + 1}: {decision['action']} - {decision['reason']}")
            
            # Execute action
            action_success = await self._execute_action(decision['action'])
            
            if not action_success:
                log.error(f"Failed to execute action: {decision['action']}")
                continue
                
            # Check if we found and selected the app
            if decision['action'] == 'press_ok' and decision['confidence'] > 0.8:
                log.info(f"Successfully selected {app_name}")
                await asyncio.sleep(2)  # Wait for app to start loading
                return True
                
            # Small delay between navigation attempts
            await asyncio.sleep(0.5)
            
        log.error(f"Failed to navigate to {app_name} after {max_attempts} attempts")
        return False
        
    async def wait_for_screen(
        self,
        expected_screens: List[str],
        timeout: int = 30,
        check_interval: float = 1.0
    ) -> Optional[str]:
        """Wait for one of the expected screens to appear"""
        log.info(f"Waiting for screens: {expected_screens} (timeout: {timeout}s)")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            frame = self.video_capture.get_frame()
            if frame is None:
                await asyncio.sleep(check_interval)
                continue
                
            # Analyze screen
            analysis = await self.vision_agent.analyze_screen(frame)
            if not analysis:
                await asyncio.sleep(check_interval)
                continue
                
            # Check if current screen matches any expected screen
            current_screen = analysis.screen_type
            
            for expected in expected_screens:
                if expected.lower() in current_screen.lower():
                    log.info(f"Found expected screen: {current_screen}")
                    return current_screen
                    
            # Also check app name for app-specific screens
            if analysis.app_name:
                for expected in expected_screens:
                    if expected.lower() in analysis.app_name.lower():
                        log.info(f"Found expected app: {analysis.app_name}")
                        return analysis.app_name
                        
            await asyncio.sleep(check_interval)
            
        log.warning(f"Timeout waiting for screens: {expected_screens}")
        return None
        
    async def verify_no_anomalies(self, screenshot_on_failure: bool = True) -> bool:
        """Verify the current screen has no anomalies"""
        frame = self.video_capture.get_frame()
        if frame is None:
            log.error("No frame available for verification")
            return False
            
        analysis = await self.vision_agent.analyze_screen(frame)
        if not analysis:
            log.error("Failed to analyze screen for anomalies")
            return False
            
        anomalies = analysis.anomalies
        has_anomalies = any(anomalies.values())
        
        if has_anomalies:
            log.error(f"Screen anomalies detected: {anomalies}")
            
            if screenshot_on_failure:
                screenshot_path = self.video_capture.capture_screenshot(
                    f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                log.info(f"Screenshot saved: {screenshot_path}")
                
        return not has_anomalies
        
    async def _execute_action(self, action: str) -> bool:
        """Execute a navigation action"""
        try:
            action_lower = action.lower()
            
            # Map action to key command
            if action_lower == "navigate_up":
                return await self.device_controller.press_key(KeyCommand.UP)
            elif action_lower == "navigate_down":
                return await self.device_controller.press_key(KeyCommand.DOWN)
            elif action_lower == "navigate_left":
                return await self.device_controller.press_key(KeyCommand.LEFT)
            elif action_lower == "navigate_right":
                return await self.device_controller.press_key(KeyCommand.RIGHT)
            elif action_lower in ["press_ok", "select"]:
                return await self.device_controller.select_ok()
            elif action_lower == "go_back":
                return await self.device_controller.go_back()
            elif action_lower == "go_home":
                return await self.device_controller.go_home()
            elif action_lower == "wait":
                await asyncio.sleep(1)
                return True
            else:
                log.warning(f"Unknown action: {action}")
                return False
                
        except Exception as e:
            log.error(f"Error executing action {action}: {e}")
            return False
            
    async def go_to_home(self) -> bool:
        """Navigate to home screen"""
        log.info("Navigating to home screen")
        
        # Press home button
        if not await self.device_controller.go_home():
            return False
            
        # Wait for home screen
        screen = await self.wait_for_screen(["home", "home_screen", "main_menu"], timeout=10)
        
        return screen is not None
        
    async def launch_app_from_home(self, app_name: str) -> bool:
        """Complete flow to launch an app from home screen"""
        log.info(f"Launching {app_name} from home screen")
        
        # Ensure we're at home
        if not await self.go_to_home():
            log.error("Failed to navigate to home screen")
            return False
            
        # Navigate to app in rail
        if not await self.navigate_to_app(app_name):
            log.error(f"Failed to navigate to {app_name}")
            return False
            
        # Wait for app to launch
        expected_screens = [
            app_name.lower(),
            "login",
            "profile_selection",
            "loading",
            "splash_screen"
        ]
        
        screen = await self.wait_for_screen(expected_screens, timeout=30)
        
        if screen:
            log.info(f"{app_name} launched successfully, current screen: {screen}")
            
            # Verify no black screen or other anomalies
            return await self.verify_no_anomalies()
        else:
            log.error(f"Failed to launch {app_name}")
            return False
            
    async def get_current_screen_info(self) -> Dict[str, Any]:
        """Get detailed information about current screen"""
        frame = self.video_capture.get_frame()
        if frame is None:
            return {"error": "No frame available"}
            
        analysis = await self.vision_agent.analyze_screen(frame)
        if not analysis:
            return {"error": "Failed to analyze screen"}
            
        return {
            "screen_type": analysis.screen_type,
            "app_name": analysis.app_name,
            "detected_elements": analysis.detected_elements,
            "detected_text": analysis.detected_text,
            "content_playing": analysis.content_playing,
            "anomalies": analysis.anomalies,
            "confidence": analysis.confidence,
            "timestamp": analysis.timestamp.isoformat()
        }