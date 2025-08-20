import asyncio
import aiohttp
import cv2
import numpy as np
from typing import Optional, AsyncGenerator
import io
from PIL import Image
import time
from datetime import datetime
import threading
from queue import Queue, Empty

from src.config import settings
from src.utils.logger import log


class HttpVideoCapture:
    """HTTP/HTTPS video capture with authentication support"""
    
    def __init__(self, stream_url: Optional[str] = None, auth_token: Optional[str] = None):
        self.stream_url = stream_url or settings.video_stream_url
        self.auth_token = auth_token or getattr(settings, 'device_auth_token', None)
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.capture_task: Optional[asyncio.Task] = None
        self.last_frame: Optional[np.ndarray] = None
        self.fps_actual = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """Start video capture in a separate thread with async loop"""
        try:
            self.is_running = True
            self._thread = threading.Thread(target=self._start_async_loop)
            self._thread.daemon = True
            self._thread.start()
            
            # Wait a bit to see if startup succeeds
            time.sleep(2)
            
            log.info("HTTP video capture started successfully")
            return True
            
        except Exception as e:
            log.error(f"Error starting HTTP video capture: {e}")
            return False
    
    def _start_async_loop(self):
        """Start the async event loop in thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._capture_loop())
        except Exception as e:
            log.error(f"Error in capture loop: {e}")
        finally:
            self._loop.close()
    
    async def _capture_loop(self):
        """Async capture loop"""
        frame_interval = 1.0 / settings.video_fps
        last_capture_time = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        # Set up session with authentication
        headers = {}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'
            # Try different auth header formats
            headers['authToken'] = self.auth_token
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            self.session = session
            
            while self.is_running:
                current_time = time.time()
                
                # Control capture rate
                if current_time - last_capture_time < frame_interval:
                    await asyncio.sleep(0.001)
                    continue
                
                try:
                    # Capture frame via HTTP request
                    frame = await self._capture_frame_http(session)
                    
                    if frame is not None:
                        # Update FPS counter
                        fps_counter += 1
                        if current_time - fps_start_time >= 1.0:
                            self.fps_actual = fps_counter
                            fps_counter = 0
                            fps_start_time = current_time
                        
                        # Store frame
                        self.last_frame = frame
                        
                        # Try to add to queue (drop old frames if full)
                        if self.frame_queue.full():
                            try:
                                self.frame_queue.get_nowait()
                            except Empty:
                                pass
                        
                        self.frame_queue.put(frame)
                        last_capture_time = current_time
                    else:
                        log.warning("Failed to capture frame via HTTP")
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    log.error(f"Error in HTTP capture: {e}")
                    await asyncio.sleep(1)
    
    async def _capture_frame_http(self, session: aiohttp.ClientSession) -> Optional[np.ndarray]:
        """Capture a single frame via HTTP request"""
        try:
            # Add timestamp to avoid caching
            url = f"{self.stream_url}&_t={int(time.time() * 1000)}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    # Read image data
                    image_data = await response.read()
                    
                    # Convert to OpenCV format
                    image = Image.open(io.BytesIO(image_data))
                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    return frame
                else:
                    log.warning(f"HTTP capture failed: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            log.warning("HTTP capture timeout")
            return None
        except Exception as e:
            log.error(f"Error in HTTP frame capture: {e}")
            return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        try:
            # Try to get from queue first
            frame = self.frame_queue.get_nowait()
            self.last_frame = frame
            return frame
        except Empty:
            # Return last known frame if queue is empty
            return self.last_frame
    
    def capture_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """Capture current frame as screenshot"""
        frame = self.get_frame()
        if frame is None:
            log.error("No frame available for screenshot")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
        
        import os
        filepath = os.path.join(settings.screenshot_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            log.info(f"Screenshot saved: {filepath}")
            return filepath
        except Exception as e:
            log.error(f"Error saving screenshot: {e}")
            return None
    
    def get_frame_info(self) -> dict:
        """Get information about current frame and stream"""
        if not self.is_running:
            return {"status": "not_running"}
        
        frame = self.last_frame
        if frame is not None:
            height, width = frame.shape[:2]
            resolution = (width, height)
        else:
            resolution = (0, 0)
        
        return {
            "status": "running",
            "fps_target": settings.video_fps,
            "fps_actual": self.fps_actual,
            "resolution": resolution,
            "has_frame": frame is not None,
            "stream_url": self.stream_url
        }
    
    def stop(self):
        """Stop video capture"""
        log.info("Stopping HTTP video capture")
        self.is_running = False
        
        if self._thread:
            self._thread.join(timeout=5)
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        log.info("HTTP video capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()