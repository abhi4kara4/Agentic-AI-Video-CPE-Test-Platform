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
        self.current_resolution = None  # Track current resolution
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.capture_task: Optional[asyncio.Task] = None
        self.last_frame: Optional[np.ndarray] = None
        self.fps_actual = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._should_start_streaming = False  # Flag to control when to start streaming
        
    def start(self) -> bool:
        """Start video capture in a separate thread with async loop"""
        try:
            self.is_running = True
            self._thread = threading.Thread(target=self._start_async_loop)
            self._thread.daemon = True
            self._thread.start()
            
            # If stream reading is disabled on start, just initialize but don't wait for frames
            if not settings.read_stream_on_start:
                log.info("HTTP video capture initialized but stream reading disabled on startup")
                return True
            
            # Wait for frames to be available before returning
            max_wait_time = 30  # seconds - increased timeout
            wait_interval = 0.2  # smaller intervals for more responsive checking
            elapsed = 0
            
            log.info("Waiting for HTTP video frames to be available...")
            while elapsed < max_wait_time:
                time.sleep(wait_interval)
                elapsed += wait_interval
                
                # Check if we have frames - check both queue and last_frame
                has_queued_frame = False
                try:
                    # Peek at queue without removing frame
                    if not self.frame_queue.empty():
                        has_queued_frame = True
                except:
                    pass
                
                if self.last_frame is not None or has_queued_frame:
                    log.info(f"✓ HTTP video capture started and frames available after {elapsed:.1f}s")
                    return True
                
                if elapsed % 3 == 0:  # Log every 3 seconds
                    log.info(f"Still waiting for frames... ({elapsed:.1f}s) - queue size: {self.frame_queue.qsize()}")
            
            log.warning(f"HTTP video capture started but no frames after {max_wait_time}s")
            return False  # Return False if no frames after timeout
            
        except Exception as e:
            log.error(f"Error starting HTTP video capture: {e}")
            return False
    
    def _start_async_loop(self):
        """Start the async event loop in thread"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            log.info("Starting HTTP video capture async loop")
            self._loop.run_until_complete(self._capture_loop())
        except Exception as e:
            log.error(f"Error in capture loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            log.info("HTTP video capture async loop ended")
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
        
        log.info("Setting up HTTP session for video capture")
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            self.session = session
            log.info("HTTP session created, starting capture loop")
            
            frame_count = 0
            while self.is_running:
                current_time = time.time()
                
                # If stream reading is disabled, wait until it's enabled
                if not settings.read_stream_on_start and not self._should_start_streaming:
                    # Wait in idle mode - only start capturing when first frame is requested
                    await asyncio.sleep(0.1)
                    continue
                
                # Control capture rate
                if current_time - last_capture_time < frame_interval:
                    await asyncio.sleep(0.001)
                    continue
                
                try:
                    # Capture frame via HTTP request
                    frame = await self._capture_frame_http(session)
                    
                    if frame is not None:
                        frame_count += 1
                        
                        # Log first few frames for debugging
                        if frame_count <= 3:
                            log.info(f"✓ Captured frame #{frame_count}: {frame.shape}")
                        elif frame_count == 10:
                            log.info(f"✓ HTTP capture working normally, captured {frame_count} frames")
                        
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
                        if frame_count == 0:
                            log.warning("Still waiting for first frame via HTTP...")
                        await asyncio.sleep(0.1)
                        
                except Exception as e:
                    log.error(f"Error in HTTP capture: {e}")
                    await asyncio.sleep(1)
    
    async def _capture_frame_http(self, session: aiohttp.ClientSession) -> Optional[np.ndarray]:
        """Capture a single frame via HTTP request"""
        try:
            # Test different approaches for the stream
            urls_to_try = [
                self.stream_url,  # Original URL
                f"{self.stream_url}&_t={int(time.time() * 1000)}",  # With timestamp
            ]
            
            for url in urls_to_try:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            content_type = response.headers.get('Content-Type', '').lower()
                            log.debug(f"Response Content-Type: {content_type}")
                            
                            # Check if it's an MJPEG stream
                            if 'multipart' in content_type:
                                log.debug("Detected MJPEG multipart stream")
                                # For MJPEG streams, we need to parse the multipart content
                                return await self._parse_mjpeg_frame(response)
                            
                            # Try as single image
                            image_data = await response.read()
                            
                            if len(image_data) == 0:
                                log.warning("Received empty response")
                                continue
                            
                            # Try to decode as image
                            try:
                                # Convert to OpenCV format
                                nparr = np.frombuffer(image_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None:
                                    log.debug(f"Successfully decoded frame: {frame.shape}")
                                    return frame
                                else:
                                    # Try PIL as fallback
                                    image = Image.open(io.BytesIO(image_data))
                                    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                                    return frame
                                    
                            except Exception as decode_error:
                                log.warning(f"Failed to decode image data: {decode_error}")
                                continue
                        else:
                            log.warning(f"HTTP capture failed: {response.status}")
                            continue
                            
                except Exception as url_error:
                    log.warning(f"Failed to fetch {url}: {url_error}")
                    continue
            
            return None
                    
        except asyncio.TimeoutError:
            log.warning("HTTP capture timeout")
            return None
        except Exception as e:
            log.error(f"Error in HTTP frame capture: {e}")
            return None
    
    async def _parse_mjpeg_frame(self, response) -> Optional[np.ndarray]:
        """Parse a frame from MJPEG multipart stream"""
        try:
            # Read the multipart content
            boundary = None
            content_type = response.headers.get('Content-Type', '')
            
            # Extract boundary from content type
            if 'boundary=' in content_type:
                boundary = content_type.split('boundary=')[1].split(';')[0].strip()
                if boundary.startswith('"') and boundary.endswith('"'):
                    boundary = boundary[1:-1]
            
            if not boundary:
                log.warning("Could not find boundary in MJPEG stream")
                return None
            
            # Read until we find a complete frame
            buffer = b''
            chunk_size = 8192
            
            while len(buffer) < 1024 * 1024:  # Max 1MB per frame
                chunk = await response.content.read(chunk_size)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Look for JPEG frame boundaries
                if b'\xff\xd8' in buffer and b'\xff\xd9' in buffer:
                    # Found start and end of JPEG
                    start = buffer.find(b'\xff\xd8')
                    end = buffer.find(b'\xff\xd9', start) + 2
                    
                    if start >= 0 and end > start:
                        jpeg_data = buffer[start:end]
                        
                        # Decode JPEG
                        nparr = np.frombuffer(jpeg_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            log.info(f"✓ Successfully decoded MJPEG frame: {frame.shape}")
                            return frame
            
            log.warning("Could not extract frame from MJPEG stream")
            return None
            
        except Exception as e:
            log.error(f"Error parsing MJPEG frame: {e}")
            return None
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        # If stream reading was disabled on start, enable it now on first frame request
        if not settings.read_stream_on_start and not self._should_start_streaming:
            log.info("First frame requested - enabling stream reading")
            self._should_start_streaming = True
        
        try:
            # Try to get from queue first
            frame = self.frame_queue.get_nowait()
            self.last_frame = frame
            return frame
        except Empty:
            # Return last known frame if queue is empty
            return self.last_frame
    
    def wait_for_frame(self, max_wait_seconds: float = 10.0) -> Optional[np.ndarray]:
        """Wait for a frame to be available"""
        start_time = time.time()
        wait_interval = 0.1
        
        while time.time() - start_time < max_wait_seconds:
            frame = self.get_frame()
            if frame is not None:
                return frame
            time.sleep(wait_interval)
        
        log.warning(f"No frame available after waiting {max_wait_seconds}s")
        return None
    
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
            "stream_url": self.stream_url,
            "current_resolution": self.current_resolution
        }
    
    def update_stream_url(self, new_url: str):
        """Update the stream URL dynamically"""
        if new_url != self.stream_url:
            log.info(f"Updating stream URL from {self.stream_url} to {new_url}")
            self.stream_url = new_url
            # Extract resolution from URL for tracking
            import re
            resolution_match = re.search(r'resolution_w=(\d+).*?resolution_h=(\d+)', new_url)
            if resolution_match:
                width, height = resolution_match.groups()
                self.current_resolution = f"{width}x{height}"
                log.info(f"Stream resolution updated to: {self.current_resolution}")
    
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