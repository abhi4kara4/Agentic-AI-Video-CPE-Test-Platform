import cv2
import numpy as np
from typing import Optional, Tuple, Generator
import time
from datetime import datetime
import os
from src.utils.logger import log
from src.config import settings
import threading
from queue import Queue, Empty


class VideoCapture:
    def __init__(self, stream_url: Optional[str] = None):
        self.stream_url = stream_url or settings.video_stream_url
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.frame_queue = Queue(maxsize=10)
        self.capture_thread: Optional[threading.Thread] = None
        self.last_frame: Optional[np.ndarray] = None
        self.fps_actual = 0
        
    def start(self) -> bool:
        """Start video capture from stream"""
        try:
            log.info(f"Starting video capture from: {self.stream_url}")
            
            # For network streams, try different approaches
            success = False
            
            if self.stream_url.startswith(('http://', 'https://')):
                # Try different OpenCV backends for HTTP/HTTPS streams
                backends_to_try = [
                    (cv2.CAP_FFMPEG, "FFmpeg"),
                    (cv2.CAP_GSTREAMER, "GStreamer"), 
                    (cv2.CAP_ANY, "Auto-detect")
                ]
                
                for backend, name in backends_to_try:
                    try:
                        log.info(f"Trying OpenCV backend: {name} ({backend})")
                        self.cap = cv2.VideoCapture(self.stream_url, backend)
                        
                        if self.cap.isOpened():
                            log.info(f"Stream opened with {name} backend")
                            
                            # Set properties before testing frame read
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            self.cap.set(cv2.CAP_PROP_FPS, settings.video_fps)
                            
                            # Give stream some time to initialize
                            import time
                            time.sleep(1)
                            
                            # Test multiple frame reads (sometimes first fails)
                            for attempt in range(3):
                                ret, test_frame = self.cap.read()
                                if ret and test_frame is not None:
                                    log.info(f"✓ Successfully read frame with {name}: {test_frame.shape}")
                                    success = True
                                    break
                                else:
                                    log.warning(f"Frame read attempt {attempt + 1} failed with {name}")
                                    time.sleep(0.5)
                            
                            if success:
                                break
                            else:
                                log.warning(f"✗ {name} opened stream but all frame reads failed")
                                self.cap.release()
                                continue
                        else:
                            log.warning(f"✗ {name} failed to open stream")
                            continue
                    except Exception as e:
                        log.warning(f"✗ {name} backend error: {e}")
                        if hasattr(self, 'cap') and self.cap:
                            self.cap.release()
                        continue
                
                if not success:
                    log.error("All OpenCV backends failed for HTTP/HTTPS stream")
                    return False
            else:
                # For RTSP or other protocols, use default
                log.info("Using default OpenCV backend for non-HTTP stream")
                self.cap = cv2.VideoCapture(self.stream_url)
                
                if self.cap.isOpened():
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        success = True
                    else:
                        log.error("Default backend opened stream but cannot read frames")
                        return False
                else:
                    log.error("Default backend failed to open stream")
                    return False
            
            # Set capture properties for better performance
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, settings.video_fps)
                
                # Additional settings for network streams
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                # Test reading a frame to verify stream works
                ret, test_frame = self.cap.read()
                if not ret or test_frame is None:
                    log.error("Stream opened but cannot read frames")
                    log.error("Possible causes:")
                    log.error("1. Stream format not supported")
                    log.error("2. Network connectivity issues") 
                    log.error("3. Stream requires specific headers")
                    return False
                
                log.info(f"Stream test successful. Frame shape: {test_frame.shape}")
                
                self.is_running = True
                self.capture_thread = threading.Thread(target=self._capture_loop)
                self.capture_thread.daemon = True
                self.capture_thread.start()
                
                log.info("Video capture started successfully")
                return True
            else:
                log.error(f"Failed to open video stream: {self.stream_url}")
                return False
                
        except Exception as e:
            log.error(f"Error starting video capture: {e}")
            return False
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread"""
        frame_interval = 1.0 / settings.video_fps
        last_capture_time = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            
            # Control capture rate
            if current_time - last_capture_time < frame_interval:
                time.sleep(0.001)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                log.warning("Failed to read frame from stream")
                time.sleep(0.1)
                continue
            
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
    
    def get_frame_stream(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames at specified FPS"""
        frame_interval = 1.0 / settings.video_fps
        last_yield_time = 0
        
        while self.is_running:
            current_time = time.time()
            
            if current_time - last_yield_time >= frame_interval:
                frame = self.get_frame()
                if frame is not None:
                    yield frame
                    last_yield_time = current_time
            else:
                time.sleep(0.001)
    
    def capture_screenshot(self, filename: Optional[str] = None) -> Optional[str]:
        """Capture current frame as screenshot"""
        frame = self.get_frame()
        if frame is None:
            log.error("No frame available for screenshot")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
        
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
        if self.cap is None or not self.cap.isOpened():
            return {"status": "not_connected"}
        
        return {
            "status": "connected",
            "fps_target": settings.video_fps,
            "fps_actual": self.fps_actual,
            "resolution": (
                int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ),
            "frame_count": int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
            "has_frame": self.last_frame is not None
        }
    
    def stop(self):
        """Stop video capture"""
        log.info("Stopping video capture")
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                break
        
        log.info("Video capture stopped")
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()