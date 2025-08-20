import cv2
import numpy as np
from typing import Optional, Tuple, List
from PIL import Image
import base64
import io
from src.utils.logger import log


class FrameProcessor:
    """Process video frames for AI analysis"""
    
    @staticmethod
    def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Resize frame while maintaining aspect ratio"""
        height, width = frame.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Pad to target size if needed
        if new_width < target_width or new_height < target_height:
            top = (target_height - new_height) // 2
            bottom = target_height - new_height - top
            left = (target_width - new_width) // 2
            right = target_width - new_width - left
            
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        
        return resized
    
    @staticmethod
    def enhance_frame(frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better AI recognition"""
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    @staticmethod
    def frame_to_base64(frame: np.ndarray) -> str:
        """Convert frame to base64 string for API transmission"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG", optimize=True)
        
        # Encode to base64
        img_bytes = buffer.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        return img_base64
    
    @staticmethod
    def detect_screen_anomalies(frame: np.ndarray) -> dict:
        """Detect common screen anomalies"""
        anomalies = {
            "black_screen": False,
            "frozen_frame": False,
            "low_quality": False,
            "artifacts": False
        }
        
        # Check for black screen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 10:
            anomalies["black_screen"] = True
        
        # Check for low quality (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            anomalies["low_quality"] = True
        
        # Check for artifacts (high frequency noise)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        if edge_ratio > 0.3:
            anomalies["artifacts"] = True
        
        return anomalies
    
    @staticmethod
    def extract_regions(frame: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
        """Extract specific regions from frame"""
        extracted = []
        
        for x, y, w, h in regions:
            region = frame[y:y+h, x:x+w]
            extracted.append(region)
        
        return extracted
    
    @staticmethod
    def compare_frames(frame1: np.ndarray, frame2: np.ndarray, threshold: float = 0.95) -> bool:
        """Compare two frames for similarity"""
        # Resize to same size if different
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
        
        # Calculate structural similarity
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram correlation
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation > threshold
    
    @staticmethod
    def prepare_for_ai(frame: np.ndarray, enhance: bool = True) -> Tuple[np.ndarray, str]:
        """Prepare frame for AI analysis"""
        # Resize to optimal size for AI
        processed = FrameProcessor.resize_frame(frame, (1280, 720))
        
        # Enhance if requested
        if enhance:
            processed = FrameProcessor.enhance_frame(processed)
        
        # Convert to base64
        base64_image = FrameProcessor.frame_to_base64(processed)
        
        return processed, base64_image