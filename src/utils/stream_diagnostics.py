import requests
import cv2
import asyncio
import aiohttp
from src.config import settings
from src.utils.logger import log


def diagnose_stream():
    """Comprehensive stream diagnostics"""
    stream_url = settings.video_stream_url
    log.info(f"=== STREAM DIAGNOSTICS ===")
    log.info(f"URL: {stream_url}")
    
    # Test 1: Basic HTTP connectivity
    log.info("\n1. Testing basic HTTP connectivity...")
    try:
        response = requests.head(stream_url, timeout=10)
        log.info(f"✓ HTTP HEAD: {response.status_code}")
        log.info(f"Headers: {dict(response.headers)}")
    except Exception as e:
        log.error(f"✗ HTTP HEAD failed: {e}")
    
    # Test 2: Content type detection
    log.info("\n2. Testing content type...")
    try:
        response = requests.get(stream_url, timeout=10, stream=True)
        content_type = response.headers.get('Content-Type', 'unknown')
        content_length = response.headers.get('Content-Length', 'unknown')
        
        log.info(f"✓ HTTP GET: {response.status_code}")
        log.info(f"Content-Type: {content_type}")
        log.info(f"Content-Length: {content_length}")
        
        # Read first chunk to analyze
        chunk = next(response.iter_content(chunk_size=2048))
        log.info(f"First chunk size: {len(chunk)} bytes")
        
        # Check for common formats
        if chunk.startswith(b'\xff\xd8'):
            log.info("✓ Detected JPEG format")
        elif chunk.startswith(b'\x89PNG'):
            log.info("✓ Detected PNG format")
        elif b'multipart' in content_type.lower():
            log.info("✓ Detected multipart stream (likely MJPEG)")
        else:
            log.info(f"? Unknown format. First 20 bytes: {chunk[:20]}")
            
    except Exception as e:
        log.error(f"✗ Content analysis failed: {e}")
    
    # Test 3: OpenCV compatibility
    log.info("\n3. Testing OpenCV compatibility...")
    
    backends = [
        (cv2.CAP_FFMPEG, "FFmpeg"),
        (cv2.CAP_GSTREAMER, "GStreamer"),
        (cv2.CAP_ANY, "Auto-detect")
    ]
    
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(stream_url, backend)
            if cap.isOpened():
                log.info(f"✓ {name} backend opened stream")
                
                # Get stream properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                log.info(f"  Properties: {width}x{height} @ {fps} FPS")
                
                # Try to read frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    log.info(f"  ✓ Successfully read frame: {frame.shape}")
                else:
                    log.warning(f"  ✗ Could not read frame")
                
                cap.release()
            else:
                log.warning(f"✗ {name} backend failed to open stream")
                
        except Exception as e:
            log.error(f"✗ {name} backend error: {e}")
    
    log.info("\n=== DIAGNOSTICS COMPLETE ===")


async def diagnose_async_stream():
    """Async stream diagnostics"""
    stream_url = settings.video_stream_url
    log.info(f"\n4. Testing async HTTP access...")
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stream_url) as response:
                log.info(f"✓ Async GET: {response.status}")
                log.info(f"Headers: {dict(response.headers)}")
                
                # Read first chunk
                chunk = await response.content.read(2048)
                log.info(f"First chunk: {len(chunk)} bytes")
                
                if chunk.startswith(b'\xff\xd8'):
                    log.info("✓ JPEG detected in async response")
                
    except Exception as e:
        log.error(f"✗ Async stream test failed: {e}")


if __name__ == "__main__":
    diagnose_stream()
    asyncio.run(diagnose_async_stream())