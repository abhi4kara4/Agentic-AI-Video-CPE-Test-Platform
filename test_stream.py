#!/usr/bin/env python3
"""
Quick test script to debug the video stream
"""
import requests
import cv2
from src.config import settings

def test_stream_direct():
    """Test the stream URL directly"""
    stream_url = settings.video_stream_url
    print(f"Testing stream URL: {stream_url}")
    
    # Test 1: HTTP HEAD request
    try:
        response = requests.head(stream_url, timeout=10)
        print(f"HEAD status: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
    except Exception as e:
        print(f"HEAD request failed: {e}")
    
    # Test 2: HTTP GET request (first few bytes)
    try:
        response = requests.get(stream_url, timeout=10, stream=True)
        print(f"GET status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type')}")
        
        # Read first chunk
        chunk = next(response.iter_content(chunk_size=1024))
        print(f"First chunk size: {len(chunk)} bytes")
        print(f"First 50 bytes: {chunk[:50]}")
        
    except Exception as e:
        print(f"GET request failed: {e}")
    
    # Test 3: OpenCV with different approaches
    print("\nTesting OpenCV approaches:")
    
    # Try direct URL
    cap = cv2.VideoCapture(stream_url)
    if cap.isOpened():
        print("✓ OpenCV opened stream")
        ret, frame = cap.read()
        if ret:
            print(f"✓ Successfully read frame: {frame.shape}")
        else:
            print("✗ Could not read frame")
        cap.release()
    else:
        print("✗ OpenCV could not open stream")
    
    # Try with FFmpeg backend
    cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
    if cap.isOpened():
        print("✓ OpenCV FFmpeg opened stream")
        ret, frame = cap.read()
        if ret:
            print(f"✓ FFmpeg successfully read frame: {frame.shape}")
        else:
            print("✗ FFmpeg could not read frame")
        cap.release()
    else:
        print("✗ OpenCV FFmpeg could not open stream")

if __name__ == "__main__":
    test_stream_direct()