import aiohttp
import asyncio
from urllib.parse import urlparse
import socket
from src.config import settings
from src.utils.logger import log


async def test_stream_connectivity():
    """Test if the video stream URL is accessible"""
    stream_url = settings.video_stream_url
    
    log.info(f"Testing stream connectivity to: {stream_url}")
    
    # Parse URL
    parsed = urlparse(stream_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    
    # Test DNS resolution
    try:
        ip = socket.gethostbyname(host)
        log.info(f"DNS resolution successful: {host} -> {ip}")
    except socket.gaierror as e:
        log.error(f"DNS resolution failed for {host}: {e}")
        return False
    
    # Test TCP connection
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((ip, port))
        sock.close()
        
        if result == 0:
            log.info(f"TCP connection successful to {host}:{port}")
        else:
            log.error(f"TCP connection failed to {host}:{port}")
            return False
    except Exception as e:
        log.error(f"TCP connection test error: {e}")
        return False
    
    # Test HTTP request
    try:
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.head(stream_url) as response:
                log.info(f"HTTP HEAD request status: {response.status}")
                log.info(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
                log.info(f"Content-Length: {response.headers.get('Content-Length', 'unknown')}")
                
                if response.status == 200:
                    log.info("Stream URL is accessible via HTTP")
                    return True
                else:
                    log.warning(f"Stream returned status {response.status}")
                    return False
                    
    except asyncio.TimeoutError:
        log.error("HTTP request timeout")
        return False
    except Exception as e:
        log.error(f"HTTP request error: {e}")
        return False


async def test_stream_content():
    """Test reading actual content from stream"""
    stream_url = settings.video_stream_url
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(stream_url) as response:
                if response.status == 200:
                    # Try to read first chunk
                    chunk = await response.content.read(1024)
                    log.info(f"Successfully read {len(chunk)} bytes from stream")
                    log.info(f"First few bytes: {chunk[:50]}")
                    return True
                else:
                    log.error(f"Stream content request failed: {response.status}")
                    return False
                    
    except Exception as e:
        log.error(f"Stream content test error: {e}")
        return False


if __name__ == "__main__":
    async def main():
        connectivity_ok = await test_stream_connectivity()
        if connectivity_ok:
            content_ok = await test_stream_content()
            return content_ok
        return False
    
    result = asyncio.run(main())
    if result:
        print("✅ Stream test successful")
    else:
        print("❌ Stream test failed")