#!/usr/bin/env python3
"""
Basic test to verify the platform is working
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.control.test_orchestrator import TestOrchestrator
from src.utils.logger import log

async def test_platform_basic():
    """Test basic platform functionality"""
    log.info("🧪 Starting basic platform test...")
    
    orchestrator = TestOrchestrator()
    
    try:
        # Test initialization
        log.info("1. Testing platform initialization...")
        if await orchestrator.initialize(require_device_lock=False):
            log.info("✅ Platform initialized successfully")
        else:
            log.error("❌ Platform initialization failed")
            return False
        
        # Test video capture info
        log.info("2. Testing video capture...")
        video_info = orchestrator.video_capture.get_frame_info()
        log.info(f"Video info: {video_info}")
        
        # Test frame capture
        log.info("3. Testing frame capture...")
        frame = orchestrator.video_capture.get_frame()
        if frame is not None:
            log.info(f"✅ Frame captured successfully: {frame.shape}")
        else:
            log.warning("⚠️  No frame available yet (may need more time)")
        
        # Test screen analysis
        log.info("4. Testing screen analysis...")
        screen_info = await orchestrator.get_current_screen_info()
        log.info(f"Screen analysis: {screen_info}")
        
        log.info("✅ Basic platform test completed successfully!")
        return True
        
    except Exception as e:
        log.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    success = asyncio.run(test_platform_basic())
    if success:
        print("\n🎉 Platform is working correctly!")
        sys.exit(0)
    else:
        print("\n💥 Platform has issues!")
        sys.exit(1)