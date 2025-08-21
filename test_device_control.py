#!/usr/bin/env python3
"""
Basic device control test without AI dependency
"""
import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.control.device_controller import DeviceController
from src.control.key_commands import KeyCommand
from src.utils.logger import log

async def test_device_control():
    """Test basic device control functionality"""
    log.info("🎮 Testing device control...")
    
    controller = DeviceController()
    
    # Initialize
    if not await controller.initialize():
        log.error("Failed to initialize device controller")
        return False
    
    # Test basic key presses
    keys_to_test = [
        (KeyCommand.HOME, "HOME button"),
        (KeyCommand.OK, "OK button"),
        (KeyCommand.DOWN, "DOWN arrow"),
        (KeyCommand.RIGHT, "RIGHT arrow"),
    ]
    
    for key, description in keys_to_test:
        log.info(f"Testing {description}...")
        try:
            success = await controller.press_key(key)
            if success:
                log.info(f"✓ {description} pressed successfully")
            else:
                log.warning(f"⚠️ {description} press may have failed")
            
            # Small delay between key presses
            await asyncio.sleep(1)
            
        except Exception as e:
            log.error(f"✗ {description} failed: {e}")
    
    # Cleanup
    await controller.cleanup()
    
    log.info("🎮 Device control test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_device_control())
    if success:
        print("\n✅ Device control test completed!")
        sys.exit(0)
    else:
        print("\n❌ Device control test failed!")
        sys.exit(1)