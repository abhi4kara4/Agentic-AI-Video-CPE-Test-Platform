import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from urllib.parse import quote
from tenacity import retry, stop_after_attempt, wait_exponential
import time

from src.config import settings
from src.utils.logger import log
from src.control.key_commands import KeyCommand


class DeviceController:
    """Controller for device operations via REST API"""
    
    def __init__(self, mac_address: Optional[str] = None):
        self.base_url = settings.device_api_base_url
        self.mac_address = mac_address or settings.device_mac_address
        self.auth_token = settings.device_auth_token
        self.key_set = settings.device_key_set
        self.allocation_category = settings.device_allocation_category
        self.is_locked = False
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
        
    async def initialize(self):
        """Initialize the controller"""
        self._session = aiohttp.ClientSession()
        log.info(f"Device controller initialized for MAC: {self.mac_address}")
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.is_locked:
            await self.unlock_device()
        if self._session:
            await self._session.close()
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def lock_device(self) -> bool:
        """Lock the device for exclusive control"""
        if self.is_locked:
            log.warning("Device already locked")
            return True
            
        try:
            url = f"{self.base_url}/rest/settop/{quote(self.mac_address)}/allocation/lock"
            headers = {
                "accept": "application/json",
                "authToken": self.auth_token
            }
            params = {"category": self.allocation_category}
            
            async with self._session.post(url, headers=headers, params=params) as response:
                if response.status == 200:
                    self.is_locked = True
                    log.info(f"Device locked successfully: {self.mac_address}")
                    return True
                else:
                    text = await response.text()
                    log.error(f"Failed to lock device: {response.status} - {text}")
                    return False
                    
        except Exception as e:
            log.error(f"Error locking device: {e}")
            return False
            
    async def unlock_device(self) -> bool:
        """Unlock the device"""
        if not self.is_locked:
            log.warning("Device not locked")
            return True
            
        try:
            url = f"{self.base_url}/rest/settop/{quote(self.mac_address)}/allocation/lock"
            headers = {
                "accept": "application/json",
                "authToken": self.auth_token
            }
            params = {"category": self.allocation_category}
            
            async with self._session.delete(url, headers=headers, params=params) as response:
                if response.status == 204:
                    self.is_locked = False
                    log.info(f"Device unlocked successfully: {self.mac_address}")
                    return True
                else:
                    text = await response.text()
                    log.error(f"Failed to unlock device: {response.status} - {text}")
                    return False
                    
        except Exception as e:
            log.error(f"Error unlocking device: {e}")
            return False
            
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    async def press_key(self, key: KeyCommand, hold_time_ms: int = 0) -> bool:
        """Press a key on the device"""
        try:
            url = f"{self.base_url}/rest/settop/{self.mac_address}/ir/pressKey"
            headers = {
                "accept": "text/plain",
                "authToken": self.auth_token,
                "allocationCategory": self.allocation_category
            }
            params = {
                "command": key.value,
                "keySet": self.key_set
            }
            
            if hold_time_ms > 0:
                params["holdTime"] = str(hold_time_ms)
                
            async with self._session.post(url, headers=headers, params=params) as response:
                if response.status == 200:
                    log.info(f"Key pressed successfully: {key.value}")
                    return True
                else:
                    text = await response.text()
                    log.error(f"Failed to press key {key.value}: {response.status} - {text}")
                    return False
                    
        except Exception as e:
            log.error(f"Error pressing key {key.value}: {e}")
            return False
            
    async def press_key_sequence(self, keys: List[KeyCommand], delay_ms: int = 500) -> bool:
        """Press a sequence of keys with delay between them"""
        success = True
        
        for i, key in enumerate(keys):
            if not await self.press_key(key):
                success = False
                break
                
            # Add delay between keys (except after last key)
            if i < len(keys) - 1 and delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
                
        return success
        
    async def navigate_to_element(self, direction: str, max_presses: int = 10) -> int:
        """Navigate in a direction up to max_presses times"""
        try:
            key = KeyCommand.from_string(direction)
            if key not in KeyCommand.get_navigation_keys():
                log.error(f"Invalid navigation direction: {direction}")
                return 0
                
            presses = 0
            for _ in range(max_presses):
                if await self.press_key(key):
                    presses += 1
                    await asyncio.sleep(0.3)  # Small delay between presses
                else:
                    break
                    
            return presses
            
        except ValueError as e:
            log.error(f"Invalid key command: {e}")
            return 0
            
    async def power_on(self) -> bool:
        """Power on the device"""
        try:
            url = f"{self.base_url}/rest/settop/{self.mac_address}/power/on"
            headers = {
                "accept": "text/plain",
                "authToken": self.auth_token,
                "allocationCategory": self.allocation_category
            }
            
            async with self._session.post(url, headers=headers) as response:
                if response.status == 200:
                    log.info("Device powered on successfully")
                    # Wait for device to fully boot
                    await asyncio.sleep(5)
                    return True
                else:
                    text = await response.text()
                    log.error(f"Failed to power on device: {response.status} - {text}")
                    return False
                    
        except Exception as e:
            log.error(f"Error powering on device: {e}")
            return False
            
    async def power_off(self) -> bool:
        """Power off the device"""
        try:
            url = f"{self.base_url}/rest/settop/{self.mac_address}/power/off"
            headers = {
                "accept": "text/plain",
                "authToken": self.auth_token,
                "allocationCategory": self.allocation_category
            }
            
            async with self._session.post(url, headers=headers) as response:
                if response.status == 200:
                    log.info("Device powered off successfully")
                    return True
                else:
                    text = await response.text()
                    log.error(f"Failed to power off device: {response.status} - {text}")
                    return False
                    
        except Exception as e:
            log.error(f"Error powering off device: {e}")
            return False
            
    async def reboot(self) -> bool:
        """Reboot the device"""
        log.info("Rebooting device...")
        
        # Power off
        if not await self.power_off():
            return False
            
        # Wait a bit
        await asyncio.sleep(3)
        
        # Power on
        return await self.power_on()
        
    async def go_home(self) -> bool:
        """Navigate to home screen"""
        return await self.press_key(KeyCommand.HOME)
        
    async def go_back(self) -> bool:
        """Go back/exit current screen"""
        return await self.press_key(KeyCommand.BACK)
        
    async def select_ok(self) -> bool:
        """Press OK/Select button"""
        return await self.press_key(KeyCommand.OK)
        
    async def enter_text(self, text: str, delay_ms: int = 200) -> bool:
        """Enter text using number keys (for search, etc)"""
        # This is a simplified version - real implementation would need
        # to handle T9 or on-screen keyboard navigation
        success = True
        
        for char in text:
            if char.isdigit():
                try:
                    key = KeyCommand(char)
                    if not await self.press_key(key):
                        success = False
                        break
                except ValueError:
                    log.warning(f"Cannot enter character: {char}")
                    
            await asyncio.sleep(delay_ms / 1000.0)
            
        return success
        
    def get_status(self) -> Dict[str, Any]:
        """Get controller status"""
        return {
            "mac_address": self.mac_address,
            "is_locked": self.is_locked,
            "base_url": self.base_url,
            "key_set": self.key_set,
            "allocation_category": self.allocation_category
        }