import pytest
import asyncio
from typing import Generator
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.control.test_orchestrator import PlatformOrchestrator
from src.utils.logger import log


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_orchestrator(event_loop):
    """Create and initialize test orchestrator for the session"""
    orchestrator = PlatformOrchestrator()
    
    # Initialize
    async def init():
        if not await orchestrator.initialize():
            pytest.fail("Failed to initialize test orchestrator")
        return orchestrator
    
    # Run initialization
    event_loop.run_until_complete(init())
    
    yield orchestrator
    
    # Cleanup
    async def cleanup():
        await orchestrator.cleanup()
    
    event_loop.run_until_complete(cleanup())


@pytest.fixture
def device_at_home(test_orchestrator):
    """Ensure device is at home screen before test"""
    # For now, just return the orchestrator
    # TODO: Implement actual home navigation when device control is working
    return test_orchestrator


@pytest.fixture
def screenshot_on_failure(request, test_orchestrator):
    """Capture screenshot on test failure"""
    yield
    
    if request.node.rep_call.failed:
        try:
            screenshot_path = test_orchestrator.video_capture.capture_screenshot(
                f"failure_{request.node.name}.png"
            )
            log.info(f"Failure screenshot saved: {screenshot_path}")
        except Exception as e:
            log.error(f"Failed to capture screenshot: {e}")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Make test result available to fixtures"""
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)