import pytest
from pytest_bdd import scenarios, given, when, then, parsers
import asyncio
import os

from src.control.test_orchestrator import TestOrchestrator
from src.utils.logger import log

# Load scenarios from the correct path
feature_file = os.path.join(os.path.dirname(__file__), '..', 'features', 'netflix_launch.feature')
scenarios(feature_file)


@given('the test platform is initialized')
def platform_initialized(test_orchestrator):
    """Test platform should be initialized by fixture"""
    assert test_orchestrator is not None
    log.info("Test platform is initialized")


@given('the device is powered on')
async def device_powered_on(test_orchestrator):
    """Ensure device is powered on"""
    # The device should already be on, but we can verify
    current_info = await test_orchestrator.get_current_screen_info()
    assert "error" not in current_info, "Device appears to be offline"
    log.info("Device is powered on")


@given('the device is on home screen')
async def device_on_home(device_at_home):
    """Device should be at home screen (handled by fixture)"""
    log.info("Device is on home screen")


@when(parsers.parse('I navigate to {app_name} in app rail'))
async def navigate_to_app(test_orchestrator, app_name):
    """Navigate to app in rail"""
    success = await test_orchestrator.navigate_to_app(app_name)
    assert success, f"Failed to navigate to {app_name}"
    log.info(f"Navigated to {app_name} in app rail")


@when('I press OK')
async def press_ok(test_orchestrator):
    """Press OK button"""
    success = await test_orchestrator.device_controller.select_ok()
    assert success, "Failed to press OK"
    log.info("Pressed OK button")


@when(parsers.parse('I launch the {app_name} app'))
async def launch_app(test_orchestrator, app_name):
    """Launch app from home screen"""
    success = await test_orchestrator.launch_app_from_home(app_name)
    assert success, f"Failed to launch {app_name}"
    log.info(f"Launched {app_name} app")


@then(parsers.parse('{app_name} should launch'))
async def app_should_launch(test_orchestrator, app_name):
    """Verify app launched"""
    # Wait for app-specific screens
    expected_screens = [
        app_name.lower(),
        "loading",
        "splash"
    ]
    
    screen = await test_orchestrator.wait_for_screen(expected_screens, timeout=15)
    assert screen is not None, f"{app_name} did not launch"
    log.info(f"{app_name} launched successfully")


@then('I should see either login screen or profile selection or home screen')
async def verify_app_screens(test_orchestrator):
    """Verify expected app screens appear"""
    expected_screens = [
        "login",
        "sign_in", 
        "profile",
        "profile_selection",
        "who_watching",
        "home_screen",
        "browse",
        "content"
    ]
    
    screen = await test_orchestrator.wait_for_screen(expected_screens, timeout=20)
    assert screen is not None, f"Did not see any expected screens. Current screen: {await test_orchestrator.get_current_screen_info()}"
    log.info(f"Found expected screen: {screen}")


@then('I should not see black screen')
async def no_black_screen(test_orchestrator):
    """Verify no black screen"""
    current_info = await test_orchestrator.get_current_screen_info()
    
    assert current_info.get("screen_type") != "black_screen", "Black screen detected"
    assert not current_info.get("anomalies", {}).get("black_screen", False), "Black screen anomaly detected"
    
    log.info("No black screen detected")


@then('the app should load without anomalies')
async def no_anomalies(test_orchestrator):
    """Verify no screen anomalies"""
    success = await test_orchestrator.verify_no_anomalies(screenshot_on_failure=True)
    assert success, "Screen anomalies detected"
    log.info("No anomalies detected")


@then('no buffering indicator should be present')
async def no_buffering(test_orchestrator):
    """Verify no buffering"""
    current_info = await test_orchestrator.get_current_screen_info()
    
    # Check anomalies
    assert not current_info.get("anomalies", {}).get("buffering", False), "Buffering detected"
    
    # Check detected text for buffering keywords
    detected_text = current_info.get("detected_text", [])
    buffering_keywords = ["buffering", "loading", "please wait"]
    
    for text in detected_text:
        for keyword in buffering_keywords:
            assert keyword not in text.lower(), f"Buffering indicator found: {text}"
    
    log.info("No buffering indicator present")


@then('the screen should not be frozen')
async def screen_not_frozen(test_orchestrator):
    """Verify screen is not frozen by checking for changes"""
    # Take two screenshots with delay and compare
    frame1 = test_orchestrator.video_capture.get_frame()
    await asyncio.sleep(2)
    frame2 = test_orchestrator.video_capture.get_frame()
    
    assert frame1 is not None and frame2 is not None, "Failed to capture frames"
    
    # Use frame processor to compare
    from src.capture.frame_processor import FrameProcessor
    
    # If frames are too similar, might be frozen
    is_similar = FrameProcessor.compare_frames(frame1, frame2, threshold=0.99)
    
    assert not is_similar, "Screen appears to be frozen (no changes detected)"
    log.info("Screen is not frozen")