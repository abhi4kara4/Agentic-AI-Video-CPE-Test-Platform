#!/usr/bin/env python3
"""
Main entry point for the AI Video Test Platform
"""

import asyncio
import uvicorn
from src.config import settings
from src.utils.logger import log


def run_api_server():
    """Run the FastAPI server"""
    log.info("Starting AI Video Test Platform API Server")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )


async def run_standalone_test():
    """Run a standalone test for demonstration"""
    from src.control.test_orchestrator import TestOrchestrator
    
    log.info("Starting standalone test demonstration")
    
    orchestrator = TestOrchestrator()
    
    try:
        # Initialize
        if not await orchestrator.initialize():
            log.error("Failed to initialize orchestrator")
            return
        
        log.info("Orchestrator initialized successfully")
        
        # Get current screen info
        screen_info = await orchestrator.get_current_screen_info()
        log.info(f"Current screen: {screen_info}")
        
        # Navigate to home
        if await orchestrator.go_to_home():
            log.info("Successfully navigated to home")
        else:
            log.error("Failed to navigate to home")
        
        # Try to launch Netflix
        if await orchestrator.launch_app_from_home("Netflix"):
            log.info("Successfully launched Netflix")
        else:
            log.error("Failed to launch Netflix")
        
    except Exception as e:
        log.error(f"Error in standalone test: {e}")
    finally:
        await orchestrator.cleanup()
        log.info("Standalone test completed")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "api":
            run_api_server()
        elif command == "test":
            asyncio.run(run_standalone_test())
        else:
            print("Usage: python -m src.main [api|test]")
            print("  api  - Start the FastAPI server")
            print("  test - Run standalone test demonstration")
    else:
        # Default to API server
        run_api_server()