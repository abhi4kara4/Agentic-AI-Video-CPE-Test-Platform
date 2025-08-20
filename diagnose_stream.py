#!/usr/bin/env python3
"""
Stream diagnostics script - Run this to debug video stream issues
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.stream_diagnostics import diagnose_stream, diagnose_async_stream
import asyncio

def main():
    print("🔍 Starting stream diagnostics...")
    print("This will test your video stream connectivity and compatibility.")
    print("=" * 60)
    
    try:
        # Run synchronous diagnostics
        diagnose_stream()
        
        # Run asynchronous diagnostics
        asyncio.run(diagnose_async_stream())
        
        print("\n" + "=" * 60)
        print("✅ Diagnostics complete! Check the logs above for details.")
        
    except Exception as e:
        print(f"\n❌ Diagnostics failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()