#!/usr/bin/env python3
"""
Test runner script for BDD scenarios
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse


def run_pytest_bdd(feature_file=None, verbose=False, capture=True):
    """Run pytest-bdd tests"""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add capture options
    if not capture:
        cmd.append("-s")
    
    # Add coverage
    cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])
    
    # Add specific feature file if provided
    if feature_file:
        feature_path = Path("tests/features") / feature_file
        if not feature_path.exists():
            print(f"Feature file not found: {feature_path}")
            return False
        cmd.append(str(feature_path))
    else:
        cmd.append("tests/")
    
    # Add HTML report
    cmd.extend(["--html=reports/test_report.html", "--self-contained-html"])
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run BDD tests for AI Video Test Platform")
    
    parser.add_argument(
        "--feature",
        help="Specific feature file to run (e.g., netflix_launch.feature)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--no-capture", "-s",
        action="store_true",
        help="Don't capture output (useful for debugging)"
    )
    
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="List available feature files"
    )
    
    args = parser.parse_args()
    
    # Create reports directory
    os.makedirs("reports", exist_ok=True)
    
    if args.list_features:
        features_dir = Path("tests/features")
        if features_dir.exists():
            print("Available feature files:")
            for feature in features_dir.glob("*.feature"):
                print(f"  - {feature.name}")
        else:
            print("No features directory found")
        return
    
    # Run tests
    success = run_pytest_bdd(
        feature_file=args.feature,
        verbose=args.verbose,
        capture=not args.no_capture
    )
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()