#!/usr/bin/env python3
"""
ApexBase Test Runner

This script provides a convenient way to run the ApexBase test suite
with various options and configurations.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_pytest(args):
    """Run pytest with the given arguments"""
    # Change to the project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add pytest arguments
    if args.verbose:
        cmd.append("-v")
    
    if args.coverage:
        cmd.extend(["--cov=apexbase", "--cov-report=html", "--cov-report=term-missing"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    if args.skip_slow:
        if args.markers:
            cmd[-1] = f"{args.markers} and not slow"
        else:
            cmd.extend(["-m", "not slow"])
    
    if args.file:
        cmd.append(args.file)
    
    if args.pytest_args:
        cmd.extend(args.pytest_args)
    
    # Add test directory if no specific file specified
    if not args.file and not any(arg.startswith("test/") for arg in cmd):
        cmd.append("test/")
    
    print(f"Running: {' '.join(cmd)}")
    
    # Run pytest
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        return e.returncode

def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import pytest
    except ImportError:
        missing_deps.append("pytest")
    
    # Check optional dependencies
    optional_deps = {
        "pytest-cov": "coverage reporting",
        "pytest-xdist": "parallel execution", 
        "pytest-timeout": "test timeouts",
        "pandas": "pandas tests",
        "polars": "polars tests",
        "pyarrow": "pyarrow tests",
    }
    
    available_optional = []
    missing_optional = []
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep.replace("-", "_"))
            available_optional.append(f"✓ {dep} ({description})")
        except ImportError:
            missing_optional.append(f"✗ {dep} ({description})")
    
    print("Dependency Check:")
    print("=" * 50)
    
    if missing_deps:
        print("Missing required dependencies:")
        for dep in missing_deps:
            print(f"  ✗ {dep}")
        print("\nInstall with: pip install " + " ".join(missing_deps))
        return False
    
    print("✓ All required dependencies available")
    
    if available_optional:
        print("\nAvailable optional dependencies:")
        for dep in available_optional:
            print(f"  {dep}")
    
    if missing_optional:
        print("\nMissing optional dependencies:")
        for dep in missing_optional:
            print(f"  {dep}")
        print("\nInstall optional dependencies with:")
        print("  pip install " + " ".join(missing_optional))
    
    print()
    return True

def list_tests():
    """List all available tests"""
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "test/"],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing tests: {e}")
        return 1
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="ApexBase Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                          # Run all tests
  python run_tests.py --verbose                # Verbose output
  python run_tests.py --coverage               # Run with coverage
  python run_tests.py --parallel               # Run in parallel
  python run_tests.py --markers "fts"         # Run only FTS tests
  python run_tests.py --skip-slow              # Skip slow tests
  python run_tests.py --file test_init_config.py  # Run specific file
  python run_tests.py --list                   # List all tests
  python run_tests.py --check-deps             # Check dependencies
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage reporting"
    )
    
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--markers", "-m",
        type=str,
        help="Pytest markers to run (e.g., 'fts', 'not slow')"
    )
    
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip tests marked as slow"
    )
    
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Run specific test file"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available tests"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies"
    )
    
    parser.add_argument(
        "--pytest-args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return 0
    
    # List tests if requested
    if args.list:
        return list_tests()
    
    # Quick dependency check
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests
    return run_pytest(args)

if __name__ == "__main__":
    sys.exit(main())
