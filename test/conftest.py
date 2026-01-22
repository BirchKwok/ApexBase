"""
Pytest configuration for ApexBase tests

Supports switching between ApexClient and V3Client via environment variable:
- USE_V3=1 pytest test/  # Run tests with V3Client
- pytest test/          # Run tests with ApexClient (default)
"""

import os
import pytest
import sys

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))

# Check if we should use V3Client
USE_V3 = os.environ.get('USE_V3', '0') == '1'

if USE_V3:
    print("\n>>> Running tests with V3Client (on-demand storage) <<<\n")


def pytest_configure(config):
    """Configure pytest to optionally use V3Client"""
    if USE_V3:
        # Monkey-patch ApexClient with V3Client for all tests
        import apexbase
        from apexbase import V3Client
        
        # Store original for reference
        apexbase._OriginalApexClient = apexbase.ApexClient
        
        # Replace ApexClient with V3Client
        apexbase.ApexClient = V3Client
        
        # Also update the module-level import
        import importlib
        for module_name in list(sys.modules.keys()):
            if module_name.startswith('test_'):
                # Will be re-imported with new ApexClient
                pass


def pytest_unconfigure(config):
    """Restore original ApexClient after tests"""
    if USE_V3:
        import apexbase
        if hasattr(apexbase, '_OriginalApexClient'):
            apexbase.ApexClient = apexbase._OriginalApexClient
