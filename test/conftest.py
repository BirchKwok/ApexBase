"""
Pytest configuration for ApexBase tests
"""

import os
import sys

# Add the apexbase python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'apexbase', 'python'))
