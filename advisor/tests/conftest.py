"""Shared test configuration for advisor tests.

Adds advisor/ and toolkit/ to sys.path once, so individual test files
don't need sys.path.insert hacks.
"""

import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_advisor = os.path.join(_here, "..")
_toolkit = os.path.join(_here, "..", "..", "toolkit")

for p in (_advisor, _toolkit):
    if p not in sys.path:
        sys.path.insert(0, p)
