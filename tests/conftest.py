"""Shared test configuration."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path so `from tools.xxx import ...` works in tests
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
