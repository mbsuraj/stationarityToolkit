"""
StationarityToolkit v2.0
========================

A comprehensive toolkit for detecting and handling non-stationarity in time series data.

This version includes:
- Proper variance stationarity tests (Levene, Bartlett, White, ARCH)
- Improved trend stationarity tests (ADF, KPSS, Phillips-Perron)
- Clean, modular architecture
- Type hints and comprehensive documentation
- Better error handling and logging

Quick Start:
-----------
>>> from stationarity_toolkit_v2 import StationarityToolkit
>>> toolkit = StationarityToolkit(alpha=0.05)
>>> result = toolkit.make_stationary(data)
>>> print(f"Is stationary: {result.is_stationary}")
"""

from .toolkit import StationarityToolkit
from .results import StationarityResult, TestResult, TransformationResult
from .config import StationarityConfig

__version__ = "2.0.0"
__all__ = [
    "StationarityToolkit",
    "StationarityResult",
    "TestResult",
    "TransformationResult",
    "StationarityConfig",
]
