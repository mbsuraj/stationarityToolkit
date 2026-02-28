"""Stationarity Toolkit v3

Detection-only toolkit for identifying time series non-stationarity.

Usage:
    from stationarity_toolkit_v3 import StationarityToolkit
    
    toolkit = StationarityToolkit(alpha=0.05)
    
    # Quick check
    result = toolkit.detect(ts, verbosity='minimal')
    print(result.trend_stationary)
    
    # Full report
    result = toolkit.detect(ts, verbosity='detailed')
    print(result.report())
    result.report(filepath='report.md')
"""

from .toolkit import StationarityToolkit
from .results import DetectionResult, TestResult

__all__ = ['StationarityToolkit', 'DetectionResult', 'TestResult']