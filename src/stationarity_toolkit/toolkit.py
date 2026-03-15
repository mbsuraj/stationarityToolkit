import pandas as pd
from stationarity_toolkit.results import DetectionResult
from stationarity_toolkit.tests.trend import run_all_trend_tests
from stationarity_toolkit.tests.variance import run_all_variance_tests
from stationarity_toolkit.tests.seasonal import run_all_seasonal_tests


class StationarityToolkit:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def detect(self, ts: pd.Series, verbosity: str = 'detailed') -> DetectionResult:
        """Run all stationarity tests and return results.
        
        Args:
            ts: Time series as pandas Series with datetime index
            verbosity: 'minimal' or 'detailed'
        
        Returns:
            DetectionResult with flags and test results
        """
        # Input validation
        if not isinstance(ts, pd.Series):
            raise ValueError("Input must be a pandas Series")
        
        if not isinstance(ts.index, pd.DatetimeIndex):
            raise ValueError("Series must have a datetime index")
        
        if len(ts) < 50:
            raise ValueError("Series must have at least 50 observations")
        
        # Run all tests
        trend_tests = run_all_trend_tests(ts, self.alpha)
        variance_tests = run_all_variance_tests(ts, self.alpha)
        seasonal_tests = run_all_seasonal_tests(ts, self.alpha)
        
        # Aggregation: ANY test fails → flag non-stationary
        trend_stationary = all(t.is_stationary for t in trend_tests)
        variance_stationary = all(t.is_stationary for t in variance_tests)
        seasonal_stationary = all(t.is_stationary for t in seasonal_tests)
        
        # Return based on verbosity
        if verbosity == 'minimal':
            return DetectionResult(
                trend_stationary=trend_stationary,
                variance_stationary=variance_stationary,
                seasonal_stationary=seasonal_stationary,
                tests={}
            )
        else:  # detailed
            return DetectionResult(
                trend_stationary=trend_stationary,
                variance_stationary=variance_stationary,
                seasonal_stationary=seasonal_stationary,
                tests={
                    'trend': trend_tests,
                    'variance': variance_tests,
                    'seasonal': seasonal_tests
                }
            )
