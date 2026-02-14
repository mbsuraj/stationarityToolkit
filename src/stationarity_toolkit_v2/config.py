"""Configuration classes for StationarityToolkit."""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class StationarityConfig:
    """
    Configuration for stationarity analysis and transformations.
    
    Attributes:
        alpha: Significance level for hypothesis tests (default: 0.05)
        seasonal_period: Period for seasonal differencing (default: None for auto-detection)
        max_differencing: Maximum number of differencing operations (default: 2)
        variance_test: Test to use for variance stationarity 
                      ('levene', 'bartlett', 'white', 'arch')
        trend_test: Test to use for trend stationarity ('adf', 'kpss', 'pp')
        auto_detect_seasonality: Whether to automatically detect seasonal period
        min_observations: Minimum number of observations required
        transformation_methods: List of variance transformations to try
        verbose: Whether to print detailed logging information
    """
    
    alpha: float = 0.05
    seasonal_period: Optional[int] = None
    max_differencing: int = 2
    variance_test: str = "levene"
    trend_test: str = "adf"
    auto_detect_seasonality: bool = True
    min_observations: int = 50
    transformation_methods: List[str] = field(
        default_factory=lambda: ["log", "sqrt", "boxcox"]
    )
    verbose: bool = False
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be between 0 and 1, got {self.alpha}")
        
        if self.seasonal_period is not None and self.seasonal_period < 2:
            raise ValueError(
                f"seasonal_period must be >= 2, got {self.seasonal_period}"
            )
        
        if self.max_differencing < 0:
            raise ValueError(
                f"max_differencing must be >= 0, got {self.max_differencing}"
            )
        
        valid_variance_tests = {"levene", "bartlett", "white", "arch"}
        if self.variance_test not in valid_variance_tests:
            raise ValueError(
                f"variance_test must be one of {valid_variance_tests}, "
                f"got {self.variance_test}"
            )
        
        valid_trend_tests = {"adf", "kpss", "pp"}
        if self.trend_test not in valid_trend_tests:
            raise ValueError(
                f"trend_test must be one of {valid_trend_tests}, "
                f"got {self.trend_test}"
            )
        
        valid_transformations = {"log", "sqrt", "boxcox", "none"}
        invalid = set(self.transformation_methods) - valid_transformations
        if invalid:
            raise ValueError(
                f"Invalid transformation methods: {invalid}. "
                f"Valid options: {valid_transformations}"
            )
