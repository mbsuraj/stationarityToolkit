"""
Main StationarityToolkit class.

This is the primary interface for stationarity analysis and transformation.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
import logging
import warnings

from .config import StationarityConfig
from .results import StationarityResult, TestResult, TransformationResult
from .tests import (
    test_trend_stationarity,
    test_variance_stationarity,
)
from .transformations import (
    apply_variance_transformation,
    apply_trend_transformation,
)


class StationarityToolkit:
    """
    Comprehensive toolkit for time series stationarity analysis and transformation.
    
    This toolkit provides:
    - Proper variance stationarity tests (Levene, Bartlett, White, ARCH)
    - Trend stationarity tests (ADF, KPSS, Phillips-Perron)
    - Variance stabilizing transformations (log, sqrt, Box-Cox)
    - Trend removal transformations (differencing, seasonal differencing)
    - Automatic transformation selection
    - Comprehensive result reporting
    
    Example:
        >>> from stationarity_toolkit_v2 import StationarityToolkit
        >>> toolkit = StationarityToolkit(alpha=0.05)
        >>> result = toolkit.make_stationary(data)
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        config: Optional[StationarityConfig] = None
    ):
        """
        Initialize the StationarityToolkit.
        
        Args:
            alpha: Significance level for hypothesis tests
            config: Optional configuration object (overrides alpha if provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = StationarityConfig(alpha=alpha)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger(__name__)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(levelname)s] %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        return logger
    
    def test_stationarity(
        self,
        timeseries: pd.Series,
        test_variance: bool = True,
        test_trend: bool = True
    ) -> StationarityResult:
        """
        Test time series for stationarity without applying transformations.
        
        Args:
            timeseries: Input time series
            test_variance: Whether to test variance stationarity
            test_trend: Whether to test trend stationarity
            
        Returns:
            StationarityResult with test results
        """
        self.logger.info("Testing stationarity...")
        
        variance_result = None
        trend_result = None
        
        # Test variance stationarity
        if test_variance:
            self.logger.info(f"Performing {self.config.variance_test} test for variance...")
            try:
                variance_result = test_variance_stationarity(
                    timeseries,
                    method=self.config.variance_test,
                    alpha=self.config.alpha
                )
                self.logger.info(
                    f"Variance stationary: {variance_result.is_stationary} "
                    f"(p={variance_result.p_value:.4f})"
                )
            except Exception as e:
                self.logger.error(f"Variance test failed: {e}")
                warnings.warn(f"Variance stationarity test failed: {e}")
        
        # Test trend stationarity
        if test_trend:
            self.logger.info(f"Performing {self.config.trend_test} test for trend...")
            try:
                trend_result = test_trend_stationarity(
                    timeseries,
                    method=self.config.trend_test,
                    alpha=self.config.alpha
                )
                self.logger.info(
                    f"Trend stationary: {trend_result.is_stationary} "
                    f"(p={trend_result.p_value:.4f})"
                )
            except Exception as e:
                self.logger.error(f"Trend test failed: {e}")
                warnings.warn(f"Trend stationarity test failed: {e}")
        
        # Determine overall stationarity
        is_stationary = True
        if variance_result is not None:
            is_stationary = is_stationary and variance_result.is_stationary
        if trend_result is not None:
            is_stationary = is_stationary and trend_result.is_stationary
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            variance_result,
            trend_result,
            None,
            None
        )
        
        return StationarityResult(
            is_stationary=is_stationary,
            trend_stationary=trend_result.is_stationary if trend_result else True,
            variance_stationary=variance_result.is_stationary if variance_result else True,
            trend_test_result=trend_result,
            variance_test_result=variance_result,
            trend_transformation=None,
            variance_transformation=None,
            final_data=timeseries,
            original_data=timeseries,
            recommendations=recommendations
        )
    
    def make_stationary(
        self,
        timeseries: pd.Series,
        handle_variance: bool = True,
        handle_trend: bool = True,
        seasonal_period: Optional[int] = None
    ) -> StationarityResult:
        """
        Analyze and transform time series to achieve stationarity.
        
        This is the main method that:
        1. Tests for variance and trend stationarity
        2. Applies appropriate transformations if needed
        3. Returns comprehensive results
        
        Args:
            timeseries: Input time series
            handle_variance: Whether to handle variance non-stationarity
            handle_trend: Whether to handle trend non-stationarity
            seasonal_period: Seasonal period (None for auto-detection)
            
        Returns:
            StationarityResult with complete analysis and transformations
        """
        self.logger.info("="*70)
        self.logger.info("Starting stationarity analysis...")
        self.logger.info("="*70)
        
        # Validate input
        if len(timeseries) < self.config.min_observations:
            raise ValueError(
                f"Time series too short. Need at least {self.config.min_observations} "
                f"observations, got {len(timeseries)}"
            )
        
        # Use provided seasonal period or config
        if seasonal_period is None:
            seasonal_period = self.config.seasonal_period
        
        # Auto-detect seasonality if enabled and not provided
        if seasonal_period is None and self.config.auto_detect_seasonality:
            seasonal_period = self._detect_seasonality(timeseries)
            if seasonal_period:
                self.logger.info(f"Auto-detected seasonal period: {seasonal_period}")
        
        current_data = timeseries.copy()
        variance_transformation = None
        trend_transformation = None
        
        # Step 1: Handle variance non-stationarity
        if handle_variance:
            self.logger.info("\n" + "-"*70)
            self.logger.info("Step 1: Analyzing variance stationarity...")
            self.logger.info("-"*70)
            
            variance_result = test_variance_stationarity(
                current_data,
                method=self.config.variance_test,
                alpha=self.config.alpha
            )
            
            self.logger.info(
                f"Variance stationary: {variance_result.is_stationary} "
                f"(p={variance_result.p_value:.4f})"
            )
            
            if not variance_result.is_stationary:
                self.logger.info("Applying variance stabilizing transformation...")
                
                # Create test function for transformation selection
                def var_test_func(ts, alpha):
                    return test_variance_stationarity(ts, self.config.variance_test, alpha)
                
                variance_transformation = apply_variance_transformation(
                    current_data,
                    method="auto",
                    test_func=var_test_func,
                    alpha=self.config.alpha
                )
                
                current_data = variance_transformation.transformed_data
                self.logger.info(
                    f"Applied {variance_transformation.transformation_name} "
                    f"(improvement: {variance_transformation.improvement_metric:.2%})"
                )
                
                # Re-test
                variance_result = test_variance_stationarity(
                    current_data,
                    method=self.config.variance_test,
                    alpha=self.config.alpha
                )
                self.logger.info(
                    f"After transformation - Variance stationary: {variance_result.is_stationary} "
                    f"(p={variance_result.p_value:.4f})"
                )
        else:
            variance_result = None
        
        # Step 2: Handle trend non-stationarity
        if handle_trend:
            self.logger.info("\n" + "-"*70)
            self.logger.info("Step 2: Analyzing trend stationarity...")
            self.logger.info("-"*70)
            
            trend_result = test_trend_stationarity(
                current_data,
                method=self.config.trend_test,
                alpha=self.config.alpha
            )
            
            self.logger.info(
                f"Trend stationary: {trend_result.is_stationary} "
                f"(p={trend_result.p_value:.4f})"
            )
            
            if not trend_result.is_stationary:
                self.logger.info("Applying trend removal transformation...")
                
                # Create test function for transformation selection
                def trend_test_func(ts, alpha):
                    return test_trend_stationarity(ts, self.config.trend_test, alpha)
                
                trend_transformation = apply_trend_transformation(
                    current_data,
                    method="auto",
                    seasonal_period=seasonal_period,
                    test_func=trend_test_func,
                    alpha=self.config.alpha,
                    max_order=self.config.max_differencing
                )
                
                current_data = trend_transformation.transformed_data
                self.logger.info(
                    f"Applied {trend_transformation.transformation_name} "
                    f"(improvement: {trend_transformation.improvement_metric:.2%})"
                )
                
                # Re-test
                trend_result = test_trend_stationarity(
                    current_data,
                    method=self.config.trend_test,
                    alpha=self.config.alpha
                )
                self.logger.info(
                    f"After transformation - Trend stationary: {trend_result.is_stationary} "
                    f"(p={trend_result.p_value:.4f})"
                )
        else:
            trend_result = None
        
        # Determine overall stationarity
        is_stationary = True
        if variance_result is not None:
            is_stationary = is_stationary and variance_result.is_stationary
        if trend_result is not None:
            is_stationary = is_stationary and trend_result.is_stationary
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            variance_result,
            trend_result,
            variance_transformation,
            trend_transformation
        )
        
        self.logger.info("\n" + "="*70)
        self.logger.info(f"Analysis complete. Overall stationary: {is_stationary}")
        self.logger.info("="*70)
        
        return StationarityResult(
            is_stationary=is_stationary,
            trend_stationary=trend_result.is_stationary if trend_result else True,
            variance_stationary=variance_result.is_stationary if variance_result else True,
            trend_test_result=trend_result,
            variance_test_result=variance_result,
            trend_transformation=trend_transformation,
            variance_transformation=variance_transformation,
            final_data=current_data,
            original_data=timeseries,
            recommendations=recommendations
        )
    
    def _detect_seasonality(self, timeseries: pd.Series) -> Optional[int]:
        """
        Automatically detect seasonal period using ACF.
        
        Args:
            timeseries: Input time series
            
        Returns:
            Detected seasonal period or None
        """
        try:
            from statsmodels.tsa.stattools import acf
            
            # Calculate ACF
            max_lag = min(len(timeseries) // 2, 365)
            acf_values = acf(timeseries.dropna(), nlags=max_lag, fft=True)
            
            # Find peaks in ACF (potential seasonal periods)
            # Look for first significant peak after lag 1
            threshold = 1.96 / np.sqrt(len(timeseries))  # 95% confidence
            
            for lag in range(2, len(acf_values)):
                if acf_values[lag] > threshold and acf_values[lag] > acf_values[lag-1]:
                    # Found a peak
                    if lag >= 4:  # Minimum reasonable seasonal period
                        return lag
            
            return None
        
        except Exception as e:
            self.logger.warning(f"Seasonality detection failed: {e}")
            return None
    
    def _generate_recommendations(
        self,
        variance_result: Optional[TestResult],
        trend_result: Optional[TestResult],
        variance_transformation: Optional[TransformationResult],
        trend_transformation: Optional[TransformationResult]
    ) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            variance_result: Variance stationarity test result
            trend_result: Trend stationarity test result
            variance_transformation: Applied variance transformation
            trend_transformation: Applied trend transformation
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Check if series is now stationary
        is_stationary = True
        if variance_result:
            is_stationary = is_stationary and variance_result.is_stationary
        if trend_result:
            is_stationary = is_stationary and trend_result.is_stationary
        
        if is_stationary:
            recommendations.append(
                "✓ Series is stationary and ready for modeling with ARIMA, VAR, or similar methods."
            )
        else:
            recommendations.append(
                "⚠ Series is still non-stationary. Consider additional transformations or use models "
                "that can handle non-stationarity (e.g., Prophet, LSTM, Gradient Boosting)."
            )
        
        # Variance-specific recommendations
        if variance_result and not variance_result.is_stationary:
            if variance_transformation:
                recommendations.append(
                    f"Applied {variance_transformation.transformation_name} for variance stabilization. "
                    f"Remember to use inverse transformation when making predictions."
                )
            else:
                recommendations.append(
                    "Consider applying variance stabilizing transformations (log, sqrt, Box-Cox) "
                    "or use models robust to heteroskedasticity (e.g., GARCH, weighted regression)."
                )
        
        # Trend-specific recommendations
        if trend_result and not trend_result.is_stationary:
            if trend_transformation:
                recommendations.append(
                    f"Applied {trend_transformation.transformation_name} for trend removal. "
                    f"Remember to use inverse transformation when making predictions."
                )
            else:
                recommendations.append(
                    "Consider applying differencing transformations or use models that can "
                    "handle trends directly (e.g., Prophet, ETS, Holt-Winters)."
                )
        
        # Model recommendations
        if variance_transformation or trend_transformation:
            recommendations.append(
                "For forecasting, consider: ARIMA (if now stationary), Prophet (handles trends/seasonality), "
                "or ML models (Gradient Boosting, Random Forest, Neural Networks)."
            )
        
        return recommendations
