"""
Basic usage examples for StationarityToolkit v2.0

This script demonstrates the key features and improvements of the new toolkit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stationarity_toolkit_v2 import StationarityToolkit, StationarityConfig


def generate_sample_data():
    """Generate sample time series with different characteristics."""
    np.random.seed(42)
    n = 200
    
    # 1. Stationary series (white noise)
    stationary = pd.Series(
        np.random.normal(0, 1, n),
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
        name='Stationary'
    )
    
    # 2. Series with trend
    trend = pd.Series(
        np.arange(n) * 0.1 + np.random.normal(0, 1, n),
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
        name='With Trend'
    )
    
    # 3. Series with changing variance
    changing_var = pd.Series(
        np.random.normal(0, 1 + np.linspace(0, 2, n), n),
        index=pd.date_range('2020-01-01', periods=n, freq='D'),
        name='Changing Variance'
    )
    
    # 4. Series with trend and seasonality
    t = np.arange(n)
    seasonal = pd.Series(
        0.1 * t + 5 * np.sin(2 * np.pi * t / 52) + np.random.normal(0, 1, n),
        index=pd.date_range('2020-01-01', periods=n, freq='W'),
        name='Trend + Seasonality'
    )
    
    return {
        'stationary': stationary,
        'trend': trend,
        'changing_variance': changing_var,
        'seasonal': seasonal
    }


def example_1_basic_testing():
    """Example 1: Basic stationarity testing without transformations."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Stationarity Testing")
    print("="*80)
    
    data = generate_sample_data()
    toolkit = StationarityToolkit(alpha=0.05)
    
    # Test stationary series
    print("\n--- Testing Stationary Series ---")
    result = toolkit.test_stationarity(data['stationary'])
    print(result.summary())


def example_2_variance_transformation():
    """Example 2: Handling variance non-stationarity."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Variance Stabilization")
    print("="*80)
    
    data = generate_sample_data()
    
    # Configure toolkit to use Levene's test for variance
    config = StationarityConfig(
        alpha=0.05,
        variance_test='levene',
        verbose=True
    )
    toolkit = StationarityToolkit(config=config)
    
    print("\n--- Handling Changing Variance ---")
    result = toolkit.make_stationary(
        data['changing_variance'],
        handle_variance=True,
        handle_trend=False
    )
    
    print(result.summary())
    
    # Plot before and after
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].plot(result.original_data)
    axes[0].set_title('Original Series (Non-constant Variance)')
    axes[0].set_ylabel('Value')
    
    axes[1].plot(result.final_data)
    axes[1].set_title('After Variance Stabilization')
    axes[1].set_ylabel('Transformed Value')
    axes[1].set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig('variance_transformation.png')
    print("\nPlot saved as 'variance_transformation.png'")


def example_3_trend_removal():
    """Example 3: Handling trend non-stationarity."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Trend Removal")
    print("="*80)
    
    data = generate_sample_data()
    
    config = StationarityConfig(
        alpha=0.05,
        trend_test='adf',
        verbose=True
    )
    toolkit = StationarityToolkit(config=config)
    
    print("\n--- Removing Trend ---")
    result = toolkit.make_stationary(
        data['trend'],
        handle_variance=False,
        handle_trend=True
    )
    
    print(result.summary())


def example_4_full_pipeline():
    """Example 4: Complete pipeline with variance and trend handling."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Complete Stationarity Pipeline")
    print("="*80)
    
    data = generate_sample_data()
    
    config = StationarityConfig(
        alpha=0.05,
        variance_test='levene',
        trend_test='adf',
        auto_detect_seasonality=True,
        verbose=True
    )
    toolkit = StationarityToolkit(config=config)
    
    print("\n--- Processing Seasonal Series ---")
    result = toolkit.make_stationary(
        data['seasonal'],
        handle_variance=True,
        handle_trend=True
    )
    
    print(result.summary())
    
    # Demonstrate inverse transformation
    if result.variance_transformation or result.trend_transformation:
        print("\n--- Testing Inverse Transformation ---")
        inverse_func = result.get_inverse_transform()
        
        # Apply inverse to final data
        reconstructed = inverse_func(result.final_data.values)
        
        # Compare with original
        mse = np.mean((reconstructed - result.original_data.values) ** 2)
        print(f"MSE between original and reconstructed: {mse:.6f}")
        print("(Should be close to 0 if inverse transformation is correct)")


def example_5_comparison_old_vs_new():
    """Example 5: Demonstrate the key improvement - proper variance tests."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Old vs New Approach")
    print("="*80)
    
    data = generate_sample_data()
    
    print("\n--- OLD APPROACH (Phillips-Perron for 'variance') ---")
    print("Phillips-Perron actually tests for TREND stationarity (unit roots),")
    print("NOT variance stationarity. This was the main issue in the old toolkit.")
    
    from arch.unitroot import PhillipsPerron
    pp_test = PhillipsPerron(data['changing_variance'].dropna())
    print(f"Phillips-Perron p-value: {pp_test.pvalue:.4f}")
    print(f"Conclusion: {'Stationary' if pp_test.pvalue < 0.05 else 'Non-stationary'}")
    print("⚠ This is testing TREND, not VARIANCE!")
    
    print("\n--- NEW APPROACH (Levene's Test for variance) ---")
    print("Levene's test CORRECTLY tests for variance homogeneity.")
    
    toolkit = StationarityToolkit(alpha=0.05)
    from stationarity_toolkit_v2.tests import levene_test
    
    levene_result = levene_test(data['changing_variance'], alpha=0.05)
    print(f"Levene's test p-value: {levene_result.p_value:.4f}")
    print(f"Conclusion: {'Constant variance' if levene_result.is_stationary else 'Non-constant variance'}")
    print("✓ This CORRECTLY identifies the variance issue!")


def example_6_different_tests():
    """Example 6: Compare different variance tests."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comparing Different Variance Tests")
    print("="*80)
    
    data = generate_sample_data()
    
    from stationarity_toolkit_v2.tests import (
        levene_test, bartlett_test, white_test, arch_test
    )
    
    print("\n--- Testing Series with Changing Variance ---")
    
    tests = [
        ('Levene', levene_test),
        ('Bartlett', bartlett_test),
        ('White', white_test),
        ('ARCH', arch_test),
    ]
    
    for name, test_func in tests:
        try:
            result = test_func(data['changing_variance'], alpha=0.05)
            print(f"\n{name} Test:")
            print(f"  P-value: {result.p_value:.4f}")
            print(f"  Stationary: {result.is_stationary}")
        except Exception as e:
            print(f"\n{name} Test: Failed - {e}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("StationarityToolkit v2.0 - Examples")
    print("="*80)
    print("\nThis demonstrates the improved toolkit with:")
    print("  ✓ Proper variance tests (Levene, Bartlett, White, ARCH)")
    print("  ✓ Clear separation of trend vs variance testing")
    print("  ✓ Better code structure and documentation")
    print("  ✓ Comprehensive result objects")
    
    # Run examples
    example_1_basic_testing()
    example_2_variance_transformation()
    example_3_trend_removal()
    example_4_full_pipeline()
    example_5_comparison_old_vs_new()
    example_6_different_tests()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)
