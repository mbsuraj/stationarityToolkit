
import pandas as pd
import numpy as np


def return_trend_variance_non_stationary_series():
    # Set a random seed for reproducibility
    np.random.seed(0)

    # Define the number of time points and the time interval
    n = 100
    time_interval = 1

    # Create a time index
    time_index = pd.date_range(start='2023-01-01', periods=n, freq=f'{time_interval}D')

    # Generate a time series with a linear trend and increasing variance
    trend = 0.1 * np.arange(n)  # Trend component

    # Generate noise with increasing variance
    noise_variance = np.linspace(0.1, 2.0, n)  # Increasing variance
    noise = np.random.normal(0, noise_variance, n)

    # Combine the trend and noise to create the time series
    time_series = trend + noise

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Value': time_series}, index=time_index)

    return df
