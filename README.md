# StationarityToolkit

The StationarityToolkit is a Python class designed to help you analyze and prepare time series data for stationarity. It offers a set of powerful tools for dealing with both trend and variance non-stationarity in your time series data. Below, we'll describe its key features and how to use them:

## Features:

### 1. Test for Variance Non-Stationarity
   - Use the Phillips-Perron test to assess variance non-stationarity in your time series data.

### 2. Test for Trend Non-Stationarity
   - Employ the Augmented Dickey-Fuller (ADF) and Kwiatkowski-Phillips-Schmidt-Shin (KPSS) tests to identify trend non-stationarity.

### 3. Remove Trend Non-Stationarity
   - Choose from various methods to eliminate trend non-stationarity, including trend differencing, seasonal differencing, or a combination of both.

### 4. Remove Variance Non-Stationarity
   - Apply data transformations such as logarithm, square, or Box-Cox to address variance non-stationarity.

### 5. Remove Both Trend and Variance Non-Stationarity
   - Combine the trend and variance non-stationarity removal techniques to make your time series data stationary.

## How to Use:

1. **Import the StationarityToolkit:**
   - Import the StationarityToolkit class in your Python script or Jupyter Notebook.

   ```python
   from StationarityToolkit import StationarityToolkit
2. **Initialize the Toolkit:**

- Begin by creating an instance of the StationarityToolkit class, passing your time series data as an argument.

   ```python
   from StationarityToolkit import StationarityToolkit

   # Replace `your_time_series_data` with your actual time series data
   toolkit = StationarityToolkit(alpha)
2. **Test for Stationarity:**

- Utilize the toolkit's methods to assess stationarity in your time series data. The toolkit offers the following testing options:

   ```python
   toolkit.perform_pp_test()  # Phillips-Perron Test for variance non-stationarity
   toolkit.adf_test()              # Test for trend non-stationarity using ADF
   toolkit.kpss_test()             # Test for trend non-stationarity using KPSS

These steps will help you get started with the StationarityToolkit and analyze your time series data for stationarity.