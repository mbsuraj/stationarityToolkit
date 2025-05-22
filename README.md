# StationarityToolkit

The StationarityToolkit is a Python library designed to help you analyze and prepare time series data for stationarity. It offers a set of powerful tools for dealing with both trend and variance non-stationarity in your time series data. Below, we'll describe its key features and how to use them:

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

 
1. **Install Stationarity Toolkit:**
   ```cmd
   pip install StationarityToolkit
2. **Import the StationarityToolkit:**
   - Import the StationarityToolkit library in your Python script or Jupyter Notebook.

   ```python
    from stationarity_toolkit.stationarity_toolkit import StationarityToolkit
3. **Initialize the Toolkit:**

   - Begin by creating an instance of the StationarityToolkit class, passing your time series data as an argument.

   ```python
    from StationarityToolkit import StationarityToolkit
    
    toolkit = StationarityToolkit(alpha)
   
After this point you can try different things with the toolkit. Such as:
1. **Test for Stationarity:**

- Utilize the toolkit's methods to assess stationarity in your time series data. The toolkit offers the following testing options:

   ```python
   toolkit.perform_pp_test(ts)  # Phillips-Perron Test for variance non-stationarity
   toolkit.adf_test(ts)              # Test for trend non-stationarity using ADF
   toolkit.kpss_test(ts)             # Test for trend non-stationarity using KPSS
2. **Remove Variance Non-Stationarity**
- The toolkit will perform log, square root, and box-cox transformations
while testing for Phillips-Perron Test. The transformation with least p value 
is chosen.
- Even before testing the transformations, the model shall test the original
data to see if it really has any variance nonstationarity. If not, it will skip
transformations altogether.
   ```python
  toolkit.remove_var_nonstationarity(ts_as_a_dataframe)

3. **Remove Trend Non-Stationarity**
- The toolkit will perform lag term differencing, seasonal differencing
or a combination of both while iteratively doing adf_test and kpss_test 
to determine the best transformation to achieve trend non-stationarity. 
- NOTE: CURRENT VERSION ONLY DEALS WITH WEEKLY SEASONAL DATA.
   ```python
  toolkit.remove_var_nonstationarity(ts_as_a_dataframe)
4. **Remove Non-Stationarity**
- The toolkit will test and remove both trend and variance
non-stationarity. First it will begin with variance non-stationarity testing
and removal and then proceed to trend non-stationarity testing 
and removal.
    ```python
  toolkit.remove_nonstationarity(ts_as_a_dataframe)