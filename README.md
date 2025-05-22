# 📊 StationarityToolkit

**StationarityToolkit** is a Python library designed to help you **analyze and transform time series data for stationarity**. It offers a suite of statistical tests and automated transformations to detect and handle both **trend** and **variance** non-stationarity.

Whether you're building a forecasting model or preparing data for analysis, this toolkit makes your preprocessing easier and more reliable.

## 🚀 Features


### ✅ 1. Test for Variance Non-Stationarity
- Use the **Phillips-Perron test** to detect variance instability.

### ✅ 2. Test for Trend Non-Stationarity
- Use both **ADF (Augmented Dickey-Fuller)** and **KPSS (Kwiatkowski-Phillips-Schmidt-Shin)** tests to check for trend-based non-stationarity.

### 🔧 3. Remove Trend Non-Stationarity
- Automatically apply:
  - **Trend differencing**
  - **Seasonal differencing**
  - Or a combination of both
- Optimized for **weekly seasonal data**.

### 🔧 4. Remove Variance Non-Stationarity
- Automatically apply transformations like:
  - **Logarithmic**
  - **Square root**
  - **Box-Cox**
- Selects the best transformation based on statistical significance.
- Skips transformation if variance is already stationary.

### 🧹 5. Remove All Non-Stationarity
- Combine both variance and trend removal in one pipeline:
  - Detect and remove variance issues first
  - Then proceed to handle trend non-stationarity

---

## 🛠️ Installation
    pip install StationarityToolkit

## 🧪 Quick Start:

1. **Import the toolkit:**
   ```python
    from stationarity_toolkit.stationarity_toolkit import StationarityToolkit
2. **Initialize the Toolkit:**
   ```python 
    toolkit = StationarityToolkit(alpha=0.05)
   
## ⚙️ Usage Guide
1. **✅ Test for Stationarity:**
   ```python
    toolkit.perform_pp_test(ts)     # Phillips-Perron test for variance non-stationarity
    toolkit.adf_test(ts)            # Augmented Dickey-Fuller test for trend
    toolkit.kpss_test(ts)           # KPSS test for trend
2. **🔧 Remove Variance Non-Stationarity**
   ```python
    toolkit.remove_var_nonstationarity(ts_as_a_dataframe)
- Checks if variance non-stationarity exists. 
- Applies log, square root, and Box-Cox transformations. 
- Selects the transformation that produces the lowest p-value. 
- Skips transformation if unnecessary.

3. **🔧 Remove Trend Non-Stationarity**
   ```python
    toolkit.remove_var_nonstationarity(ts_as_a_dataframe)
- Applies differencing techniques:
  - Lag differencing 
  - Seasonal differencing 
  - Combination of both
- Evaluates each using ADF and KPSS tests to find the best transformation. 
- ⚠️ Currently supports weekly seasonality only.

4. **🧹 Remove All Non-Stationarity**
    ```python
    toolkit.remove_nonstationarity(ts_as_a_dataframe)
- Runs both variance and trend checks/removal:
  - Removes variance non-stationarity (if present)
  - Then removes trend non-stationarity

## 💡 Why Stationarity Matters
- Most classical and deep learning time series models (ARIMA, VAR, Prophet, LSTM) assume that the data is stationary. Non-stationary data can lead to:
  - Spurious regressions 
  - Poor model accuracy 
  - Invalid statistical inferences

StationarityToolkit helps you automate this critical preprocessing step with minimal manual intervention.

