import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Load the dataset
file_path = 'training_data_all_oem_code_sample.xlsx'
data = pd.read_excel(file_path)

# Aggregate the revenue data by the 'dt_input' date column
time_series_data = data.groupby('dt_input')['amt_revenue'].sum().sort_index()

# Perform an Augmented Dickey-Fuller test to check stationarity
adf_test_result = adfuller(time_series_data)
print('ADF Statistic:', adf_test_result[0])
print('p-value:', adf_test_result[1])

# Plot the ACF and PACF to help identify the order of the ARIMA model
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
plot_acf(time_series_data, lags=20, ax=ax1)
plot_pacf(time_series_data, lags=20, ax=ax2)
plt.show()

# Fit an ARIMA(1,0,1) model
arima_model = ARIMA(time_series_data, order=(1, 0, 1))
arima_result = arima_model.fit()

# Display the summary of the ARIMA model
print(arima_result.summary())

# Get the residuals from the ARIMA model
residuals = pd.DataFrame(arima_result.resid)

# Plot residuals
plt.figure(figsize=(12,6))
plt.plot(residuals)
plt.title('Residuals from ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# Calculate and plot the actual vs predicted values
predictions = arima_result.predict(start=0, end=len(time_series_data)-1)
plt.figure(figsize=(12,6))
plt.plot(time_series_data, label='Actual')
plt.plot(predictions, label='Predicted', alpha=0.7)
plt.title('Actual vs. Predicted Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.show()
