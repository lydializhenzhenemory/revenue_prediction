import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the dataset
file_path = 'normalized_data.csv'
data = pd.read_csv(file_path)

# Assuming that the 'dt_input' is the datetime column used for indexing,
# and 'amt_revenue' is the target variable.

# Replace NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Drop non-numeric columns for simplicity, including target variable 'amt_revenue' and datetime 'dt_input'
# In a real case scenario, you would encode or extract features from these columns.
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.drop(columns=['row_id', 'period_input', 'period_fcst', 'qty_fcst', 'amount_fcst', 
                               'qty_revenue', 'amt_cost_group', 'amt_fcst_min', 'amt_fcst_max',
                               'qty_fcst_min', 'qty_fcst_max', 'amt_backlog', 'qty_backlog',
                               'qty_FGI_inventory', 'amt_FGI_inventory', 'amt_projection', 
                               'amt_budget'], inplace=True)  # Drop non-relevant numeric columns

# Normalize all the numeric features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(numeric_features)

# Define the target variable
target = data['amt_revenue'].values.reshape(-1, 1)
scaled_target = scaler.fit_transform(target)

# Combine the scaled features and target into one array
combined_data = np.hstack((scaled_features, scaled_target))

# Create the dataset for LSTM
def create_dataset_with_all_features(dataset, target_index, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back), :target_index])
        Y.append(dataset[i + look_back, target_index])
    return np.array(X), np.array(Y)

# Target variable is at the last index
target_index = combined_data.shape[1] - 1
X, Y = create_dataset_with_all_features(combined_data, target_index, look_back=1)

# Split the data into training and test sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predictions to original scale for the target variable only
train_predict = scaler.inverse_transform(np.hstack((X_train.reshape(X_train.shape[0], -1), train_predict)))
Y_train_inv = scaler.inverse_transform(np.hstack((X_train.reshape(X_train.shape[0], -1), Y_train.reshape(-1, 1))))[:, -1]
test_predict = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], -1), test_predict)))
Y_test_inv = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], -1), Y_test.reshape(-1, 1))))[:, -1]

# Calculate root mean squared error (RMSE) for the predictions
train_rmse = np.sqrt(mean_squared_error(Y_train_inv, train_predict[:, -1]))
test_rmse = np.sqrt(mean_squared_error(Y_test_inv, test_predict[:, -1]))

# Print RMSE results
print('Train RMSE:', train_rmse)
print('Test RMSE:', test_rmse)

# Plotting the results
# Plot of actual vs. predicted training values
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(Y_train_inv, label='True')
plt.plot(train_predict[:, -1], label='Predicted')
plt.title('Train Data: True vs Predicted')
plt.legend()

# Plot of actual vs. predicted test values
plt.subplot(1, 2, 2)
plt.plot(Y_test_inv, label='True')
plt.plot(test_predict[:, -1], label='Predicted')
plt.title('Test Data: True vs Predicted')
plt.legend()

# Save the plot to a file
plt.savefig('LSTM_Full_Features_True_vs_Predicted.png')
plt.show()
