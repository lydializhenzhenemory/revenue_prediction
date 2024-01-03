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

# Replace NaN values with the mean of each column
data.fillna(data.mean(), inplace=True)

# Select numeric columns, excluding identifiers and non-relevant columns
numeric_features = data.select_dtypes(include=[np.number])
numeric_features.drop(columns=['row_id', 'period_input', 'period_fcst', 'qty_fcst', 'amount_fcst', 
                               'qty_revenue', 'amt_cost_group', 'amt_fcst_min', 'amt_fcst_max',
                               'qty_fcst_min', 'qty_fcst_max', 'amt_backlog', 'qty_backlog',
                               'qty_FGI_inventory', 'amt_FGI_inventory', 'amt_projection', 
                               'amt_budget'], inplace=True)

# Normalize the numeric features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data['amt_revenue'].values.reshape(-1, 1))

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

# Define the target variable index and look_back period
target_index = combined_data.shape[1] - 1
look_back = 1
X, Y = create_dataset_with_all_features(combined_data, target_index, look_back)

# Split the data into training and test sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Reshape input for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[2]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Fit the model to the training data
model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

# Predict for 2024
latest_data = combined_data[-look_back:, :target_index]
latest_data_reshaped = np.reshape(latest_data, (1, look_back, latest_data.shape[1]))
prediction_2024 = model.predict(latest_data_reshaped)
prediction_2024_inversed = scaler.inverse_transform(np.hstack((latest_data_reshaped.reshape(latest_data_reshaped.shape[0], -1), prediction_2024)))[:, -1]
print("Predicted Revenue for 2024:", prediction_2024_inversed[0])

# Optional: Code for evaluating the model, plotting results, or saving the model
# ...

# Save the plot to a file
plt.savefig('LSTM_Full_Features_True_vs_Predicted.png')
plt.show()
