import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load your data
x_train = pd.read_csv('xTrain.csv')
y_train = pd.read_csv('yTrain.csv')
x_test = pd.read_csv('xTest.csv')
y_test = pd.read_csv('yTest.csv')

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit on training data and transform both training and test data
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Initialize Neural Network Regressor
nn_model = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=500)

# Perform cross-validation
cv_scores = cross_val_score(nn_model, x_train_scaled, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')

# Calculating Mean Squared Error for each fold
mse_scores = -cv_scores  # Convert to positive mean squared error
print("MSE for each fold:", mse_scores)

# Calculating average and standard deviation of MSE
print("Average MSE:", mse_scores.mean())
print("Standard Deviation of MSE:", mse_scores.std())

# Training the model with the entire training set
nn_model.fit(x_train_scaled, y_train.values.ravel())

# Predicting on the test set
y_pred = nn_model.predict(x_test_scaled)

# Calculating Mean Squared Error on the test set
mse_test = mean_squared_error(y_test, y_pred)
print("Mean Squared Error on Test Set:", mse_test)

# Plotting Predicted vs Actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Values')
plt.text(0.05, 0.95, f'MSE: {mse_test:.4f}', ha='left', va='top', transform=plt.gca().transAxes) 
plt.grid(True)
plt.savefig('nn_predicted_plot.png', format='png')
plt.show()
