import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# File paths
x_train_path = 'xTrain.csv'
x_test_path = 'xTest.csv'
y_train_path = 'yTrain.csv'
y_test_path = 'yTest.csv'

# Loading the datasets
x_train = pd.read_csv(x_train_path)
x_test = pd.read_csv(x_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

# Imputing NaN values with the mean value of each column in x_train and x_test
x_train_filled = x_train.fillna(x_train.mean())
x_test_filled = x_test.fillna(x_test.mean())

# Training the Random Forest model with the imputed data
rf_model = RandomForestRegressor(random_state=334)

# Perform 5-fold Cross Validation
cv_scores = cross_val_score(rf_model, x_train_filled, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')

# Calculating Mean Squared Error for each fold
mse_scores = -cv_scores  # Convert to positive mean squared error
print("MSE for each fold:", mse_scores)

# Calculating average and standard deviation of MSE
print("Average MSE:", mse_scores.mean())
print("Standard Deviation of MSE:", mse_scores.std())

# Training the model with the entire training set
rf_model.fit(x_train_filled, y_train.values.ravel())

# Predicting on the test set
y_pred = rf_model.predict(x_test_filled)

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
plt.savefig('rf_predicted_plot.png', format='png')
plt.show()
