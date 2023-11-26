from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import pandas as pd
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

# Imputing missing values in x_train with the mean of each column
x_train_filled = x_train.fillna(x_train.mean())

# Initialize the Linear Regression model
model = LinearRegression()

# Perform 5-fold Cross Validation
cv_scores = cross_val_score(model, x_train_filled, y_train.values.ravel(), cv=5, scoring='neg_mean_squared_error')

# Convert to positive mean squared error
mse_scores_cv = -cv_scores  

# Calculating average and standard deviation of MSE
average_mse_cv = mse_scores_cv.mean()
std_mse_cv = mse_scores_cv.std()

# Fit the model with the entire training set
model.fit(x_train_filled, y_train.values.ravel())

# Predict on the test set (needs to handle x_test as well)
x_test = pd.read_csv('xTest.csv')
y_test = pd.read_csv('yTest.csv')

# Impute missing values in x_test with the mean of each column of x_train
x_test_filled = x_test.fillna(x_train.mean())

y_pred = model.predict(x_test_filled)

# Calculate Mean Squared Error on the test set
mse_test = mean_squared_error(y_test, y_pred)

average_mse_cv, std_mse_cv, mse_test

# Print out the results
print("Cross-Validation Results:", mse_scores_cv)
print(f"Average Mean Squared Error (MSE): {average_mse_cv:.4f}")
print(f"Standard Deviation of MSE: {std_mse_cv:.4f}")
print("\nTest Set Results:")
print(f"Mean Squared Error on Test Set: {mse_test:.4f}")

# Plotting Predicted vs Actual values for the test set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.text(0.05, 0.95, f'MSE: {mse_test:.4f}', ha='left', va='top', transform=plt.gca().transAxes)
plt.grid(True)

# Saving the plot to a file
plot_filename = 'linear_regression_plot.png'
plt.savefig(plot_filename)
plt.close()

plot_filename

plt.show()
