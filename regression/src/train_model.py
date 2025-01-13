import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import preprocess
filepath = "/Users/saumyamishra/Desktop/Ashoka/sem 5/Introduction to Machine Learning/Saumya_Mishra_A1/regression_task/data/train_data.csv"

dataset = preprocess(filepath)

y = dataset['FUEL CONSUMPTION']

X = dataset.drop('FUEL CONSUMPTION', axis=1)

shuffled_indices = np.random.permutation(len(X))
test_size = int(len(X) * 0.2)
test_indices = shuffled_indices[:test_size]
train_indices = shuffled_indices[test_size:]

X_train, X_test = X.iloc[train_indices].to_numpy(), X.iloc[test_indices].to_numpy()
y_train, y_test = y.iloc[train_indices].to_numpy().reshape(-1,1), y.iloc[test_indices].to_numpy().reshape(-1,1)
def add_polynomial_features(X, degree):
    """
    add polynomial features up to specified degree
    """
    X_poly = X
    for d in range(2, degree + 1):
        X_poly = np.hstack([X_poly, np.power(X, d)])
    return X_poly

#  calculate the weights using the normal equation
def train_polynomial_regression(X_train, y_train, degree):
    """
     solving polynomial regression model using normal equation
    """
    X_train_poly = add_polynomial_features(X_train, degree)
    X_train_poly = np.hstack([np.ones((X_train_poly.shape[0], 1)), X_train_poly])  # Add intercept term
    theta = np.linalg.inv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ y_train
    return theta

# Function to make predictions
def predict(X, theta, degree):
    """
    run infernce using the trained model
    """
    X_poly = add_polynomial_features(X, degree)
    X_poly = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
    return X_poly @ theta

# Function to compute regression metrics
def compute_metrics(y_true, y_pred):
    """
    compute MSE, RMSE, and R-squared
    """
    mse = np.mean((y_true - y_pred) ** 2)  # MSE
    rmse = np.sqrt(mse)  # RMSE
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_residual / ss_total)  # R^2
    return mse, rmse, r2_score


degree = 2  
theta = train_polynomial_regression(X_train, y_train, degree)
y_pred = predict(X_test, theta, degree)
mse, rmse, r2_score = compute_metrics(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²) Score: {r2_score}")
plt.scatter(y_test, y_pred, color='red', label='Predicted vs Actual')
plt.title('Predicted vs Actual Fuel Consumption')
plt.xlabel('Actual Fuel Consumption')
plt.ylabel('Predicted Fuel Consumption')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='blue', linestyle='--', label='Ideal Fit')
plt.legend()
plt.show()

import pickle

#create information dictionary for pickled module 
model_info = {
    'weights': theta,
    'degree': degree  
}

print(model_info)

model_path = 'Regression_model_final.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model_info, file)

print(f"Model saved to {model_path}")