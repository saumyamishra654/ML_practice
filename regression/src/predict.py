import numpy as np
import pandas as pd
import pickle
import argparse
from data_preprocessing import preprocess

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

# make predictions
def predict(X, theta, degree):
    """
    run infernce using the trained model
    """
    X_poly = add_polynomial_features(X, degree)
    X_poly = np.hstack([np.ones((X_poly.shape[0], 1)), X_poly])
    return X_poly @ theta

# compute regression metrics
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

#  add the bias term
def add_bias_term(X):
    return np.column_stack([np.ones(X.shape[0]), X])

# load the saved model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


#  function to load the model, preprocess data, and make predictions
def main(model_path, data_path, metrics_output_path, predictions_output_path):
    
    # Preprocess the data
    data = preprocess(data_path)
    
    # Separate features and labels
    y_true = data['FUEL CONSUMPTION']

    X = data.drop('FUEL CONSUMPTION', axis=1)

    # Load the model
    model = load_model(model_path)
    weights = model['weights']
    degree = model['degree']

    y_pred = predict(X, weights, degree)
    y_pred = y_pred.flatten()

    mse, rmse, r_squared = compute_metrics(y_true, y_pred)

    np.savetxt(predictions_output_path, y_pred, delimiter=',', fmt='%f')

    with open(metrics_output_path, 'w') as f:
        f.write(f"Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"R-squared (R²) Score: {r_squared:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using a trained regression model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file that includes features and true labels')
    parser.add_argument('--metrics_output_path', type=str, required=True, help='Path where the evaluation metrics will be saved')
    parser.add_argument('--predictions_output_path', type=str, required=True, help='Path where the predictions will be saved')
    
    args = parser.parse_args()

    main(args.model_path, args.data_path, args.metrics_output_path, args.predictions_output_path)
