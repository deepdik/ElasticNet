import os
import sys
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
#import numpy as np

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():

    # Load the dataset
    data = fetch_california_housing()
    X_real = data.data
    y_real = data.target

    # -------------------------------------------------------
    # Standardize the features for custom model
    X_mean = np.mean(X_real, axis=0)
    X_std = np.std(X_real, axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X_real_standardized = (X_real - X_mean) / X_std

    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)

    # Fit the model
    r = model.fit(X_real_standardized, y_real)

    # Make predictions
    y_pred_real = r.predict(X_real_standardized)

    print()
    # Evaluate performance
    mse_custom = mean_squared_error(y_real, y_pred_real)
    print(f"Custom ElasticNetRegression MSE: {mse_custom}")
    print()
    # Scikit-learn ElasticNet Model
    # -------------------------------------------------------
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_real)

    # Initialize scikit-learn's ElasticNet model
    sklearn_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
    sklearn_model.fit(X_scaled, y_real)
    y_pred_sklearn = sklearn_model.predict(X_scaled)

    # Evaluate performance
    mse_sklearn = mean_squared_error(y_real, y_pred_sklearn)
    print(f"Scikit-learn ElasticNet MSE: {mse_sklearn}")