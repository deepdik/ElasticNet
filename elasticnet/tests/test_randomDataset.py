import sys
import os

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet


from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():

    # Dataset 1: Small dataset
    X_small = np.random.randn(100, 5)
    y_small = X_small @ np.array([1.5, -2., 0., 0.5, 3.]) + np.random.randn(100) * 0.1


    # Dataset 2: Larger dataset with more features
    X_large = np.random.randn(1000, 50)
    y_large = X_large @ np.random.randn(50) + np.random.randn(1000) * 0.5

    # Dataset 3: Dataset with multicollinearity
    X_multi = np.random.randn(200, 10)
    X_multi[:, 5:] = X_multi[:, :5] + np.random.randn(200, 5) * 0.01  # Add correlated features
    y_multi = X_multi @ np.random.randn(10) + np.random.randn(200) * 0.2

    ### Test model ####

    # Initialize model
    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)

    # Fit on small dataset
    r = model.fit(X_small, y_small)
    y_pred_small = r.predict(X_small)

    # Fit on large dataset
    r = model.fit(X_large, y_large)
    y_pred_large = r.predict(X_large)

    # Fit on multicollinear dataset
    r = model.fit(X_multi, y_multi)
    y_pred_multi = r.predict(X_multi)

    print()


    ###### Initialize scikit-learn's ElasticNet ######
    sk_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    sk_model.fit(X_small, y_small)
    y_pred_sk = sk_model.predict(X_small)

    print("-------- Compare prediction -------------")
    print()
    print("-----Small data set------")
    # Compare predictions
    mse_custom = mean_squared_error(y_small, y_pred_small)
    print(f"Custom ElasticNetRegression MSE: {mse_custom}")
    mse_custom = mean_squared_error(y_small, y_pred_sk)
    print(f"Sklearn ElasticNetRegression MSE: {mse_custom}")
    print()
    print("-----Large data set------")
    ###### Initialize scikit-learn's ElasticNet ######
    sk_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    sk_model.fit(X_large, y_large)
    y_pred_sk = sk_model.predict(X_large)

    mse_custom = mean_squared_error(y_large, y_pred_large)
    print(f"Custom ElasticNetRegression MSE: {mse_custom}")
    mse_custom = mean_squared_error(y_large, y_pred_sk)
    print(f"Sklearn ElasticNetRegression MSE: {mse_custom}")
    print()


    print("-----Multicollinearity data set------")

    ###### Initialize scikit-learn's ElasticNet ######
    sk_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-4)
    sk_model.fit(X_multi, y_multi)
    y_pred_sk = sk_model.predict(X_multi)


    mse_custom = mean_squared_error(y_multi, y_pred_multi)
    print(f"Custom ElasticNetRegression MSE: {mse_custom}")
    mse_custom = mean_squared_error(y_multi, y_pred_sk)
    print(f"Sklearn ElasticNetRegression MSE: {mse_custom}")
