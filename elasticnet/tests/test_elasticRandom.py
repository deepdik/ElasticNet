import os
import sys
import numpy as np

# only for testing purpose
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score



from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
   

    # Load the dataset
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
    dataframe = pd.read_csv(url, header=None)

    data = dataframe.values
    X, y = data[:, :-1], data[:, -1]

    # Standardize the features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X_standardized = (X - X_mean) / X_std

    # Define Scikit-Learn model
    sklearn_model = ElasticNet(alpha=0.1, l1_ratio=0.5)

    # Fit the model
    sklearn_model.fit(X, y)

    # Predict on training data
    sklearn_pred = sklearn_model.predict(X)

    # Compute Mean Squared Error and R-squared
    sklearn_mse = mean_squared_error(y, sklearn_pred)
    sklearn_r2 = r2_score(y, sklearn_pred)

    print("Scikit-Learn Model:")
    print("Mean Squared Error:", sklearn_mse)
    print("R-squared:", sklearn_r2)


    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5)

    # Fit the model with standardized features
    result = model.fit(X_standardized, y)

    # Predict on training data
    normal_pred = result.predict(X_standardized)

    # Compute Mean Squared Error and R-squared
    normal_mse = mean_squared_error(y, normal_pred)
    normal_r2 = r2_score(y, normal_pred)

    print()
    print("Custom ElasticNet Model:")
    print("Mean Squared Error:", normal_mse)
    print("R-squared:", normal_r2)
