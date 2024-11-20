import os
import sys
import numpy as np

from sklearn.linear_model import ElasticNet as SklearnElasticNet
import matplotlib.pyplot as plt

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    X = np.random.randn(n_samples, n_features)

    # Known coefficients with sparsity
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10)

    # Generate target variable with noise
    y = X.dot(true_coef) + np.random.randn(n_samples) * 0.1

    # Initialize and fit the custom ElasticNet model
    model = ElasticNetModel(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-6, learning_rate=0.5)
    result = model.fit(X, y)

    # Predict using the custom model
    y_pred = result.predict(X)

    # Initialize and fit Scikit-Learn's ElasticNet model for comparison
    sklearn_model = SklearnElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-6, fit_intercept=True)
    sklearn_model.fit(X, y)
    y_pred_sklearn = sklearn_model.predict(X)

    # Compare the coefficients
    print("Custom ElasticNet Coefficients:\n", model.coef)
    print("Scikit-Learn ElasticNet Coefficients:\n", sklearn_model.coef_)
    print("True Coefficients:\n", true_coef)

    # Plotting coefficients comparison
    plt.figure(figsize=(12, 6))
    plt.plot(model.coef, 'o', label='Custom ElasticNet Coefficients')
    plt.plot(sklearn_model.coef_, 'x', label='Scikit-Learn ElasticNet Coefficients')
    plt.plot(true_coef, '^', label='True Coefficients')
    plt.legend()
    plt.title('Comparison of Coefficients')
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.show()

