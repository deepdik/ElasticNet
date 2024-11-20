import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet as SklearnElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from elasticnet.models.ElasticNet import ElasticNetModel

def test_predict():
    
    # Set random seed for reproducibility
    
    np.random.seed(42)

    # Parameters for synthetic data
    n_samples = 100  # Number of samples
    n_features = 500  # High number of features (p >> n)

    # Generate independent features
    X = np.random.randn(n_samples, n_features)

    # Generate sparse coefficients (only a few non-zero)
    true_coef = np.zeros(n_features)
    true_coef[:10] = np.random.randn(10)  # Only first 10 features are relevant

    # Generate target variable with some noise
    y = X @ true_coef + np.random.randn(n_samples) * 0.5  # '@' is matrix multiplication

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the Elastic Net model
    #model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, learning_rate=0.001)
    model = ElasticNetModel(alpha=1.0, l1_ratio=0.5, max_iter=1000, learning_rate=0.001)

    # Fit the model
    result = model.fit(X_scaled, y)

    # Predict on training data
    y_pred = result.predict(X_scaled)

    # Compute Mean Squared Error and R-squared
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    print(f"\nElasticNetRegression Model Performance:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared Score: {r2:.4f}")

    # Extract learned coefficients
    learned_coef = model.coef

    # Compare the first 20 coefficients
    print("\nComparison of True and Learned Coefficients (first 20 features):")
    print("Feature Index\tTrue Coefficient\tLearned Coefficient")
    for i in range(20):
        print(f"{i}\t\t{true_coef[i]:.4f}\t\t{learned_coef[i]:.4f}")

    # Plot the true and learned coefficients
    plt.figure(figsize=(12, 6))
    plt.plot(true_coef, label='True Coefficients')
    plt.plot(learned_coef, label='Learned Coefficients', linestyle='--')
    plt.legend()
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of True and Learned Coefficients')
    plt.show()

    # ============================================
    # Comparison with scikit-learn's ElasticNet
    # ============================================

    # Initialize scikit-learn's ElasticNet
    sklearn_model = SklearnElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4)

    # Fit the model
    sklearn_model.fit(X_scaled, y)

    # Predict on training data
    sklearn_pred = sklearn_model.predict(X_scaled)

    # Compute Mean Squared Error and R-squared
    sklearn_mse = mean_squared_error(y, sklearn_pred)
    sklearn_r2 = r2_score(y, sklearn_pred)

    print(f"\nScikit-Learn ElasticNet Model Performance:")
    print(f"Mean Squared Error: {sklearn_mse:.4f}")
    print(f"R-squared Score: {sklearn_r2:.4f}")

    # Compare coefficients
    plt.figure(figsize=(12, 6))
    plt.plot(learned_coef, label='ElasticNetRegression Coefficients')
    plt.plot(sklearn_model.coef_, label='Scikit-Learn Coefficients', linestyle='--')
    plt.legend()
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Comparison of ElasticNetRegression and Scikit-Learn Coefficients')
    plt.show()

    # Print difference in coefficients
    coef_difference = learned_coef - sklearn_model.coef_
    max_diff = np.max(np.abs(coef_difference))
    print(f"\nMaximum difference in coefficients between implementations: {max_diff:.4e}")

