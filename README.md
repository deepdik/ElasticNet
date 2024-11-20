# Elastic Net Regression from Scratch in Python

This project implements a **Linear Regression with ElasticNet Regularization** from scratch in Python, without using pre-built libraries like Scikit-Learn or Statsmodels. The implementation uses NumPy for numerical computations.

## Group Members

| Name                  | A-Number     | Name                   |
|-----------------------|--------------|----------------------- |
| Deepak Kumar          | A20547017    | dkumar15@hawk.iit.edu  |
| Kuldeep Patel         | A20540269    | kpatel181@hawk.iit.edu |
| Shubham Dhanavade     | A20541092    | sdhanavade@hawk.iit.edu|
| Kartheek Kumar Basava | A20553532    | kbasava@hawk.iit.edu   |
|                       |              |                        |

## Table of Contents

- [Introduction](#introduction)
- [Things we have done above and beyond?](#things-we-have-done-above-and-beyond)
- [Usage](#usage)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
  - [Example](#example)
  - [Running Testcases](#running-the-tests)
- [Answers to the Questions](#answers-to-the-questions)
  - [1. What does the model you have implemented do and when should it be used?](#1-what-does-the-model-you-have-implemented-do-and-when-should-it-be-used)
  - [2. How did you test your model to determine if it is working reasonably correctly?](#2-how-did-you-test-your-model-to-determine-if-it-is-working-reasonably-correctly)
  - [3. What parameters have you exposed to users of your implementation in order to tune performance?](#3-what-parameters-have-you-exposed-to-users-of-your-implementation-in-order-to-tune-performance)
  - [4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?](#4-are-there-specific-inputs-that-your-implementation-has-trouble-with-given-more-time-could-you-work-around-these-or-is-it-fundamental-to-the-model)
- [Structure of the Implementation](#structure-of-the-implementation)


## Introduction

**Elastic Net Regression** combines both L1 (Lasso) and L2 (Ridge) regularization techniques. It is particularly useful when dealing with datasets that have highly correlated features or when performing feature selection.

This implementation provides a custom `ElasticNetRegression` class with `fit` and `predict` methods, similar to the interface provided by Scikit-Learn models.

# Things we have done above and beyond?
- **Comprehensive Input Validation:**   We've implemented detailed checks for input data in the `fit` method, including checks for NaNs, infinite values, and dimension mismatches. This makes  model robust and user-friendly.

- **Proximal Gradient Descent Implementation:**   Instead of a basic gradient descent, we implemented proximal gradient descent by incorporating the soft-thresholding function for L1 regularization. This is a more sophisticated optimization technique suitable for models with L1 penalties.

- **Modular Design:**  
  By structuring  code with clear, separate methods (e.g., `_soft_thresholding`), we've made it more maintainable and easier to extend.

- **Convergence Check with Tolerance:**  
  We've included a convergence criterion based on the L2 norm of the weight updates, which enhances the efficiency of the training process by potentially reducing unnecessary iterations.

- **Visualization Tools: Convergence Plots:** 
Added functionality to plot the loss function or weight updates over iterations to visualize convergence. Visualize how coefficients change with different values of alpha and l1_ratio.

- **Model Evaluation Metrics:** 
 Implement methods to calculate evaluation metrics such as Mean Squared Error (MSE) and R-squared.

- **Comprehensive Testing Across Multiple Datasets**

    - **Small Dataset:**  
        We've tested our model on a small synthetic dataset to verify basic functionality.

    - **Large Dataset with More Features:**  
      By scaling up to a larger dataset with more features, we've demonstrated our model's capability to handle higher-dimensional data efficiently.

    - **Dataset with Multicollinearity:**  
      Testing on a dataset with multicollinear features showcases the strength of Elastic Net in handling multicollinearity, which is one of its primary advantages over other regularization methods.
     - **Real-World Dataset:**  
        Using the California Housing dataset to compare performances demonstrates the applicability of our model to real-world scenarios and datasets.

- **Comparison with Scikit-Learn's ElasticNet:**
  - **Performance Evaluation:**  
    We've compared the Mean Squared Error (MSE) of our custom implementation with scikit-learn's ElasticNet, providing quantitative evidence of our model's accuracy.
  

### Installation

1. **Clone the Repository**:

   ```bash
   git clone <repo-link>
   ```

2. **Install Dependencies**:

   This implementation only requires NumPy, which is usually installed by default. To run our test cases we need some more library. 
   <span style="color: red;">We are using scikit-learn, scipy, and matplotlib for testing purposes only. This is just to compare our model results with the standard library results.</span>

   ```bash
   pip install -r requirements.txt
   ```


## Data Preparation

- **Features (`X`):** Should be a 2D array-like structure with shape `(n_samples, n_features)`.
- **Target (`y`):** Should be a 1D array-like structure with length `n_samples`.
- **Standardization:** It's recommended to standardize or normalize features for better performance. Elastic Net regression often performs better when the input features are standardized or normalized. This is because the regularization terms penalize the coefficients, and if features are on different scales, it can bias the model.

    ##### How to Standardize Features
    To standardize feature matrix `X`, calculate the mean and standard deviation for each feature, then subtract the mean and divide by the standard deviation:
    
    ```python
    # Assuming X is your feature matrix
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_standardized = (X - X_mean) / X_std
    ```

## Model Parameters

- **alpha (float):** Regularization strength. Must be a positive float.
- **l1_ratio (float):** The mix of L1 and L2 regularization. `l1_ratio=0` corresponds to L2 (Ridge), `l1_ratio=1` to L1 (Lasso).
- **max_iter (int):** Maximum number of iterations for the optimization algorithm.
- **tol (float):** Tolerance for stopping criteria.
- **learning_rate (float):** Step size for gradient descent updates.


### Example

Here's how to use the `ElasticNetModel` class:
### Note: 
If you get error like python module elesticnet not found then export this module in python path by this command -
```
"export PYTHONPATH="$PYTHONPATH:<project-root-location>"
```

Like for me it was -
```
"export PYTHONPATH="$PYTHONPATH:/Volumes/MyLab/assignments/cs-584/project-01-remote/Project1"
```

```python

import numpy as np
from elasticnet.models.ElasticNet import ElasticNetModel

# Generate synthetic data
np.random.seed(0)
n_samples, n_features = 100, 5
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, -2.0, 0.0, 0.0, 3.0])
y = np.dot(X, true_coef) + np.random.randn(n_samples) * 0.5  # Add noise

# Split into training and testing sets
split_ratio = 0.8
split_index = int(n_samples * split_ratio)
X_train = X[:split_index]
y_train = y[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]

# Initialize and fit the model
model = ElasticNetModel(alpha=0.1, l1_ratio=0.7)
result = model.fit(X_train, y_train)

# Predict on test set
y_pred = result.predict(X_test)

# Evaluate the model
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

mse = mean_squared_error(y_test, y_pred)
print("Mean squared error:", mse)
print("Coefficients:", model.coef)
print("Intercept:", model.intercept)
```

## Running the Tests

To run the tests using pytest, you can execute the following command:

#### 1. **Professor-Given Test Cases**
To run the test cases go inside **tests** folder. I am using pytes t(python library) to run the test cases.

- **Test Case 1**: Verifies the initialization of the model with default parameters.
    ```bash
    pytest -s test_ElasticNetModel.py
    ```
#### 2. **Custom Test Cases (Added by Us)**

- **Custom Test Case 1**: This test evaluates the performance of a custom ElasticNetModel compared to Scikit-Learn's ElasticNet on a housing dataset from the UCI Machine Learning Repository. The features are standardized for the custom model, while the Scikit-Learn model uses raw data. Both models are fit to the data, and their predictions are compared using mean squared error (MSE) and R-squared (R²) scores to assess their accuracy and goodness of fit.
    ```bash
    pytest -s test_elasticRandom.py
    ```
- **Custom Test Case 2**: This test evaluates the prediction accuracy and coefficient similarity of a custom ElasticNetModel compared to Scikit-Learn's ElasticNet, using synthetic data with 100 samples and 50 features. It compares model coefficients and predictions against known true coefficients. Create Graphical output for compare.
    ```bash
    pytest -s test_graphical.py
    ```
- **Custom Test Case 3**: Ensures proper handling of a dataset with a high number of features compared to samples. This test evaluates the performance of a custom ElasticNetModel compared to Scikit-Learn's ElasticNet, using synthetic data with 100 samples and 500 features (where only the first 10 features are relevant). It measures the model's mean squared error (MSE) and R-squared (R²) scores, compares the true and learned coefficients, and identifies the maximum difference between the two models' coefficients.
    ```bash
    pytest -s test_high-dimension.py
    ```
- **Custom Test Case 4**: This test evaluates the performance of a custom ElasticNetModel on three different datasets: a small dataset, a larger dataset with more features, and a dataset with multicollinearity. The test compares the predictions of the custom model to Scikit-Learn's ElasticNet model by computing the mean squared error (MSE) for each dataset. This helps assess the consistency and performance of the custom model across different scenarios, such as small data, large data, and data with correlated features.
    ```bash
    pytest -s test_randomDataset.py
    ```
- **Custom Test Case 5**: This test evaluates the performance of a custom ElasticNetModel and Scikit-Learn's ElasticNet on the real-world California Housing dataset. The features are standardized, and both models are fit to the dataset. After predictions are made, the models' performances are compared by calculating the mean squared error (MSE) for each, providing insight into how well the custom model performs compared to the established Scikit-Learn implementation on real data.
    ```bash
    pytest -s test_real_world.py
    ```




## Answers to the Questions

### 1. What does the model you have implemented do and when should it be used?

**Answer:**

The implemented model is **Linear Regression with ElasticNet regularization**. This model extends ordinary linear regression by adding a regularization term that is a combination of both L1 (Lasso) and L2 (Ridge) penalties. The objective is to minimize the following function:

The Elastic Net regression model minimizes the following objective function:
(1 / 2n) * Σ (i=1 to n) [ (yᵢ - Xᵢᵀ β)² ] + λ [ α * |β|₁ + (1 - α) / 2 * |β|₂² ]
Where:
- n is the number of samples.
- yᵢ is the target variable for the i-th sample.
- Xᵢ is the feature vector for the i-th sample.
- β is the coefficient vector (model weights).
- λ is the regularization strength (corresponds to `alpha` in the code).
- α (corresponds to `l1_ratio` in the code) controls the balance between L1 (Lasso) and L2 (Ridge) penalties:
  - When α = 1, the model reduces to Lasso (pure L1 regularization).
  - When α = 0, the model reduces to Ridge (pure L2 regularization).
  - Intermediate values of α mix both L1 and L2 regularization.


**When to use this model:**

- **Feature Selection and Multicollinearity**: ElasticNet is useful when there are multiple correlated features. It can select groups of correlated features and perform variable selection due to the L1 penalty.
- **High-dimensional Data**: In datasets where the number of features exceeds the number of samples, ElasticNet helps prevent overfitting by regularizing the coefficients.
- **Balance Between Lasso and Ridge**: If neither pure Lasso nor pure Ridge regression gives satisfactory results, ElasticNet provides a compromise by combining both penalties.

### 2. How did you test your model to determine if it is working reasonably correctly?

**Answer:**

I tested the model using the following steps:

- **Synthetic Data Generation**: Created a synthetic dataset with known coefficients to simulate a linear relationship. This allows for direct comparison between the true coefficients and those estimated by the model.
- **Training and Testing Split**: Split the dataset into training and testing sets to evaluate the model's ability to generalize to unseen data.
- **Model Fitting**: Used the `fit` method to train the model on the training data.
- **Prediction and Evaluation**: Predicted the target variable on the test set using the `predict` method and calculated the Mean Squared Error (MSE) to assess performance.
- **Coefficient Analysis**: Compared the estimated coefficients to the true coefficients to check if the model accurately identified the significant features.

The results indicated that the model could recover the underlying relationship in the data, and the MSE was within an acceptable range, demonstrating that the implementation works correctly.

### 3. What parameters have you exposed to users of your implementation in order to tune performance?

**Answer:**

The implementation exposes the following parameters:

- **`alpha`** (Regularization strength λ): Controls the overall amount of regularization applied. A higher value of `alpha` increases the penalty for large coefficients, promoting simpler models.
- **`l1_ratio`** (Mixing parameter α): Determines the balance between L1 and L2 penalties.
  - `l1_ratio = 0` corresponds to pure Ridge regression (L2 penalty).
  - `l1_ratio = 1` corresponds to pure Lasso regression (L1 penalty).
  - Values between 0 and 1 provide a mix of both penalties (ElasticNet).
- **`tol`** (Tolerance for optimization): Sets the threshold for the maximum change in coefficients between iterations. The algorithm stops when the change is less than `tol`.
- **`max_iter`** (Maximum number of iterations): Limits the number of iterations the algorithm runs to prevent excessive computation time in cases where convergence is slow.
- **`learning_rate (float)`** Step size for gradient descent updates.

By adjusting these parameters, users can fine-tune the model to balance bias and variance, control overfitting, and handle specific characteristics of their dataset.

### 4. Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental to the model?

**Answer:**

**Potential issues:**

- **Non-convergence with Certain Data**: The coordinate descent algorithm may struggle to converge on datasets with certain properties, such as extremely high multicollinearity or poorly scaled features.
- **Sensitivity to Feature Scaling**: Although the implementation standardizes features internally, datasets with features on vastly different scales may still pose challenges.
- **Large-scale Datasets**: For very large datasets (in terms of samples or features), the implementation may be computationally inefficient due to the simplicity of the algorithm.

**Possible Workarounds:**

- **Improved Optimization Algorithms**: Implementing more advanced optimization techniques like stochastic coordinate descent or using accelerated gradient methods could enhance convergence and efficiency.
- **Adaptive Learning Rates**: Incorporating adaptive learning rates or line search methods might improve the convergence behavior on challenging datasets.
- **Parallelization**: Utilizing parallel computing techniques could speed up computations, especially for high-dimensional data.

These issues are not fundamental limitations of the ElasticNet model itself but are related to the specific implementation and optimization approach. With additional time and resources, the implementation can be enhanced to handle these challenges more effectively.

## Structure of the Implementation

The implementation consists of a single Python class:

- **`ElasticNetModel`**: The main class that encapsulates the Elastic Net regression model.
  - **Methods**:
    - `__init__(self, alpha=1.0, l1_ratio=0.5, tol=1e-4, max_iter=1000)`: Initializes the model with the given parameters.
    - `fit(self, X, y)`: Fits the model to the training data using the coordinate descent algorithm.
  - **Attributes**:
    - `self.coef`: Stores the learned coefficients.
    - `self.intercept`: Stores the learned intercept term.

- **`ElasticNetModelResults`**: Result class to predict method.
    - `predict(self, X)`: Predicts target values for the given input data.

**Key Components:**

- **Soft Thresholding Function**: Implements the soft thresholding operator required for L1 regularization.
  
  ```python
  def soft_threshold(rho, lam):
      """Soft thresholding operator for L1 penalty."""
      if rho > lam:
          return rho - lam
      elif rho < -lam:
          return rho + lam
      else:
          return 0.0
  ```

- **Feature Scaling**: The `fit` method standardizes the features to have zero mean and unit variance, which is crucial for the regularization penalties to work effectively.

- **Coordinate Descent Algorithm**: Iteratively updates each coefficient by fixing others, which is efficient for L1(Lasso) and L2(ridge) penalties.