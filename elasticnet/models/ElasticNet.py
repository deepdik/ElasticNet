
import numpy as np


class ElasticNetModel():
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4, learning_rate=0.01):
        """
            Elastic Net linear regression model.
        
            This model combines both L1 and L2 regularization to prevent overfitting and handle multicollinearity.
        
            **Note:** Elastic Net often performs better when the input features are standardized or normalized.
            It's recommended to preprocess your data accordingly before fitting the model.
        
            Parameters:
            - alpha: float, regularization strength (must be >= 0)
            - l1_ratio: float, between 0 and 1 (proportion of L1 regularization)
            - max_iter: int, maximum number of iterations
            - tol: float, tolerance for the optimization
            - learning_rate: float, step size for the gradient descent
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.coef= None
        self.intercept = None

    def _soft_thresholding(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0.0)


    def fit(self, feature, target):
        """

        Fit Elastic Net model.
    
        Parameters:
        - feature: array-like of shape (n_samples, n_features)
          Training data. Can be a NumPy array or any array-like structure.
        - target: array-like of shape (n_samples,)
          Target values. Should be a 1D array.
    
        Note:
        - Ensure that features and target have matching numbers of samples.
        - features should not contain NaN or infinite values.
        """
        ## Data Input Validations ####
        
        #Convert inputs to NumPy arrays
        feature = np.asarray(feature)
        target = np.asarray(target)

        # Convert to float if data is not already numeric
        if not np.issubdtype(feature.dtype, np.number):
            try:
                feature = feature.astype(float)
            except ValueError:
                raise ValueError("Feature array contains non-numeric values that cannot be converted to float.")

        if not np.issubdtype(target.dtype, np.number):
            try:
                target = target.astype(float)
            except ValueError:
                raise ValueError("Target array contains non-numeric values that cannot be converted to float.")
        

        # Ensure target is 1D (flatten if necessary)
        if target.ndim != 1:
            target = target.ravel()


        if np.any(np.isnan(feature)) or np.any(np.isnan(target)):
            raise ValueError("Input data feature and target  must not contain NaN values.")

        if np.any(np.isinf(feature)) or np.any(np.isinf(target)):
            raise ValueError("Input data feature and target  must not contain infinite values.")

        if self.alpha < 0:
            raise ValueError("Regularization strength alpha must be non-negative.")

        if not 0 <= self.l1_ratio <= 1:
            raise ValueError("l1_ratio must be between 0 and 1.")
            
        # Check dimensions
        if feature.ndim != 2:
            raise ValueError("Input features must be a 2D array.")
        
        if target.ndim != 1:
            raise ValueError("Input target must be a 1D array.")
        
        if feature.shape[0] != target.shape[0]:
            raise ValueError("Number of samples in features and target must be equal.")
        if feature.size == 0 or target.size == 0:
            raise ValueError("Input data features and target must not be empty.")

        n_samples, n_features = feature.shape

        # if n_samples == 0:
        #     raise ValueError("The input data features must contain at least one sample.")


            
        feature = np.hstack([np.ones((n_samples, 1)), feature])  # Add intercept term
        w = np.zeros(n_features + 1)
        target = target.flatten()

        alpha = self.alpha
        l1_ratio = self.l1_ratio
        learning_rate = self.learning_rate

        # Precompute some constants
        l1_reg = alpha * l1_ratio
        l2_reg = alpha * (1 - l1_ratio)

        for iteration in range(self.max_iter):
            w_old = w.copy()

            # Compute prediction
            y_pred = feature.dot(w)

            # Compute gradient
            error = y_pred - target
            grad = (1.0 / n_samples) * feature.T.dot(error) + l2_reg * w

            # Update weights with gradient descent step
            w = w - learning_rate * grad

            # Apply proximal operator (soft-thresholding) for L1 regularization
            w[1:] = self._soft_thresholding(w[1:], learning_rate * l1_reg)

            # Check for convergence
            if np.linalg.norm(w - w_old, ord=2) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

        self.intercept = w[0]
        self.coef= w[1:]
        return ElasticNetModelResults(self.coef, self.intercept)


class ElasticNetModelResults():
    """
    """
    def __init__(self, coef, intercept):
        self.coef = coef
        self.intercept = intercept


    def predict(self, feature):
        """
        Predict using the Elastic Net model.

        Parameters:
        - X: array-like of shape (n_samples, n_features), samples

        Returns:
        - y_pred: array of shape (n_samples,), predicted values
        """

         # Convert feature to a NumPy array if it isn't already
        feature = np.asarray(feature)

        # Convert to float if data is not already numeric
        if not np.issubdtype(feature.dtype, np.number):
            try:
                feature = feature.astype(float)
            except ValueError:
                raise ValueError("Feature array contains non-numeric values that cannot be converted to float.")

        # Ensure self.coef is a valid numeric array
        if self.coef is None:
            raise ValueError("Model coefficients (self.coef) are not set. Ensure that the model is fitted before prediction.")
        
        self.coef = np.asarray(self.coef)

        if self.coef.dtype == object:
            try:
                self.coef = self.coef.astype(float)  # Convert to float
            except ValueError as e:
                raise ValueError(f"Model coefficients contain non-convertible values: {e}")


        return feature.dot(self.coef) + self.intercept
