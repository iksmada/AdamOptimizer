from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class LinearRegressor(BaseEstimator, RegressorMixin):
    weights_ = None

    def __init__(self, gamma=1):
        self.gamma = gamma

    def fit(self, X, y):
        # label validation
        y = check_array(y, ensure_2d=False)
        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True)
        # add bias to X
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        if X.shape[0] > X.shape[1]:
            #  w = (x_t*x + gamma*I)^-1 * x_t*y
            self.weights_ = np.linalg.inv(X.transpose().dot(X) + self.gamma * np.identity(X.shape[1])).dot(X.transpose().dot(y))
        else:
            #  w = x_t * (x * x_t)^-1 * y
            self.weights_ = X.T.dot(np.linalg.inv(X.dot(X.T))).dot(y)
        return self

    def predict(self, X: np.ndarray):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X, ensure_min_features=self.weights_.shape[0]-1)
        # add bias to X and remove colums in order to match lines in W
        X = np.hstack((np.ones((X.shape[0], 1)), X[:, :self.weights_.shape[0]-1]))
        # x * w
        return X.dot(self.weights_)
