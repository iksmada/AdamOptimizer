from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
import math


class AdamRegressor(BaseEstimator, RegressorMixin):
    coef_ = None
    t_ = None
    loss_hist_ = None

    def __init__(self, n_iter=500, eta0=0.1, power_t=0.5):
        self.n_iter = n_iter
        self.eta0 = eta0
        self.power_t = power_t

    def fit(self, X, Y: np.uint8, coef_init):
        # coef_init validation
        if coef_init is not None:
            coef_init = np.asarray(coef_init, dtype=np.float64, order="C")
            coef_init = coef_init.ravel()
            if coef_init.shape != (X.shape[1],):
                raise ValueError("Provided coef_init does not match dataset.")
            coef_init = np.r_[np.zeros(1), coef_init]
        else:
            coef_init = np.zeros((X.shape[1] + 1,))

        # label validation
        Y = check_array(Y, dtype='uint8', ensure_2d=False)
        # get classes
        self.classes_ = np.unique(Y)
        # Check that X and Y have correct shape
        X, Y = check_X_y(X, Y, y_numeric=True)
        # add bias to X
        X = np.c_[np.ones((X.shape[0])), X]
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.e-8
        self.t_ = 0
        self.loss_hist_ = []
        m_t = 0
        v_t = 0
        theta = coef_init
        for i in range(int(self.n_iter)):
            np.random.shuffle(X)
            for x, y in zip(X, Y):
                self.t_ += 1
                error = x.dot(theta) - y
                self.loss_hist_.append(np.sum(error ** 2))
                gradient = (x.T.dot(error))
                m_t = beta_1*m_t + (1 - beta_1) * gradient
                v_t = beta_2*v_t + (1 - beta_2) * gradient * gradient
                m_hat = m_t / (1 - (beta_1**self.t_))
                v_hat = v_t / (1 - (beta_2**self.t_))
                theta = theta - self.eta0*m_hat / (v_hat**self.power_t + epsilon)

        self.coef_ = theta
        return self

    def predict(self, X: np.ndarray):
        # Check is fit had been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X, ensure_min_features=self.coef_.shape[0] - 1)
        # add bias to X and remove colums in order to match lines in W
        X = np.hstack((np.ones((X.shape[0], 1)), X[:, :self.coef_.shape[0] - 1]))
        # x * w
        return X.dot(self.coef_)
