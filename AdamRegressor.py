from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np
from utils import data_iter


class AdamRegressor(BaseEstimator, RegressorMixin):
    coef_ = None
    t_ = None
    loss_hist_ = None
    loss_epoch_ = None
    n_iter_ = None

    def __init__(self, max_iter=500, eta0=0.1, power_t=0.5, tol=1e-3, n_iter_no_change=2):
        self.max_iter = max_iter
        self.eta0 = eta0
        self.power_t = power_t
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change

    def fit(self, X, Y, batch_size, coef_init=None):
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
        Y = check_array(Y, ensure_2d=False)
        # Check that X and Y have correct shape
        X, Y = check_X_y(X, Y, y_numeric=True)
        # add bias to X
        X = np.c_[np.ones((X.shape[0])), X]
        beta_1 = 0.9
        beta_2 = 0.999
        epsilon = 1.e-8
        self.t_ = 0
        self.n_iter_ = 0
        self.loss_hist_ = []
        self.loss_epoch_ = []
        loss_count = 0
        loss = float("inf")
        m_t = 0
        v_t = 0
        theta = coef_init
        for i in range(int(self.max_iter)):
            self.n_iter_ += 1
            loss_epoch = []
            for x, y in data_iter(X, Y, batch_size):
                self.t_ += 1
                error = x.dot(theta) - y
                loss = error.dot(error) / x.shape[0]
                loss_epoch.append(loss)
                gradient = (x.T.dot(error)) / x.shape[0]
                m_t = beta_1*m_t + (1 - beta_1) * gradient
                v_t = beta_2*v_t + (1 - beta_2) * gradient * gradient
                m_hat = m_t / (1 - (beta_1**self.t_))
                v_hat = v_t / (1 - (beta_2**self.t_))
                theta = theta - self.eta0*m_hat / (v_hat**self.power_t + epsilon)

            if self.loss_hist_:
                loss_prev = min(self.loss_hist_[-self.n_iter_no_change + 1:])
            else:
                loss_prev = float("inf")
            loss = np.average(loss_epoch)
            self.loss_epoch_.append(loss_epoch)
            self.loss_hist_.append(loss)
            if loss + self.tol > loss_prev:
                if loss_count == self.n_iter_no_change - 1:
                    break
                else:
                    loss_count += 1
            else:
                loss_count = 0
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
