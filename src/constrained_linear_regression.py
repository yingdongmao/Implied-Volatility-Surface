"""
Coordinate-descent linear regression with box constraints on the coefficients.

Used by ``state_variables.py`` to fit the volatility-smile regression

    I(x, tau)^2 - I(ATM, tau)^2 = beta_1 * (2 z_+) + beta_2 * (z_+ z_-)

under the economic restriction beta_2 = omega^2 >= 0 (a variance must be
non-negative).  Plain OLS occasionally returns a small negative beta_2 because
of interpolation noise; the constraint keeps omega real.

Self-contained re-implementation (no ``normalize`` kwarg, so it is robust to
scikit-learn >= 1.2).  Adapted from
https://github.com/avidale/weighted-quantiles / public coordinate-descent recipes.
"""
try:
    from sklearn.linear_model._base import LinearModel
except ImportError:                                   # older sklearn
    from sklearn.linear_model.base import LinearModel
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y
import numpy as np


class ConstrainedLinearRegression(LinearModel, RegressorMixin):
    def __init__(self, fit_intercept=True, copy_X=True, nonnegative=False,
                 ridge=0, lasso=0, tol=1e-15, learning_rate=1.0, max_iter=10000):
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.nonnegative = nonnegative
        self.ridge = ridge
        self.lasso = lasso
        self.tol = tol
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y, min_coef=None, max_coef=None):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=False)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        # Manual centering (replaces sklearn's _preprocess_data, whose
        # ``normalize`` kwarg was removed in sklearn >= 1.2).
        if self.fit_intercept:
            X_offset = X.mean(axis=0)
            y_offset = y.mean()
            X = X - X_offset
            y = y - y_offset
        else:
            X_offset = np.zeros(X.shape[1], dtype=float)
            y_offset = 0.0
        X_scale = np.ones(X.shape[1], dtype=float)

        self.min_coef_ = min_coef if min_coef is not None else np.repeat(-np.inf, X.shape[1])
        self.max_coef_ = max_coef if max_coef is not None else np.repeat(np.inf, X.shape[1])
        if self.nonnegative:
            self.min_coef_ = np.clip(self.min_coef_, 0, None)

        beta = np.zeros(X.shape[1]).astype(float)
        prev_beta = beta + 1
        hessian = np.dot(X.transpose(), X)
        if self.ridge:
            hessian += np.eye(X.shape[1]) * self.ridge
        loss_scale = len(y)

        step = 0
        while not (np.abs(prev_beta - beta) < self.tol).all():
            if step > self.max_iter:
                print('THE MODEL DID NOT CONVERGE')
                break
            step += 1
            prev_beta = beta.copy()
            for i in range(len(beta)):
                grad = np.dot(np.dot(X, beta) - y, X)
                if self.ridge:
                    grad += beta * self.ridge
                prev_value = beta[i]
                new_value = beta[i] - grad[i] / hessian[i, i] * self.learning_rate
                if self.lasso:
                    new_value2 = beta[i] - (grad[i] + np.sign(prev_value or new_value)
                                            * self.lasso * loss_scale) / hessian[i, i] * self.learning_rate
                    new_value = 0 if new_value2 * new_value < 0 else new_value2
                beta[i] = np.clip(new_value, self.min_coef_[i], self.max_coef_[i])

        self.coef_ = beta
        self._set_intercept(X_offset, y_offset, X_scale)
        return self
