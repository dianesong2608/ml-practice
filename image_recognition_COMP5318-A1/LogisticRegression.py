import numpy as np
import warnings

class LogisticRegression():
    def __init__(self):
        self.theta = None
        self.intercept_ = None
        self.coef_ = None

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def _J(self, X, y, theta):
        y_hat = self._sigmoid(np.dot(X, theta))
        try:
            return - np.sum(y * np.log(y_hat) + (1 - y_hat) * np.log(1 - y_hat)) / len(y)
        except:
            return float("inf")

    def _dJ(self, X, y, theta):
        y_hat = self._sigmoid(np.dot(X, theta))
        return np.dot(X.T, (y_hat - y)) / len(y)

    def _gradient_descent(self, X, y, initial_theta, learning_rate, n_iters=1e4, epsilon=1e-8):
        theta = initial_theta
        i_iter = 0

        while (i_iter < n_iters):
            pre_theta = theta
            theta = theta - learning_rate * self._dJ(X, y, theta)

            if (abs(self._J(X, y, theta) - self._J(X, y, pre_theta)) < epsilon):
                break

            i_iter += 1

        return theta

    def fit(self, X, y, learning_rate=0.01, n_iters=1e4, epsilon=1e-8):
        if len(X) != len(y):
            raise ValueError("The size of X_train and y_train should be the same.")

        X_new = np.hstack([np.ones((X.shape[0], 1)), X])
        initial_theta = np.zeros(X_new.shape[1])

        # Find out the best theta through gradient descent.
        self.theta = self._gradient_descent(X_new, y, initial_theta, learning_rate, n_iters, epsilon)

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self

    def predict_proba(self, X):
        if self.theta is None:
            raise ValueError("The classifier should be fitted before.")

        X_new = np.hstack([np.ones((X.shape[0], 1)), X])

        return self._sigmoid(X_new.dot(self.theta))

    def predict(self, X):
        if self.theta is None:
            raise ValueError("The classifier should be fitted before.")

        proba = self.predict_proba(X)
        return np.array(proba >= 0.5, dtype='int')

    def __repr__(self):
        return "LogisticRegression()"
