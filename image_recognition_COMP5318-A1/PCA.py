import numpy as np
import warnings

class PCA():
    def __init__(self, n_components):
        if n_components < 1:
            raise ValueError("The number of components must be greater than 1.")

        self._X_demean = None
        self._n_components = n_components
        self.components_ = None

    def _demean(self, X):
        return X - np.mean(X, axis=0)

    def fit(self, X):
        # step1: demean
        X_demean = self._demean(X)

        # step2: get eigenvalue & eigenvector through covariance matrix
        eigenvalue, eigenvector = np.linalg.eig(np.cov(X_demean, rowvar=0))

        # step3: get indices of eigen value by descending order
        idx_sorted_desc = np.argsort(eigenvalue)[::-1]

        self.components_ = eigenvector[:, idx_sorted_desc[:self._n_components]].T

        return self

    def transform(self, X):
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("The dimension of input matrix cannot match with that of components.")

        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        if X.shape[1] != self.components_.shape[0]:
            raise ValueError("The dimension of input matrix cannot match with that of components.")

        return np.dot(X, self.components_)

    def __repr__(self):
        return "PCA(n_components = %d)" % self._n_components