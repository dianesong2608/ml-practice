import numpy as np
from collections import Counter
from math import sqrt
import warnings


class KNeighborsClassifier():
    def __init__(self, n_neighbors=5):
        self._k = n_neighbors
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The size of X_train and Y_train should be the same.")

        self._X_train = X_train
        self._y_train = y_train
        return self

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearestK_idx = np.argsort(distances)[:self._k]
        votes = Counter(self._y_train[nearestK_idx]).most_common()
        return votes[0][0]

    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def __repr__(self):
        return "KNeighborsClassifier(n_neighbors=%d)" % self._k