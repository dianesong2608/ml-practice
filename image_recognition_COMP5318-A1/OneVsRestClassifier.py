import numpy as np
import copy

class OneVsRestClassifier():

    def __init__(self, algorithm):
        self.estimator = algorithm
        self.n_classes = None
        self.estimators_ = None

    def fit(self, X, y):
        ## step1: calculate the number of classifiers in total
        classes = np.unique(y)

        ## step2: training a classifier for each label
        estimators = {}
        for label in classes:
            y_binary = np.array([1 if y_i == label else 0 for y_i in y])

            estimator = copy.copy(self.estimator)
            estimator.fit(X, y_binary)
            estimators[label] = estimator

        self.n_classes = classes
        self.estimators_ = estimators
        return self

    def predict(self, X):
        y_predict = []

        for i in np.arange(len(X)):
            best_proba = 0.0

            best_y = -1

            for label, estimator in self.estimators_.items():
                proba = estimator.predict_proba(X[i].reshape(1, -1))

                if best_proba < proba:
                    best_proba = proba
                    best_y = label

            y_predict.append(best_y)

        return np.array(y_predict)

    def __repr__(self):
        return "OneVsRestClassifier()"