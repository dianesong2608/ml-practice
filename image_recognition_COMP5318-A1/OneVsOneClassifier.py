import numpy as np
import copy

class OneVsOneClassifier():

    def __init__(self, estimator):
        if estimator is None:
            raise ValueError("Parameter estimator cannot be None.")

        self.estimator = estimator
        self.classes = None
        self.n_classes = None
        self.estimators_ = None

    def _fit_ovo_binary(self, X, y, label0, label1):
        cond = np.logical_or((y == label0), (y == label1))
        y = y[cond]

        y_binary = np.empty(y.shape, int)
        y_binary[y == label0] = 0
        y_binary[y == label1] = 1

        indcond = np.arange(X.shape[0])[cond]
        X_binary = X[indcond]

        estimator = copy.copy(self.estimator)
        estimator.fit(X_binary, y_binary)

        return estimator

    def fit(self, X, y):
        classes = np.unique(y)
        estimators = {}

        for i in range(len(classes) - 1):
            for j in range(i + 1, len(classes)):
                label_pair = str(i) + "-" + str(j)
                estimators[label_pair] = self._fit_ovo_binary(X, y, classes[i], classes[j])

        self.classes = classes
        self.n_classes = len(classes)
        self.estimators_ = estimators

        return self

    def _transformed_confidences(self, confidences):
        transformed_confidences = {}

        for label, confidence in confidences.items():
            transformed_confidences[label] = confidence / (3 * (np.abs(confidence) + 1))

        return transformed_confidences

    def _predict(self, votes, confidences):
        best_result = 0.0
        best_label = -1
        transformed_confidences = self._transformed_confidences(confidences)

        for label, vote in votes.items():
            if (best_result < (vote + transformed_confidences[label])):
                best_result = vote + transformed_confidences[label]
                best_label = label

        return best_label

    def predict(self, X):

        y_predict = []

        for i in range(len(X)):
            x = X[i].reshape(1, -1)

            votes = {}
            confidences = {}

            for label_pair, estimator in self.estimators_.items():

                labels = label_pair.split("-")

                if estimator.predict(x) == 1:
                    label = int(labels[1])
                else:
                    label = int(labels[0])

                if label in votes:
                    votes[label] += 1
                else:
                    votes[label] = 1

                if label in confidences:
                    confidences[label] += estimator.predict_proba(x)
                else:
                    confidences[label] = estimator.predict_proba(x)

            y_predict.append(self._predict(votes, confidences))

        return np.array(y_predict)

    def __repr__(self):
        return "OneVsOneClassifier()"