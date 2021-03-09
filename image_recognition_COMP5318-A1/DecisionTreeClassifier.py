import numpy as np
from collections import Counter

class Node:
    """
    A Tree Node

    Parameters: (for current node)
        gini - The calculated gini number
        n_samples - The number of samples
        n_samples_per_class - The number of samples for each label
        predict_class - The label predicted.
    """

    def __init__(self, gini, n_samples, n_samples_per_class, predict_class):
        self.gini = gini
        self.n_samples = n_samples
        self.n_samples_per_class = n_samples_per_class
        self.predict_class = predict_class

        ## Split from which feature/dimension and related value
        self.feature_idx = 0
        self.feature_value = 0

        ## the left and right children
        self.left = None
        self.right = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def _gini(self, y):
        # gini = 1 - Σ p^2
        y_counter = Counter(y)
        G = 1.0
        for count in y_counter.values():
            G -= (count / len(y)) ** 2
        return G

    def _search_split_val(self, x, y, y_parent_node):
        """
        Search for the best value (minimum gini) to split the x.
        """
        # step1: get index of sorted data.
        sorted_idx = np.argsort(x)

        # Go through all samples of the target column.
        y_left = [0] * self.n_classes
        y_right = y_parent_node.copy()

        ginis, dim_vals = [], []

        # Go through all possible split positions.
        for i in range(1, len(y)):
            label = y[sorted_idx[i - 1]]
            y_left[label] += 1
            y_right[label] -= 1

            # Calculate gini
            gini_left = 1.0 - sum(
                (y_left[y_i] / i) ** 2 for y_i in range(self.n_classes)
            )
            gini_right = 1.0 - sum(
                (y_right[y_i] / (len(y) - i)) ** 2 for y_i in range(self.n_classes)
            )

            gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)

            if (x[sorted_idx[i - 1]] == x[sorted_idx[i]]):
                continue

            ginis.append(gini)
            dim_vals.append((x[sorted_idx[i - 1]] + x[sorted_idx[i]]) / 2)

        return ginis, dim_vals

    def _search_split_dim(self, X, y):
        """
        To find out the best dimension(feature/column) and value of that dimension to split the dataset.
        """

        # The number of Samples per class in the parent node.
        y_parent_node = [np.sum(y == y_i) for y_i in range(self.n_classes)]

        # Gini of current node.
        best_gini = self._gini(y_parent_node)
        best_dim_idx, best_dim_val = None, None

        # Go through all features.
        for dim_idx in range(self.n_features):

            # Go through all samples of the target column.
            ginis, dim_vals = self._search_split_val(X[:, dim_idx], y, y_parent_node)

            if np.min(ginis) < best_gini:
                min_idx = np.argmin(ginis)
                best_gini = ginis[min_idx]
                best_dim_idx = dim_idx
                best_dim_val = dim_vals[min_idx]

        return best_dim_idx, best_dim_val

    def _best_splitter(self, X, y):

        # Precondition to split the node: the size of y is no less than 2.
        if len(y) <= 1:
            return None, None

        # Go through all features and all possible split values of related feature.

        return self._search_split_dim(X, y)

    def _split_single_node(self, X, y, dim_idx, dim_val):
        """
        Split tree based on given dimension and its value.
        """
        # indices of the left node
        # indices of the right node: ~idx_left
        idx_left = (X[:, dim_idx] < dim_val)

        return X[idx_left], y[idx_left], X[~idx_left], y[~idx_left]

    def _build_tree(self, X, y, depth=0):

        """
        Build a decision tree by recursively finding the best dimension and value to split the tree.

        Here best means that finding calculated gini as small as possible
        by going through all dimension and all possible value of that dimension.
        """

        ## step1: init a node
        n_samples_per_class = [np.sum(y == y_i) for y_i in range(self.n_classes)]

        node = Node(
            gini=self._gini(y),
            n_samples=len(y),
            n_samples_per_class=n_samples_per_class,
            predict_class=np.argmax(n_samples_per_class)
        )

        """
        Build the tree recursively until reaching the given maximum depth.
        """
        ## step2: build trees recursively.
        if depth < self.max_depth:
            # Search the best way (the dimenstion has minimum gini) to split the node.
            dim_idx, dim_val = self._best_splitter(X, y)

            if dim_idx is not None:
                X_left, y_left, X_right, y_right = self._split_single_node(X, y, dim_idx, dim_val)

                node.feature_idx = dim_idx
                node.feature_value = dim_val

                node.left = self._build_tree(X_left, y_left, depth + 1)
                node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        """Fit decision tree classifier."""
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree_ = self._build_tree(X, y)
        return self

    def predict(self, X):
        """Predict classes for X."""
        return [self._predict(inputs) for inputs in X]

    def _predict(self, x_predict):
        """Predict a class for a sample."""
        node = self.tree_
        while node.left:
            if x_predict[node.feature_idx] < node.feature_value:
                node = node.left
            else:
                node = node.right
        return node.predict_class

    def __repr__(self):
        return "DecisionTreeClassifier()"

