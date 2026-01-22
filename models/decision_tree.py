import numpy as np
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        prob = counts / counts.sum()
        return 1 - np.sum(prob ** 2)
    def _best_split(self, X, y):
        best_gini = float("inf")
        best_idx, best_thr = None, None
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for thr in thresholds:
                left_mask = X[:, feature_idx] <= thr
                right_mask = X[:, feature_idx] > thr
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gini_left = self._gini(y[left_mask])
                gini_right = self._gini(y[right_mask])
                weighted_gini = (
                    (left_mask.sum() * gini_left + right_mask.sum() * gini_right)
                    / n_samples
                )
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_idx = feature_idx
                    best_thr = thr
        return best_idx, best_thr
    def _build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        num_labels = len(np.unique(y))
        if (
            depth >= self.max_depth
            or num_labels == 1
            or num_samples < self.min_samples_split
        ):
            return np.bincount(y).argmax()
        feature_idx, threshold = self._best_split(X, y)
        if feature_idx is None:
            return np.bincount(y).argmax()
        left_mask = X[:, feature_idx] <= threshold
        right_mask = X[:, feature_idx] > threshold
        left_subtree = self._build_tree(
            X[left_mask], y[left_mask], depth + 1
        )
        right_subtree = self._build_tree(
            X[right_mask], y[right_mask], depth + 1
        )
        return (feature_idx, threshold, left_subtree, right_subtree)
    def fit(self, X, y):
        self.classes, y_encoded = np.unique(y, return_inverse=True)
        self.tree = self._build_tree(X, y_encoded, 0)
    def _predict_one(self, x, tree):
        if not isinstance(tree, tuple):
            return tree
        feature_idx, threshold, left, right = tree
        if x[feature_idx] <= threshold:
            return self._predict_one(x, left)
        else:
            return self._predict_one(x, right)
    def predict(self, X):
        preds = [self._predict_one(x, self.tree) for x in X]
        return self.classes[np.array(preds)]
