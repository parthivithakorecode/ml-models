import numpy as np
class BinarySVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (
                        2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    )
                    self.b -= self.lr * y_[idx]
    def project(self, X):
        return np.dot(X, self.w) - self.b
class SVM:
    def fit(self, X, y):
        self.classes = np.unique(y)
        if len(self.classes) == 2:
            self.models = {}
            c1, c2 = self.classes
            y_binary = np.where(y == c1, 1, 0)
            model = BinarySVM()
            model.fit(X, y_binary)
            self.models[c1] = model
        else:
            self.models = {}
            for c in self.classes:
                y_binary = np.where(y == c, 1, 0)
                model = BinarySVM()
                model.fit(X, y_binary)
                self.models[c] = model
    def predict(self, X):
        scores = {}
        for c, model in self.models.items():
            scores[c] = model.project(X)
        scores_matrix = np.column_stack(list(scores.values()))
        predicted_indices = np.argmax(scores_matrix, axis=1)
        classes_list = list(scores.keys())
        return np.array([classes_list[i] for i in predicted_indices])
