import numpy as np
class SingleLayerPerceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
    def fit(self, X, y):
        self.classes = np.unique(y)
        if len(self.classes) != 2:
            raise ValueError("Single Layer Perceptron supports binary classification only.")
        y_binary = np.where(y == self.classes[0], -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.w) + self.b
                y_pred = np.sign(linear_output)
                if y_pred == 0:
                    y_pred = -1
                if y_binary[idx] != y_pred:
                    self.w += self.lr * y_binary[idx] * x_i
                    self.b += self.lr * y_binary[idx]
    def predict(self, X):
        linear_output = np.dot(X, self.w) + self.b
        preds = np.sign(linear_output)
        preds[preds == -1] = self.classes[0]
        preds[preds == 1] = self.classes[1]
        return preds
