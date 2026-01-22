import numpy as np
class MultiLayerPerceptron:
    def __init__(self, hidden_size=16, lr=0.01, n_iters=1000):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_iters = n_iters
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    def fit(self, X, y):
        self.classes, y_encoded = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)
        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_encoded] = 1
        self.W1 = np.random.randn(n_features, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, n_classes) * 0.01
        self.b2 = np.zeros((1, n_classes))
        for _ in range(self.n_iters):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self._sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._sigmoid(z2)
            error = a2 - y_onehot
            dW2 = np.dot(a1.T, error)
            db2 = np.sum(error, axis=0, keepdims=True)
            d_hidden = np.dot(error, self.W2.T) * self._sigmoid_derivative(a1)
            dW1 = np.dot(X.T, d_hidden)
            db1 = np.sum(d_hidden, axis=0, keepdims=True)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
    def predict(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._sigmoid(z2)
        predictions = np.argmax(a2, axis=1)
        return self.classes[predictions]
