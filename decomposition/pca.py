import numpy as np
class PCA:
    def __init__(self, n_components=2):
        self.n_components = n_components
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_idx[:self.n_components]]
    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)