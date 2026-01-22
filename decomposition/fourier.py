import numpy as np
import math
class FourierTransform:
    def __init__(self):
        pass
    def _DFT(self, x):
        N = len(x)
        X = np.zeros(N, dtype=complex)
        for k in range(N):
            summation = 0
            for n in range(N):
                angle = -2j * np.pi * k * n / N
                summation += x[n] * np.exp(angle)
            X[k] = summation / math.sqrt(N)
        return X
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        spectra = []
        for row in X:
            spectrum = self._DFT(row)
            spectra.append(spectrum)
        return np.array(spectra)
