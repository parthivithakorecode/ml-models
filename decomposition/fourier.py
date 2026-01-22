import numpy as np

class FourierTransform:
    """
    Applies Discrete Fourier Transform (DFT) to input signals.
    Works on each sample independently.
    """

    def transform(self, X):
        """
        Parameters:
            X : numpy array of shape (n_samples, n_features)

        Returns:
            X_ft : numpy array (complex), Fourier transformed data
        """
        # Apply FFT along feature axis
        X_ft = np.fft.fft(X, axis=1)
        return X_ft
