import numpy as np

class ThermometerEncoder:
    def __init__(self, minimum, maximum, resolution: int = 16):
        self.minimum = minimum
        self.maximum = maximum
        self.resolution = resolution

    def __repr__(self):
        return f"ThermometerEncoder(minimum={self.minimum}, maximum={self.maximum}, resolution={self.resolution})"

    def __str__(self):
        return repr(self)

    def encode(self, X):
        X = np.asarray(X)

        if X.ndim == 0:
            f = lambda i: X > self.minimum + i*(self.maximum - self.minimum)/self.resolution
        elif X.ndim == 1:
            f = lambda i, j: X[j] > self.minimum + i*(self.maximum - self.minimum)/self.resolution
        else:
            f = lambda i, j, k: X[k, j] > self.minimum + i*(self.maximum - self.minimum)/self.resolution 
        return  np.fromfunction(
            f,
            (self.resolution, *reversed(X.shape)),
            dtype=int
        ).astype(int)

    def decode(self, pattern):
        pattern = np.asarray(pattern)
        # TODO: Check if pattern is at least a vector
        # TODO: Check if pattern length or number of rows is equal to resolution
        # TODO: Check if pattern is a binary array
        if pattern.ndim == 1:
            # TODO: Test np.count_nonzero
            popcount = np.sum(pattern)
            return self.minimum + popcount*(self.maximum - self.minimum)/self.resolution
        return np.asarray([self.decode(pattern[..., i]) for i in range(pattern.shape[-1])])
