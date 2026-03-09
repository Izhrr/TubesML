import numpy as np

class Initializer:
    @staticmethod
    def zero_initialization(shape):
        """Inisialisasi bobot dengan nilai 0."""
        return np.zeros(shape)

    @staticmethod
    def uniform_initialization(shape, lower_bound, upper_bound, seed=None):
        """Inisialisasi bobot dengan distribusi uniform."""
        if seed is not None:
            np.random.seed(seed)
        return np.random.uniform(lower_bound, upper_bound, shape)

    @staticmethod
    def normal_initialization(shape, mean, variance, seed=None):
        """Inisialisasi bobot dengan distribusi normal."""
        if seed is not None:
            np.random.seed(seed)
        std_dev = np.sqrt(variance)
        return np.random.normal(mean, std_dev, shape)