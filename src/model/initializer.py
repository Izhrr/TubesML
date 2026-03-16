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

    @staticmethod
    def xavier_initialization(shape, n_in, n_out, seed=None):
        """Inisialisasi Xavier (Glorot) untuk aktivasi seperti Sigmoid atau Tanh."""
        if seed is not None:
            np.random.seed(seed)
        # Xavier Normal: mean=0, variance=2 / (n_in + n_out)
        std_dev = np.sqrt(2.0 / (n_in + n_out))
        return np.random.normal(0.0, std_dev, shape)

    @staticmethod
    def he_initialization(shape, n_in, seed=None):
        """Inisialisasi He untuk aktivasi berbasis ReLU."""
        if seed is not None:
            np.random.seed(seed)
        # He Normal: mean=0, variance=2 / n_in
        std_dev = np.sqrt(2.0 / n_in)
        return np.random.normal(0.0, std_dev, shape)