import numpy as np

class Regularizer:
    @staticmethod
    def l1(weights, lam):
        """Penalti L1: jumlah nilai absolut bobot."""
        return lam * np.sum(np.abs(weights))

    @staticmethod
    def l1_derivative(weights, lam):
        """Turunan L1 untuk update bobot."""
        return lam * np.sign(weights)

    @staticmethod
    def l2(weights, lam):
        """Penalti L2: jumlah kuadrat bobot."""
        return (lam / 2) * np.sum(np.square(weights))

    @staticmethod
    def l2_derivative(weights, lam):
        """Turunan L2 untuk weight decay."""
        return lam * weights