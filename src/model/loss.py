import numpy as np

class Loss:
    @staticmethod
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    @staticmethod
    def mse_derivative(y_true, y_pred):
        n = y_true.shape[0]
        return -(2/n) * (y_true - y_pred)

    @staticmethod
    def bce(y_true, y_pred):
        epsilon = 1e-15 #nambahin epsilon supaya ga log 0
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) # [cite: 17, 20]

    @staticmethod
    def bce_derivative(y_true, y_pred):
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (1/n) * ((y_pred - y_true) / (y_pred * (1 - y_pred))) # 

    @staticmethod
    def cce(y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1)) # [cite: 17, 20]

    @staticmethod
    def cce_derivative(y_true, y_pred):
        n = y_true.shape[0]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -(1/n) * (y_true / y_pred)