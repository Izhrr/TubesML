import numpy as np
import matplotlib.pyplot as plt
from .activation import Activations
from .loss import Loss
from .regularizer import Regularizer
from .initializer import Initializer

import pickle

class Layer:
    def __init__(self, n_input, n_output, activation_name):
        self.n_input = n_input
        self.n_output = n_output
        self.activation_name = activation_name.lower()

        # Inisialisasi placeholder untuk bobot dan bias 
        self.W = None 
        self.b = None
        
        # Placeholder untuk gradien 
        self.dW = None
        self.db = None
        
        # Cache untuk menyimpan nilai selama forward pass
        self.input_cache = None
        self.net_cache = None
        self.output_cache = None

    def forward(self, x):
        """Hitung output = f(Wx + b)"""
        self.input_cache = x
        
        # Linear transformation: net = Wx + b
        self.net_cache = np.dot(x, self.W) + self.b
        
        # Activation: o = f(net)
        activation_func = getattr(Activations, self.activation_name)
        self.output_cache = activation_func(self.net_cache)
        
        return self.output_cache
    
class FFNN:
    def __init__(self, loss_name, regularization_type=None, lam=0):
        self.layers = []
        self.loss_name = loss_name.lower()
        self.regularization_type = regularization_type # 'l1', 'l2', None
        self.lam = lam
        self.history = {'train_loss': [], 'val_loss': []}

    def add_layer(self, layer):
        """Tambah layer"""
        self.layers.append(layer)

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out
    
    def backward(self, y_true, y_pred):
        """
        Backpropagation untuk menghitung gradien (dW dan db) 
        pada setiap layer.
        """
        # 1. Hitung delta untuk output layer
        # dE/do (Turunan Loss)
        loss_derivative_func = getattr(Loss, f"{self.loss_name}_derivative")
        error_signal = loss_derivative_func(y_true, y_pred)
        
        # Iterasi mundur melalui layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            # dO/dNet (Turunan Aktivasi)
            activation_deriv_func = getattr(Activations, f"{layer.activation_name}_derivative")
            da_dnet = activation_deriv_func(layer.net_cache)
            
            # Hitung Delta (Error Term)
            # Untuk output layer: delta = (dE/do) * (do/dnet)
            # Untuk hidden layer: delta = (sum delta_next * W_next) * (do/dnet)
            if i == len(self.layers) - 1:
                # Output Layer case
                delta = error_signal * da_dnet
            else:
                # Hidden Layer case: delta dari layer depannya
                next_layer = self.layers[i+1]
                delta = np.dot(delta_next, next_layer.W.T) * da_dnet
            
            # 2. Hitung gradien Bobot (dW) dan Bias (db)

            # dW = delta * input (transpose untuk batch processing)
            layer.dW = np.dot(layer.input_cache.T, delta)

            # db = jumlah delta di sepanjang batch
            layer.db = np.sum(delta, axis=0, keepdims=True)
            
            # Save delta untuk digunakan oleh layer sebelumnya
            delta_next = delta

    def update_weights(self, learning_rate):
        """
        Memperbarui W dan b menggunakan Gradient Descent standar.
        """
        for layer in self.layers:
            reg_penalty = 0
            if self.regularization_type == 'l1':
                reg_penalty = Regularizer.l1_derivative(layer.W, self.lam)
            elif self.regularization_type == 'l2':
                reg_penalty = Regularizer.l2_derivative(layer.W, self.lam)
            
            # Update: W_new = W_old - alpha * (dW + penalty)
            layer.W -= learning_rate * (layer.dW + reg_penalty)
            layer.b -= learning_rate * layer.db

    def fit(self, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, verbose=1):
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # 1. Shuffle data untuk setiap epoch agar model tidak belajar urutan
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # 2. Mini-batch processing
            for i in range(0, n_samples, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # Forward pass
                y_pred_batch = self.forward(X_batch)
                
                # Backward pass & Gradien
                self.backward(y_batch, y_pred_batch)
                
                # Perbarui bobot
                self.update_weights(learning_rate)
            
            # 3. Hitung Loss untuk History
            train_pred = self.forward(X_train)
            val_pred = self.forward(X_val)
            
            loss_func = getattr(Loss, self.loss_name)
            train_loss = loss_func(y_train, train_pred)
            val_loss = loss_func(y_val, val_pred)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # 4. Verbose output
            if verbose == 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    def plot_weights_distribution(self, layer_indices):
        """Menampilkan plot distribusi bobot untuk layer tertentu"""
        for idx in layer_indices:
            weights = self.layers[idx].W.flatten()
            plt.hist(weights, bins=30, alpha=0.7, label=f'Layer {idx}')
        plt.title("Distribusi Bobot")
        plt.xlabel("Nilai Bobot")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.show()

    def plot_gradients_distribution(self, layer_indices):
        """Menampilkan plot distribusi gradien untuk layer tertentu"""
        for idx in layer_indices:
            gradients = self.layers[idx].dW.flatten()
            plt.hist(gradients, bins=30, alpha=0.7, label=f'Layer {idx}')
        plt.title("Distribusi Gradien")
        plt.xlabel("Nilai Gradien")
        plt.ylabel("Frekuensi")
        plt.legend()
        plt.show()

    def save(self, filename):
        """Menyimpan instance model ke file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """Memuat instance model dari file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
    def initialize_weights(self, method='uniform', seed=None, **kwargs):
        """
        Melakukan inisialisasi bobot dan bias untuk seluruh layer[cite: 21].
        method: 'zero', 'uniform', atau 'normal'[cite: 23, 24, 27].
        """
        for i, layer in enumerate(self.layers):
            # Tentukan dimensi: (jumlah_input, jumlah_neuron) [cite: 11]
            # Kita perlu tahu n_input dari layer sebelumnya atau parameter input_dim
            if i == 0:
                n_in = kwargs.get('input_dim')
            else:
                n_in = self.layers[i-1].W.shape[1]
                
            n_out = layer.n_output # Diambil dari atribut layer
            shape = (n_in, n_out)

            if method == 'zero':
                layer.W = Initializer.zero_initialization(shape)
                layer.b = Initializer.zero_initialization((1, n_out))
            elif method == 'uniform':
                lb = kwargs.get('lower_bound', -0.05)
                ub = kwargs.get('upper_bound', 0.05)
                layer.W = Initializer.uniform_initialization(shape, lb, ub, seed)
                layer.b = Initializer.uniform_initialization((1, n_out), lb, ub, seed)
            elif method == 'normal':
                mean = kwargs.get('mean', 0.0)
                var = kwargs.get('variance', 0.01)
                layer.W = Initializer.normal_initialization(shape, mean, var, seed)
                layer.b = Initializer.normal_initialization((1, n_out), mean, var, seed)