import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

class LogisticRegressionNumPy:
    def __init__(self, input_dim, lr=0.01):
        self.w = np.zeros((input_dim, 1))
        self.b = 0.0
        self.lr = lr

    def fit(self, X, y, epochs=100):
        m = X.shape[0]
        y = np.asarray(y).reshape(-1, 1)
        X = np.asarray(X)
        for _ in range(epochs):
            z = np.dot(X, self.w) + self.b
            a = sigmoid(z)
            dw = (1 / m) * np.dot(X.T, (a - y))
            db = (1 / m) * np.sum(a - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

class DeepNeuralNetworkNumPy:
    def __init__(self, layers_dims, dropout_rate=0.2):
        """
        layers_dims: lista com número de neurónios por camada, ex: [input_dim, 64, 32, 1]
        """
        self.dropout_rate = dropout_rate
        self.params = {}
        for l in range(1, len(layers_dims)):
            self.params[f'W{l}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
            self.params[f'b{l}'] = np.zeros((layers_dims[l], 1))

    def forward(self, X, training=True):
        cache = {'A0': X.T}
        L = len(self.params) // 2
        
        for l in range(1, L):
            Z = np.dot(self.params[f'W{l}'], cache[f'A{l-1}']) + self.params[f'b{l}']
            A = relu(Z)
            # Dropout
            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) >= self.dropout_rate) / (1 - self.dropout_rate)
                A *= mask
                cache[f'D{l}'] = mask
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
            
        # Última camada (Sigmoid para classificação binária)
        ZL = np.dot(self.params[f'W{L}'], cache[f'A{L-1}']) + self.params[f'b{L}']
        AL = sigmoid(ZL)
        cache[f'Z{L}'] = ZL
        cache[f'A{L}'] = AL
        return AL, cache

    def train_step(self, X, y, lr=0.01):
        m = X.shape[0]
        y = y.reshape(1, -1)
        AL, cache = self.forward(X, training=True)
        L = len(self.params) // 2
        
        grads = {}
        dZ = AL - y
        
        for l in range(L, 0, -1):
            grads[f'dW{l}'] = (1 / m) * np.dot(dZ, cache[f'A{l-1}'].T)
            grads[f'db{l}'] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dA = np.dot(self.params[f'W{l}'].T, dZ)
                if self.dropout_rate > 0:
                    dA *= cache[f'D{l-1}']
                dZ = dA * relu_derivative(cache[f'Z{l-1}'])
                
        # Atualização dos parâmetros
        for l in range(1, L + 1):
            self.params[f'W{l}'] -= lr * grads[f'dW{l}']
            self.params[f'b{l}'] -= lr * grads[f'db{l}']