import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps = 1e-5):
        self.gamma = np.ones((d_model,), dtype = np.float32)
        self.beta  = np.zeros((d_model,), dtype = np.float32)
        self.eps = eps

    def __call__(self, x):
        # x: (B, S, D)
        mean = x.mean(axis = -1, keepdims = True)
        var  = x.var(axis = -1, keepdims = True)
        x_hat = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta
