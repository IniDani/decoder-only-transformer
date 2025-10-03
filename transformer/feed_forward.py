import numpy as np
from .utils import relu  # atau gelu

class FeedForward:
    def __init__(self, d_model, d_hidden, activation = "gelu"):
        self.W1 = np.random.randn(d_model, d_hidden).astype(np.float32) / np.sqrt(d_model)
        self.b1 = np.zeros((d_hidden,), dtype = np.float32)
        self.W2 = np.random.randn(d_hidden, d_model).astype(np.float32) / np.sqrt(d_hidden)
        self.b2 = np.zeros((d_model,), dtype = np.float32)
        self.activation = activation

    def __call__(self, x):
        h = x @ self.W1 + self.b1
        if self.activation == "gelu":
            # inline gelu utk menghindari import siklus
            h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * np.power(h, 3))))
        else:
            h = relu(h)
        y = h @ self.W2 + self.b2
        return y
