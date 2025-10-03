import numpy as np

class SinusoidalPositionalEncoding:
    def __init__(self, max_len, d_model):
        pe = np.zeros((max_len, d_model), dtype = np.float32)
        position = np.arange(0, max_len, dtype = np.float32)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2, dtype = np.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe    # (S, D)
    
    def __call__(self, x):
        # x: (B, S, D)
        S = x.shape[1]
        return x + self.pe[None, :S, :]