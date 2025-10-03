import numpy as np
from .utils import split_heads, combine_heads, softmax

def scaled_dot_product_attention(Q, K, V, mask = None):
    # Q,K,V: (B, H, S, Dh)
    Dh = Q.shape[-1]
    scores = np.matmul(Q, np.transpose(K, (0,1,3,2))) / np.sqrt(Dh)  # (B,H,S,S)

    if mask is not None:
        # mask True = disallowed; set ke -inf
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores, axis = -1)  # (B,H,S,S)
    out = np.matmul(attn, V)         # (B,H,S,Dh)
    return out, attn

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        Dh = d_model

        # Parameter matriks proyeksi
        self.Wq = np.random.randn(Dh, Dh).astype(np.float32) / np.sqrt(Dh)
        self.Wk = np.random.randn(Dh, Dh).astype(np.float32) / np.sqrt(Dh)
        self.Wv = np.random.randn(Dh, Dh).astype(np.float32) / np.sqrt(Dh)
        self.Wo = np.random.randn(Dh, Dh).astype(np.float32) / np.sqrt(Dh)
        self.bq = np.zeros((Dh,), dtype = np.float32)
        self.bk = np.zeros((Dh,), dtype = np.float32)
        self.bv = np.zeros((Dh,), dtype = np.float32)
        self.bo = np.zeros((Dh,), dtype = np.float32)

    def __call__(self, x, mask = None):
        # x: (B, S, D)
        Q = x @ self.Wq + self.bq
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        Q = split_heads(Q, self.num_heads)  # (B,H,S,Dh)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)  # (B,H,S,Dh), (B,H,S,S)
        out = combine_heads(out)  # (B,S,D)

        out = out @ self.Wo + self.bo  # (B,S,D)
        return out, attn
