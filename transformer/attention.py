import numpy as np
from .utils import split_heads, combine_heads, softmax

def scaled_dot_product_attention(Q, K, V, mask=None):
    Q = np.asarray(Q); K = np.asarray(K); V = np.asarray(V)
    assert Q.ndim == K.ndim == V.ndim == 4, f"Q/K/V harus 4D, dapat {Q.ndim}/{K.ndim}/{V.ndim}"
    Dh = Q.shape[-1]

    scores = np.matmul(Q, np.transpose(K, (0,1,3,2))) / np.sqrt(Dh)  # (B,H,S,S)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        scores = np.where(mask, -1e9, scores)

    attn = softmax(scores, axis=-1)             # (B,H,S,S)
    out  = np.matmul(attn, V)                   # (B,H,S,Dh)

    out  = np.asarray(out)
    attn = np.asarray(attn)
    assert out.ndim == 4, f"SDPA out harus 4D, dapat {out.ndim}"
    return out, attn

class MultiHeadAttention:
    def __init__(self, d_model, num_heads, rope = None):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope
        self.last_attn = None   # untuk visualisasi

        Dh = d_model
        s = 1/np.sqrt(Dh)
        self.Wq = (np.random.randn(Dh, Dh)*s).astype(np.float32)
        self.Wk = (np.random.randn(Dh, Dh)*s).astype(np.float32)
        self.Wv = (np.random.randn(Dh, Dh)*s).astype(np.float32)
        self.Wo = (np.random.randn(Dh, Dh)*s).astype(np.float32)
        self.bq = np.zeros((Dh,), dtype=np.float32)
        self.bk = np.zeros((Dh,), dtype=np.float32)
        self.bv = np.zeros((Dh,), dtype=np.float32)
        self.bo = np.zeros((Dh,), dtype=np.float32)

    def __call__(self, x, mask = None, positions = None):
        # x: (B, S, D)
        x = np.asarray(x)
        assert x.ndim == 3, f"Input MHA harus 3D (B,S,D), dapat {x.ndim}D"

        Q = x @ self.Wq + self.bq
        K = x @ self.Wk + self.bk
        V = x @ self.Wv + self.bv

        Q = split_heads(Q, self.num_heads)  # (B,H,S,Dh)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        if self.rope is not None:
            Q = self.rope.apply(Q, positions)
            K = self.rope.apply(K, positions)

        out, attn = scaled_dot_product_attention(Q, K, V, mask)  # (B,H,S,Dh)
        out = combine_heads(out)  # (B,S,D)

        # Sanity checks
        if not isinstance(out, np.ndarray):
            raise TypeError(f"MHA out tipe {type(out)} (harus np.ndarray)")
        if out.ndim != 3:
            raise ValueError(f"MHA out harus 3D, dapat {out.ndim}D dengan shape {getattr(out, 'shape', None)}")
        B, S, D = out.shape
        if D != self.d_model:
            raise ValueError(f"D mismatch setelah combine_heads: {D} vs d_model {self.d_model}")

        # Simpan attention untuk visualisasi
        self.last_attn = attn   # (B,H,S,S)

        # Proyeksi akhir: (B,S,D) x (D,D) -> (B,S,D)
        out = np.einsum('bsd,dd->bsd', out, self.Wo) + self.bo[None, None, :]
        return out, attn