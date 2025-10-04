import numpy as np

class RotaryEmbedding:
    """
    RoPE: Rotary Position Embedding
    Referensi: Su et al., 2021 (RoFormer)
    Implementasi ringkas untuk head_dim genap.
    """
    def __init__(self, dim, max_len=2048, base=10000.0):
        assert dim % 2 == 0, "RoPE membutuhkan dimensi head (Dh) genap"
        self.dim = dim
        self.max_len = max_len
        self.base = base

        half = dim // 2
        # frekuensi (θ) per dimensi genap
        inv_freq = (base ** (-np.arange(0, half, dtype=np.float32) / half)).astype(np.float32)
        t = np.arange(max_len, dtype=np.float32)  # [0..S-1]
        freqs = np.outer(t, inv_freq)  # (S, half)
        # precompute cos/sin tabel: (S, half)
        self.cos = np.cos(freqs).astype(np.float32)
        self.sin = np.sin(freqs).astype(np.float32)

    def apply(self, x, positions=None):
        """
        x: (B, H, S, Dh) — RoPE diaplikasikan ke dim terakhir (Dh).
        positions: (S,) optional; default arange(S)
        Return: (B, H, S, Dh) rotated
        """
        B, H, S, Dh = x.shape
        assert Dh == self.dim and Dh % 2 == 0
        if positions is None:
            positions = np.arange(S, dtype=np.int32)

        cos = self.cos[positions]   # (S, half)
        sin = self.sin[positions]   # (S, half)

        # pisahkan jadi (B,H,S,half) + (B,H,S,half)
        x1 = x[..., :Dh//2]
        x2 = x[..., Dh//2:]

        # broadcasting cos/sin: (1,1,S,half)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

        # rotasi kompleks: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return np.concatenate([y1, y2], axis=-1)
