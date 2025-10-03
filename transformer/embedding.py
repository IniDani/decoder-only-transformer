import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # normal(0, 1) / sqrt(d_model) sering dipakai
        self.weight = (np.random.randn(vocab_size, d_model) / np.sqrt(d_model)).astype(np.float32)

        def __call__(self, token_ids):
            # token_ids: (B, S) int
            return self.weight[token_ids]   # (B, S, D)