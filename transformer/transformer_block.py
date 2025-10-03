import numpy as np
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .embedding import Embedding
from .positional import SinusoidalPositionalEncoding
from .utils import softmax, causal_mask

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation = "gelu")

    def __call__(self, x, mask):
        # pre-norm
        y, _ = self.mha(self.ln1(x), mask)   # (B,S,D)
        x = x + y
        z = self.ffn(self.ln2(x))
        x = x + z
        return x

class Transformer:
    def __init__(self, vocab_size, max_len, d_model = 128, num_heads = 8, d_ff = 512, num_layers = 2):
        self.embed = Embedding(vocab_size, d_model)
        self.posenc = SinusoidalPositionalEncoding(max_len, d_model)
        self.blocks = [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        # output projection
        self.Wo = np.random.randn(d_model, vocab_size).astype(np.float32) / np.sqrt(d_model)
        self.bo = np.zeros((vocab_size,), dtype = np.float32)

    def __call__(self, token_ids):
        # token_ids: (B,S) int
        x = self.embed(token_ids)          # (B,S,D)
        x = self.posenc(x)                 # (B,S,D)
        mask = causal_mask(token_ids.shape[1])  # (1,1,S,S)

        for blk in self.blocks:
            x = blk(x, mask)

        logits = x @ self.Wo + self.bo     # (B,S,V)
        # distribusi untuk token terakhir (auto-regressive)
        last_logits = logits[:, -1, :]     # (B,V)
        probs = softmax(last_logits, axis = -1)
        return logits, probs
