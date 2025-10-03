import numpy as np
from transformer import Transformer

if __name__ == "__main__":
    np.random.seed(42)

    # Hyperparams contoh
    vocab_size = 1000
    max_len    = 64
    d_model    = 128
    num_heads  = 8
    d_ff       = 512
    num_layers = 2

    model = Transformer(vocab_size, max_len, d_model, num_heads, d_ff, num_layers)

    # Dummy batch token ids
    B, S = 2, 10
    tokens = np.random.randint(0, vocab_size, size=(B, S), dtype=np.int32)

    logits, probs = model(tokens)
    print("logits shape:", logits.shape)  # (B, S, vocab_size)
    print("probs shape :", probs.shape)   # (B, vocab_size)
    print("probs[0].sum():", probs[0].sum())  # ~1.0
    print("first 5 probs:", probs[0][:5])
