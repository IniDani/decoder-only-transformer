import numpy as np
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm
from .embedding import Embedding
from .positional import SinusoidalPositionalEncoding
from .rotary import RotaryEmbedding
from .utils import softmax, causal_mask

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, rope = None):
        self.ln1 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, num_heads, rope = rope)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, activation = "gelu")

    def __call__(self, x, mask, positions = None):
        # pre-norm
        y, _ = self.mha(self.ln1(x), mask, positions)   # (B,S,D)
        x = x + y
        z = self.ffn(self.ln2(x))
        x = x + z
        return x

class Transformer:
    def __init__(
        self,
        vocab_size,
        max_len,
        d_model = 128,
        num_heads = 8,
        d_ff = 512,
        num_layers = 2,
        positional_type = "sinusoidal",   # "sinusoidal" | "rope"
        weight_tying = True               # True: Wo = E^T
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.weight_tying = weight_tying

        self.embed = Embedding(vocab_size, d_model)
        if positional_type == "sinusoidal":
            self.posenc = SinusoidalPositionalEncoding(max_len, d_model)
            rope = None
        elif positional_type == "rope":
            self.posenc = None  # RoPE tidak ditambahkan ke x; diaplikasikan ke Q/K
            rope = RotaryEmbedding(dim = d_model // num_heads, max_len = max_len)
        else:
            raise ValueError("positional_type harus 'sinusoidal' atau 'rope'")

        self.blocks = [TransformerBlock(d_model, num_heads, d_ff, rope = rope) for _ in range(num_layers)]

        if weight_tying:
            # tidak buat Wo terpisah; pakai embedding.weight.T saat proyeksi
            self.Wo = None
            self.bo = np.zeros((vocab_size,), dtype = np.float32)
        else:
            self.Wo = np.random.randn(d_model, vocab_size).astype(np.float32) / np.sqrt(d_model)
            self.bo = np.zeros((vocab_size,), dtype = np.float32)

        # tempat menyimpan attention (untuk visualisasi)
        self.collected_attn = []

    def __call__(self, token_ids, collect_attn = False):
        # token_ids: (B,S) int
        B, S = token_ids.shape
        x = self.embed(token_ids)          # (B,S,D)

        positions = np.arange(S, dtype=np.int32)

        # Sinusoidal: tambahkan ke x; RoPE: tidak (nanti di Q/K)
        if self.posenc is not None:
            x = self.posenc(x)

        mask = causal_mask(S)  # (1,1,S,S)
        self.collected_attn = []

        for blk in self.blocks:
            x = blk(x, mask, positions)
            # simpan attention setiap blok (dari MHA di dalamnya)
            if collect_attn:
                self.collected_attn.append(blk.mha.last_attn)  # (B,H,S,S)

        # Output projection (weight tying atau tidak)
        if self.weight_tying:
            # logits = x @ E^T
            logits = np.einsum('bsd,dv->bsv', x, self.embed.weight.T) + self.bo[None, None, :]
        else:
            logits = np.einsum('bsd,dv->bsv', x, self.Wo) + self.bo[None, None, :]

        last_logits = logits[:, -1, :]     # (B,V)
        probs = softmax(last_logits, axis=-1)

        if collect_attn:
            return logits, probs, self.collected_attn  # list of (B,H,S,S)
        return logits, probs