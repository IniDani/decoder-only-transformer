import numpy as np
from transformer import Transformer

if __name__ == "__main__":
    np.random.seed(42)

    vocab_size = 1000
    max_len    = 64
    d_model    = 128
    num_heads  = 8
    d_ff       = 512
    num_layers = 2

    # === Coba RoPE + weight tying + kumpulkan attention ===
    model = Transformer(
        vocab_size, max_len,
        d_model, num_heads, d_ff, num_layers,
        positional_type = "rope",    # "sinusoidal" atau "rope"
        weight_tying = True          # aktifkan weight tying
    )

    B, S = 2, 10
    tokens = np.random.randint(0, vocab_size, size = (B, S), dtype = np.int32)

    logits, probs, attn_list = model(tokens, collect_attn=True)
    print("logits shape:", logits.shape)     # (B,S,V)
    print("probs shape :", probs.shape)      # (B,V)
    print("attn blocks:", len(attn_list))    # = num_layers
    print("attn[0] shape:", attn_list[0].shape)  # (B,H,S,S)

    # === Visualisasi ===
    # Simpan head 0, block 0 ke .npy untuk diinspeksi/plot belakangan
    np.save("attn_block0_head0.npy", attn_list[0][0, 0])  # (S,S)

    # === (Opsional) Plot heatmap kalau matplotlib tersedia ===
    try:
        import matplotlib.pyplot as plt
        plt.imshow(attn_list[0][0, 0], aspect='auto')
        plt.title("Attention Heatmap (Block 0, Head 0, Sample 0)")
        plt.xlabel("Key positions")
        plt.ylabel("Query positions")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig("attn_block0_head0.png")
        print("Saved heatmap to attn_block0_head0.png")
    except Exception as e:
        print("Matplotlib not available or plotting failed:", e)
