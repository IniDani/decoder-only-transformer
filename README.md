Nama  : Muhammad Rafli Ramadani

NIM   : 22/497787/TK/54571

---

# Decoder-Only Transformer (NumPy Implementation)

Proyek ini merupakan implementasi **Transformer Decoder-Only (GPT-style)** dari nol menggunakan **NumPy**, tanpa library deep learning seperti PyTorch atau TensorFlow.  
Tujuannya adalah memahami alur **forward pass** model transformer, mulai dari embedding hingga menghasilkan distribusi probabilitas token berikutnya.



## 📌 Fitur Utama
- Token Embedding
- Positional Encoding (Sinusoidal atau Rotary/ROPE)
- Scaled Dot-Product Attention dengan Softmax
- Multi-Head Attention
- Feed-Forward Network (FFN)
- Residual Connection + Layer Normalization
- Causal Masking (untuk autoregressive prediction)
- Output Layer: proyeksi ke vocab + distribusi softmax
- **Bonus:** 
  - Visualisasi Attention Heatmap
  - Weight Tying
  - Rotary Positional Embedding (RoPE)



## 🚀 Cara Menggunakan
1. **Kloning repository ini**:
   ```bash
   git clone https://github.com/IniDani/decoder-only-transformer.git
   cd decoder-only-transformer

2. **Jalankan program utama**:
    ```bash
    python main.py

3. **Output yang dihasilkan**:
- logits dengan shape (batch, seq_len, vocab_size)
- probs distribusi probabilitas token berikutnya (batch, vocab_size)
- (opsional) Attention heatmap tersimpan sebagai file .png

Contoh output:
    ```bash
    logits shape: (2, 10, 1000)
    probs shape : (2, 1000)
    attn blocks: 2
    attn[0] shape: (2, 8, 10, 10)
    Saved heatmap to attn_block0_head0.png



## Hyperparameter
1. **vocab_size**:
Jumlah token unik dalam kosakata. Ukuran output layer model.
(Misal 1000 artinya model mengenali 1000 token unik.)

2. **max_len**:
Panjang maksimum sequence/token yang bisa diproses.
Dipakai untuk membangun positional encoding.

3. **d_model**:
Dimensi embedding/token representation.
Semakin besar → representasi lebih kaya, tapi komputasi lebih berat.

4. **num_heads**:
Jumlah attention head di Multi-Head Attention.
Masing-masing head memperhatikan konteks berbeda dalam sequence.

5. **d_ff**:
Ukuran hidden layer di Feed Forward Network.
Biasanya ≈ 4 × d_model.

6. **num_layers**:
Jumlah stack Transformer block.
Semakin banyak layer → model lebih dalam dan kompleks.



## Dependensi
- Python 3.9+
- NumPy
- Matplotlib (opsional untuk visualisasi)



## Catatan
Implementasi ini belum dilatih pada dataset NLP nyata, sehingga distribusi probabilitas masih mendekati uniform. Tujuan utamanya adalah untuk memahami mekanisme internal Transformer dengan implementasi manual.
