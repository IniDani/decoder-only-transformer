import numpy as np

def xavier_init(fan_in, fan_out):
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*np.power(x,3))))

def causal_mask(seq_len):
    m = np.triu(np.ones((seq_len, seq_len), dtype=np.bool_), k=1)
    return m[None, None, :, :]

def split_heads(x, num_heads):
    # x: (B, S, D) -> (B, H, S, Dh)
    x = np.asarray(x)
    B, S, D = x.shape
    assert D % num_heads == 0, "D must be divisible by num_heads"
    Dh = D // num_heads
    x = x.reshape(B, S, num_heads, Dh)
    x = np.transpose(x, (0, 2, 1, 3))
    return np.ascontiguousarray(x)

def combine_heads(x):
    # x: (B, H, S, Dh) -> (B, S, H*Dh)
    x = np.asarray(x)
    B, H, S, Dh = x.shape
    x = np.transpose(x, (0, 2, 1, 3)).reshape(B, S, H*Dh)
    return np.ascontiguousarray(x)