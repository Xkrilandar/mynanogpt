import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class GPTConfig:
    """Configuration class for the GPT model."""
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = False
    n_kv_head: int = 2
    window_size: int = 128

def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # 2nd parameter acts like a lambda

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    x_half_rotated = x_rot.flatten(-2)
    return (x * cos) + (x_half_rotated * sin)

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2 
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with optional Flash Attention support."""
    def __init__(self, config, block_size, rope_base=10000.0):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.n_kv_head = config.n_kv_head
        self.dropout = config.dropout
        self.n_embd = config.n_embd
        C = self.n_embd
        Dh = self.head_size
        Hq = self.n_head
        Hkv = self.n_kv_head
        T = block_size
        W = config.window_size

        assert Dh % 2 == 0, "RoPE requires head_dim to be even."
        assert Hq % Hkv == 0, "Number of query heads must be divisible by number of key/value heads."

        self.c_attn = nn.Linear(C, (Hq + 2 * Hkv) * Dh, bias=config.bias)
        self.c_proj = nn.Linear(C, C, bias=config.bias)

        self.dropout_1 = nn.Dropout(config.dropout)
        self.dropout_2 = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, 1, T, T)
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        sliding_mask = ~torch.tril(torch.ones(T, T, dtype=torch.bool), diagonal=-W).view(1, 1, T, T)
        self.register_buffer("sliding_window", causal_mask & sliding_mask, persistent=False)

        inv_freq = 1.0 / (rope_base ** (torch.arange(0, Dh, 2, dtype=torch.float32) / Dh))
        rope_freqs = inv_freq[None, None, :, None] * torch.arange(T, dtype=torch.float32)[None, None, None, :]
        self.register_buffer("rope_freqs", rope_freqs.permute(0, 1, 3, 2), persistent=False)


    def forward(self, x, use_cache=False, past_kv=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd) 
        present_kv = past_kv
        print("B, T, C", B, T, C)
        Dh = self.head_size
        Hq = self.n_head
        Hkv = self.n_kv_head
        assert C % self.n_head == 0
        print("Dh, Hq, Hkv", Dh, Hq, Hkv)
        q, k, v = self.c_attn(x).split([Hq*Dh, Hkv*Dh, Hkv*Dh], dim=2)

        Q = q.view(B, T, Hq, Dh)
        Q = Q.transpose(1,2)

        K = k.view(B, T, Hkv, Dh).transpose(1,2)
        V = v.view(B, T, Hkv, Dh).transpose(1,2)
        print("Q, K, V shape", (Q.shape, K.shape, V.shape))
        K = K.repeat_interleave(Hq // Hkv, dim=1)
        V = V.repeat_interleave(Hq // Hkv, dim=1)
        print("Q, K, V shape", (Q.shape, K.shape, V.shape))

        print("self rope freqs shape", self.rope_freqs.shape)
        cos = torch.repeat_interleave(self.rope_freqs[:,:,:T,:].cos(), 2, dim=-1)
        sin = torch.repeat_interleave(self.rope_freqs[:,:,:T,:].sin(), 2, dim=-1)
        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)
        Q, K = norm(Q), norm(K)
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_size)
        print("causal mask shape", self.causal_mask.shape)
        print("sliding window shape", self.sliding_window.shape)
        print("Q, K, V shape", (Q.shape, K.shape, V.shape))

        attn = scores.masked_fill(self.sliding_window[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, -1)
        attn = attn @ V 

        y = attn.transpose(2,1).reshape(B, T, C)
        y = self.c_proj(y) 
        print("Attention output shape", y.shape)
        return y


class MLP(nn.Module):
    """Feedforward network used in the Transformer block."""
    def __init__(self, config):
        super().__init__()
        self.lin = nn.Linear(config.n_embd, 4 * config.n_embd, config.bias)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, config.bias)

    def forward(self, x):
        print("MLP input shape", x.shape)
        x = self.lin(x)
        x = F.relu(x).square()  # relu^2 activation in MLP
        x = self.proj(x)
        print("MLP output shape", x.shape)
        return x

class Block(nn.Module):
    """Transformer block: self-attn + MLP + residual connections."""
    def __init__(self, config, block_size):
        super().__init__()
        self.attn = CausalSelfAttention(config, block_size)
        self.MLP = MLP(config)

    def forward(self, x, use_cache=False, past_kv=None):
        print("Block input shape", x.shape)
        x = x + self.attn(norm(x), use_cache, past_kv)
        x = x + self.MLP(norm(x))
        print("Block output shape", x.shape)
        return x

class GPT(nn.Module):
    """GPT Language Model."""
    def __init__(self, config, vocab_size, block_size):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config, block_size) for _ in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        #self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def forward(self, idx, targets=None, use_cache=False, past_kv=None):
        print("===")
        print("GPT input shape", idx.shape)
        print("===")
        """idx and targets are (B,T), returns logits (B,T,vocab_size)"""
        device = idx.device
        b, t = idx.size()
        print("idx:", idx)

        tok_embd = self.transformer.wte(idx)
        x = norm(tok_embd)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, use_cache, past_kv)
        x = norm(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.permute(0, 2, 1).reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=False, past_kv=None):
        """return idx + new tokens predicted"""
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx, None)
            if top_k is not None:
                v, _ = torch.topk(logits, k=top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf')
            logits = logits[:,-1,:] / max(temperature,1e-8)
            probs = F.softmax(logits, -1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat((idx,next_token),dim=1)
        return idx




device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 100
eval_interval = 10
eval_iters = 10
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

model = GPT(config=GPTConfig(), vocab_size=vocab_size, block_size=block_size)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
print("block size", block_size)

for iter in range(1):
    print("start")
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    optimizer.step()
    print("step ", iter)
    



    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    print("end")


# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))