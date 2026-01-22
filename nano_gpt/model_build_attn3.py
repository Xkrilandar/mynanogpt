import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

@dataclass
class GPTConfig:
    """Configuration class for the GPT model."""
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    bias: bool = True
    n_kv_head: int = 2
    window = 4

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    x:   (B, n_head, T, head_dim)
    cos: (1, 1, T, head_dim)
    sin: (1, 1, T, head_dim)
    """
    x_even = x[..., ::2]
    x_odd  = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    x_half_rotated = x_rot.flatten(-2)
    return (x * cos) + (x_half_rotated * sin)



class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with optional Flash Attention support."""
    def __init__(self,config,block_size):
        super().__init__()
        T = block_size
        C = config.n_embd
        W = config.window
        Hq = config.n_head
        Hkv = config.n_kv_head
        Dh = C // Hq


        self.c_q = nn.Linear(C, Hq * Dh, bias=False)
        self.c_k = nn.Linear(C, Hkv * Dh, bias=False)
        self.c_v = nn.Linear(C, Hkv * Dh, bias=False)
        self.c_proj = nn.Linear(C, C, bias=False)

        causal_mask = torch.tril(torch.ones(T,T,dtype=torch.bool))

        sliding_mask = torch.tril(torch.ones(T,T,dtype=torch.bool), diagonal=-W)
        #self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.register_buffer("sliding_window", ~causal_mask & sliding_mask, persistent=False)

        self.Hq = Hq
        self.Hkv = Hkv
        self.Dh = Dh
        

    def forward(self, x):
        B, T, C = x.size()
        Hq = self.Hq
        Hkv = self.Hkv
        Dh = self.Dh

        Q = self.c_q(x).view(B,T,Hq,Dh).transpose(1,2) #B,Hq,T,Dh
        K = self.c_k(x).view(B,T,Hkv,Dh).transpose(1,2) #B,Hkv,T,Dh
        V = self.c_v(x).view(B,T,Hkv,Dh).transpose(1,2) #B,Hkv,T,Dh

        Q = norm(Q)
        K = norm(V)

        g = Hq // Hkv
        Q = Q.view(B,Hkv,g,T,Dh)
        K = K.unsqueeze(2)
        scores = Q @ K.transpose(-1,-2) #B,Hkv,g,T,T
        masked_scores = scores.masked_fill(self.sliding_window, float("-inf")) 
        y = F.softmax(masked_scores, dim=-1) #B,Hkv,g,T,T

        V = V.unsqueeze(2) 
        y = y @ V #B,Hkv,g,T,Dh
        y = y.contiguous().view(B,Hq,T,Dh)
        y = y.transpose(1,2) #B,T,Hq,Dh
        y = y.contiguous().view(B,T,C) #B,T,C
        y = self.c_proj(y) #B,T,C
        y = norm(y)
        return y






class MLP(nn.Module):
    """Feedforward network used in the Transformer block."""
    def __init__(self, config):
        super().__init__()
        self.expand = nn.Linear(config.n_embd, 4 * config.n_embd, config.bias)
        self.reduce = nn.Linear(4 * config.n_embd, config.n_embd, config.bias)

    def forward(self, x):
        x = self.expand(x)
        x = F.relu(x).square()
        x = self.reduce(x)
        return x

class Block(nn.Module):
    """Transformer block: self-attn + MLP + residual connections."""
    def __init__(self, config, block_size):
        super().__init__()
        self.attn = CausalSelfAttention(config, block_size)
        self.MLP = MLP(config)

    def forward(self, x):
        x = x + self.attn(norm(x))
        x = x + self.MLP(norm(x))
        return x

class GPT(nn.Module):
    """GPT Language Model."""
    def __init__(self, config, vocab_size, block_size):
        super().__init__()

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config, block_size) for _ in range(config.n_layer)]),
        ))

        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    def forward(self, idx, targets=None):
        """idx and targets are (B,T), returns logits (B,T,vocab_size)"""
        device = idx.device
        b, t = idx.size()

        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """return idx + new tokens predicted"""
        for _ in range(max_new_tokens):
            logits, loss = self.forward(idx, None)
            logits = logits[:,-1,:] / max(temperature,1e-8)
            if top_k is not None:
                logits = torch.topk(logits, top_k)
                logits = logits.values

            probs = F.softmax(logits, -1)
            next_token = torch.multinomial(probs, 1)
            idx = torch.cat([idx,next_token],dim=1)
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

for iter in range(100):
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