import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 100
eval_iters = 10
batch_size = 64  # sequences processed in parallel
block_size = 256  # context length

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 3
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.5
    bias: bool = True

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        Q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        K = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        V = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        attn = (Q @ K.transpose(-2, -1)) / math.sqrt(self.n_embd)
        #attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('inf')) # bug 2
        attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        sm = F.softmax(attn, -1)
        attn = self.attn_dropout(sm) @ V
        out = attn.transpose(2, 1).reshape(B, T, C)

        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lin = nn.Linear(config.n_embd, 4 * config.n_embd, config.bias)
        self.GELU = nn.GELU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.lin(x)
        #x = self.dropout(x) #bug 3 
        # no dropout

        x = self.GELU(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.MLP = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        #x = self.ln2(x) + self.MLP(x) # first bug
        x = x + self.MLP(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()

        tok_embd = self.transformer.wte(idx)

        #pos = torch.arange(0, t, dtype=torch.long) # bug 4
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        pos_embd = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_embd + pos_embd)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return idx


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i:i + block_size] for i in ix])
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

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = GPT(config=GPTConfig)
m = model.to(device)
print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)

for iter in range(5000):
    xb, yb = get_batch('train')

    print(iter)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
