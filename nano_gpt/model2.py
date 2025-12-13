import math
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_iters = 5000
eval_interval = 10
eval_iters = 10
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?

@dataclass
class GPTConfig:
    """Configuration class for the GPT model."""
    block_size: int = 1024
    vocab_size: int = 65
    n_layer: int = 1
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention with optional Flash Attention support."""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.ln1 = nn.Linear(self.n_embd, self.n_embd*3)
        self.d1 = nn.Dropout(config.dropout)
        self.ln2 = nn.Linear(self.n_embd, self.n_embd)
        self.d2 = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # x 64 256 768
        B,T,C = x.size()
        x = self.ln1(x)
        q,k,v = x.split(self.n_embd, dim=-1)
        Q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)#64 3072 256
        K = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        V = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        K_t = K.transpose(-2,-1) #64, 256, 3072 what is the diff between transpose and permute (i would guess contiguity)
        attn = Q @ K_t / math.sqrt(self.n_head) #64, 3072, 256
        attn = F.softmax(attn, dim=-1) 
        attn = attn @ V # 64, 3072, 256
        attn = attn.transpose(1,2).reshape(B,T,C)
        attn = self.d1(attn) 
        attn = self.ln2(attn) # 64, 256, 768
        attn = self.d2(attn)
        return attn
        #1 linear, 1 attn = Q @ K.T / sqrt(head_size), 1 mask, 1 softmax, 1 dropout, attn @ V, 1 linear, 1 dropout

class MLP(nn.Module):
    """Feedforward network used in the Transformer block."""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.ln1 = nn.Linear(self.n_embd, self.n_embd*4)
        self.gelu = nn.GELU()
        self.ln2 = nn.Linear(self.n_embd*4, self.n_embd)
        self.d1 = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.ln1(x)
        x = self.gelu(x)
        x = self.ln2(x)
        x = self.d1(x)
        return x
        # 1 norm, 1 gelu, 1 norm, 1 dropout

class Block(nn.Module):
    """Transformer block: self-attn + MLP + residual connections."""
    def __init__(self, config):
        super().__init__()
        self.n_embd = config.n_embd
        self.ln1 = nn.LayerNorm(self.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(self.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
        # 1 norm, 1 attention w residual, 1 norm, 1 MLP w residual

class GPT(nn.Module):
    """GPT Language Model."""
    def __init__(self, config):
        super().__init__()
        #init weights mean =0, std = 0.02
        self.n_embd = config.n_embd
        self.tok_e = nn.Embedding(config.vocab_size, self.n_embd)
        self.pos_e = nn.Embedding(config.block_size, self.n_embd)
        self.d1 = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for i in range(config.n_layer))
        self.norm = nn.LayerNorm(self.n_embd)

        self.lm_head = nn.Linear(self.n_embd, config.vocab_size, bias=config.bias)
        #self.tok_e.weigth = self.lm_head.weigth
        self.apply(self.init_weigths)

    def init_weigths(self, module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, 0, 0.2)
            
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, 0, 0.2)


    def forward(self, idx, targets=None):
        """Forward pass of the model."""  
        tok_e = self.tok_e(idx)
        b,t=idx.size()
        pos = torch.arange(0, t, 1, dtype=torch.long, device=device)
        pos_e = self.pos_e(pos)
        x = self.d1(tok_e + pos_e)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        if targets == None:
            loss = None
        else:
            loss = F.cross_entropy(logits.transpose(1,2), targets)
        return logits, loss
        # 1 token embedding (nn.Embedding( vocab_size) of idx), 1 pos embedding (block_size) of arange(0,idx.size())
        # dropout (tok_e + pos_e)
        # blocks * n_layer
        # 1 layer norm
        # 1 linear layer
        # 1 F.cross_entropy



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
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(vocab_size)
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


model = GPT(config=GPTConfig)
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

for iter in range(200):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

