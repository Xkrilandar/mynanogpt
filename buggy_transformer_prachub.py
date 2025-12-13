import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

# 15 BUGS TO FIND!!!!

class TinyDecoder(nn.Module):
    def init(self, vocab_size, d_model=64, n_heads=4):
        super().init()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.tok = nn.Embedding(vocab_size, d_model) 
        self.pos = nn.Parameter(torch.zeros(d_model)) 
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, attn_mask=None):
        B, S = x.shape
        h = self.tok(x) + self.pos 
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        H = self.n_heads
        q = q.view(B, S, H, -1)
        k = k.view(B, S, H, -1)
        v = v.view(B, S, H, -1)
        attn = torch.matmul(q, k.transpose(-2, -1)) 
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, -1e9) 
        w = F.softmax(attn, dim=0) 
        z = torch.matmul(w, v).view(B, S, -1)
        h2 = self.proj(z)
        h3 = self.ln(h + self.drop(h2)) 
        return self.out(h3).softmax(-1) 

    def train():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = TinyDecoder(vocab_size=100).to('cpu')
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.eval()
        for step in range(200):
            x = torch.randint(1, 100, (32, 16), device=device)
            y = x 
            logits = model(x) 
            loss = F.cross_entropy(F.log_softmax(logits, 2), 
            y.float()) 
            with torch.no_grad():
                loss.backward() 
            opt.step() 
            if step % 50 == 0:
                print(step, loss.item())

    train()