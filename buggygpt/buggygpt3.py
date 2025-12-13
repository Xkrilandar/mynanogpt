import math
import argparse
from dataclasses import dataclass
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ToySequenceDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, seed: int = 1337):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.rng = random.Random(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = [self.rng.randint(1, self.vocab_size - 1) for _ in range(self.seq_len)]
        src = torch.tensor(x, dtype=torch.long)
        tgt = torch.empty_like(src)
        tgt[:-1] = src[1:]
        tgt[-1] = 0
        return src, tgt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2,
                 d_ff: int = 256, dropout: float = 0.1, max_len: int = 128, pad_idx: int = 0,
                 tie_weights: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            #batch_first=False, # bug 8
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.head.weight = self.tok_emb.weight

    def generate_mask(self, T: int, device: torch.device):
        mask = torch.triu(torch.ones(T, T), diagonal=1)
        mask = mask.to(device)
        #mask = mask.masked_fill(mask == 1, float("inf")) # bug 6
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, src: torch.Tensor):
        B, T = src.shape
        x = self.tok_emb(src) * math.sqrt(self.d_model)
        x = self.pos_enc(x) 
        #x = x.transpose(0, 1) # bug 10

        src_key_padding_mask = (src == self.pad_idx)
        attn_mask = self.generate_mask(T, src.device)

        h = self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )

        #h = h.transpose(0, 1) # bug 11
        h = self.ln_f(h)
        logits = self.head(h)
        return logits


@dataclass
class Config:
    vocab_size: int = 50
    seq_len: int = 24
    batch_size: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.01
    pad_idx: int = 0
    max_len: int = 128


def shift_targets(x: torch.Tensor, pad_idx: int) -> Tuple[torch.Tensor, torch.Tensor]: # what is the point of this
    # padding is not shifting :)
    # bug 7
    #inp = x[:, :] 
    #tgt = x[:, :]
    inp = x[:, :-1] 
    tgt = x[:, 1:]

    #tgt[:, -1] = pad_idx 
    return inp, tgt


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int) -> float:
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        mask = targets != pad_idx
        correct = (preds == targets) & mask
        denom = mask.sum().clamp(min=1).item()
        return correct.sum().item() / denom


def train_step(model, batch, optimizer, criterion, device):
    #model.eval() # bug 3
    model.train()
    src, _ = batch
    src = src.to(device)
    inp, tgt = shift_targets(src, pad_idx=model.pad_idx)
    logits = model(inp)
    #loss = criterion(logits, tgt.to(device)) # bug 1
    loss = criterion(logits.permute(0,2,1), tgt.to(device))
    # bug 5 missing zero grad
    optimizer.zero_grad(set_to_none=True)
    loss.backward() 
    optimizer.step() 
    acc = accuracy_from_logits(logits, tgt.to(device), pad_idx=model.pad_idx)
    return loss.item(), acc


def eval_step(model, batch, criterion, device):
    #model.train() # bug 4
    model.eval()
    with torch.no_grad():
        src, _ = batch
        src = src.to(device)
        inp, tgt = shift_targets(src, pad_idx=model.pad_idx)
        logits = model(inp)
        #loss = criterion(logits, tgt.to(device)) # bug 2
        loss = criterion(logits.permute(0,2,1), tgt.to(device))
        acc = accuracy_from_logits(logits, tgt.to(device), pad_idx=model.pad_idx)
    return loss.item(), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = Config()
    device = torch.device(args.device)

    train_ds = ToySequenceDataset(num_samples=2000, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=args.seed)
    val_ds = ToySequenceDataset(num_samples=256, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size, seed=args.seed + 1)

    #train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False) # bug 13
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True) 
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MiniTransformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=cfg.max_len,
        pad_idx=cfg.pad_idx,
        tie_weights=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    step = 0
    train_iter = iter(train_loader)
    while step < args.train_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        loss, acc = train_step(model, batch, optimizer, criterion, device)

        if (step + 1) % 50 == 0:
            val_batch = next(iter(val_loader))
            vloss, vacc = eval_step(model, val_batch, criterion, device)
            print(f"step {step+1:04d} | train loss {loss:.3f} acc {acc:.3f} | val loss {vloss:.3f} acc {vacc:.3f}")
        step += 1

    x, _ = val_ds[0]
    x = x.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits[:, -1], dim=-1)
        topk = torch.topk(probs, k=5).indices.squeeze(0).tolist()
    print("Top-5 next-token preds for sample 0:", topk)


if __name__ == "__main__":
    main()
