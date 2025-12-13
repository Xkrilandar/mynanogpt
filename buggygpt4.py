import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Linear transformations
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        if mask is not None:
            #scores = scores.masked_fill(mask == 0, -1e9) # bug 1
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        return self.W_o(attention_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # bug 2 
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=5000, dropout=0.1):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)
        
        return self.output_projection(x)

def create_causal_mask(seq_len):
    """Create a causal mask for decoder attention"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.bool()

# Test the model
if __name__ == "__main__":
    # Model parameters
    vocab_size = 10000
    d_model = 512
    n_heads = 8
    n_layers = 6
    d_ff = 2048
    seq_len = 100
    batch_size = 32
    
    # Create model
    model = SimpleTransformer(vocab_size, d_model, n_heads, n_layers, d_ff)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create causal mask
    mask = create_causal_mask(seq_len)
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    # Forward pass
    try:
        output = model(input_ids, mask)
        print(f"Output shape: {output.shape}")
        print("Model ran successfully!")
    except Exception as e:
        print(f"Error: {e}")
        
    # Simple training loop test
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(2):
        target = input_ids
        
        optimizer.zero_grad()
        output = model(input_ids, mask)
        
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")