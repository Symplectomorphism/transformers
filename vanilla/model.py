import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, seq_len: int = 5000) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin function to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply the cos function to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.size(1), :]).required_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # scale (learnable)
        self.bias = nn.Parameter(torch.zeros(1)) # shift (learnable)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) # don't flatten the dimension on which mean is applied
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1 (bias is set to True by default)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  
        self.h = h
        assert d_model & h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, self.d_k * h)
        self.w_k = nn.Linear(d_model, self.d_k * h)
        self.w_v = nn.Linear(d_model, self.d_k * h)

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout: nn.Dropout=None):
        d_k = query.size(-1)

        # (batch_size, h, seq_len, d_k) -> (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # transpose the last two dims.
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return torch.matmul(attention_scores, value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, d_k * h)
        key = self.w_k(k)   # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, d_k * h)
        value = self.w_v(v) # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, d_k * h)
 
        # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(-1, query.size(1), self.h, self.d_k).transpose(1, 2) 
        # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        key = key.view(-1, key.size(1), self.h, self.d_k).transpose(1, 2) 
        # (batch_size, seq_len, d_k * h) -> (batch_size, seq_len, h, d_k) -> (batch_size, h, seq_len, d_k)
        value = value.view(-1, value.size(1), self.h, self.d_k).transpose(1, 2) 

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h * d_k
        x = x.transpose(1, 2) # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k)
        x = x.contiguous().view(x.size(0), -1, self.h * self.d_k) # (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, h * d_k)
        
        return self.w_o(x) # (batch_size, seq_len, h * d_k) -> (batch_size, seq_len, d_model)


class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(2))

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))