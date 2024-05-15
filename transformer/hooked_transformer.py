import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos_enc = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = 1.0 / torch.pow(10_000, torch.arange(0, d_model, 2).float() / d_model)

        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer("pos_enc", pos_enc)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)

        x = x + self.pos_enc[:, : x.size(1), :]
        x = self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.scale)

        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, bias=True, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

        self.attn = ScaledDotProductAttention(self.depth)

        self.out = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
    
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)

        attn, attn_weights = self.attn(q, k, v, mask=mask)

        attn = self.attn_dropout(attn.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model))

        attn = self.residual_dropout(self.out(attn))

        return attn, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout)

        self.ff = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model))

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, attn_weights = self.attn(x, x, x, mask=mask)
        x = self.ln1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))

        return x, attn_weights


class TransformerClassifierConfig:
    def __init__(self, vocab_size, d_model, n_heads, dim_ff, n_layers, n_classes, max_seq_len):
        self.vocab_size = vocab_size + 3
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_ff = dim_ff
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.max_seq_len = max_seq_len


class TransformerClassifier(nn.Module):
    def __init__(self, config: TransformerClassifierConfig):
        super().__init__()

        self.d_model = config.d_model
        self.n_layers = config.n_layers
        self.n_classes = config.n_classes
        self.n_heads = config.n_heads
        self.vocab_size = config.vocab_size
        self.dim_ff = config.dim_ff

        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoder(self.d_model, config.max_seq_len)

        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    dim_ff=self.dim_ff,
                )
                for _ in range(self.n_layers)
            ]
        )

        self.fc = nn.Linear(self.d_model, self.n_classes)
        self.attn_head_weights = []

    def forward(self, x, mask=None):
        x = x.long()
        x = self.embedding(x)

        for encoder in self.encoder_layers:
            x, _ = encoder(x, mask=mask)

        x = x[:, 0]
        x = self.fc(x)

        return x


def pad_token_mask(input_ids, pad_token=1):
    return (input_ids != pad_token).unsqueeze(1).type(torch.uint8)