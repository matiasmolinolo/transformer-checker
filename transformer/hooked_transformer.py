import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        print(q.shape, k.shape, v.shape)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
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

        self.wq = nn.Linear(d_model, d_model, bias=bias)
        self.wk = nn.Linear(d_model, d_model, bias=bias)
        self.wv = nn.Linear(d_model, d_model, bias=bias)

        self.wo = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(scale=math.sqrt(self.depth))

        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def _split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.n_heads, self.depth)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.n_heads, seq_len, self.depth)
        )

    def _concat_heads(self, x):
        # When concatenating back, since n_heads = 1, it should revert to the original d_model
        batch_size, seq_len, in_features = x.size()
        batch_size //= self.n_heads
        return (
            x.reshape(batch_size, seq_len, self.n_heads, in_features)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, self.d_model)
        )

    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        scaled_attn, attn_weights = self.attn(q=q, k=k, v=v, mask=mask)

        attn = self._concat_heads(scaled_attn)

        out = self.wo(attn)
        out = self.dropout(out)

        out = self.ln(out + q)  # add + norm, residual connection

        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads=n_heads, d_model=d_model, dropout=dropout
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model), nn.ReLU()
        )

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
    def __init__(self, vocab_size, d_model, n_heads, dim_ff, n_layers, n_classes):
        self.vocab_size = vocab_size + 3
        self.d_model = d_model
        self.n_heads = n_heads
        self.dim_ff = dim_ff
        self.n_layers = n_layers
        self.n_classes = n_classes


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
        print("Shape after embedding:", x.shape)

        for encoder in self.encoder_layers:
            x, _ = encoder(x, mask=mask)

        x = x[:, 0]
        x = self.fc(x)

        return x


def pad_token_mask(input_ids, pad_token=1):
    pad_mask = input_ids == pad_token

    return pad_mask.logical_not().float()
