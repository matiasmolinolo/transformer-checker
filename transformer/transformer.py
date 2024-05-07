import math

import torch
import torch.nn as nn
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig, train
from transformers import PretrainedConfig, PreTrainedModel


def generate_config(
    n_ctx,
    d_model,
    d_head,
    n_heads,
    d_mlp,
    n_layers,
    attention_dir,
    act_fn,
    d_vocab,
    d_vocab_out,
    use_attn_result,
    device,
    use_hook_tokens,
):
    return HookedTransformerConfig(
        n_ctx=n_ctx,
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        d_mlp=d_mlp,
        n_layers=n_layers,
        attention_dir=attention_dir,
        act_fn=act_fn,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab_out,
        use_attn_result=use_attn_result,
        device=device,
        use_hook_tokens=use_hook_tokens,
    )


def generate_model(config):
    return HookedTransformer(config)


def train_model(model, n_epochs, batch_size, lr, dataset):
    train_cfg = HookedTransformerTrainConfig(
        num_epochs=n_epochs, batch_size=128, lr=0.001, device="cuda:0"
    )

    return train(model, train_cfg, dataset)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) * 1 / self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.depth = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        self.attn = ScaledDotProductAttention(scale=math.sqrt(self.depth))

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.wq(q).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        k = self.wk(k).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)
        v = self.wv(v).view(batch_size, -1, self.n_heads, self.depth).transpose(1, 2)

        attn_out, _ = self.attn(q, k, v, mask=mask)
        attn_out = (
            attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        out = self.dense(attn_out)

        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_heads, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, x, x, mask=mask)
        x = self.ln1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))

        return x


class TransformerClassifierConfig(PretrainedConfig):
    model_type = "transformer-checker"

    def __init__(
        self,
        in_dim=512,
        d_model=256,
        n_heads=8,
        ff_dim=2048,
        n_layers=6,
        n_classes=2,
        **kwargs,
    ):
        self.in_dim = in_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.n_layers = n_layers
        self.n_classes = n_classes

        super().__init__(**kwargs)


class TransformerClassifier(PreTrainedModel):
    config_class = TransformerClassifierConfig

    def __init__(self, config: TransformerClassifierConfig):
        super().__init__(config)
        self.embedding = nn.Linear(config.in_dim, config.d_model)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderLayer(config.d_model, config.n_heads, config.ff_dim)
                for _ in range(config.n_layers)
            ]
        )
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for encoder in self.encoders:
            x = encoder(x, mask=mask)

        x = self.classifier(x[:, 0])

        return x
