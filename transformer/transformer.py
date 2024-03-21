from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.train import HookedTransformerTrainConfig, train


def generate_config(n_ctx, d_model, d_head, n_heads, d_mlp, n_layers, attention_dir, act_fn, d_vocab, d_vocab_out, use_attn_result, device, use_hook_tokens):
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
        use_hook_tokens=use_hook_tokens
    )

def generate_model(config):
    return HookedTransformer(config)

def train_model(model, n_epochs, batch_size, lr, dataset):
    train_cfg = HookedTransformerTrainConfig(num_epochs=n_epochs, batch_size=128, lr=0.001, device='cuda:0')

    return train(model, train_cfg, dataset)