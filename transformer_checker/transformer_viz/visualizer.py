import matplotlib.pyplot as plt
import numpy as np

from transformer_checker.dataset.dataset import DyckLanguageTokenizer


def z_score_normalize(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return 2 * normalized_matrix - 1


def plot_attn_matrices(vocab, batch, model, norm, mask):
    _, _, tokens = batch

    # Filter unique samples
    unique_tokens = {}
    for i, token_seq in enumerate(tokens):
        token_tuple = tuple(token_seq.cpu().numpy())
        if token_tuple not in unique_tokens:
            unique_tokens[token_tuple] = i

    unique_indices = list(unique_tokens.values())
    tokens = tokens[unique_indices]

    attn_matrices = model.get_attn_matrices(tokens, mask=mask(tokens))
    tokenizer = DyckLanguageTokenizer(vocab)

    num_layers = len(attn_matrices)
    num_heads = attn_matrices[0].shape[1]
    num_samples = len(unique_indices)

    # Plot individual samples
    for sample_idx in range(num_samples):
        fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads * 6, num_layers * 6), dpi=150)
        if num_layers == 1 and num_heads == 1:
            axes = np.array([[axes]])
        elif num_layers == 1 or num_heads == 1:
            axes = axes.reshape(num_layers, num_heads)

        labels = tokenizer.decode_single(tokens[sample_idx], remove_special_tokens=False).split()

        for layer in range(num_layers):
            for head in range(num_heads):
                ax = axes[layer, head]

                matrix = attn_matrices[layer][sample_idx, head].cpu().detach().numpy()
                norm_matrix = norm(matrix)

                heatmap = ax.imshow(norm_matrix, cmap="coolwarm", interpolation="nearest", aspect="auto")

                ax.set_xticks(range(len(labels)))
                ax.set_yticks(range(len(labels)))
                ax.set_xticklabels(labels, fontsize=6)
                ax.set_yticklabels(labels, fontsize=6)

                ax.set_title(f"Layer {layer}, Head {head}", fontsize=8)

                for i in range(len(labels)):
                    for j in range(len(labels)):
                        ax.text(j, i, f"{norm_matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=6)

        plt.tight_layout()
        plt.show()

__all__ = ['plot_attn_matrices', 'z_score_normalize', 'min_max_normalize']
