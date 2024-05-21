import matplotlib.pyplot as plt
import numpy as np

from dataset.dataset import DyckLanguageTokenizer


def z_score_normalize(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    normalized_matrix = (matrix - mean) / std
    return normalized_matrix


def min_max_normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def plot_attn_matrices(vocab, batch, model, norm, mask):
    _, labels, tokens = batch
    attn_matrices = model.get_attn_matrices(tokens, mask=mask(tokens))
    tokenizer = DyckLanguageTokenizer(vocab)

    unique_matrices = []
    unique_labels_set = set()

    for matrix in range(len(attn_matrices[0])):
        labels = tokenizer.decode_single(tokens[matrix], remove_special_tokens=False).split(
            " "
        )
        labels_tuple = tuple(labels)  # Convert list to tuple for set comparison

        if labels_tuple not in unique_labels_set:
            unique_labels_set.add(labels_tuple)
            unique_matrices.append((matrix, labels))

    num_unique_matrices = len(unique_matrices)

    # Calculate the number of rows and columns for the subplots
    num_cols = 2  # Adjust this based on your preference
    num_rows = (num_unique_matrices + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 10, num_rows * 10))

    # Flatten axes array for easy iteration
    axes = axes.flatten()

    for idx, (matrix, labels) in enumerate(unique_matrices):
        ax = axes[idx]
        norm_matrix = norm(attn_matrices[0][matrix][0].cpu().detach().numpy())
        heatmap = ax.imshow(
            norm_matrix,
            cmap="coolwarm",
            interpolation="nearest",
        )

        x_ticks = list(range(attn_matrices[0][matrix][0].shape[0]))
        y_ticks = list(range(attn_matrices[0][matrix][0].shape[1]))

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(labels, fontsize=14)
        ax.xaxis.set_ticks_position("top")  # Position x-ticks on top
        ax.xaxis.set_label_position("top")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(labels, fontsize=14)

        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label("Attention weights", fontsize=14)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(
                    j,
                    i,
                    f"{norm_matrix[i, j]:.4f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )

    # Hide any unused subplots
    for ax in axes[num_unique_matrices:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()