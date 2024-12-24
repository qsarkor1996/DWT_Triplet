import os
import numpy as np
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import torch

def save_plot(fig, output_dir, filename):
    """
    Save a matplotlib figure to a specified directory.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        output_dir (str): The directory to save the figure.
        filename (str): The filename for the saved plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, filename)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {fig_path}")


def plot_sample_signals(data, indices, output_dir):
    """
    Plot and save sample signals from the dataset.

    Args:
        data (numpy.ndarray): Input signal data (2D array: samples x time).
        indices (list): List of indices for the samples to be plotted.
        output_dir (str): Directory to save the plots.

    Saves:
        A plot of the specified signals in the output directory.
    """
    if data is None or len(data) == 0:
        print("No data available for plotting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Create subplots
    num_plots = len(indices)
    fig, axs = plt.subplots(num_plots, 1, figsize=(10, 8))

    # Ensure axs is always iterable
    if num_plots == 1:
        axs = [axs]

    for i, idx in enumerate(indices):
        if idx >= len(data):
            print(f"Index {idx} is out of bounds for data with shape {data.shape}.")
            continue
        axs[i].plot(data[idx])
        axs[i].set_title(f"Sample Signal @ Index {idx}")
        axs[i].set_ylabel("Amplitude")
        axs[i].grid(True)

    axs[-1].set_xlabel("Time")
    plt.tight_layout()

    # Save the plot
    filename = "sample_signals.png"
    save_plot(fig, output_dir, filename)


def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """
    Plot and save the confusion matrix as percentages.

    Args:
        conf_matrix (numpy.ndarray): Confusion matrix to visualize.
        class_names (list): List of class names for labeling.
        output_dir (str): Directory to save the confusion matrix plot.
    """
    # Normalize confusion matrix to percentages
    conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
    conf_matrix_percentage = np.nan_to_num(conf_matrix_percentage)  # Handle division by zero cases

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix_percentage,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix (in %)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "confusion_matrix_percentage.png")
    plt.savefig(file_path, bbox_inches="tight")
    print(f"Saved confusion matrix visualization to {file_path}")
    plt.close()

def visualize_tsne_embeddings(model, loader, device, class_to_idx, output_dir):
    """
    Visualize embeddings using t-SNE and save the plot.

    Args:
        model: Trained model to generate embeddings.
        loader: DataLoader for the data to be visualized.
        device: Device to run the model on.
        class_to_idx: Mapping of class names to indices.
        output_dir: Directory to save the t-SNE plot.
    """
    from sklearn.manifold import TSNE

    model.eval()
    embeddings, labels = [], []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model.forward_once(inputs).cpu().numpy()  # Extract embeddings
            embeddings.append(outputs)
            labels.extend(targets.cpu().numpy())

    embeddings = np.vstack(embeddings)  # Combine all embeddings
    labels = np.array(labels)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Plot t-SNE results
    fig, ax = plt.subplots(figsize=(12, 10))
    unique_classes = np.unique(labels)
    idx_to_class = {v: k for k, v in class_to_idx.items()}  # Reverse the mapping

    for class_idx in unique_classes:
        class_name = idx_to_class.get(class_idx, f"Class {class_idx}")
        class_points = embeddings_2d[labels == class_idx]
        ax.scatter(
            class_points[:, 0],
            class_points[:, 1],
            label=class_name,
            alpha=0.6
        )

    ax.legend()
    ax.set_title("t-SNE Visualization of Embeddings")
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.grid(True)

    # Save the plot
    filename = "tsne_embeddings.png"
    save_plot(fig, output_dir, filename)


def visualize_dwt_signals(signal, wavelet, output_dir, signal_idx):
    """
    Visualize and save the noisy signal and its DWT decompositions.

    Args:
        signal (numpy.ndarray): The input noisy signal (1D array).
        wavelet (str): The wavelet to use for decomposition.
        output_dir (str): Directory to save the visualization.
        signal_idx (int): Index of the signal for labeling the plot.
    """
    # Perform DWT decomposition up to level 3
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=3)
    approx3 = coeffs[0]  # Approximation at level 3
    detail3 = coeffs[1]  # Detail at level 3
    detail2 = coeffs[2]  # Detail at level 2
    detail1 = coeffs[3]  # Detail at level 1

    # Create plots
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle(f"DWT Visualization for Signal #{signal_idx}", fontsize=16)

    # Plot the original noisy signal
    axs[0].plot(signal, label="Noisy Signal", color="blue")
    axs[0].set_title("Noisy Signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # Plot DWT level 3
    axs[1].plot(approx3, label="DWT Level 3 - Approximation", color="green")
    axs[1].plot(detail3, label="DWT Level 3 - Detail", color="red")
    axs[1].set_title("DWT Level 3")
    axs[1].set_ylabel("Amplitude")
    axs[1].legend()

    # Plot DWT level 2
    axs[2].plot(detail2, label="DWT Level 2 - Detail", color="purple")
    axs[2].set_title("DWT Level 2")
    axs[2].set_ylabel("Amplitude")
    axs[2].legend()

    # Plot DWT level 1
    axs[3].plot(detail1, label="DWT Level 1 - Detail", color="orange")
    axs[3].set_title("DWT Level 1")
    axs[3].set_ylabel("Amplitude")
    axs[3].legend()

    # Layout and save the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit title
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"dwt_visualization_signal_{signal_idx}.png")
    fig.savefig(file_path, bbox_inches="tight")
    print(f"Saved DWT visualization to {file_path}")
    plt.close(fig)