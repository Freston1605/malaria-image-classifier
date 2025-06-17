import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE


def plot_training_metrics(log_dir, save_path="training_metrics.png"):
    """
    Plots training and validation loss and accuracy curves from Lightning CSV logs.

    Args:
        log_dir (str): Path to the lightning_logs/version_x/ directory containing metrics.csv.
        save_path (str): Path to save the plot image.
    """

    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        print(f"metrics.csv not found in {log_dir}")
        return

    df = pd.read_csv(metrics_path)
    # Remove rows with NaN values in key columns
    df = df.dropna(subset=["epoch"])

    train_loss = df[df["train_loss"].notna()].groupby("epoch")["train_loss"].mean()
    val_loss = df[df["val_loss"].notna()].groupby("epoch")["val_loss"].mean()
    val_acc = df[df["val_acc"].notna()].groupby("epoch")["val_acc"].mean()

    epochs = np.arange(1, max(df["epoch"]) + 2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs[: len(train_loss)], train_loss, label="Train Loss")
    plt.plot(epochs[: len(val_loss)], val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs[: len(val_acc)], val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    # plt.close()


def plot_tsne(embeddings, labels, save_path="tsne.png"):
    """
    Generates and saves a t-SNE visualization of high-dimensional embeddings.
    Parameters:
        embeddings (array-like): High-dimensional data to be reduced and visualized, shape (n_samples, n_features).
        labels (array-like): Class labels corresponding to each embedding, shape (n_samples,).
        save_path (str, optional): File path to save the generated plot. Defaults to "tsne.png".
    The function reduces the dimensionality of the input embeddings to 2D using t-SNE,
    creates a scatter plot colored by the provided labels, and saves the plot to the specified path.
    A legend is included to distinguish between classes (0=Healthy, 1=Infected).
    """

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.title("t-SNE Visualization of Learned Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(*scatter.legend_elements(), title="Class (0=Healthy, 1=Infected)")
    plt.savefig(save_path)
    #plt.show()


def extract_embeddings(model, dataloader, device):
    """
    Extracts embeddings and corresponding labels from a model using a given dataloader.
    Args:
        model (torch.nn.Module): The neural network model that outputs embeddings.
        dataloader (torch.utils.data.DataLoader): DataLoader providing batches of images and labels.
        device (torch.device): The device (CPU or GPU) to perform computations on.
    Returns:
        Tuple[np.ndarray, List[int]]:
            - all_embeddings: Numpy array containing all extracted embeddings.
            - all_labels: List of labels corresponding to each embedding.
    """

    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _, emb = model(images)
            all_embeddings.append(emb.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    return all_embeddings, all_labels


def save_submission(img_names, preds, save_path="submission.csv"):
    """
    Saves image names and their corresponding predictions to a CSV file.
    Args:
        img_names (list or array-like): List of image file names.
        preds (list or array-like): List of predicted labels corresponding to each image.
        save_path (str, optional): Path to save the CSV file. Defaults to "submission.csv".
    Returns:
        None
    """

    df = pd.DataFrame({"img_name": img_names, "label": preds})
    df.to_csv(save_path, index=False)
