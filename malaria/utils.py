import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator


class EmbeddingVisualizer:
    """
    Class for extracting embeddings and visualizing results from a model and dataloader.
    """

    def __init__(self, model, dataloader, device=None):
        self.model = model
        self.dataloader = dataloader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.embeddings = None
        self.labels = None

    def extract_embeddings(self):
        """
        Extracts embeddings and corresponding labels from the dataset using the model.
        This method sets the model to evaluation mode and iterates over the dataloader,
        passing each batch of images through the model to obtain embeddings. The embeddings
        and their associated labels are collected and concatenated into arrays.
        Returns:
            tuple: A tuple containing:
                - embeddings (np.ndarray): Array of extracted embeddings for all samples.
                - labels (np.ndarray): Array of corresponding labels for all samples.
        """

        self.model.eval()
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                _, emb = self.model(images)
                all_embeddings.append(emb.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        self.embeddings = np.concatenate(all_embeddings, axis=0)
        self.labels = np.array(all_labels)
        return self.embeddings, self.labels

    def plot_tsne(
        self,
        model=None,
        dataloader=None,
        overlay="class",
        save_path=None,
        perplexity=30,
    ):
        """
        Generates a t-SNE visualization of learned embeddings, optionally overlaying class labels or prediction outcomes.

        Parameters:
            model (torch.nn.Module, optional): Trained model used to compute prediction outcomes for overlay. Required if overlay='outcome'.
            dataloader (torch.utils.data.DataLoader, optional): DataLoader for the dataset to compute prediction outcomes. Required if overlay='outcome'.
            overlay (str, optional): Type of overlay to display on the t-SNE plot.
                'class' overlays ground truth class labels (default).
                'outcome' overlays prediction outcomes (TP, TN, FP, FN).
            save_path (str, optional): Path to save the generated plot. If None, the plot is not saved.
            perplexity (int, optional): Perplexity parameter for t-SNE. Default is 30.

        Raises:
            ValueError: If overlay is set to 'outcome' but either model or dataloader is not provided.

        Notes:
            - If embeddings or labels are not already extracted, calls self.extract_embeddings().
            - Always displays the plot. If save_path is provided, also saves the plot to the specified location.
        """
        if self.embeddings is None or self.labels is None:
            self.extract_embeddings()
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        tsne_emb = tsne.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))
        if overlay == "class":
            scatter = plt.scatter(
                tsne_emb[:, 0], tsne_emb[:, 1], c=self.labels, cmap="viridis", alpha=0.7
            )
            plt.legend(
                *scatter.legend_elements(), title="Class (0=Healthy, 1=Infected)"
            )
            plt.title(
                f"t-SNE Visualization of Learned Embeddings (Perplexity={perplexity})"
            )
        elif overlay == "outcome" and model is not None and dataloader is not None:
            TP, TN, FP, FN = self.compute_outcome_masks(model, dataloader)
            plt.scatter(
                tsne_emb[TP, 0], tsne_emb[TP, 1], c="green", label="TP", alpha=0.7
            )
            plt.scatter(
                tsne_emb[TN, 0], tsne_emb[TN, 1], c="blue", label="TN", alpha=0.7
            )
            plt.scatter(
                tsne_emb[FP, 0], tsne_emb[FP, 1], c="red", label="FP", alpha=0.7
            )
            plt.scatter(
                tsne_emb[FN, 0], tsne_emb[FN, 1], c="orange", label="FN", alpha=0.7
            )
            plt.legend()
            plt.title(
                f"t-SNE Embeddings with Prediction Outcomes (Perplexity={perplexity})"
            )
        else:
            raise ValueError(
                "To plot outcome overlay, supply both model and dataloader."
            )
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        if save_path:
            plt.savefig(save_path)
        # Always show, never save by default
        plt.show()

    @staticmethod
    def plot_training_metrics(log_dir, save_path=None):
        """
        Plots training and validation loss and accuracy curves per epoch from TensorBoard event files in the log_dir.
        The X-axis is the true epoch number, not the global step.
        """
        # Gather all event files and aggregate all epoch-level events
        event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))
        if not event_files:
            print(f"No TensorBoard event files found in {log_dir}")
            return

        def get_all_scalars(tag):
            scalars = []
            for event_file in event_files:
                event_acc = EventAccumulator(event_file)
                event_acc.Reload()
                if tag in event_acc.Tags()["scalars"]:
                    scalars.extend(event_acc.Scalars(tag))
            # Sort by wall_time to ensure correct epoch order
            scalars = sorted(scalars, key=lambda x: x.wall_time)
            return [s.value for s in scalars]

        # Prefer *_epoch tags for epoch-level metrics
        tags_to_try = [
            (
                "train_loss",
                [
                    "train_loss_epoch",
                    "train/loss_epoch",
                    "train_loss",
                    "loss",
                    "train/loss",
                ],
            ),
            ("val_loss", ["val_loss_epoch", "val/loss_epoch", "val_loss", "val/acc"]),
            (
                "train_acc",
                ["train_acc_epoch", "train/acc_epoch", "train_acc", "acc", "train/acc"],
            ),
            ("val_acc", ["val_acc_epoch", "val/acc_epoch", "val_acc", "val/acc"]),
        ]
        metrics = {}
        for name, possibilities in tags_to_try:
            for tag in possibilities:
                vals = get_all_scalars(tag)
                if vals:
                    metrics[name] = vals
                    break
            else:
                metrics[name] = None

        # X-axis: epoch number (starting from 1)
        max_len = max(len(v) for v in metrics.values() if v is not None)
        epochs = range(1, max_len + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if metrics["train_loss"]:
            plt.plot(
                list(epochs)[: len(metrics["train_loss"])],
                metrics["train_loss"],
                label="Train Loss",
            )
        if metrics["val_loss"]:
            plt.plot(
                list(epochs)[: len(metrics["val_loss"])],
                metrics["val_loss"],
                label="Val Loss",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves (per epoch)")
        plt.legend()

        plt.subplot(1, 2, 2)
        if metrics["train_acc"]:
            plt.plot(
                list(epochs)[: len(metrics["train_acc"])],
                metrics["train_acc"],
                label="Train Accuracy",
            )
        if metrics["val_acc"]:
            plt.plot(
                list(epochs)[: len(metrics["val_acc"])],
                metrics["val_acc"],
                label="Val Accuracy",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves (per epoch)")
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_umap(
        self,
        model=None,
        dataloader=None,
        overlay="class",
        save_path=None,
        random_state=42,
    ):
        """
        Generates a UMAP visualization of learned embeddings, with optional overlays for class labels or prediction outcomes.

        Parameters:
            model (torch.nn.Module, optional): Trained model used to compute prediction outcomes. Required if overlay is 'outcome'.
            dataloader (torch.utils.data.DataLoader, optional): DataLoader for the dataset. Required if overlay is 'outcome'.
            overlay (str, optional): Type of overlay to display on the UMAP plot.
                - "class": Colors points by their class labels (default).
                - "outcome": Colors points by prediction outcomes (TP, TN, FP, FN). Requires both model and dataloader.
            save_path (str, optional): Path to save the generated plot. If None, the plot is not saved.
            random_state (int, optional): Random seed for UMAP dimensionality reduction. Default is 42.

        Raises:
            ValueError: If overlay is set to 'outcome' but either model or dataloader is not provided.

        Notes:
            - If embeddings or labels are not already extracted, calls self.extract_embeddings().
            - Uses matplotlib for plotting and umap-learn for dimensionality reduction.
        """
        if self.embeddings is None or self.labels is None:
            self.extract_embeddings()
        reducer = umap.UMAP(random_state=random_state)
        umap_emb = reducer.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))
        if overlay == "class":
            scatter = plt.scatter(
                umap_emb[:, 0], umap_emb[:, 1], c=self.labels, cmap="viridis", alpha=0.7
            )
            plt.legend(
                *scatter.legend_elements(), title="Class (0=Healthy, 1=Infected)"
            )
            plt.title("UMAP Visualization of Learned Embeddings")
        elif overlay == "outcome" and model is not None and dataloader is not None:
            TP, TN, FP, FN = self.compute_outcome_masks(model, dataloader)
            plt.scatter(
                umap_emb[TP, 0], umap_emb[TP, 1], c="green", label="TP", alpha=0.7
            )
            plt.scatter(
                umap_emb[TN, 0], umap_emb[TN, 1], c="blue", label="TN", alpha=0.7
            )
            plt.scatter(
                umap_emb[FP, 0], umap_emb[FP, 1], c="red", label="FP", alpha=0.7
            )
            plt.scatter(
                umap_emb[FN, 0], umap_emb[FN, 1], c="orange", label="FN", alpha=0.7
            )
            plt.legend()
            plt.title("UMAP Embeddings with Prediction Outcomes")
        else:
            raise ValueError(
                "To plot outcome overlay, supply both model and dataloader."
            )
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_pca(
        self,
        model=None,
        dataloader=None,
        overlay="class",
        save_path=None,
        random_state=42,
    ):
        """
        Visualizes the learned embeddings using PCA (Principal Component Analysis) and plots them in 2D.

        Parameters:
            model (torch.nn.Module, optional): Trained model used to compute prediction outcomes for overlay. Required if overlay='outcome'.
            dataloader (torch.utils.data.DataLoader, optional): DataLoader providing data for prediction outcome overlay. Required if overlay='outcome'.
            overlay (str, optional): Determines the coloring of the scatter plot.
                - 'class': Colors points by their true class labels (default).
                - 'outcome': Colors points by prediction outcome (TP, TN, FP, FN). Requires both model and dataloader.
            save_path (str, optional): If provided, saves the plot to the specified file path.
            random_state (int, optional): Random seed for PCA reproducibility. Default is 42.

        Raises:
            ValueError: If overlay is set to 'outcome' but either model or dataloader is not provided.

        Notes:
            - If embeddings or labels are not already extracted, calls self.extract_embeddings().
            - For 'class' overlay, uses class labels for coloring.
            - For 'outcome' overlay, computes prediction outcomes (TP, TN, FP, FN) and colors accordingly.
        """
        if self.embeddings is None or self.labels is None:
            self.extract_embeddings()
        pca = PCA(n_components=2, random_state=random_state)
        pca_emb = pca.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))
        if overlay == "class":
            scatter = plt.scatter(
                pca_emb[:, 0], pca_emb[:, 1], c=self.labels, cmap="viridis", alpha=0.7
            )
            plt.legend(
                *scatter.legend_elements(), title="Class (0=Healthy, 1=Infected)"
            )
            plt.title("PCA Visualization of Learned Embeddings")
        elif overlay == "outcome" and model is not None and dataloader is not None:
            TP, TN, FP, FN = self.compute_outcome_masks(model, dataloader)
            plt.scatter(
                pca_emb[TP, 0], pca_emb[TP, 1], c="green", label="TP", alpha=0.7
            )
            plt.scatter(pca_emb[TN, 0], pca_emb[TN, 1], c="blue", label="TN", alpha=0.7)
            plt.scatter(pca_emb[FP, 0], pca_emb[FP, 1], c="red", label="FP", alpha=0.7)
            plt.scatter(
                pca_emb[FN, 0], pca_emb[FN, 1], c="orange", label="FN", alpha=0.7
            )
            plt.legend()
            plt.title("PCA Embeddings with Prediction Outcomes")
        else:
            raise ValueError(
                "To plot outcome overlay, supply both model and dataloader."
            )
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def compute_outcome_masks(self, model, dataloader):
        """
        Computes boolean masks for TP, TN, FP, FN given a model and dataloader.
        Returns: TP, TN, FP, FN (all boolean numpy arrays)
        """
        model.eval()
        all_preds = []
        device = self.device
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.argmax(outputs[0], dim=1)
                all_preds.extend(preds.cpu().numpy())
        all_preds = np.array(all_preds)
        true_labels = np.array(self.labels)
        TP = (all_preds == 1) & (true_labels == 1)
        TN = (all_preds == 0) & (true_labels == 0)
        FP = (all_preds == 1) & (true_labels == 0)
        FN = (all_preds == 0) & (true_labels == 1)
        return TP, TN, FP, FN


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
