import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
import pandas as pd

def plot_training_metrics(trainer, save_path="training_metrics.png"):
    metrics = trainer.callback_metrics
    epochs = np.arange(1, len(metrics['train_loss']) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'].cpu().numpy(), 'r-', marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'].cpu().numpy(), 'b-', marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_tsne(embeddings, labels, save_path="tsne.png"):
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title("t-SNE Visualization of Learned Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(*scatter.legend_elements(), title="Class (0=Healthy, 1=Infected)")
    plt.savefig(save_path)
    plt.show()

def extract_embeddings(model, dataloader, device):
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
    df = pd.DataFrame({"img_name": img_names, "label": preds})
    df.to_csv(save_path, index=False)
