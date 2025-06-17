import os

import torch

from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import extract_embeddings, plot_training_metrics, plot_tsne

DATA_DIR = "data"
RESULTS_DIR = "results/malaria_cnn/visualization"
MODEL_PATH = "lightning_logs/malaria_cnn/version_0/checkpoints/last.ckpt"
BATCH_SIZE = 32

if __name__ == "__main__":
    # Data
    # Ensure the output directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load the data module
    data_module = MalariaDataModule(data_dir=DATA_DIR, batch_size=BATCH_SIZE)
    data_module.setup()
    train_loader = data_module.train_dataloader()

    # Model
    # Load the model from the specified checkpoint
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}")
    model = MalariaLitModel()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Embedding extraction and t-SNE visualization
    embeddings, labels = extract_embeddings(model, train_loader, device)
    model_version = os.path.basename(os.path.dirname(os.path.dirname(MODEL_PATH)))
    tsne_path = os.path.join(RESULTS_DIR, f"tsne_{model_version}.png")
    plot_tsne(embeddings, labels, save_path=tsne_path)
    print(f"t-SNE visualization saved to {tsne_path}")

    # Plot training metrics from Lightning logs
    log_dir = os.path.join("lightning_logs", "malaria_cnn", model_version)
    training_metrics_path = os.path.join(RESULTS_DIR, f"training_metrics_{model_version}.png")
    plot_training_metrics(log_dir, save_path=training_metrics_path)
    print(f"Training metrics saved to {training_metrics_path}")
