import os
import torch
from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import plot_training_metrics, plot_tsne, extract_embeddings
from lightning.pytorch import Trainer
from datetime import datetime

DATASET_DIR = "dataset"
TRAIN_DATA = os.path.join(DATASET_DIR, "train_data.csv")
TRAIN_IMG = os.path.join(DATASET_DIR, "train_images")
OUTPUTS_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUTS_DIR, "malaria_cnn.pth")
BATCH_SIZE = 32

if __name__ == "__main__":
    # Data
    data_module = MalariaDataModule(
        train_csv=TRAIN_DATA,
        train_img_dir=TRAIN_IMG,
        test_img_dir=None,
        batch_size=BATCH_SIZE
    )
    train_loader = data_module.train_dataloader()

    # Model
    model = MalariaLitModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Embedding extraction and t-SNE visualization
    embeddings, labels = extract_embeddings(model, train_loader, device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tsne_path = os.path.join(OUTPUTS_DIR, f"tsne_{timestamp}.png")
    plot_tsne(embeddings, labels, save_path=tsne_path)
    print(f"t-SNE visualization saved to {tsne_path}")

    # Optionally, add code to plot training metrics if you have access to logs
    # plot_training_metrics(...)
