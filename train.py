import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping

from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import (
    extract_embeddings,
    plot_training_metrics,
    plot_tsne,
    save_submission,
)

# Generate a timestamp for unique model saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define all paths and constants at the top
DATASET_DIR = "dataset"
TRAIN_DATA = os.path.join(DATASET_DIR, "train_data.csv")
TRAIN_IMG = os.path.join(DATASET_DIR, "train_images")
TEST_IMG = os.path.join(DATASET_DIR, "test_images")

OUTPUTS_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUTS_DIR, f"malaria_cnn_{timestamp}.pth")

# Create output directory if it doesn't exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 10
LR = 1e-3

if __name__ == "__main__":
    # Data
    data_module = MalariaDataModule(
        data_dir=DATASET_DIR,
        train_csv=TRAIN_DATA,
        train_img_dir=TRAIN_IMG,
        test_img_dir=TEST_IMG,
        batch_size=BATCH_SIZE,
    )

    # Model
    model = MalariaLitModel(lr=LR)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=20,  # "Several decades" interpreted as 20 epochs
        mode="min",  # Stop when the monitored metric stops decreasing
        verbose=True,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, datamodule=data_module)

    # Plot training metrics
    plot_training_metrics(trainer)

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)

    # Test predictions
    test_loader = data_module.test_dataloader()
    model.eval()
    img_names = []
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    with torch.no_grad():
        for batch in test_loader:
            images, names = batch
            images = images.to(device)
            logits, _ = model.forward(images)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            img_names.extend(names)
    save_submission(img_names, preds)

    # Embedding extraction and t-SNE visualization
    train_loader = data_module.train_dataloader()
    embeddings, labels = extract_embeddings(model.model, train_loader, device)
    plot_tsne(embeddings, labels)
