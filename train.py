import os
import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel

# Generate a timestamp for unique model saving
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Define all paths and constants at the top

# Input data paths
DATA_DIR = "data"
TRAIN_DATA = "train_data.csv"
TRAIN_IMG = "train_images"
TEST_IMG = "test_images"

# Output paths
OUTPUTS_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUTS_DIR, f"malaria_cnn_{timestamp}.pth")
CHECKPOINT_DIR = os.path.join(OUTPUTS_DIR, "checkpoints")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last.ckpt")

# Hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 10
LR = 1e-3

if __name__ == "__main__":
    # Data
    data_module = MalariaDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    # Model
    model = MalariaLitModel(lr=LR)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",  # Metric to monitor
        patience=20,  # "Several decades" interpreted as 20 epochs
        mode="min",  # Stop when the monitored metric stops decreasing
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    # Resume from last checkpoint if available
    resume_ckpt = LAST_CKPT_PATH if os.path.exists(LAST_CKPT_PATH) else None

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        default_root_dir=OUTPUTS_DIR,
    )
    trainer.fit(model, datamodule=data_module, ckpt_path=resume_ckpt)

    # Save model
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    # Training ends here. Visualization and testing are handled in separate scripts.
