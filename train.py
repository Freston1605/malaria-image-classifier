import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel

# Input data paths
DATA_DIR = "data"
TRAIN_DATA = "train_data.csv"
TRAIN_IMG = "train_images"
TEST_IMG = "test_images"

# Output/logging
LOGS_DIR = "lightning_logs"
EXPERIMENT_NAME = "malaria_cnn"

# Hyperparameters
BATCH_SIZE = 32
MAX_EPOCHS = 100
LR = 1e-3
SEED = 42

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(SEED)
    L.seed_everything(SEED, workers=True)

    # Logger (standard practice)
    logger = TensorBoardLogger(save_dir=LOGS_DIR, name=EXPERIMENT_NAME)

    # Data
    data_module = MalariaDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
    )

    # Model
    model = MalariaLitModel(lr=LR)

    # Callbacks
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="{epoch:02d}-{val_loss:.4f}",
        auto_insert_metric_name=False,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger,
        deterministic=True,
        default_root_dir=LOGS_DIR,
    )
    trainer.fit(model, datamodule=data_module)
    # Checkpoints and logs are now organized under lightning_logs/malaria_cnn/version_N/
