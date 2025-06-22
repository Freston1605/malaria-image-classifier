import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from malaria.data import MalariaDataModule
from malaria.yolo.model import YOLOLitModel

if __name__ == "__main__":
    # Hyperparameters
    DATA_DIR = "dataset"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    IMG_SIZE = 64
    LR = 1e-3
    MAX_EPOCHS = 100

    # Logging
    LOGS_DIR = "lightning_logs"
    EXPERIMENT_NAME = "yolo"

    # Reproducibility
    L.seed_everything(42, workers=True)

    # Data
    datamodule = MalariaDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_size=IMG_SIZE,
    )
    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Model
    num_classes = datamodule.num_classes()
    model = YOLOLitModel(num_classes=num_classes, lr=LR)

    # Logger
    logger = TensorBoardLogger(save_dir=LOGS_DIR, name=EXPERIMENT_NAME)

    # Checkpoint callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{LOGS_DIR}/{EXPERIMENT_NAME}/checkpoints",
        filename="epoch{epoch:02d}-val_acc{val_acc:.2f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        deterministic=True,
        default_root_dir=LOGS_DIR,
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    trainer.fit(
        model,
        datamodule=datamodule,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
