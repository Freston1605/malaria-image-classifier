import os
from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping

from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import (extract_embeddings, plot_training_metrics,
                           plot_tsne, save_submission)

if __name__ == "__main__":
    # Paths
    TRAIN_DATA = "dataset/train_data.csv"
    TRAIN_IMG = "dataset/train_images"
    TEST_IMG = "dataset/test_images"
    BATCH_SIZE = 32
    MAX_EPOCHS = 10
    LR = 1e-3

    # Data
    data_module = MalariaDataModule(
        TRAIN_DATA, TRAIN_IMG, TEST_IMG, batch_size=BATCH_SIZE
    )

    # Model
    model = MalariaLitModel(lr=LR)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",           # Metric to monitor
        patience=20,                  # "Several decades" interpreted as 20 epochs
        mode="min",                   # Stop when the monitored metric stops decreasing
        verbose=True
    )

    # Trainer
    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback],
    )
    trainer.fit(model, datamodule=data_module)

    # plot_training_metrics(trainer)
    plot_training_metrics(trainer)


    # Save model
    os.makedirs("outputs/pth", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"outputs/pth/malaria_cnn_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)

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
