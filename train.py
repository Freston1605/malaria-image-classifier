import pytorch_lightning as pl
from malaria.data import MalariaDataModule
from malaria.model import MalariaLitModel
from malaria.utils import plot_training_metrics, plot_tsne, extract_embeddings, save_submission
import torch
import os

if __name__ == "__main__":
    # Paths
    train_csv = "dataset/train_data.csv"
    train_img_dir = "dataset/train_images"
    test_img_dir = "dataset/test_images"
    batch_size = 32
    max_epochs = 10
    lr = 1e-3

    # Data
    data_module = MalariaDataModule(train_csv, train_img_dir, test_img_dir, batch_size=batch_size)

    # Model
    model = MalariaLitModel(lr=lr)

    # Trainer
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="auto")
    trainer.fit(model, datamodule=data_module)

    # Plot training metrics (optional, can be improved with callbacks)
    # plot_training_metrics(trainer)

    # Save model
    torch.save(model.state_dict(), "malaria_cnn.pth")

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
            logits, _ = model(images)
            batch_preds = logits.argmax(dim=1).cpu().numpy()
            preds.extend(batch_preds)
            img_names.extend(names)
    save_submission(img_names, preds)

    # Embedding extraction and t-SNE visualization
    train_loader = data_module.train_dataloader()
    embeddings, labels = extract_embeddings(model.model, train_loader, device)
    plot_tsne(embeddings, labels)
