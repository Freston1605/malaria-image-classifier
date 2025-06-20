import os

import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class MalariaDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for malaria cell classification using YOLO-style folder structure.
    Expects:
        data/
            train/
                infected/
                healthy/
            val/
                infected/
                healthy/
    """

    def __init__(self, data_dir="data", batch_size=32, num_workers=4, img_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                # Add more augmentations here if needed
            ]
        )
        self.train_dataset = ImageFolder(
            os.path.join(self.data_dir, "train"), transform=transform
        )
        self.val_dataset = ImageFolder(
            os.path.join(self.data_dir, "val"), transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def num_classes(self):
        return len(self.train_dataset.classes)
