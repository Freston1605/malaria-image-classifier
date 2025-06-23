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
    Optional data augmentations (flips, rotations, color jitter) can be enabled
    via the ``augment`` flag.
    """

    def __init__(
        self,
        data_dir="dataset",
        batch_size=32,
        num_workers=4,
        img_size=64,
        augment=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.augment = augment
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        train_transforms = [transforms.Resize((self.img_size, self.img_size))]
        if self.augment:
            train_transforms.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                ]
            )
        train_transforms.append(transforms.ToTensor())

        val_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ]

        self.train_dataset = ImageFolder(
            os.path.join(self.data_dir, "train"),
            transform=transforms.Compose(train_transforms),
        )
        self.val_dataset = ImageFolder(
            os.path.join(self.data_dir, "val"),
            transform=transforms.Compose(val_transforms),
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
