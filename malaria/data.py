import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L


class MalariaDataset(Dataset):
    """
    PyTorch Dataset for malaria cell images with labels.
    Loads image paths and labels from a CSV file and applies optional transforms.
    Each item is a tuple (image, label).
    """

    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["img_name"]
        label = int(self.data.iloc[idx]["label"])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """
    PyTorch Dataset for malaria cell test images (no labels).
    Loads image paths from a file list and applies optional transforms.
    Each item is a tuple (image, img_name).
    """

    def __init__(self, img_dir, file_list, transform=None):
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_name


class MalariaDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preprocessing the malaria dataset.
    """

    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        train_csv="train.csv",
        train_img_dir="train_images",
        test_img_dir="test_images",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Paths for CSV and image directories
        self.train_csv = os.path.join(data_dir, train_csv)
        self.train_img_dir = os.path.join(data_dir, train_img_dir)
        self.test_img_dir = os.path.join(data_dir, test_img_dir)

        # Add RandomRotation for tilting augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.RandomRotation(
                    degrees=20
                ),  # Tilting images by up to ±20 degrees
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.test_transforms = self.val_transforms
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.
        This method is called by Lightning at the appropriate time.
        """
        if stage == "fit" or stage is None:
            self.train_dataset = MalariaDataset(
                csv_file=self.train_csv,
                img_dir=self.train_img_dir,
                transform=self.train_transforms,
            )
            self.val_dataset = MalariaDataset(
                csv_file=self.train_csv,
                img_dir=self.train_img_dir,
                transform=self.val_transforms,
            )
        if stage == "test" or stage is None:
            test_files = [
                f for f in os.listdir(self.test_img_dir) if f.endswith(".png")
            ]
            self.test_dataset = TestDataset(
                self.test_img_dir, test_files, transform=self.test_transforms
            )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        This method creates and returns a PyTorch DataLoader instance for the training dataset,
        using the specified batch size and enabling shuffling of the data at each epoch.

        Returns:
            DataLoader: A DataLoader object for iterating over the training dataset in batches.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        This method creates and returns a DataLoader instance for the test dataset,
        using the specified batch size and without shuffling the data.

        Returns:
            DataLoader: A DataLoader object for iterating over the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
