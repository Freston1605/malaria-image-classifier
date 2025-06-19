import os

import lightning as L
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MalariaDataset(Dataset):
    """
    PyTorch Dataset for malaria cell images with labels.
    Loads image paths and labels from a CSV file and applies optional transforms.
    Each item is a tuple (image, label).
    Skips unreadable/corrupted images.
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
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # Optionally, print a warning or log the error
            print(f"Warning: Could not read image {img_path}, skipping.")
            # Skip this image by returning the next one (recursive call)
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    """
    PyTorch Dataset for malaria cell test images (no labels).
    Loads image paths from a file list and applies optional transforms.
    Each item is a tuple (image, img_name).
    Skips unreadable/corrupted images.
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
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            print(f"Warning: Could not read image {img_path}, skipping.")
            return self.__getitem__((idx + 1) % len(self))
        if self.transform:
            image = self.transform(image)
        return image, img_name


class MalariaDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading and preprocessing the malaria dataset.
    """

    def __init__(
        self,
        data_dir: str = "data",
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 64,
    ):
        """
        Args:
            train_csv (str): Path to the CSV file containing training/validation
            labels and image names.
            train_img_dir (str): Directory containing training/validation images.
            test_img_dir (str): Directory containing test images.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            image_size (int): Size to which images will be resized.
        """
        super().__init__()
        # Initialize paths and parameters
        self.data_dir = data_dir
        self.train_csv_name = "train_data.csv"
        self.train_img_dir_name = "train_images"
        self.test_img_dir_name = "test_images"
        self.train_csv = os.path.join(self.data_dir, self.train_csv_name)
        self.train_img_dir = os.path.join(self.data_dir, self.train_img_dir_name)
        self.test_img_dir = os.path.join(self.data_dir, self.test_img_dir_name)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.train = None
        self.val = None
        self.test = None

        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),  # Ensure all images are the same size
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
                transforms.Resize(
                    (self.image_size, self.image_size)
                ),  # Ensure all images are the same size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        self.test_transforms = self.val_transforms

    def setup(self, stage=None):
        """
        Sets up the datasets for training, validation, and testing.
        This method is called by Lightning at the appropriate time.
        """

        if stage == "fit" or stage is None:
            self.train = MalariaDataset(
                self.train_csv, self.train_img_dir, transform=self.train_transforms
            )
            self.val = MalariaDataset(
                self.train_csv, self.train_img_dir, transform=self.val_transforms
            )
        if stage == "test" or stage is None:
            test_files = [
                f for f in os.listdir(self.test_img_dir) if f.endswith(".png")
            ]
            self.test = TestDataset(
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
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Returns a DataLoader for the validation dataset.

        This method creates and returns a DataLoader instance for the validation dataset,
        using the specified batch size and without shuffling the data.

        Returns:
            DataLoader: A DataLoader object for iterating over the validation dataset.
        """
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
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
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
