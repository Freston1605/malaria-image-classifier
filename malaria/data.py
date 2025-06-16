import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
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
    Lightning DataModule for malaria cell classification.
    Handles setup and loading of training and test datasets with transforms.
    Provides train and test dataloaders for Lightning workflows.
    """

    def __init__(self, train_csv, train_img_dir, test_img_dir, batch_size=32):
        super().__init__()
        self.train_csv = train_csv
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.train_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """
        Initializes the training and test datasets for the data module.

        Args:
            stage (str, optional): Stage to set up. Defaults to None.

        Sets:
            self.train_dataset: An instance of MalariaDataset initialized with training CSV, 
            image directory, and transforms.
            self.test_dataset: An instance of TestDataset initialized with test image directory, 
            list of PNG files, and transforms.
        """
        self.train_dataset = MalariaDataset(
            self.train_csv, self.train_img_dir, transform=self.transform
        )
        test_files = [f for f in os.listdir(self.test_img_dir) if f.endswith(".png")]
        self.test_dataset = TestDataset(
            self.test_img_dir, test_files, transform=self.transform
        )

    def train_dataloader(self):
        """
        Returns a DataLoader for the training dataset.

        This method creates and returns a PyTorch DataLoader instance for the training dataset,
        using the specified batch size and enabling shuffling of the data at each epoch.

        Returns:
            DataLoader: A DataLoader object for iterating over the training dataset in batches.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        """
        Returns a DataLoader for the test dataset.

        This method creates and returns a DataLoader instance for the test dataset,
        using the specified batch size and without shuffling the data.

        Returns:
            DataLoader: A DataLoader object for iterating over the test dataset.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
