import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl

class MalariaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['img_name']
        label = int(self.data.iloc[idx]['label'])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TestDataset(Dataset):
    def __init__(self, img_dir, file_list, transform=None):
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

class MalariaDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, train_img_dir, test_img_dir, batch_size=32):
        super().__init__()
        self.train_csv = train_csv
        self.train_img_dir = train_img_dir
        self.test_img_dir = test_img_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.train_dataset = MalariaDataset(self.train_csv, self.train_img_dir, transform=self.transform)
        test_files = [f for f in os.listdir(self.test_img_dir) if f.endswith('.png')]
        self.test_dataset = TestDataset(self.test_img_dir, test_files, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
