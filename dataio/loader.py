from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import random
import glob
import os


########################################################################################################################################################
# For Classifiers
########################################################################################################################################################
        
class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.classes = [self.classes[0], self.classes[-1]]

        print(self.classes)

        self.data = []
        for class_label in self.classes:
            class_path = os.path.join(root_dir, class_label)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                self.data.append((img_path, self.classes.index(class_label)))
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')

        #image = image.resize((8,8), Image.LANCZOS)

        if self.transform: image = self.transform(image)

        return image, label

class EuroSATDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=9):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers=num_workers

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = EuroSATDataset(os.path.join('/content/dataset', 'training'), transform=transform)
        self.valid_dataset = EuroSATDataset(os.path.join('/content/dataset', 'validation'), transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)