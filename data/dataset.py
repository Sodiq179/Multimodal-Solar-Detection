"""
Dataset module for loading and preprocessing solar panel and boiler data.
"""

import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SolarPanelDataset(Dataset):
    """
    Custom Dataset for the Solar Panel and Boiler Counting Task.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image IDs and labels.
        image_dir (str): Path to directory containing image files.
        transforms (albumentations.Compose): Data augmentation pipeline.
    """
    def __init__(self, dataframe, image_dir, transforms=None):
        self.data = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = row['ID']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # One-hot encode metadata: image_origin and placement
        image_origin = row['img_origin']  # 'D' or 'S'
        placement = row['placement']     # 'roof', 'openspace', etc.

        metadata = [
            1 if image_origin == 'D' else 0,  # Drone
            1 if placement == 'roof' else 0,
            1 if placement == 'openspace' else 0,
            1 if placement == 'r_openspace' else 0,
            1 if placement == 'S-unknown' else 0
        ]

        metadata_tensor = torch.tensor(metadata, dtype=torch.float)

        if self.transforms:
            image = self.transforms(image=image)['image']

        target = torch.tensor([row['boil_nbr'], row['pan_nbr']], dtype=torch.float)

        return image, metadata_tensor, target


def get_train_transforms(img_size=512):
    """
    Return training data augmentation pipeline.
    """
    return A.Compose([
        A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussNoise(p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        A.HueSaturationValue(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_valid_transforms(img_size=512):
    """
    Return validation/test data augmentation pipeline.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])