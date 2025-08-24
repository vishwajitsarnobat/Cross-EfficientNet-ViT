# deepfakes_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, 
    HueSaturationValue, OneOf, ToGray, Affine,
    ImageCompression, GaussNoise, Resize
)

class DeepFakesDataset(Dataset):
    def __init__(self, images, labels, image_size, mode='train'):
        self.x = images
        self.y = torch.from_numpy(labels)
        self.image_size = image_size
        self.mode = mode
        self.n_samples = images.shape[0]

    def create_train_transforms(self, size):
        return Compose([
            Resize(height=size, width=size),
            ImageCompression(quality_range=(60, 100), p=0.2),
            GaussNoise(p=0.3),
            HorizontalFlip(),
            OneOf([
                RandomBrightnessContrast(),
                FancyPCA(),
                HueSaturationValue()
            ], p=0.4),
            ToGray(p=0.2),
            Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # shift ±10%
                scale=(0.8, 1.2),                                        # zoom out/in
                rotate=(-5, 5),                                          # small rotations
                border_mode=cv2.BORDER_CONSTANT,                         # padding mode
                fill=0,                                                  # fill empty pixels with black
                fill_mask=0,                                             # fill empty mask pixels with black
                p=0.5,
            )
        ])

    def create_val_transform(self, size):
        # The validation transform is now just a simple, direct resize.
        return Compose([
            Resize(height=size, width=size),
        ])

    def __getitem__(self, index):
        image = self.x[index]

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)

        image = transform(image=image)['image']
        
        # The original code used torch.tensor(), but this can be slow.
        # Let's convert to a numpy array first for efficiency if needed, though ToTensor would be better.
        # For now, let's keep it simple.
        return torch.from_numpy(image).float(), self.y[index]

    def __len__(self):
        return self.n_samples