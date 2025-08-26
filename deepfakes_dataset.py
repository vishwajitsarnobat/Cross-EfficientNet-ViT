# deepfakes_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip, FancyPCA, 
    HueSaturationValue, OneOf, ToGray, Affine,
    ImageCompression, GaussNoise, Resize, Normalize
)
from albumentations.pytorch import ToTensorV2

class DeepFakesDataset(Dataset):
    def __init__(self, frame_label_list, image_size, mode='train'):
        """
        MODIFIED: Now accepts a list of (frame_path, label) tuples
        instead of a giant array of image data. This is memory-safe.
        """
        self.frame_label_list = frame_label_list
        self.image_size = image_size
        self.mode = mode

    def create_train_transforms(self, size):
        """
        Keeps all of your original, advanced data augmentations.
        Adds the necessary Normalization and ToTensor conversion for PyTorch.
        """
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
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-5, 5),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            # Essential steps for PyTorch
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def create_val_transform(self, size):
        """
        Validation transform now correctly includes Normalization and ToTensor.
        """
        return Compose([
            Resize(height=size, width=size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        """
        MODIFIED: Image is now loaded from disk here, one at a time.
        """
        frame_path, label = self.frame_label_list[index]
        
        image = cv2.imread(frame_path)
        if image is None:
            print(f"Warning: Could not read image {frame_path}. Returning a black image.")
            return torch.zeros((3, self.image_size, self.image_size)), torch.tensor(0.0)

        # Albumentations expects RGB images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mode == 'train':
            transform = self.create_train_transforms(self.image_size)
        else:
            transform = self.create_val_transform(self.image_size)

        # The transform pipeline now returns a ready-to-use tensor
        image = transform(image=image)['image']
        
        return image, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.frame_label_list)