"""
Module for handling spectrogram datasets, including dataset loading and 
DataLoader preparation for training, validation, and testing.
"""

import os
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from .config import VALID_ACCESS_LABELS, TRAIN_DIR, VAL_DIR, TEST_DIR

class SpectrogramDataset(Dataset):
    """
    Dataset class for loading spectrogram images and assigning labels.

    Parameters:
        directory (str): Path to the directory containing spectrogram images.
        transform (callable, optional): Transformation to apply to images.

    Attributes:
        files (list[str]): List of paths to spectrogram image files.
        transform (callable): Transformations to apply to images.
    """
    def __init__(self, directory, transform=None):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".png")
        ]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label.

        Parameters:
            idx (int): Index of the image in the dataset.

        Returns:
            tuple: A tuple containing the transformed image and its label.
        """
        img_path = self.files[idx]
        speaker_id = img_path.split("/")[-1].split("_")[0]
        label = int(speaker_id in VALID_ACCESS_LABELS)

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_dataset_loaders(transform, batch_size):
    """
    Creates data loaders for the training, validation, and test datasets.

    Parameters:
        transform (callable): Transformations to apply to the data.
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training, 
        validation, and testing datasets.
    """
    train_dataset = SpectrogramDataset(TRAIN_DIR, transform=transform)
    val_dataset = SpectrogramDataset(VAL_DIR, transform=transform)
    test_dataset = SpectrogramDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
