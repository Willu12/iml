"""
Module for handling spectrogram datasets, including dataset loading and 
DataLoader preparation for training, validation, and testing.
"""

import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
from PIL import Image
from .config import VALID_ACCESS_LABELS, TRAIN_DIR, VAL_DIR, TEST_DIR


def path_to_label(img_path):
    speaker_id = os.path.basename(img_path).split("_")[0]
    return int(speaker_id in VALID_ACCESS_LABELS)


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

    def __init__(self, directory, transform=None, rgb=False):
        self.files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith(".png")
        ]
        self.transform = transform
        self.img_convert = "RGB" if rgb else "L"

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
        label = path_to_label(img_path)
        image = Image.open(img_path).convert(self.img_convert)

        if self.transform:
            image = self.transform(image)

        return image, label


class RGBSpectrogramDataset(SpectrogramDataset):
    def __init__(self, directory, transform=None):
        SpectrogramDataset.__init__(self, directory, transform, rgb=True)


class BalancedBatchSampler(Sampler):
    """
    Custom sampler to balance classes in each batch using undersampling.

    Parameters:
        dataset (Dataset): The dataset to sample from.
        batch_size (int): Number of samples per batch.

    Attributes:
        indices_per_class (dict): Mapping of class labels to list of indices.
    """

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices_per_class = self._group_indices_by_class()
        self.classes = list(self.indices_per_class.keys())
        self.num_classes = len(self.classes)

        if batch_size % self.num_classes != 0:
            raise ValueError("Batch size must be divisible by the number of classes.")

        self.samples_per_class = batch_size // self.num_classes
        self.min_class_samples = min(
            len(indices) for indices in self.indices_per_class.values()
        )

    def _group_indices_by_class(self):
        """
        Groups dataset indices by class label.

        Returns:
            dict: Mapping of class labels to list of indices.
        """
        indices_per_class = {}
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if label not in indices_per_class:
                indices_per_class[label] = []
            indices_per_class[label].append(idx)
        return indices_per_class

    def __iter__(self):
        """
        Yields indices for batches with balanced classes using undersampling.
        """
        class_cyclers = {
            cls: iter(np.random.permutation(indices))
            for cls, indices in self.indices_per_class.items()
        }

        for _ in range(self.min_class_samples // self.samples_per_class):
            batch = []
            for cls in self.classes:
                for _ in range(self.samples_per_class):
                    try:
                        batch.append(next(class_cyclers[cls]))
                    except StopIteration as exc:
                        # shouldn't happen due to `min_class_samples` - safety assured
                        raise StopIteration(
                            "One of the classes ran out of samples."
                        ) from exc

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        """
        Number of batches in the dataset.
        """
        return self.min_class_samples // self.samples_per_class


def prepare_dataset_loaders(
    transform, batch_size, dataset_class=SpectrogramDataset, balance=True
):
    """
    Creates data loaders for the training, validation, and test datasets.

    Parameters:
        transform (callable): Transformations to apply to the data.
        batch_size (int): Number of samples per batch to load.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Data loaders for training,
        validation, and testing datasets.
    """
    train_dataset = dataset_class(TRAIN_DIR, transform=transform)
    val_dataset = dataset_class(VAL_DIR, transform=transform)
    test_dataset = dataset_class(TEST_DIR, transform=transform)

    if balance:
        sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def prepare_dataset_loader(directory, transform, batch_size,dataset_class=SpectrogramDataset):
    """
    Creates data loader for data from given directory.

    Parameters:
        directory (string): Path to the dataset
        transform (callable): Transformations to apply to the data.
        batch_size (int): Number of samples per batch to load.

    Returns:
        DataLoader: DataLoader for given dataset.
    """
    dataset = dataset_class(directory, transform = transform)
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataLoader