import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import VALID_ACCESS_LABELS, TRAIN_DIR, VAL_DIR, TEST_DIR


class SpectrogramDataset(Dataset):
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
        img_path = self.files[idx]
        speaker_id = img_path.split("/")[-1].split("_")[0]
        label = int(speaker_id in VALID_ACCESS_LABELS)
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_dataset_loaders(transform, batch_size):
    """
    Creates data loaders for training, validation and testing datasets.

    Parameters:
        transform: Combination of transforms to perform on data
        batch_size (int): Size of data batch to load

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Training, validation and testing data loaders
    """
    train_dataset = SpectrogramDataset(TRAIN_DIR, transform=transform)
    val_dataset = SpectrogramDataset(VAL_DIR, transform=transform)
    test_dataset = SpectrogramDataset(TEST_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
