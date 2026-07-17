"""Data loading utilities. Expects ImageFolder format:
data/train/ClassName/*.jpg, data/val/ClassName/*.jpg
"""

import json
from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src import config


def get_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD),
    ])


def get_dataloaders(
    train_dir: Path = config.TRAIN_DIR,
    val_dir: Path = config.VAL_DIR,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, list]:
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Expected data at {train_dir} and {val_dir}. See README for setup."
        )

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset.classes


def save_class_names(class_names: list, path: Path = config.CLASS_NAMES_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(class_names, f, indent=2)


def load_class_names(path: Path = config.CLASS_NAMES_PATH) -> list:
    with open(path) as f:
        return json.load(f)