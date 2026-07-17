"""One-time script to split the PlantVillage dataset into train/val folders."""

import random
import shutil
from pathlib import Path

# UPDATE THIS to match where you extracted the dataset
SOURCE_DIR = Path(r"C:\Users\USER\Downloads\archive\PlantVillage")

DEST_TRAIN = Path("data/train")
DEST_VAL = Path("data/val")
VAL_SPLIT = 0.2  # 20% of images go to validation
SEED = 42

random.seed(SEED)

class_folders = [f for f in SOURCE_DIR.iterdir() if f.is_dir() and f.name != "PlantVillage"]
print(f"Found {len(class_folders)} classes")

for class_folder in class_folders:
    images = list(class_folder.glob("*.*"))
    random.shuffle(images)

    split_idx = int(len(images) * (1 - VAL_SPLIT))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    train_dest = DEST_TRAIN / class_folder.name
    val_dest = DEST_VAL / class_folder.name
    train_dest.mkdir(parents=True, exist_ok=True)
    val_dest.mkdir(parents=True, exist_ok=True)

    for img in train_images:
        shutil.copy(img, train_dest / img.name)
    for img in val_images:
        shutil.copy(img, val_dest / img.name)

    print(f"{class_folder.name}: {len(train_images)} train, {len(val_images)} val")

print("Done splitting dataset.")