import os
import shutil
import random
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image


CLASSES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def prepare_dataset(raw_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    random.seed(seed)

    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        cls_dir = raw_dir / cls
        if not cls_dir.is_dir():
            continue
        images = [f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        for img in images[:n_train]:
            shutil.copy2(img, output_dir / "train" / cls / img.name)
        for img in images[n_train:n_train + n_val]:
            shutil.copy2(img, output_dir / "val" / cls / img.name)
        for img in images[n_train + n_val:]:
            shutil.copy2(img, output_dir / "test" / cls / img.name)

    print(f"Dataset split saved to {output_dir}")


def get_dataloaders(data_dir, batch_size=32, num_workers=0):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=val_transform)
    test_ds = datasets.ImageFolder(str(data_dir / "test"), transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
