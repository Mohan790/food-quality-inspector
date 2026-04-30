"""
dataset.py — Food Quality Dataset Loader
Loads fresh/rotten food images and prepares them for training.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


# ── Transforms ─────────────────────────────────────────────────────────────────
def get_transforms(split: str = "train"):
    """
    Train: augmentation to make model robust.
    Val: only resize + normalize for fair evaluation.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std  = [0.229, 0.224, 0.225]  # ImageNet std

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Dataset class ──────────────────────────────────────────────────────────────
class FoodQualityDataset(Dataset):
    """
    Custom PyTorch Dataset for fresh/rotten food images.
    Scans folder names — if folder starts with 'fresh' → label 0
                       — if folder starts with 'rotten' → label 1
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# ── DataLoader factory ─────────────────────────────────────────────────────────
def build_dataloaders(data_root: str, batch_size: int = 32, val_split: float = 0.2):
    """
    Scans data_root for class sub-folders.
    Folders starting with 'fresh' → label 0 (Fresh)
    Folders starting with 'rotten' → label 1 (Rotten)

    Returns: train_loader, val_loader, class_names
    """
    records = []
    class_names = ["Fresh", "Rotten"]

    for folder in sorted(os.listdir(data_root)):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue

        # Assign label based on folder name
        if folder.lower().startswith("fresh"):
            label = 0
        elif folder.lower().startswith("rotten") or folder.lower().startswith("stale"):
            label = 1
        else:
            continue

        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "image_path": os.path.join(folder_path, fname),
                    "label": label,
                    "folder": folder,
                })

    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Total images found: {len(df)}")
    print(f"Fresh: {len(df[df.label==0])} | Rotten: {len(df[df.label==1])}")

    # Train / Val split
    split_idx  = int(len(df) * (1 - val_split))
    train_df   = df.iloc[:split_idx]
    val_df     = df.iloc[split_idx:]

    train_ds = FoodQualityDataset(train_df, transform=get_transforms("train"))
    val_ds   = FoodQualityDataset(val_df,   transform=get_transforms("val"))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names