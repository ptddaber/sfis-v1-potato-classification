import os
import csv
import random
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Default paths, can be modified as needed
DEFAULT_ROOT = "xx/Datasetforpt/Merged_Dataset"
DEFAULT_SPLITS_DIR = "xx/Datasetforpt/splits"


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def split_dataset(
    root_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Stratify by category, split the class folders under root_dir into train/val/test CSVs.

    CSV format: path,label
    path uses absolute path, making it easy to run training scripts from any location.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    splits = {"train": [], "val": [], "test": []}  # type: ignore[var-annotated]

    class_names = [
        d
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
    ]

    for cls in class_names:
        class_dir = os.path.join(root_dir, cls)
        images = [
            os.path.join(class_dir, f)
            for f in sorted(os.listdir(class_dir))
            if _is_image_file(f)
        ]
        if not images:
            print(f"[Warning] No images found in class folder: {class_dir}")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_imgs = images[:n_train]
        val_imgs = images[n_train : n_train + n_val]
        test_imgs = images[n_train + n_val :]

        for p in train_imgs:
            splits["train"].append((os.path.abspath(p), cls))
        for p in val_imgs:
            splits["val"].append((os.path.abspath(p), cls))
        for p in test_imgs:
            splits["test"].append((os.path.abspath(p), cls))

        print(
            f"Class {cls}: total={n}, train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}"
        )

    # Write CSV
    for split_name, rows in splits.items():
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(rows)
        print(f"Saved {split_name} split to {csv_path} (n={len(rows)})")


class PotatoLeafDataset(Dataset):
    """General Dataset based on CSV (path,label)."""

    def __init__(self, csv_path: str, transform=None, classes=None, class_to_idx=None) -> None:
        self.csv_path = csv_path
        self.transform = transform

        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx = {}

        if classes is not None and class_to_idx is not None:
            self.classes = list(classes)
            self.class_to_idx = dict(class_to_idx)

        self._load_csv()

    def _load_csv(self) -> None:
        paths: List[str] = []
        labels_str: List[str] = []

        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row["path"]
                label = row["label"]
                paths.append(path)
                labels_str.append(label)

        if not self.classes:
            self.classes = sorted(set(labels_str))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for path, label_str in zip(paths, labels_str):
            if label_str not in self.class_to_idx:
                continue
            self.samples.append((path, self.class_to_idx[label_str]))

        print(
            f"Loaded {len(self.samples)} samples from {self.csv_path}. "
            f"Num classes={len(self.classes)}"
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, idx: int):  # type: ignore[override]
        path, target = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except (OSError, UserWarning) as e:
            # Catch exceptions for corrupted images
            print(f"[Warning] Loading error for {path}: {e}. Replacing with random sample.")
            # Pick a random sample instead
            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def create_dataloaders(
    splits_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
):
    """Create DataLoader based on train/val/test CSV under splits_dir."""
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    test_csv = os.path.join(splits_dir, "test.csv")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # Random rotation ±15°
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = PotatoLeafDataset(train_csv, transform=train_transform)
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    val_dataset = PotatoLeafDataset(
        val_csv,
        transform=eval_transform,
        classes=class_names,
        class_to_idx=class_to_idx,
    )
    test_dataset = PotatoLeafDataset(
        test_csv,
        transform=eval_transform,
        classes=class_names,
        class_to_idx=class_to_idx,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders, class_names


if __name__ == "__main__":
    # Perform dataset split when run directly
    print(
        f"Splitting dataset from {DEFAULT_ROOT} into {DEFAULT_SPLITS_DIR} (train/val/test CSV)"
    )
    split_dataset(DEFAULT_ROOT, DEFAULT_SPLITS_DIR)
