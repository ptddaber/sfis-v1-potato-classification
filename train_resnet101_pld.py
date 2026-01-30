import os
import sys
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

# Allow importing existing training/evaluation functions from project root
sys.path.append('xx/Datasetforpt')
sys.path.append(os.path.dirname(__file__))
from train_resnet18 import train_model, evaluate_on_test  # noqa: E402


class SafeImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder that automatically skips corrupted images (replaces with random sample).
    """
    def __getitem__(self, index: int):
        try:
            return super().__getitem__(index)
        except (IOError, OSError) as e:
            print(f"[Warning] Error loading image at index {index}: {e}. Replacing with random sample.")
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)


def create_pld_dataloaders(
    pld_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 224,
) -> Dict[str, DataLoader]:
    """
    PLD dataset already has Training/Validation/Testing folders.
    Use ImageFolder to build three DataLoaders directly.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # Random rotation ±15°
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(pld_root, 'Training')
    val_dir = os.path.join(pld_root, 'Validation')
    test_dir = os.path.join(pld_root, 'Testing')

    train_ds = SafeImageFolder(train_dir, transform=train_tf)
    val_ds = SafeImageFolder(val_dir, transform=eval_tf)
    test_ds = SafeImageFolder(test_dir, transform=eval_tf)

    print(f"Classes: {train_ds.classes}")

    dataloaders: Dict[str, DataLoader] = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }
    return dataloaders


def main():
    pld_root = 'xx/Datasetforpt/dataset_origin/1_PLD_3_Classes_256'
    batch_size = 32
    num_workers = 4
    num_epochs = 150

    dataloaders = create_pld_dataloaders(pld_root, batch_size=batch_size, num_workers=num_workers, img_size=224)
    class_names = dataloaders['train'].dataset.classes  # type: ignore[attr-defined]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Num classes: {len(class_names)} -> {class_names}")

    # ResNet101 pretrained model
    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(num_ftrs, len(class_names))  # type: ignore[attr-defined]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=num_epochs,
    )

    ckpt_dir = 'xx/Datasetforpt/dataset_origin/duzixunlian/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'resnet101_pld_best.pth')

    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, ckpt_path)
    print(f"Saved best model to {ckpt_path}")

    evaluate_on_test(model, dataloaders, device)


if __name__ == '__main__':
    main()
