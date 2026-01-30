#!/usr/bin/env python3
"""
ConvNeXt Training Script
Supports: convnext_tiny, convnext_small, convnext_base, convnext_large
Usage: python train_convnext.py --model convnext_tiny --dataset pld
"""
import os
import sys
import random
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(__file__))
from trainer import train_model, evaluate_on_test
from data_utils import split_dataset, create_dataloaders


# ===================== Model Configuration =====================
CONVNEXT_MODELS = {
    'convnext_tiny': models.convnext_tiny,
    'convnext_small': models.convnext_small,
    'convnext_base': models.convnext_base,
    'convnext_large': models.convnext_large,
}


class SafeImageFolder(datasets.ImageFolder):
    """Custom ImageFolder that automatically skips corrupted images."""
    def __getitem__(self, index: int):
        try:
            return super().__getitem__(index)
        except (IOError, OSError) as e:
            print(f"[Warning] Error loading image at index {index}: {e}. Replacing with random sample.")
            new_index = random.randint(0, len(self) - 1)
            return self.__getitem__(new_index)


def get_transforms(img_size: int = 224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    return train_tf, eval_tf


def create_pld_dataloaders(pld_root: str, batch_size: int, num_workers: int, img_size: int) -> Dict[str, DataLoader]:
    train_tf, eval_tf = get_transforms(img_size)
    train_ds = SafeImageFolder(os.path.join(pld_root, 'Training'), transform=train_tf)
    val_ds = SafeImageFolder(os.path.join(pld_root, 'Validation'), transform=eval_tf)
    test_ds = SafeImageFolder(os.path.join(pld_root, 'Testing'), transform=eval_tf)
    print(f"Classes: {train_ds.classes}")
    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    }


def main():
    parser = argparse.ArgumentParser(description='ConvNeXt Training')
    parser.add_argument('--model', type=str, default='convnext_tiny', 
                        choices=list(CONVNEXT_MODELS.keys()),
                        help='Model variant')
    parser.add_argument('--dataset', type=str, default='pld', 
                        choices=['pld', 'ue', 'pv'],
                        help='Dataset to use: pld, ue (uncontrolled env), pv (plant village)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    dataset_paths = 'xx/Datasetforpt/train_1and3/dataset'
    splits_dir = 'xx/Datasetforpt/train_1and3/splits'
    if not os.path.exists(os.path.join(splits_dir, 'train.csv')):
        os.makedirs(splits_dir, exist_ok=True)
        split_dataset(dataset_paths, splits_dir)
    dataloaders, class_names = create_dataloaders(splits_dir, args.batch_size, args.num_workers)
    num_classes = len(class_names)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Using device: {device}")
    print(f"Num classes: {num_classes} -> {class_names}")

    # Create model
    model = CONVNEXT_MODELS[args.model](pretrained=True)
    # The classifier head for ConvNeXt is model.classifier[2]
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=args.epochs)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.model}_best.pth')
    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, ckpt_path)
    print(f"Saved best model to {ckpt_path}")

    evaluate_on_test(model, dataloaders, device, model_name="ConvNeXt")


if __name__ == '__main__':
    main()
