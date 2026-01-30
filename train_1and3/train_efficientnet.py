#!/usr/bin/env python3
"""
EfficientNet V1 & V2 Training Script
Supports: efficientnet_b0~b7, efficientnet_v2_s/m/l
Usage: python train_efficientnet.py --model efficientnet_b0 --dataset pld
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
EFFICIENTNET_MODELS = {
    # EfficientNet V1
    'efficientnet_b0': models.efficientnet_b0,
    'efficientnet_b1': models.efficientnet_b1,
    'efficientnet_b2': models.efficientnet_b2,
    'efficientnet_b3': models.efficientnet_b3,
    'efficientnet_b4': models.efficientnet_b4,
    'efficientnet_b5': models.efficientnet_b5,
    'efficientnet_b6': models.efficientnet_b6,
    'efficientnet_b7': models.efficientnet_b7,
    # EfficientNet V2
    'efficientnet_v2_s': models.efficientnet_v2_s,
    'efficientnet_v2_m': models.efficientnet_v2_m,
    'efficientnet_v2_l': models.efficientnet_v2_l,
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
    parser = argparse.ArgumentParser(description='EfficientNet Training')
    parser.add_argument('--model', type=str, default='efficientnet_b0', 
                        choices=list(EFFICIENTNET_MODELS.keys()),
                        help='Model variant')
    parser.add_argument('--dataset', type=str, default='pld', 
                        choices=['pld', 'ue', 'pv'],
                        help='Dataset to use: pld, ue (uncontrolled env), pv (plant village)')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Dataset root
    """
    dataset_root = 'xx/Datasetforpt/train_1and3/dataset'
    dataset_paths = {
        'pld': os.path.join(dataset_root, '1_PLD_3_Classes_256'),
        'ue': os.path.join(dataset_root, '2_Potato Leaf Disease Dataset in Uncontrolled Environment'),
        'pv': os.path.join(dataset_root, '3_plant village', 'color'),
    }
    
    # Create DataLoader
    if args.dataset == 'pld':
        dataloaders = create_pld_dataloaders(dataset_paths['pld'], args.batch_size, args.num_workers, img_size=224)
        class_names = dataloaders['train'].dataset.classes
    else:
        splits_dir = os.path.join(os.path.dirname(__file__), f'splits_{args.dataset}')
        if not os.path.exists(os.path.join(splits_dir, 'train.csv')):
            os.makedirs(splits_dir, exist_ok=True)
            split_dataset(dataset_paths[args.dataset], splits_dir)
        dataloaders, class_names = create_dataloaders(splits_dir, args.batch_size, args.num_workers)
    """
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
    model = EFFICIENTNET_MODELS[args.model](pretrained=True)
    # The classifier head for EfficientNet is model.classifier[1]
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, device, num_epochs=args.epochs)

    # Save model
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'{args.model}_best.pth')
    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, ckpt_path)
    print(f"Saved best model to {ckpt_path}")

    evaluate_on_test(model, dataloaders, device, model_name=args.model)


if __name__ == '__main__':
    main()
