import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

sys.path.append('xx/Datasetforpt')
sys.path.append(os.path.dirname(__file__))
from data_utils import split_dataset, create_dataloaders  # noqa: E402
from train_resnet18 import train_model, evaluate_on_test  # noqa: E402


def main():
    root = 'xx/Datasetforpt/dataset_origin/3_plant village/color'
    splits_dir = 'xx/Datasetforpt/dataset_origin/duzixunlian/splits_pv'
    os.makedirs(splits_dir, exist_ok=True)

    need_split = False
    for f in ['train.csv', 'val.csv', 'test.csv']:
        if not os.path.exists(os.path.join(splits_dir, f)):
            need_split = True
            break
    if need_split:
        split_dataset(root, splits_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    dataloaders, class_names = create_dataloaders(splits_dir, batch_size=32, num_workers=4, img_size=224)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Num classes: {len(class_names)} -> {class_names}')

    model = models.resnet101(pretrained=True)
    num_ftrs = model.fc.in_features  # type: ignore[attr-defined]
    model.fc = nn.Linear(num_ftrs, len(class_names))  # type: ignore[attr-defined]
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=150,
    )

    ckpt_dir = 'xx/Datasetforpt/dataset_origin/duzixunlian/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'resnet101_pv_best.pth')
    torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, ckpt_path)
    print(f'Saved best model to {ckpt_path}')

    evaluate_on_test(model, dataloaders, device)


if __name__ == '__main__':
    main()
