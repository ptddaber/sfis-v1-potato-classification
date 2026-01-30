import os
import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from data_utils import create_dataloaders
from tqdm import tqdm
import time
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    num_epochs: int = 20,
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    hist_epoch = []
    hist_train_loss = []
    hist_train_acc = []
    hist_val_loss = []
    hist_val_acc = []
    hist_lr = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        tr_loss = 0.0
        tr_acc = 0.0
        vl_loss = 0.0
        vl_acc = 0.0
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in tqdm(
                dataloaders[phase],
                desc=f"{phase} {epoch + 1}/{num_epochs}",
                leave=False,
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Gradient initialization
                optimizer.zero_grad()
                
                # Enable gradients for training phase
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)             # Forward pass
                    _, preds = torch.max(outputs, 1)    # Predict classes
                    loss = criterion(outputs, labels)   # Calculate loss

                    if phase == "train":
                        loss.backward()          # Backward pass
                        optimizer.step()         # Update model parameters

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                total += labels.size(0)

            epoch_loss = running_loss / total if total > 0 else 0.0
            epoch_acc = running_corrects / total if total > 0 else 0.0

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step()    # Adjust learning rate
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                vl_loss = epoch_loss
                vl_acc = epoch_acc
            else:
                tr_loss = epoch_loss
                tr_acc = epoch_acc

        print()
        hist_epoch.append(epoch + 1)
        hist_train_loss.append(tr_loss)
        hist_train_acc.append(tr_acc)
        hist_val_loss.append(vl_loss)
        hist_val_acc.append(vl_acc)
        hist_lr.append(optimizer.param_groups[0]["lr"])

    print(f"Best val Acc: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    log = {
        "epoch": hist_epoch,
        "train_loss": hist_train_loss,
        "train_acc": hist_train_acc,
        "val_loss": hist_val_loss,
        "val_acc": hist_val_acc,
        "lr": hist_lr,
        "best_val_acc": best_acc,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(out_dir, "last_training_log.json"), "w") as f:
        json.dump(log, f)
    with open(os.path.join(out_dir, f"training_log_{ts}.json"), "w") as f:
        json.dump(log, f)

    # Plot training/validation loss and accuracy curves
    fig1 = plt.figure(figsize=(7, 5))
    plt.plot(hist_epoch, hist_train_loss, label="Train Loss")
    plt.plot(hist_epoch, hist_val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = os.path.join(out_dir, f"training_loss_{ts}.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=200)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(hist_epoch, hist_train_acc, label="Train Accuracy")
    plt.plot(hist_epoch, hist_val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    acc_path = os.path.join(out_dir, f"training_accuracy_{ts}.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.close(fig2)

    cap_path = os.path.join(out_dir, "captions.txt")
    with open(cap_path, "a") as fcap:
        fcap.write("Training and validation loss over epochs. Lower is better.\n")
        fcap.write("Training and validation accuracy over epochs. Higher is better.\n")
    return model


def evaluate_on_test(
    model: nn.Module,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: torch.device,
):
    model.eval()
    class_names = None
    if "train" in dataloaders and hasattr(dataloaders["train"].dataset, "classes"):
        class_names = list(dataloaders["train"].dataset.classes)  # type: ignore[attr-defined]
    running_corrects = 0
    total = 0
    y_true = []
    y_pred = []
    start_time = time.time()
    infer_time = 0.0

    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time += time.time() - t0
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data).item()
            total += labels.size(0)
            y_true.extend(labels.view(-1).detach().cpu().tolist())
            y_pred.extend(preds.view(-1).detach().cpu().tolist())
            # collect probabilities for ROC/PR curves
            probs = torch.softmax(outputs, dim=1)
            if 'probs_all' not in locals():
                probs_all = probs.detach().cpu()
            else:
                probs_all = torch.cat([probs_all, probs.detach().cpu()], dim=0)

    acc = running_corrects / total if total > 0 else 0.0

    num_classes = max(max(y_true, default=-1), max(y_pred, default=-1)) + 1 if total > 0 else 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    diag = torch.diag(cm).to(torch.float64)
    support = cm.sum(dim=1).to(torch.float64)
    pred_sum = cm.sum(dim=0).to(torch.float64)

    precision_c = torch.where((pred_sum > 0), diag / torch.clamp(pred_sum, min=1.0), torch.zeros_like(diag))
    recall_c = torch.where((support > 0), diag / torch.clamp(support, min=1.0), torch.zeros_like(diag))
    f1_c = torch.where(
        (precision_c + recall_c) > 0,
        2 * precision_c * recall_c / torch.clamp(precision_c + recall_c, min=1e-12),
        torch.zeros_like(precision_c),
    )

    macro_p = precision_c.mean().item() if num_classes > 0 else 0.0
    macro_r = recall_c.mean().item() if num_classes > 0 else 0.0
    macro_f1 = f1_c.mean().item() if num_classes > 0 else 0.0

    weights = support / support.sum() if support.sum() > 0 else torch.zeros_like(support)
    weighted_p = (precision_c * weights).sum().item() if num_classes > 0 else 0.0
    weighted_r = (recall_c * weights).sum().item() if num_classes > 0 else 0.0
    weighted_f1 = (f1_c * weights).sum().item() if num_classes > 0 else 0.0

    total_time = time.time() - start_time
    avg_latency = infer_time / total if total > 0 else 0.0
    ips = total / infer_time if infer_time > 0 else 0.0

    total_params = sum(p.numel() for p in model.parameters())
    param_size_mb = total_params * 4 / (1024 ** 2)

    def _estimate_flops(m: nn.Module, input_size=(1, 3, 224, 224)):
        flops = {"total": 0}

        def conv_hook(self, inp, out):
            out = out[0] if isinstance(out, (tuple, list)) else out
            batch = out.shape[0]
            out_c = out.shape[1]
            out_h = out.shape[2]
            out_w = out.shape[3]
            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels // self.groups)
            output_elements = batch * out_h * out_w * out_c
            ops = kernel_ops * output_elements
            if self.bias is not None:
                ops += output_elements
            flops["total"] += int(ops)

        def linear_hook(self, inp, out):
            out = out[0] if isinstance(out, (tuple, list)) else out
            batch = out.shape[0]
            ops = self.in_features * self.out_features
            if self.bias is not None:
                ops += self.out_features
            flops["total"] += int(batch * ops)

        handles = []
        for mod in m.modules():
            if isinstance(mod, nn.Conv2d):
                handles.append(mod.register_forward_hook(conv_hook))
            elif isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(linear_hook))

        m_device = next(m.parameters()).device
        dummy = torch.zeros(input_size, device=m_device)
        m_prev = m.training
        m.eval()
        with torch.no_grad():
            m(dummy)
        if m_prev:
            m.train()
        for h in handles:
            h.remove()
        return flops["total"] * 2

    approx_flops = _estimate_flops(model, (1, 3, 224, 224))

    print(f"Test Acc: {acc:.4f}")
    print("Precision (macro): {:.4f}  Recall (macro): {:.4f}  F1 (macro): {:.4f}".format(macro_p, macro_r, macro_f1))
    print("Precision (weighted): {:.4f}  Recall (weighted): {:.4f}  F1 (weighted): {:.4f}".format(weighted_p, weighted_r, weighted_f1))
    if class_names and len(class_names) == num_classes:
        for i in range(num_classes):
            print("Class {}: P={:.4f} R={:.4f} F1={:.4f} Support={}".format(class_names[i], precision_c[i].item(), recall_c[i].item(), f1_c[i].item(), int(support[i].item())))
    else:
        for i in range(num_classes):
            print("Class {}: P={:.4f} R={:.4f} F1={:.4f} Support={}".format(i, precision_c[i].item(), recall_c[i].item(), f1_c[i].item(), int(support[i].item())))
    print("Params: {:,} ({:.2f} MB)".format(total_params, param_size_mb))
    print("FLOPs (approx): {:.2f} GFLOPs @224".format(approx_flops / 1e9))
    print("Inference time: total {:.2f}s, net forward {:.2f}s, latency {:.4f}s/img, {:.2f} img/s".format(total_time, infer_time, avg_latency, ips))

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    ds_name = None
    if "train" in dataloaders:
        ds = dataloaders["train"].dataset
        if hasattr(ds, "root"):
            ds_name = os.path.basename(os.path.dirname(getattr(ds, "root")))
        elif hasattr(ds, "csv_path"):
            ds_name = os.path.basename(os.path.dirname(getattr(ds, "csv_path")))
    if ds_name is None:
        ds_name = "dataset"
    ts = time.strftime("%Y%m%d_%H%M%S")
    metrics = {
        "accuracy": acc,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_weighted": weighted_p,
        "recall_weighted": weighted_r,
        "f1_weighted": weighted_f1,
        "per_class": [
            {
                "class": (class_names[i] if class_names and i < len(class_names) else str(i)),
                "precision": float(precision_c[i].item()),
                "recall": float(recall_c[i].item()),
                "f1": float(f1_c[i].item()),
                "support": int(support[i].item()),
            }
            for i in range(num_classes)
        ],
        "confusion_matrix": cm.tolist(),
        "params": int(total_params),
        "param_size_mb": float(param_size_mb),
        "approx_flops": int(approx_flops),
        "avg_latency_sec_per_image": float(avg_latency),
        "images_per_second": float(ips),
        "dataset": ds_name,
    }
    with open(os.path.join(out_dir, "last_test_metrics.json"), "w") as f:
        json.dump(metrics, f)
    with open(os.path.join(out_dir, f"test_metrics_{ds_name}_{ts}.json"), "w") as f:
        json.dump(metrics, f)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm.cpu().numpy(), interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix (Test Set) - {ds_name}")
    plt.colorbar()
    tick_marks = range(num_classes)
    labels = class_names if class_names and len(class_names) == num_classes else [str(i) for i in tick_marks]
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, f"confusion_matrix_{ds_name}_{ts}.png")
    plt.savefig(cm_path, dpi=200)
    plt.close(fig)

    cap_path = os.path.join(out_dir, "captions.txt")
    with open(cap_path, "a") as f:
        f.write(f"Confusion matrix on the test set for {ds_name}. Rows are true classes; columns are predicted classes.\n")

    # Micro-averaged ROC and PR curves (one-vs-rest)
    if num_classes > 1 and total > 0 and 'probs_all' in locals():
        y_true_bin = torch.zeros((total, num_classes), dtype=torch.float32)
        idx = torch.arange(total)
        y_true_tensor = torch.tensor(y_true, dtype=torch.long)
        y_true_bin[idx, y_true_tensor] = 1.0
        scores = probs_all.to(torch.float32)

        # flatten for micro-average
        y_flat = y_true_bin.view(-1).numpy()
        s_flat = scores.view(-1).numpy()

        # sort by score desc
        order = s_flat.argsort()[::-1]
        y_sorted = y_flat[order]
        s_sorted = s_flat[order]

        # ROC computation
        P = y_sorted.sum()
        N = len(y_sorted) - P
        tp = 0.0
        fp = 0.0
        tpr = [0.0]
        fpr = [0.0]
        prev_score = None
        for i in range(len(y_sorted)):
            if prev_score is None or s_sorted[i] != prev_score:
                tpr.append(tp / P if P > 0 else 0.0)
                fpr.append(fp / N if N > 0 else 0.0)
                prev_score = s_sorted[i]
            if y_sorted[i] > 0.5:
                tp += 1.0
            else:
                fp += 1.0
        tpr.append(1.0)
        fpr.append(1.0)

        # AUC (trapezoidal)
        import numpy as np
        fpr_arr = np.array(fpr)
        tpr_arr = np.array(tpr)
        # ensure sorted by fpr
        order2 = np.argsort(fpr_arr)
        fpr_arr = fpr_arr[order2]
        tpr_arr = tpr_arr[order2]
        auc = np.trapz(tpr_arr, fpr_arr)

        fig_roc = plt.figure(figsize=(6, 5))
        plt.plot(fpr_arr, tpr_arr, label=f"Micro ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Micro-averaged ROC Curve - {ds_name}")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        roc_path = os.path.join(out_dir, f"roc_micro_{ds_name}_{ts}.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close(fig_roc)

        # PR computation (micro)
        cum_tp = np.cumsum(y_sorted)
        cum_fp = np.cumsum(1 - y_sorted)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        recall = cum_tp / np.maximum(P, 1e-12)
        # prepend (0,1)
        precision = np.concatenate([[1.0], precision])
        recall = np.concatenate([[0.0], recall])
        # AP (area under PR)
        ap = np.trapz(precision, recall)

        fig_pr = plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"Micro PR (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Micro-averaged Precision-Recall Curve - {ds_name}")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        pr_path = os.path.join(out_dir, f"pr_micro_{ds_name}_{ts}.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=200)
        plt.close(fig_pr)

        with open(cap_path, "a") as f:
            f.write(f"Micro-averaged ROC curve on the test set for {ds_name}. The ROC shows TPR vs FPR with AUC summarizing performance.\n")
            f.write(f"Micro-averaged Precision-Recall curve on the test set for {ds_name}. The AP summarizes precision trade-offs across recalls.\n")
    return acc


def main():
    splits_dir = "xx/Datasetforpt/splits"
    batch_size = 32
    num_workers = 4
    num_epochs = 150

    dataloaders, class_names = create_dataloaders(
        splits_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=224,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Num classes: {len(class_names)} -> {class_names}")

    # ResNet18 ResNet18 pretrained model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=num_epochs,
    )

    ckpt_dir = "xx/Datasetforpt/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "resnet18_best.pth")

    torch.save({"model_state_dict": model.state_dict(), "class_names": class_names}, ckpt_path)
    print(f"Saved best model to {ckpt_path}")

    evaluate_on_test(model, dataloaders, device)


if __name__ == "__main__":
    main()
