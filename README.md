# SFIS v1 Funding Project: Advanced Potato Leaf Disease Classification

This project, developed under the **SFIS v1 Funding Program**, is a specialized deep learning framework dedicated to the accurate identification and classification of **Potato Leaf Diseases**. It provides a robust benchmarking environment to evaluate state-of-the-art models for agricultural disease detection.

## Project Focus

Specifically engineered for potato health monitoring, this framework addresses the challenges of botanical classification in both controlled laboratory settings and uncontrolled field environments.

- **Automated Data Pipeline**: Stratified dataset splitting with CSV-based management.
- **Robust Image Loading**: Built-in error handling for corrupted or invalid image files.
- **Model Zoo**: Supports multiple architectures via `torchvision`:
  - CNNs: ResNet, EfficientNet, MobileNet, ConvNeXt
  - Transformers: Swin Transformer
- **Detailed Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-Score (Macro/Weighted).
  - Visualizations: Confusion Matrices, Training Curves (Loss/Accuracy), ROC, and PR curves.
  - Performance Benchmarking: FLOPs estimation and inference latency (images/sec).
- **HPC Ready**: Optimized for SLURM environments with automated batch submission scripts.

## Datasets

The project benchmarking is conducted using three primary datasets, covering both controlled and uncontrolled environments:

1.  **Potato Disease Leaf Dataset (PLD)**
    *   Focuses on common potato leaf diseases under controlled conditions.
    *   [Link to Dataset](https://www.kaggle.com/datasets/simranjeet97/potato-disease-leaf-datasetpld)
2.  **Potato Leaf Disease Dataset in Uncontrolled Environment (UE)**
    *   Real-world images captured in natural, non-studio field settings.
    *   [Link - Mendeley Data](https://data.mendeley.com/datasets/6y6k5n3v2b/1)
3.  **PlantVillage Dataset (PV)**
    *   Large-scale botanical dataset covering various crops beyond potatoes.
    *   [Link - Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

## Configuration

> [!IMPORTANT]
> This project contains environment-specific paths that must be updated before running the scripts.

All sensitive paths and user-specific directories in the source code have been labeled as `xx` for security. Please update the following variables in the corresponding files to match your environment:

- **Data Paths**: Update `DEFAULT_ROOT` and `DEFAULT_SPLITS_DIR` in `data_utils.py`.
- **Training Paths**: Update `splits_dir` and `ckpt_dir` in `train_*.py` files.
- **HPC Configuration**: If using SLURM, review the account and partition settings in `run_*.sh` scripts.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/xx-username/sfis-v1-potato-classification.git
   cd sfis-v1-potato-classification
   ```

2. Install dependencies:
   ```bash
   pip install torch torchvision tqdm matplotlib pillow pandas numpy
   ```

## Usage

### 1. Data Preparation
Setup your dataset path in `data_utils.py`, then generate split CSVs:
```bash
python data_utils.py
```

### 2. Single Dataset Training
Update the `splits_dir` in the script, then run:
```bash
python train_resnet18.py
```

### 3. Combined Training (1 & 3)
Specialized scripts for training on merged datasets (PLD + PV) are located in:
- Directory: `train_1and3/` (Update environment paths in scripts before execution)

### 4. Batch Submission (HPC)
Submit tasks to a SLURM cluster:
```bash
bash run_all.sh
```

## Project Structure

- `data_utils.py`: Dataset splitting and loading (Update `xx` paths here).
- `train_*.py`: Model-specific training/evaluation (Update `xx` paths here).
- `run_*.sh`: SLURM batch scripts (Update `xx` account info).
- `outputs/`: Training logs and visualizations.
- `checkpoints/`: Model weights (`.pth`).

## Analytics & Outputs

After training, the `outputs/` folder will contain:
- `training_log_[timestamp].json`: Per-epoch loss and accuracy metrics.
- `test_metrics_[dataset]_[timestamp].json`: Detailed test set evaluation.
- `training_loss_*.png`: Loss curves.
- `training_accuracy_*.png`: Accuracy curves.
- `confusion_matrix_*.png`: Visual representation of model performance per class.
- `roc_micro_*.png` & `pr_micro_*.png`: Model reliability analysis.

## License

[Specify your license, e.g., MIT]
