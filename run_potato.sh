#!/bin/bash
#SBATCH -p gpu-a100-8                  # Partition (A100 8-card nodes)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --nodes=1                      # Single node
#SBATCH --job-name=pld101              # Job name
#SBATCH --output=xx/logs_multi/pld101_%j.out
#SBATCH --error=xx/logs_multi/pld101_%j.err
#SBATCH --mem=64G



# Run resnet101 training script
python train_resnet101_pld.py
