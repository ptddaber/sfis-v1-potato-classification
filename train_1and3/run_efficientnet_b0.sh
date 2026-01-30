#!/bin/bash
#SBATCH -p gpu-a100-8
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --job-name=effnet_b0
#SBATCH --output=xx/logs_lau/effnet_b0_%j.out
#SBATCH --error=xx/logs_lau/effnet_b0_%j.err
#SBATCH --mem=64G

cd xx/Datasetforpt/train_1and3

python train_efficientnet.py --model efficientnet_b0
